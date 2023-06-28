import open3d as o3d
import numpy as np
import copy
import random
import sys
from UtilityOpen3d import read_mesh, mesh_to_point_cloud, distance_mesh_points, get_tri_meshes, points_to_pointcloud, distance_point_clouds, simplify_mesh, visualise_tri_faces, read_point_cloud
import math

from skspatial.objects import Plane, Vector, Point, Line
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation
from typing import List
import time

# EPSILON = sys.float_info.epsilon
EPSILON = 0.1

# Could output the Local contact probability in 2 ways
#       The mean of the locomo probabilities
#       Using a normalizing factor which is the maximum number of points in the neighborhood of the finger
def LoCoMo_remastered(  sampling_method,
                        object_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
                        fingers_model: o3d.cpu.pybind.geometry.TriangleMesh,
                        faces_models: List[o3d.cpu.pybind.geometry.TriangleMesh],
                        sphere_radius: float=10,
                        poses_to_sample: int=10,
                        distance: float=10
                        ):
    
    print('soo')
    final_transformations = []
    final_poses = []
    locomo_probabilities = []
    
    zero_moment_shifts = list(map(lambda x: zero_moment_shift(x, sphere_radius, np.asarray(object_point_cloud.points)), object_point_cloud.points))

    for finger_face in faces_models:
        print('face=', finger_face)
        p = 0
        for point in list(object_point_cloud.points):
            point_index = np.where(np.all(np.array(object_point_cloud.points) == (point), axis=1))

            normal = np.array(object_point_cloud.normals)[point_index]

            # poses, T = sample_finger_poses_opposite(point, normal, mesh_face, mesh, amount_poses=10)
            poses, transformations, faces_oriented = sampling_method(point, normal[0], finger_face, fingers_model, poses_to_sample)

            print('     point=', p, '/', len(object_point_cloud.points))
            p +=1
            normalizing_factor = 0 

            for pose, t, face_pose in zip(poses, transformations, faces_oriented):
                points_within_d = select_within_distance(face_pose, object_point_cloud, distance)
                n_points = len(points_within_d)
                if n_points>normalizing_factor:
                    normalizing_factor = n_points
                locomo_prob = []

                for point_d in points_within_d[:round(len(points_within_d)/4)]:
                    point_index = np.where(np.all(np.array(object_point_cloud.points) == (point_d), axis=1))
                    zms1 = zero_moment_shifts[point_index[0][0]]

                    projected_point = project_point_on_surface(face_pose, point_d)
                    zms2 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=object_point_cloud.points)
                    error = zms1 - zms2
                    if len(points_within_d)>2:
                        # Sigma = np.cov(points_within_d, rowvar=False)
                        pc = points_to_pointcloud(points_within_d)

                        mu, Sigma = pc.compute_mean_and_covariance()

                        # The covariance matrix muyst be symmetric positive definite
                        # if np.linalg.det(Sigma) > 0:
                        # if np.all(np.linalg.eigvals(Sigma)) > 0 and  np.allclose(Sigma, Sigma.T):

                        #https://stackoverflow.com/a/41518536
                        # min_eig = np.min(np.real(np.linalg.eigvals(Sigma)))
                        # if min_eig < 0:
                        #     Sigma -= 10*min_eig * np.eye(*Sigma.shape)
                        #     locomo = locomo_probability_function(X=error, mu=np.zeros(3), Sigma=Sigma)
                        #     if not math.isnan(locomo):
                        #         locomo_prob.append(locomo)

                        if np.linalg.det(Sigma) > 0 and np.all(np.linalg.eigvalsh(Sigma) > 0) and np.allclose(Sigma, Sigma.T):
                            # print(Sigma)

                            try:
                                np.linalg.cholesky(Sigma)
                                locomo = locomo_probability_remastered(X=error, mu=np.zeros(3), Sigma=Sigma)
                                if(locomo>1):
                                    print('                             uh-oh ', locomo)
                                
                                if not math.isnan(locomo):
                                    locomo_prob.append(locomo)
                                # else append 0?
                            except np.linalg.LinAlgError:
                                    locomo_prob.append(0)
                        else:
                            locomo_prob.append(0)
                    else:
                        locomo_prob.append(0)

                if len(locomo_prob)>0:
                    contact_probability_face = np.sum(np.array(locomo_prob))/normalizing_factor
                    if math.isnan(contact_probability_face):
                        print('                             uh-oh!!!!', contact_probability_face)
                        print('                             uh-oh    ', np.array(locomo_prob), '_', np.sum(np.array(locomo_prob)), '_', np.sum(np.array(locomo_prob))/normalizing_factor)

                    # print('LOCOMO PROBABILITY=', contact_probability_face)
                    locomo_probabilities.append(contact_probability_face)
                    final_poses.append(pose)
                    final_transformations.append(t)


    # Filter poses satisfying kinematics
    R = ranking(locomo_probabilities, k=1, w=np.ones(len(locomo_probabilities))/len(locomo_probabilities))
    sorted_indeces = np.argsort(-np.array(locomo_probabilities))[:len(locomo_probabilities)]  
    locomo_probabilities = np.array(locomo_probabilities)[sorted_indeces]
    final_poses_sorted = np.array(final_poses)[sorted_indeces]

    final_transformations_sorted = np.array(final_transformations)[sorted_indeces]

    return final_poses_sorted, final_transformations_sorted, locomo_probabilities



def zero_moment_shift(sphere_center, sphere_radius=30, points=[]):
    '''
    n_rho   = centroid(sphere_center) - sphere_center
            = zero_moment(points in the sphere which center is the current point) - sphere_center
            = (1/N*(sum(points))) - sphere_center
    '''
    z = zero_moment(points_in_sphere(sphere_center, sphere_radius, points)) - sphere_center
    # print('sphere center ' , sphere_center, 'zero moment SHIFT ', z)
    return z

def zero_moment(set_points):
    """
    Also called the centroid fo the set of points
    """
    zero_moment = np.mean(set_points, axis=0)
    # print('zero moment=', zero_moment)
    return zero_moment


def distance(point_1:np.ndarray, point_2:np.ndarray):
    '''
    Compute the distance between 2 points in 3D
    '''
    distance = np.sqrt(np.sum((point_1-point_2)**2, axis=0))
    # print('distance=', distance)
    return distance


def points_on_sphere(sphere_center, sphere_radius, points):
    '''
    Get the points on the contour of the sphere
    '''
    # Compute the distances between the set of points and the sphere center
    distances = list(map(lambda x: distance(x, sphere_center), points))
    distances = np.array(distances)
    points_sphere = np.array(points)
    points_sphere = points_sphere[np.abs(distances-sphere_radius)<=EPSILON] 
    # print('points  on sphere', len(points_sphere))
    return points_sphere

def points_in_sphere(sphere_center, sphere_radius, points):
    '''
    Get the points inside the sphere
    '''
    # Compute the distances between the set of points and the sphere center
    distances = list(map(lambda x: distance(x, sphere_center), points))
    distances = np.array(distances)
    points_sphere = np.array(points)
    points_sphere = points_sphere[np.abs(distances)<=sphere_radius] 
    # print('points  in sphere', len(points_sphere))
    return points_sphere



def sample_finger_poses_around(point, gripper_model):
    """

    Args:
        point (point): 3d point coordinates
        gripper_model (point_cloud):    contact surface of the gripper supposed to touch objects
                                        need to either find a way to separate the components of the mesh or directtly have the files of the mesh separated
    """    

    
    # Find suitable poses of the gripper around the point
    # Possibilities:    randomly select [x, y, z, rx, ry, rz] but with the finger touching the point
    #                   with the normal of the point taken into account 
                        # -> the point belonging to the finger that is touching the point, has its normal vector opposite direction

    return -1

def sample_finger_poses_opposite(point, point_normal, finger_face, finger_mesh, amount_poses=10):
    # Rotate the gripper such that the normal vector of the point points in the opposite direction
    # than the surface normal of the gripper
    # Sample the points on the gripper such that the are in contact with the goal point 
    # Need the surface that will be around the object


    new_p = np.array([point]).astype(np.int32)
    point_goal=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_p))

    finger_pc = mesh_to_point_cloud(finger_mesh)
    face_pc = mesh_to_point_cloud(finger_face)

    # Use to first rotate so the face's normal points in the opposite direction to the point's normal

    # 1. Translate to center
    #       The mesh
    #       The face
    # -> 2 different matrices
    matrix_face_t, _ = translate_to_origin(finger_face)

    # 2. Rotate so the face faces the direction opposite to the normal
    print('normal1=',finger_face.triangle_normals[0])
    print('normal2=', -point_normal)
    align_normal_to_opposite_matrix = rotation(finger_face.triangle_normals[0], -point_normal)
    r = np.eye(4)
    r[:3, :3] = align_normal_to_opposite_matrix

    face_center = copy.deepcopy(finger_face).transform(np.matmul(r, matrix_face_t))
    
    poses = []
    T = []
    final_faces = []
    for _ in range(amount_poses):

        # Rotation happends around the center of the object
        # Move the center to the origin, then rotate, then translate back to initial position
        # Need to use the formula: T(x, y, z)*R(x, y, z)*T(-x, -y, -z)


        # Give random angle to rotate around normal as axis
        random_angle = random.uniform(0, np.pi*2)
        # Rodriguez rotation formula - matrix notation
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        k = -point_normal/np.linalg.norm(-point_normal)
        K = np.array([[0, -k[2], k[1]],
             [k[2], 0, -k[0]],
             [-k[1], k[0], 0]])
        a = math.sin(random_angle)
        random_rotation = np.eye(3) + (math.sin(random_angle)*K) + ((1-math.cos(random_angle))*np.matmul(K, K))
        rotate = np.eye(4)
        rotate[:3, :3] = random_rotation
        

        # Translate to random point on face to goal point
        # Other degrees of rotation are random
        copy_rotate_face = face_center.transform(rotate)
        copy_rotate_face_pc = mesh_to_point_cloud(copy_rotate_face)

        random_point = random.choice(copy_rotate_face_pc.points)

        translate_to_goal = np.eye(4)
        translate_to_goal[:3, 3] = get_translation(random_point, point)
        
        transformation = np.matmul(translate_to_goal, np.matmul(rotate, np.matmul(r, matrix_face_t)))

        final_face = copy.deepcopy(finger_face).transform(transformation)
        final_mesh = copy.deepcopy(finger_mesh).transform(transformation)
        final_mesh.paint_uniform_color(np.array([.5, .5, .9]))
        final_pc = mesh_to_point_cloud(final_mesh)
        final_face.paint_uniform_color(np.array([.5, .5, .9]))

        final_faces.append(copy.deepcopy(finger_face).transform(transformation))

        poses.append(final_mesh)
        T.append(transformation)
        
        visualisation = [finger_pc, finger_face, final_face, final_pc, point_goal]#, mesh_tr_pc]#, gripper_v]#, mesh_tr]
        # o3d.visualization.draw_geometries(visualisation)

    return poses, T, final_faces

def translate_to_origin(mesh:o3d.cpu.pybind.geometry.TriangleMesh):
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -mesh.get_center()
    return translate_to_origin, copy.deepcopy(mesh).transform(translate_to_origin)



def rotation(v1, v2):
    # https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724

    axis = np.cross(v1, v2)
    cosA = np.dot(v1, v2)
    k = 1/(1+cosA)

    res = np.array([[   (axis[0]*axis[0]*k)+cosA, (axis[1]*axis[0]*k)-axis[2], (axis[2]*axis[0]*k)+axis[1]],
                    [   (axis[0]*axis[1]*k)+axis[2], (axis[1]*axis[1]*k)+cosA, (axis[2]*axis[1]*k)-axis[0]],
                    [   (axis[0]*axis[2]*k)-axis[1], (axis[1]*axis[2]*k)+axis[0], (axis[2]*axis[2]*k)+cosA]])


    return res

# def get_rotation_matrix(vector_1, vector_2):
#     rotation_matrix = Rotation.align_vectors(np.array([vector_1]), np.array([vector_2]))
#     return rotation_matrix

def sample_finger_poses_random(point, finger_face, finger_mesh, amount_poses=10):
    '''
    Return the rotation matrix

    No need to project the point after
    '''
    # Basically take any point from the gripper model and put it at the same position as the selected point in the point cloud
    # - Rotate around the center of the object
    # - Choose a random point from the point cloud of the finger model and translate it to the point
            # Might want to sample multiple points for 1 rotation

    # print(finger_model.get_center())

    # Point
    new_p = np.array([point]).astype(np.int32)
    point_goal=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_p))

    finger_pc = mesh_to_point_cloud(finger_mesh)
    
    poses = []
    faces = []
    T = []
    for _ in range(amount_poses):

        # Rotation happends around the center of the object
        # Move the center to the origin, then rotate, then translate back to initial position
        # Need to use the formula: T(x, y, z)*R(x, y, z)*T(-x, -y, -z)
        center = finger_mesh.get_center()
        translate_to_center = np.eye(4)
        translate_to_center[:3, 3] = -center
        copy_finger = copy.deepcopy(finger_mesh).transform(translate_to_center)

        rotate = np.eye(4)
        R = copy_finger.get_rotation_matrix_from_xyz(np.random.uniform(0,np.pi*2,3))
        rotate[:3, :3] = R


        copy_rotate_face = copy.deepcopy(finger_face).transform(translate_to_center).transform(rotate)
        copy_rotate_face_pc = mesh_to_point_cloud(copy_rotate_face)

        random_point = random.choice(copy_rotate_face_pc.points)

        translate_to_goal = np.eye(4)
        translate_to_goal[:3, 3] = get_translation(random_point, point)
        
        transformation = np.matmul(translate_to_goal, np.matmul(rotate, translate_to_center))

        final_face = copy.deepcopy(finger_face).transform(transformation)
        final_face.paint_uniform_color(np.array([.8, .2, .2]))
        final_mesh = copy.deepcopy(finger_mesh).transform(transformation)
        final_mesh.paint_uniform_color(np.array([.8, .2, .2]))
        final_pc = mesh_to_point_cloud(final_mesh)


        poses.append(final_mesh)
        faces.append(final_face)
        T.append(transformation)
        
        visualisation = [finger_pc, finger_face, final_face, final_pc, point_goal]
        # o3d.visualization.draw_geometries(visualisation)

    return poses, T, faces

def get_translation(point_from, point_to):
    if len(point_to)==len(point_from):
        return point_to-point_from

def select_within_distance(mesh, points: o3d.cpu.pybind.geometry.PointCloud, d):
    # select the set of points that are within distance d from the mesh 
    # Should input face maybe (?)
    distances = distance_mesh_points(mesh, points)
    points = np.array(points.points)
    points = points[distances<=d]
    return points


def select_within_distance_pc(point_cloud_1: o3d.cpu.pybind.geometry.PointCloud, 
                              point_cloud_2: o3d.cpu.pybind.geometry.PointCloud, d):
    '''
    return the distances from the first point cloud 
    '''
    distances = distance_point_clouds(point_cloud_1, point_cloud_2)
    points = np.array(point_cloud_1.points)
    distances = np.array(distances)
    points = points[distances<=d]
    return points

def project_point_on_plane(mesh_surface:o3d.cpu.pybind.geometry.TriangleMesh,
                             point:np.ndarray):
    mesh_points = np.array(mesh_surface.vertices)
    plane = Plane.from_points(mesh_points[0], mesh_points[1], mesh_points[2])
    point = Point(point)
    point_projected = plane.project_point(point)
    return point_projected

def project_point_on_plane2(mesh_surface:o3d.cpu.pybind.geometry.TriangleMesh,
                             point:np.ndarray):
    # https://stackoverflow.com/a/8944143
    mesh_surface.compute_triangle_normals()
    n =  np.asarray(mesh_surface.triangle_normals)[0]
    surface_point = np.array(mesh_surface.vertices[0])
    projected_point = point - np.dot(point-surface_point, n) * n
    return projected_point

def project_point_on_surface(mesh_surface:o3d.cpu.pybind.geometry.TriangleMesh,
                             point:np.ndarray):

    # For experiments - test if influences the results
    projected_on_plane = project_point_on_plane(mesh_surface, point)
    # if distance_mesh_points(mesh_surface, o3d.geometry.PointCloud(o3d.utility.Vector3dVector([np.array(projected_on_plane)]))) > 1.1920929e-03:
    '''
    TODO implement projecting on edge of the triangle    
    Project the point on the 3 edges of the triangle and the point with the minimum distance
    '''
        # print('To be projected on an edge of the triangle')

    return projected_on_plane


def locomo_probability_remastered(X, mu, Sigma):
    '''
    ????????? should keep ???
    '''
    diff = np.array(X-mu)
    matrices_mult = np.matmul(np.matmul(diff.transpose(), Sigma), diff)
    mvg = (1/(math.sqrt(((2*math.pi)**2)*np.linalg.det(Sigma)))) *  math.exp((-1/2)*matrices_mult)
    res = 1 - ((math.sqrt(((2*math.pi)**2)*np.linalg.det(Sigma)) - mvg)/math.sqrt(((2*math.pi)**2)*np.linalg.det(Sigma)))
    # return res
    return math.exp((-1/2)*matrices_mult)

def locomo_probability_function(X, mu, Sigma):        
    multivariate_error = multivariate_normal(mean=mu, cov=Sigma)

    return multivariate_error.pdf(X)# * math.sqrt(((2*math.pi)**2)*np.linalg.det(Sigma))
    # return multivariate_normal.pdf(X, mu, Sigma)
    

def mean_covariance_points(X):
    '''3D point coordinates'''
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    return points.compute_mean_and_covariance()

def ranking(end_poses_prob, k, w):
    '''
    k = normalizing term
    w = weights satisfying w[0]+w[1]+...+w[-1] = 1
    '''
    return k*np.multiply(end_poses_prob, w)

# objects definition
# obj = read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
# box_pcd = mesh_to_point_cloud(mesh=obj, number_of_points=1000)
# box_pcd.estimate_normals()

# mesh = read_mesh("Gripper/Grasper_Locomo_scaled.STL")
# mesh_face = read_mesh('Gripper/face5.stl')
# face_normal = mesh_face.triangle_normals

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_face.vertices))
# pcd.estimate_normals()

# # Sphere creation to visualise current point better
# sphere = o3d.geometry.TriangleMesh.create_sphere(1)
# sphere.paint_uniform_color(np.array([.5, .5, .9]))
# point = np.array(box_pcd.points)[500]
# normal = np.array(box_pcd.normals)[500]
# sphere.translate(np.array(point), relative=False)

# poses, T, f = sample_finger_poses_opposite(point, normal, mesh_face, mesh, amount_poses=10)
# for pose, t, ffff in zip(poses, T, f):
#     pose.paint_uniform_color(np.array([.5, .5, .9]))
#     ffff.paint_uniform_color(np.array([.2, .8, .4]))
#     pc = mesh_to_point_cloud(pose)

#     face = copy.deepcopy(mesh_face).transform(t)
#     visualisation = [box_pcd, sphere, ffff, pc]#, final_face, final_pc, point_goal]#, mesh_tr_pc]#, gripper_v]#, mesh_tr]
#     o3d.visualization.draw_geometries(visualisation)

# gripper = read_mesh("Grasper_Locomo.STL")
# gripper.translate((0, 0, 0), relative=False)
# gripper.scale(300, center=gripper.get_center())
# # pc = mesh_to_point_cloud(gripper)

# file = 'Labeled Bin - 1x2x5 - pinballgeek'
# box1 = read_mesh('Boxes STLs/'+file+'.obj')
# box1.translate([0, 0, 0], relative=False)

# box_pcd1 = mesh_to_point_cloud(mesh=box1, number_of_points=1000)

# objects = [box_pcd1]
# mesh2 = read_mesh("Gripper/Grasper_Locomo_scaled.STL")
# pc = mesh_to_point_cloud(mesh2)

# f1 = read_mesh("Gripper/face5.stl")
# f2 = read_mesh("Gripper/face6.stl")
# f3 = read_mesh("Gripper/face10.stl")
# f4 = read_mesh("Gripper/face13.stl")
# # o3d.io.write_triangle_mesh('attempt.STL', gripper, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=False, print_progress=True)
# poses, transformation, locomo_prob = LoCoMo_remastered(sample_finger_poses_random,box_pcd1,
#                         fingers_model=mesh2,
#                         faces_models=[f1, f2, f3, f4],
#                         sphere_radius=10,
#                         poses_to_sample=10,
#                         distance=5)

# print(locomo_prob)
# o3d.visualization.draw_geometries([box_pcd1, mesh2, poses[0]])
