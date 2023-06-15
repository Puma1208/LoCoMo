import open3d as o3d
import numpy as np
import copy
import random
import sys
from UtilityOpen3d import read_mesh, mesh_to_point_cloud, distance_mesh_points, get_tri_meshes, distance_point_clouds, simplify_mesh, visualise_tri_faces
import math

from skspatial.objects import Plane, Vector, Point, Line
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation
from typing import List

# EPSILON = sys.float_info.epsilon
EPSILON = 0.1

def LoCoMo_remastered(  object_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
                        fingers_model: o3d.cpu.pybind.geometry.TriangleMesh,
                        faces_models: List[o3d.cpu.pybind.geometry.TriangleMesh],
                        sphere_radius: float=10,
                        poses_to_sample: int=10,
                        distance: float=20):
    

    final_transformations = []
    final_poses = []
    locomo_probabilities = []
    
    zero_moment_shifts = list(map(lambda x: zero_moment_shift(x, sphere_radius, np.asarray(object_point_cloud.points)), object_point_cloud.points))

    for finger_face in faces_models:
        print('face=', finger_face)
        p = 0
        for point in object_point_cloud.points[10:20]:
            poses, transformations, faces_oriented = sample_finger_poses_random(point, finger_face, fingers_model, poses_to_sample)

            print('     point=', p, '/', len(object_point_cloud.points))
            p +=1
            for pose, t, face_pose in zip(poses[:10], transformations[:10], faces_oriented[:10]):
                points_within_d = select_within_distance(face_pose, object_point_cloud, distance)

                locomo_prob = []
                # print('         pose=', pose)


                for point_d in points_within_d:

                    point_index = np.where(np.all(np.array(object_point_cloud.points) == (point_d), axis=1))
                    zms1 = zero_moment_shifts[point_index[0][0]]
                    # Project the point on the surface
                    projected_point = project_point_on_surface(face_pose, point_d)
                    zms2 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=object_point_cloud.points)
                    error = zms1 - zms2

                    if len(points_within_d)>2:
                        Sigma = np.cov(points_within_d, rowvar=False)

                        if np.linalg.det(Sigma) > 0:
                            locomo = locomo_probability_remastered(X=error, mu=np.zeros(3), Sigma=Sigma)
                            if not math.isnan(locomo):
                                locomo_prob.append(locomo)
                if len(locomo_prob)>0:
                    locomo_probabilities.append(np.mean(np.array(locomo_prob)))
                    final_poses.append(pose)
                    final_transformations.append(t)


    # Filter poses satisfying kinematics
    R = ranking(locomo_probabilities, k=1, w=np.ones(len(locomo_probabilities))/len(locomo_probabilities))
    sorted_indeces = np.argsort(locomo_probabilities)  
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
    # Rotate the gripper such that the surface normal vector of the point points in the opposite direction
    # than the surface normal of the gripper
    # Sample the points on the gripper such that the are in contact with the goal point 
    # Need the surface that will be around the object


    new_p = np.array([point]).astype(np.int32)
    point_goal=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_p))

    finger_pc = mesh_to_point_cloud(finger_mesh)
    face_pc = mesh_to_point_cloud(finger_face)
    
    poses = []
    T = []
    for _ in range(amount_poses):

        # Rotation happends around the center of the object
        # Move the center to the origin, then rotate, then translate back to initial position
        # Need to use the formula: T(x, y, z)*R(x, y, z)*T(-x, -y, -z)

        # The rotation must be done in such a way that the normal vectors should face opposite directions
        center = finger_mesh.get_center()
        translate_to_center = np.eye(4)
        translate_to_center[:3, 3] = -center
        copy_finger = copy.deepcopy(finger_mesh).transform(translate_to_center)


        # Choose the random point which normal vector should be in opposite direction of goal point
        # Other degrees of rotation are random
        rand_index = random.randint(0, len(face_pc.points))
        random_normal = np.array(face_pc.normals)[rand_index]
        rotation_matrix = get_rotation_matrix(random_normal, -point_normal)
        print(rotation_matrix)
        print(Rotation.from_matrix(rotation_matrix))
        print(Rotation.as_matrix(rotation_matrix))
        rotate = np.eye(4)
        # rotation_matrix = get_rotation_matrix(-point_normal, )
        R = copy_finger.get_rotation_matrix_from_xyz(np.random.uniform(0,np.pi*2,3))
        rotate[:3, :3] = R
        

        copy_rotate_face = copy.deepcopy(finger_face).transform(translate_to_center).transform(rotate)
        copy_rotate_face_pc = mesh_to_point_cloud(copy_rotate_face)

        random_point = random.choice(copy_rotate_face_pc.points)

        translate_to_goal = np.eye(4)
        translate_to_goal[:3, 3] = get_translation(random_point, point)
        
        transformation = np.matmul(translate_to_goal, np.matmul(rotate, translate_to_center))

        final_face = copy.deepcopy(finger_face).transform(transformation)
        final_mesh = copy.deepcopy(finger_mesh).transform(transformation)
        final_mesh.paint_uniform_color(np.array([.5, .5, .9]))
        final_pc = mesh_to_point_cloud(final_mesh)
        final_face.paint_uniform_color(np.array([.5, .5, .9]))


        poses.append(final_mesh)
        T.append(transformation)
        
        visualisation = [finger_pc, finger_face, final_face, final_pc, point_goal]#, mesh_tr_pc]#, gripper_v]#, mesh_tr]
        # o3d.visualization.draw_geometries(visualisation)

    return poses, T


def get_rotation_matrix(vector_1, vector_2):
    rotation_matrix = Rotation.align_vectors(np.array([vector_1]), np.array([vector_2]))
    return rotation_matrix

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
        diff = np.array(X-mu)
        matrices_mult = np.matmul(np.matmul(diff.transpose(), Sigma), diff)

        return math.exp((-1/2)*matrices_mult)

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


box = read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
box = read_mesh("Boxes STLs/Mega Box - 4x4x12 - ww9a.stl")
box = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")
box.translate([0, 0, 0], relative=False)
box_pcd = mesh_to_point_cloud(mesh=box, number_of_points=500)


# gripper = read_mesh("Grasper_Locomo.STL")
# gripper.scale(300, center=box.get_center())
# gripper.translate([50, 5, -10])
gripper = read_mesh("/Gripper/Grasper_Locomo_scaled.stl")



gripper_simple = simplify_mesh(gripper, simplify_amount=7)

print(gripper.get_center())

faces_models = [read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")]
# poses, transformation, locomo_prob = LoCoMo_remastered( object_point_cloud=box_pcd,
#             fingers_model=gripper_simple,
#             faces_models=faces_models,
#             sphere_radius=15,
#             poses_to_sample=10,
#             distance=5)


# print(len(poses))
# print('probabilities sorted ', locomo_prob)
# for pose, prob in zip(poses, locomo_prob):
#     o3d.visualization.draw_geometries([pose, box_pcd], ('locomo=' + str(prob)))
