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
                        distance: float=2):
    
    zero_moment_shifts = list(map(lambda x: zero_moment_shift(x, sphere_radius, np.asarray(object_point_cloud.points)), object_point_cloud.points))

    # Gripper model as triangular meshes
    finger_faces = get_tri_meshes(fingers_model)
    # visualise_tri_faces(fingers_model)
    # finger_faces = [finger_faces[i] for i in [6, 4, 10, 35]]

    for finger_face in faces_models:
        
        for point in object_point_cloud.points[:1]:
            poses, transformations, faces_oriented = sample_finger_poses_random(point, finger_face, fingers_model, poses_to_sample)
            for face_pose in faces_oriented[:1]:
                points_within_d = select_within_distance(face_pose, object_point_cloud, distance)

                pose_pc = mesh_to_point_cloud(poses[faces_oriented.index(face_pose)])
                vis_points = []
                for point_d in points_within_d:
                    sphere = o3d.geometry.TriangleMesh.create_sphere()
                    sphere.paint_uniform_color(np.array([.5, .5, .9]))
                    sphere.translate(np.array(point), relative=False)
                    vis_points.append(sphere)

                vis_points.append(pose_pc)
                vis_points.append(face_pose)

                s = o3d.geometry.TriangleMesh.create_sphere(sphere_radius)
                s.paint_uniform_color(np.array([.5, .5, .1]))
                s.translate(np.array(point), relative=False)
                vis_points.append(s)

                o3d.visualization.draw_geometries(vis_points)


    print()

def LoCoMo( object_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
            fingers_model: o3d.cpu.pybind.geometry.TriangleMesh,
            faces_models: List[o3d.cpu.pybind.geometry.TriangleMesh],
            sphere_radius: float=5,
            poses_to_sample: int=10,
            distance: float=2):
    # Compute surface normal at each point in the point cloud
    final_poses = []
    final_transformations = []
    poses_probabilities = []

    # Store the normals of the points - dictionary
    
    # points_normals = {np.array(p): normal for p, normal in zip(point_cloud.points, point_cloud.normals)}
    
    # for point in point_cloud.points:
    #     print(point)
    zero_moment_shifts = list(map(lambda x: zero_moment_shift(x, sphere_radius, np.asarray(object_point_cloud.points)), object_point_cloud.points))

    # Gripper model as triangular meshes
    finger_faces = get_tri_meshes(fingers_model)
    # visualise_tri_faces(fingers_model)
    finger_faces = [finger_faces[i] for i in [2, 18, 20, 40]]
    # If use the following mesh of the gripper, the relevant faces for finger_face are  of index
    # 2, 18, 20, 40
    # mesh2 = read_mesh("Grasper_Locomo.STL")
    # mesh_simple = simplify_mesh(mesh2, simplify_amount=7)
    # mesh_tri = get_tri_meshes(mesh_simple)

    gripper_pcd = mesh_to_point_cloud(fingers_model)


    # points_normals = {key: value for key, value in x}
    for finger_face in faces_models:
        print('finger face = ', finger_face)

        for (p, normal, zero_moment) in zip(object_point_cloud.points[:1], object_point_cloud.normals[:1], zero_moment_shifts[:1]):
            # print('     point = ', p)
            # print('normal ', p.normal)
            

            projected_pc = np.asarray(p)
            # poses, transformations = sample_finger_poses_opposite(projected_pc, normal, finger_face, fingers_model, poses_to_sample)
            poses, transformations, faces_oriented = sample_finger_poses_random(projected_pc, finger_face, fingers_model, poses_to_sample)


            for pose, t, f in zip(poses[:1], transformations[:1], faces_oriented[:1]):

                gripper_transform = copy.deepcopy(fingers_model).transform(t)

                # TODO: find a way to get the faces of the mesh
                # simplify the mesh of the finger model and for each face compute the following
                points_within_d = select_within_distance(pose, object_point_cloud, distance)
                locomo_prob = []
                vis_points = []
                pose_pc = mesh_to_point_cloud(pose)

                vis_points.append(pose_pc)
                vis_points.append(f)
                o3d.visualization.draw_geometries(vis_points)
                for point_to_project in points_within_d:
                    
                    
                    # projected_point = project_point_on_surface(f, point_to_project)
                    projected_point = point_to_project

                    
                    # Compute the Local Contact Probabilty
                    # TODO when computing the error -> does the zero moment between the gripper and the object
                    # mean that they are correlated somehow???
                    # Anw

                    Sigma = 0
                    n_1 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=points_within_d)
                    # Need to get the point cloud of the gripper/face on which the point is projected
                    # that are within a distance d from the point then compute the zms 
                    projected_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([projected_point]))
                    pose_pc = mesh_to_point_cloud(pose, 1000)
                    points_within_d_gripper = select_within_distance_pc(pose_pc, projected_pc, distance)

                    n_2 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=points_within_d_gripper)
                    # Error between the 2 zero-shift vectors
                    # TODO figure out whether the computed zero moment shifts at the beginning should be used here
                    zms_error = n_1 - zero_moment
                    # points_within_d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_within_d))
                    # mu, Sigma = points_within_d.compute_mean_and_covariance()
                    # mu, Sigma = mean_covariance_points(points_within_d)
                    if len(points_within_d) > 2:
                        Sigma = np.cov(points_within_d, rowvar=False)

                        print("shape = ", (Sigma))
                        if np.linalg.det(Sigma) > 0:

                            # print(mu, Sigma)
                            locomo_prob.append(locomo_probability(X=projected_point, Sigma=Sigma, zero_moment_shift_error=zms_error, points=np.array(object_point_cloud.points)))
                            # print('locomo probability ', locomo_probability)
                            poses_probabilities.append(np.mean(np.array(locomo_prob)))
                            final_poses.append(pose)
                            final_transformations.append(t)

    # # Among the poses find those that satisfy kinematic constraint of the gripper
    # for pose in final_poses:
        # Compute the ranking metric
    print('prob=', sorted(poses_probabilities))

    print(np.ones(len(poses_probabilities))/len(poses_probabilities))
    R = ranking(poses_probabilities, k=1, w=np.ones(len(poses_probabilities))/len(poses_probabilities))
    print('R=',R)

    # Sort end poses in decreasing order depending on R
    # Sample gripper pose from end_poses_sorted
    # Return top grasp poses
    sorted_indeces = np.argsort(R)
    final_poses_sorted = np.array(final_poses)[sorted_indeces]
    final_transformations_sorted = np.array(final_transformations)[sorted_indeces]

    return final_poses_sorted, final_transformations_sorted


def zero_moment_shift(sphere_center, sphere_radius=30, points=[]):
    '''
    n_rho   = centroid(sphere_center) - sphere_center
            = zero_moment(points in the sphere which center is the current point) - sphere_center
            = (1/N*(sum(points))) - sphere_center
    '''
    z = zero_moment(points_in_sphere(sphere_center, sphere_radius, points)) - sphere_center
    print('sphere center ' , sphere_center, 'zero moment SHIFT ', z)
    return z

def zero_moment(set_points):
    """
    Also called the centroid fo the set of points
    """
    zero_moment = np.mean(set_points, axis=0)
    # print('zero moment=', zero_moment)
    return zero_moment


def distance(point_1, point_2):
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
    print("distances = ", distances)
    points_sphere = np.array(points)
    points_sphere = points_sphere[np.abs(distances)<=sphere_radius] 
    print('points  in sphere', len(points_sphere))
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

    return -1

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
        final_mesh = copy.deepcopy(finger_mesh).transform(transformation)
        final_mesh.paint_uniform_color(np.array([.5, .5, .9]))
        final_pc = mesh_to_point_cloud(final_mesh)
        final_face.paint_uniform_color(np.array([.5, .5, .9]))


        poses.append(final_mesh)
        faces.append(final_face)
        T.append(transformation)
        
        visualisation = [finger_pc, finger_face, final_face, final_pc, point_goal]#, mesh_tr_pc]#, gripper_v]#, mesh_tr]
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

def project_point_on_surface(mesh_surface:o3d.cpu.pybind.geometry.TriangleMesh,
                             point:np.ndarray):
    mesh_points = np.array(mesh_surface.vertices)
    plane = Plane.from_points(mesh_points[0], mesh_points[1], mesh_points[2])
    point = Point(point)
    point_projected = plane.project_point(point)
    return point_projected

def locomo_probability(X, Sigma, zero_moment_shift_error, points):
    # mvg_1  = multivariate_gaussian(X=X, mu=np.zeros(3), Sigma=Sigma)
    # mvg_2 = multivariate_gaussian(X=zero_moment_shift_error, mu=np.zeros(3), Sigma=Sigma)
    # print('mvg_1', mvg_1)
    # print('mvg_2', mvg_2)
    # num = np.max(X, mvg_1)-mvg_2

    # alternative definition from https://www.researchgate.net/profile/Maxime-Adjigble/publication/334440947_An_assisted_telemanipulation_approach_combining_autonomous_grasp_planning_with_haptic_cues/links/61d5712dd4500608168d77d8/An-assisted-telemanipulation-approach-combining-autonomous-grasp-planning-with-haptic-cues.pdf
    first_term = 0

    if np.linalg.det(Sigma) > 0:
            det = np.linalg.det(Sigma)
            power = math.pow(2*math.pi, 3) 
            first_term = math.sqrt(power*det)
    # second_term = multivariate_gaussian(X=zero_moment_shift_error, mu=np.zeros(3), Sigma=Sigma)
    second_term = multivariate_gaussian(zero_moment_shift_error, np.zeros(3), Sigma)
    if (first_term*second_term)>1:
        print("     locomo probability ", first_term, '*', second_term, '=', (first_term*second_term))
    return first_term*second_term

def multivariate_gaussian(X, mu, Sigma):
    if np.linalg.det(Sigma) > 0:
        det = np.linalg.det(Sigma)
        power = math.pow(2*math.pi, 3) 
        sqrt = math.sqrt(power*det)
            
        first_term = 1/sqrt
        # first_term = 1/(math.sqrt(math.pow(2*math.pi, 3) * np.linalg.det(Sigma)))

        diff = np.array(X-mu)
        matrices_mult = np.matmul(np.matmul(diff.transpose(), Sigma), diff)
        second_term = math.exp((-1/2)*matrices_mult)
        return first_term*second_term
    return 0

# def multivariate_gaussian_alt(points, X):
#     Sigma = np.cov(points, rowvar=False)

#     multi_variate_gaussian = multivariate_normal(mean=np.mean(points), cov=Sigma)
#     return multi_variate_gaussian.pdf(X)

# def multi_variate_gaussian(X):
#     print("__________", multivariate_normal.pdf(X, mean=None, cov=1))
#     return multivariate_normal.pdf(X, mean=None, cov=1)

def mean_covariance_points(X):
    '''3D point coordinates'''
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    return points.compute_mean_and_covariance()

def ranking(end_poses_prob, k, w):
    '''
    k = normalizing term
    w = weights satisfying w[0]+w[1]+...+w[-1] = 1
    '''
    print("k=", k)
    print("dot=", np.multiply(end_poses_prob, w))
    return k*np.multiply(end_poses_prob, w)


box = read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
box = read_mesh("Boxes STLs/Mega Box - 4x4x12 - ww9a.stl")
box = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")
box.translate([0, 0, 0], relative=False)

gripper = read_mesh("Grasper_Locomo.STL")
gripper.scale(300, center=box.get_center())
gripper.translate([50, 5, -10])
pc = mesh_to_point_cloud(gripper)
box_pcd = mesh_to_point_cloud(mesh=box, number_of_points=500)


gripper_simple = simplify_mesh(gripper, simplify_amount=7)
gripper_pcd = mesh_to_point_cloud(mesh=gripper_simple, number_of_points=800)


# print(np.min(np.array(box_pcd.points)[:,0]), "_", np.min(np.array(box_pcd.points)[:,1]), "_", np.min(np.array(box_pcd.points)[:,2]))

# print(np.max(np.array(box_pcd.points)[:,0]), "_", np.max(np.array(box_pcd.points)[:,1]), "_", np.max(np.array(box_pcd.points)[:,2]))

mesh2 = read_mesh("Grasper_Locomo.STL")
mesh_simple = simplify_mesh(mesh2, simplify_amount=7)
mesh_tri = get_tri_meshes(mesh_simple)
mesh_pc = mesh_to_point_cloud(mesh_simple)


gr = read_mesh('Gripper/Grasper_Locomo_scaled.stl')
gr_pc = mesh_to_point_cloud(gr)
gr_tr = get_tri_meshes(gr)
print(gr_tr)
# o3d.visualization.draw_geometries([gr_pc])
# o3d.visualization.draw_geometries([gr_pc, read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")])
faces_models = [read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")]
poses, transformation = LoCoMo_remastered( object_point_cloud=box_pcd,
            fingers_model=gripper_simple,
            faces_models=faces_models,
            sphere_radius=5,
            poses_to_sample=10,
            distance=5)

# face5=[[0.06  0.016 0.055]
#  [0.06  0.016 0.025]
#  [0.    0.016 0.025]]

# face6 = [[0.06  0.016 0.055]
#  [0.    0.016 0.025]
#  [0.    0.016 0.055]]

# face10 = [[0.06  0.096 0.025]
#  [0.    0.096 0.055]
#  [0.    0.096 0.025]]

# face13 = [[0.06  0.096 0.055]
#  [0.    0.096 0.055]
#  [0.06  0.096 0.025]]

# for index in faces_index:
