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

# EPSILON = sys.float_info.epsilon
EPSILON = 0.1

def LoCoMo( point_cloud: o3d.cpu.pybind.geometry.PointCloud,
            fingers_model: o3d.cpu.pybind.geometry.TriangleMesh,
            sphere_radius: float=5,
            poses_to_sample: int=10,
            distance: float=5):
    # Compute surface normal at each point in the point cloud
    final_poses = []
    final_transformations = []
    poses_probabilities = []
    
    zero_moment_shifts = list(map(lambda x: zero_moment_shift(x, sphere_radius, np.asarray(point_cloud.points)), point_cloud.points))

    # Gripper model as triangular meshes
    finger_faces = get_tri_meshes(fingers_model)
    # visualise_tri_faces(fingers_model)
    finger_faces = [finger_faces[i] for i in [2, 18, 20, 40]]
    # If use the following mesh of the gripper, the relevant faces for finger_face are  of index
    # 2, 18, 20, 40
    # mesh2 = read_mesh("Grasper_Locomo.STL")
    # mesh_simple = simplify_mesh(mesh2, simplify_amount=7)
    # mesh_tri = get_tri_meshes(mesh_simple)


    # finger_model  -> could either be a mesh -> to be transformed to a list of triangle meshes to compute
    #               -> would be nice to find a way to split and get the relevant faces, not all of them
    #               -> a list of meshes constituing the whole finger model
    for finger_face in finger_faces:
        for point in np.asarray(point_cloud.points)[0:10]:

            poses, transformations = sample_finger_poses_random(point, finger_face, fingers_model, poses_to_sample)
            for pose, t in zip(poses[0:5], transformations[0:5]):
                gripper_transform = copy.deepcopy(fingers_model).transform(t)

                o3d.visualization.draw_geometries([pose, gripper_transform, point_cloud])


                # TODO: find a way to get the faces of the mesh
                # simplify the mesh of the finger model and for each face compute the following
                points_within_d = select_within_distance(pose, point_cloud, distance)
                locomo_prob = []
                for point_to_project in points_within_d[0:2]:
                    projected_point = project_point_on_surface(pose, point_to_project)
                    # Compute the Local Contact Probabilty
                    # TODO when computing the error -> does the zero moment between the gripper and the object
                    # mean that they are correlated somehow???
                    # Anw

                    Sigma = 0
                    n_1 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=points_within_d)
                    # Need to get the point cloud of the gripper/face on which the point is projected
                    # that are within a distance d from the point then compute the zms 
                    point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([projected_point]))
                    pose_pc = mesh_to_point_cloud(pose, 1000)
                    points_within_d_gripper = select_within_distance_pc(pose_pc, point, distance)

                    n_2 = zero_moment_shift(sphere_center=projected_point, sphere_radius=sphere_radius, points=points_within_d_gripper)
                    # Error between the 2 zero-shift vectors
                    # TODO figure out whether the computed zero moment shifts at the beginning should be used here
                    zms_error = n_1 -n_2
                    # points_within_d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_within_d))
                    # mu, Sigma = points_within_d.compute_mean_and_covariance()
                    mu, Sigma = mean_covariance_points(points_within_d)
                    # print(mu, Sigma)
                    locomo_prob.append(locomo_probability(X=projected_point, Sigma=Sigma, zero_moment_shift_error=zms_error))
                    # print('locomo probability ', locomo_probability)
                poses_probabilities.append(np.mean(np.array(locomo_prob)))
                final_poses.append(pose)
                final_transformations.append(t)

    # # Among the poses find those that satisfy kinematic constraint of the gripper
    # for pose in final_poses:
        # Compute the ranking metric
    print('prob=', (poses_probabilities))

    print(np.ones(len(poses_probabilities))/len(poses_probabilities))
    R = ranking(poses_probabilities, k=1, w=np.ones(len(poses_probabilities))/len(poses_probabilities))
    print(R)
    # Sort end poses in decreasing order depending on R
    # Sample gripper pose from end_poses_sorted
    # Return top grasp poses
    
    return final_poses, final_transformations


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


def distance(point_1, point_2):
    '''
    Compute the distance between 2 points in 3D
    '''
    sum = 0
    for coord_1, coord_2 in zip(point_1, point_2):
        sum += (coord_1-coord_2)**2
    distance = math.sqrt(sum)
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

    # Need a method to select face that is touching the object - need from mesh
    
    # Find suitable poses of the gripper around the point
    # Possibilities:    randomly select [x, y, z, rx, ry, rz] but with the finger touching the point
    #                   with the normal of the point taken into account 
                        # -> the point belonging to the finger that is touching the point, has its normal vector opposite direction

    return -1

def sample_finger_poses_opposite(point, finger_model):
    # Rotate the gripper such that the surface normal vector of the point points in the opposite direction
    # than the surface normal of the gripper
    # Sample the other rotation -> choose if uniform/random/etc
    # Sample the points on the gripper such that the are in contact with the goal point 
    # Need the surface that will be around the object
    return -1

def sample_finger_poses_random(point, finger_face, finger_mesh, amount_poses=10):
    '''
    Return the rotation matrix
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
        T.append(transformation)
        
        visualisation = [finger_pc, finger_face, final_face, final_pc, point_goal]#, mesh_tr_pc]#, gripper_v]#, mesh_tr]
        # o3d.visualization.draw_geometries(visualisation)

    return poses, T

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

def locomo_probability(X, Sigma, zero_moment_shift_error):
    # mvg_1  = multivariate_gaussian(X=X, mu=np.zeros(3), Sigma=Sigma)
    # mvg_2 = multivariate_gaussian(X=zero_moment_shift_error, mu=np.zeros(3), Sigma=Sigma)
    # print('mvg_1', mvg_1)
    # print('mvg_2', mvg_2)
    # num = np.max(X, mvg_1)-mvg_2

    # alternative definition from https://www.researchgate.net/profile/Maxime-Adjigble/publication/334440947_An_assisted_telemanipulation_approach_combining_autonomous_grasp_planning_with_haptic_cues/links/61d5712dd4500608168d77d8/An-assisted-telemanipulation-approach-combining-autonomous-grasp-planning-with-haptic-cues.pdf

    first_term = math.sqrt(math.pow(2*math.pi, 3) * np.linalg.det(Sigma))
    second_term = multivariate_gaussian(X=zero_moment_shift_error, mu=np.zeros(3), Sigma=Sigma)
    return first_term*second_term

# def mean_points(X):
#     return np.mean(X, axis=0)

# def sigma(X):
#     '''
#     Return the covariance matrix of the 3d points 
#     '''

def multivariate_gaussian(X, mu, Sigma):
    first_term = 1/(math.sqrt(math.pow(2*math.pi, 3) * np.linalg.det(Sigma)))
    diff = np.array(X-mu)
    matrices_mult = np.matmul(np.matmul(diff.transpose(), Sigma), diff)
    second_term = math.exp((-1/2)*matrices_mult)
    return first_term*second_term

def multivariate_gaussian_alt(X, mu, Sigma):
    multi_variate_gaussian = multivariate_normal(mean=mu, cov=Sigma)
    return multi_variate_gaussian.pdf(X)

def mean_covariance_points(X):
    '''3D point coordinates'''
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    return points.compute_mean_and_covariance()

def ranking(end_poses_prob, k, w):
    '''
    k = normalizing term
    w = weights satisfying w[0]+w[1]+...+w[-1] = 1
    '''
    return k*np.dot(end_poses_prob, w)


box = read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
box.translate([0, 0, 0], relative=False)

gripper = read_mesh("Grasper_Locomo.STL")
gripper.scale(300, center=box.get_center())
gripper.translate([50, 5, -10])
box_pcd = mesh_to_point_cloud(mesh=box, number_of_points=1000)



gripper_simple = simplify_mesh(gripper, simplify_amount=7)


gripper_pcd = mesh_to_point_cloud(mesh=gripper_simple, number_of_points=800)
index = 0

poses, transformation = LoCoMo(point_cloud=box_pcd, fingers_model=gripper_simple)