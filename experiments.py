import open3d as o3d
import copy 
from itertools import product
from LoCoMo import LoCoMo, sample_finger_poses_random, sample_finger_poses_opposite
from UtilityOpen3d import read_mesh, mesh_to_point_cloud, read_point_cloud
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation
import time

    
def print_kwargs(**kwargs):
    print(kwargs)
    

def transformation_matrix_to_angles(transformation_matrix):
    rotation = np.array(transformation_matrix)[:3, :3] 
    angles = Rotation.from_matrix(rotation).as_euler('xyz')
    return angles[0], angles[1], angles[2]

def transformation_matrix_to_pos(transformation_matrix):
    return transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]

print(o3d.__version__)
sampling_methods = [sample_finger_poses_random]#, LoCoMo.sample_finger_poses_opposite]
file = 'Labeled Bin - 1x2x5 - pinballgeek'
# file = 'DatasetImages/whiteStainer/9_whiteStainer.ply'
box1 = read_mesh('Boxes STLs/'+file+'.obj')
# box1 = read_point_cloud(file)
box1.translate([0, 0, 0], relative=False)
files = ['sphere_r=1.ply', 'box_w=20_h=10_d=15.ply', 'box_w=40_h=20_d=30.ply']

box_pcd1 = mesh_to_point_cloud(mesh=box1, number_of_points=800)
sphere = o3d.geometry.TriangleMesh.create_sphere(10)
sphere.paint_uniform_color(np.array([.5, .5, .9]))
sphere.translate([0, 0, 0], relative=False)
sphere_pc = mesh_to_point_cloud(mesh=sphere, number_of_points=500)
# o3d.io.write_triangle_mesh("sphere_r=1.ply", sphere)

box = o3d.geometry.TriangleMesh.create_box(20, 10, 15)
box.paint_uniform_color(np.array([.7, .7, .5]))
box.translate([0, 0, 0], relative=False)
box_pc = mesh_to_point_cloud(mesh=box, number_of_points=500)
# o3d.io.write_triangle_mesh("box_w=20_h=10_d=15.ply", sphere)


box2 = o3d.geometry.TriangleMesh.create_box(40, 20, 30)
box2.paint_uniform_color(np.array([.7, .7, .5]))
box2.translate([0, 0, 0], relative=False)
box2_pc = mesh_to_point_cloud(mesh=box2, number_of_points=500)
# o3d.io.write_triangle_mesh("box_w=40_h=20_d=30.ply", sphere)

objects = [sphere_pc, box_pc, box2_pc]

mesh = read_mesh("Gripper/Grasper_Locomo_scaled.STL")

finger_model = [mesh]
# o3d.visualization.draw_geometries([box2, mesh])

faces_models = [[read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")]]
sphere_radius = [5, 10 , 15]
poses_to_sample = [10]
distance = [5, 10, 20]

dynamic_params = {
"sampling_method": sampling_methods,
"object_point_cloud": objects,
"fingers_model": finger_model,
"faces_models": faces_models,
"sphere_radius": sphere_radius,
"poses_to_sample":poses_to_sample,
"distance":distance
}


param_values = (zip(list(dynamic_params.keys()), x) for x in product(*dynamic_params.values()))
path_dir = 'LoCoMoExperimentsSamplingFunction/'

if not os.path.exists(path_dir):
    print('Creating a new path: ', path_dir)
    os.mkdir(path_dir)

object_index = 0
i = 0
for paramset in param_values:
    i += 1
    if i>9:
        object_index = 1

    if i>18:
        object_index = 2
    file = files[object_index]

    kwargs = dict(paramset)
    d = dict(paramset)

    # Need to store the features - title + TIME
    # Create a different dataframe for each setting
    # Store the transformations as matrices + the Locomo scores

    start_time = time.time()
    _, transformation, locomo_prob = LoCoMo(**kwargs)

    duration = time.time()-start_time

    rotation = map(transformation_matrix_to_angles, transformation)
    translation = map(transformation_matrix_to_pos, transformation)

    rx, ry, rz = zip(*rotation)
    x, y, z = zip(*translation)
    title =  str(object_index) + ' ' + file + ', ' + str(kwargs['sampling_method']) + ', ' + str(kwargs['sphere_radius']) + ', ' + str(kwargs['poses_to_sample']) + ', ' + str(kwargs['distance']) + ', ' + str(duration)

    df = pd.DataFrame({'Probabilities': (locomo_prob), 'x' : (x), 'y' : (y), 'z' : (z), 'rx' : (rx), 'ry' : (ry), 'rz' : (rz)})
    df.to_csv(path_dir+title+'.csv')
