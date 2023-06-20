import open3d as o3d
import copy 
from itertools import product
from LoCoMo import LoCoMo_remastered, sample_finger_poses_random
from UtilityOpen3d import read_mesh, mesh_to_point_cloud
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
    print('angles=', angles)
    return angles[0], angles[1], angles[2]

def transformation_matrix_to_pos(transformation_matrix):
    print('translate=', transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3])

    return transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]

def locomo_test(parameters_dictionary):
    
    poses, transformation, locomo_prob = LoCoMo_remastered(**parameters_dictionary)
    rx, ry, rz = transformation_matrix_to_angles(transformation_matrix=transformation)
    x, y, z = transformation_matrix_to_pos(transformation_matrix=transformation)


def test():
    file = 'Labeled Bin - 1x2x5 - pinballgeek'
    box1 = read_mesh('Boxes STLs/'+file+'.obj')
    box2 = read_mesh("Boxes STLs/Mega Box - 4x4x12 - ww9a.stl")
    box3 = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")
    box1.translate([0, 0, 0], relative=False)
    box2.translate([0, 0, 0], relative=False)
    box3.translate([0, 0, 0], relative=False)

    box_pcd1 = mesh_to_point_cloud(mesh=box1, number_of_points=1000)
    box_pcd2 = mesh_to_point_cloud(mesh=box2, number_of_points=900)
    box_pcd3 = mesh_to_point_cloud(mesh=box3, number_of_points=800)
    pc3 = mesh_to_point_cloud(box3, 5000)

    # objects = {1:{box_pcd1:'box1'},2:{ box_pcd2:'box2'}, 3:{box_pcd3:'box3'}}
    objects = [box_pcd1]
    mesh = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")

    finger_model = read_mesh('Grasper_Locomo.STL')
    finger_model.scale(300, center=box1.get_center())
    finger_model.translate([50, 5, -10])
    faces_models = [read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")]
    
    p, t, prob = LoCoMo_remastered(sample_finger_poses_random, box_pcd3, finger_model, faces_models, 10, 1, 10)


    p[1].paint_uniform_color(np.array([.5, .8, .5]))
    pc = mesh_to_point_cloud(p[1])

    mesh_t = copy.deepcopy(finger_model).transform(t[1])
    mesh_t.paint_uniform_color(np.array([.8, .6, .5]))
    pc2 = mesh_to_point_cloud(mesh_t)


    o3d.visualization.draw_geometries([finger_model, pc3, pc, pc2])


    
# def locomo_exp():
    # sampling_method,
    #                     object_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    #                     fingers_model: o3d.cpu.pybind.geometry.TriangleMesh,
    #                     faces_models: List[o3d.cpu.pybind.geometry.TriangleMesh],
    #                     sphere_radius: float=10,
    #                     poses_to_sample: int=10,
    #                     distance: float=20,



sampling_methods = [sample_finger_poses_random]#, LoCoMo.sample_finger_poses_opposite]
file = 'Labeled Bin - 1x2x5 - pinballgeek'
box1 = read_mesh('Boxes STLs/'+file+'.obj')
box2 = read_mesh("Boxes STLs/Mega Box - 4x4x12 - ww9a.stl")
box3 = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")
box1.translate([0, 0, 0], relative=False)
box2.translate([0, 0, 0], relative=False)
box3.translate([0, 0, 0], relative=False)

box_pcd1 = mesh_to_point_cloud(mesh=box1, number_of_points=1000)
box_pcd2 = mesh_to_point_cloud(mesh=box2, number_of_points=900)
box_pcd3 = mesh_to_point_cloud(mesh=box3, number_of_points=800)

# objects = {1:{box_pcd1:'box1'},2:{ box_pcd2:'box2'}, 3:{box_pcd3:'box3'}}
objects = [box_pcd1]
mesh = read_mesh("Boxes STLs/Labeled Divided Bin x2 - 1x3x6 - ZackFreedman.stl")

finger_model = [mesh]
faces_models = [[read_mesh("Gripper/face5.stl"), read_mesh("Gripper/face6.stl"), read_mesh("Gripper/face10.stl"), read_mesh("Gripper/face13.stl")]]
sphere_radius = [5, 10 , 15]
poses_to_sample = [1, 2, 3]
distance = [20, 25]

dynamic_params = {
"sampling_method": sampling_methods,
"object_point_cloud": objects,
"fingers_model": finger_model,
"faces_models": faces_models,
"sphere_radius": sphere_radius,
"poses_to_sample":poses_to_sample,
}


param_values = (zip(list(dynamic_params.keys()), x) for x in product(*dynamic_params.values()))
# print(sum(1 for _ in param_values))


path_dir = 'LoCoMoExperiments/'



# if(not os.path.exists(path_dir)):
#     print('Creating a new path: ', path_dir)
#     os.mkdir(path_dir)
for paramset in param_values:
#     # use the dict from iterator of tuples constructor

    kwargs = dict(paramset)
    d = dict(paramset)

    # Need to store the features - title + TIME
    # Create a different dataframe for each setting
    # Store the transformations as matrices + the Locomo scores

    start_time = time.time()

    _, transformation, locomo_prob = LoCoMo_remastered(**kwargs)

    duration = time.time()-start_time

    rotation = map(transformation_matrix_to_angles, transformation)
    translation = map(transformation_matrix_to_pos, transformation)

    print('transformation')
    print(transformation)
    rx, ry, rz = zip(*rotation)
    x, y, z = zip(*translation)

    print('rx=', list(rx))
    print('ry=', list(ry))
    print('rz=', list(rz))
    
    title =  file + ', ' + str(kwargs['sampling_method']) + ', ' + str(kwargs['sphere_radius']) + ', ' + str(kwargs['poses_to_sample']) + ', ' + str(duration)

    df = pd.DataFrame({'Probabilities': (locomo_prob), 'x' : (x), 'y' : (y), 'z' : (z), 'rx' : (rx), 'ry' : (ry), 'rz' : (rz)})
    df.to_csv(path_dir+title+'.csv')
