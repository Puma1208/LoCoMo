# Euler's angles
import numpy as np
from scipy.spatial.transform import Rotation
import UtilityOpen3d
import open3d as o3d
import copy
# import LoCoMo


def rotation(v1, v2):
    # https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724

    axis = np.cross(v1, v2)
    cosA = np.dot(v1, v2)
    k = 1/(1+cosA)

    res = np.array([[   (axis[0]*axis[0]*k)+cosA, (axis[1]*axis[0]*k)-axis[2], (axis[2]*axis[0]*k)+axis[1]],
                    [   (axis[0]*axis[1]*k)+axis[2], (axis[1]*axis[1]*k)+cosA, (axis[2]*axis[1]*k)-axis[0]],
                    [   (axis[0]*axis[2]*k)-axis[1], (axis[1]*axis[2]*k)+axis[0], (axis[2]*axis[2]*k)+cosA]])


    return res
def idrk():
    # objects definition
    obj = UtilityOpen3d.read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
    box_pcd = UtilityOpen3d.mesh_to_point_cloud(mesh=obj, number_of_points=1000)

    mesh = UtilityOpen3d.read_mesh("Gripper/Grasper_Locomo_scaled.STL")
    mesh_face = UtilityOpen3d.read_mesh('Gripper/face5.stl')
    face_normal = mesh_face.triangle_normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_face.vertices))
    pcd.estimate_normals()

    # Sphere creation to visualise current point better
    sphere = o3d.geometry.TriangleMesh.create_sphere(1)
    sphere.paint_uniform_color(np.array([.5, .5, .9]))
    point = np.array(box_pcd.points)[1]
    normal = np.array(box_pcd.normals)[1]
    sphere.translate(np.array(point), relative=False)


    ### Rotate face to face same direction as normal point
    # R1 = mesh_face.get_rotation_matrix_from_xyz((0, 0, 0))
    print(face_normal)
    matrix = rotation(face_normal[0], -normal)
    m = copy.deepcopy(mesh_face)
    r = copy.deepcopy(m).rotate(matrix, center=m.get_center())
    r.paint_uniform_color(np.array([.8, .2, .2]))


    # Correct transformation where first 
    #       Translate to center
    #       Rotate
    #       Translate to right position
    translate_to_center = np.eye(4)
    translate_to_center[:3, 3] = -m.get_center()

    rm = copy.deepcopy(mesh).transform(translate_to_center)
    rm.paint_uniform_color(np.array([.3, .2, .7]))
    prc = UtilityOpen3d.mesh_to_point_cloud(rm)

    rotate = np.eye(4)
    rotate[:3, :3] = matrix


    #last translation
    translate_to_goal = np.eye(4)
    translate_to_goal[:3, 3] = np.array(point)
    transformation = np.matmul(translate_to_goal, np.matmul(rotate, translate_to_center))
    final_face = copy.deepcopy(mesh_face).transform(transformation)
    final_face.paint_uniform_color(np.array([.2, .9, .9]))

    rm2 = copy.deepcopy(mesh).transform(np.matmul(translate_to_goal, np.matmul(rotate, translate_to_center)))
    rm2.paint_uniform_color(np.array([.3, .2, .7]))
    prc = UtilityOpen3d.mesh_to_point_cloud(rm2)

    mesh.transform(transformation)


    transformation = np.eye(4)
    # transformation[:3, 3] = np.array(point)
    transformation[:3, :3] = matrix
    transformed = copy.deepcopy(m).transform(transformation)
    transformed.paint_uniform_color(np.array([.6, .5, .9]))
    # mesh.transform(transformation)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
    pcd2.estimate_normals()
    o3d.visualization.draw_geometries([box_pcd, sphere, r, transformed, mesh_face, final_face, prc, mesh])

    # o3d.visualization.draw_geometries([pc, sphere, mesh_face, pcd, pcd2, m])



def get_translation(point_from, point_to):
    if len(point_to)==len(point_from):
        return point_to-point_from
    
def construct_rotation_matrix(rx, ry, rz):
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    r_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    r_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Was used just to test if implementation correct
    alt = np.array([[np.cos(ry)*np.cos(rz), np.sin(rx)*np.sin(ry)*np.cos(rz) - np.cos(rx)*np.sin(rz), np.cos(rx)*np.sin(ry)*np.cos(rz) + np.sin(rx)*np.sin(rz)], 
                    [np.cos(ry)*np.sin(rz), np.sin(rx)*np.sin(ry)*np.sin(rz) + np.cos(rx)*np.cos(rz), np.cos(rx)*np.sin(ry)*np.sin(rz) - np.sin(rx)*np.cos(rz)],
                    [-np.sin(ry), np.sin(rx)*np.cos(ry), np.cos(rx)*np.cos(ry)]])
    
    return np.matmul(r_z, np.matmul(r_y, r_x))

def transformation_matrix(x, y, z, rx, ry, rz):
    rotation = construct_rotation_matrix(rx, ry, rz)
    transformation = np.eye(4)
    transformation[:3, 3] = [x, y, z]
    transformation[:3, :3] = rotation
    return transformation

def test():
    x, y, z, rx, ry, rz = 17.811325826769803,-37.225519958085236,-39.16957494060413,-2.097374234576773,-0.7892227998030232,1.6846462525822603
    transformation = transformation_matrix(x, y, z, rx, ry, rz)

    # print(transformation)
    file = 'Labeled Bin - 1x2x5 - pinballgeek'
    box1 = UtilityOpen3d.read_mesh('Boxes STLs/'+file+'.obj')
    box1.translate([0, 0, 0], relative=False)

    box_pcd1 = UtilityOpen3d.mesh_to_point_cloud(mesh=box1, number_of_points=1000)

    objects = [box_pcd1]
    mesh = UtilityOpen3d.read_mesh("Gripper/Grasper_Locomo_scaled.STL")
    pc = UtilityOpen3d.mesh_to_point_cloud(mesh)
    # mesh = /home/swarmlab/camera_realsense/LoCoMo/

    f1 = UtilityOpen3d.read_mesh("Gripper/face5.stl")
    f2 = UtilityOpen3d.read_mesh("Gripper/face6.stl")
    f3 = UtilityOpen3d.read_mesh("Gripper/face10.stl")
    f4 = UtilityOpen3d.read_mesh("Gripper/face13.stl")

    final_face = copy.deepcopy(mesh).transform(transformation)
    o3d.visualization.draw_geometries([box_pcd1, pc, f1, f2, f3, f4, final_face])

idrk()