# Euler's angles
import numpy as np
from scipy.spatial.transform import Rotation
import UtilityOpen3d
import open3d as o3d
import copy
import LoCoMo

def rotation(v1, v2):
    # https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724

    axis = np.cross(v1, v2)
    cosA = np.dot(v1, v2)
    k = 1/(1+cosA)

    res = np.array([[   (axis[0]*axis[0]*k)+cosA, (axis[1]*axis[0]*k)-axis[2], (axis[2]*axis[0]*k)+axis[1]],
                    [   (axis[0]*axis[1]*k)+axis[2], (axis[1]*axis[1]*k)+cosA, (axis[2]*axis[1]*k)-axis[0]],
                    [   (axis[0]*axis[2]*k)-axis[1], (axis[1]*axis[2]*k)+axis[0], (axis[2]*axis[2]*k)+cosA]])


    return res

obj = UtilityOpen3d.read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
box_pcd = UtilityOpen3d.mesh_to_point_cloud(mesh=obj, number_of_points=1000)

mesh = UtilityOpen3d.read_mesh("grasper_scaled.STL")
mesh_face = UtilityOpen3d.read_mesh('Gripper/face5.stl')
face_normal = mesh_face.triangle_normals

# n = o3d.compute_triangle_normals(mesh_face)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh_face.vertices))
pcd.estimate_normals()

pc = UtilityOpen3d.mesh_to_point_cloud(mesh, 100)

sphere = o3d.geometry.TriangleMesh.create_sphere(1)
sphere.paint_uniform_color(np.array([.5, .5, .9]))
point = np.array(box_pcd.points)[1]
normal = np.array(box_pcd.normals)[1]
matrix = rotation(face_normal[0], normal)
print(matrix)

sphere.translate(np.array(point), relative=False)


### Rotate face to face same direction as normal point
# R1 = mesh_face.get_rotation_matrix_from_xyz((0, 0, 0))
print('normal=', normal)

R2 = o3d.geometry.get_rotation_matrix_from_xyz(normal)
m = copy.deepcopy(mesh_face)
m.rotate(matrix, center=m.get_center())
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
pcd2.estimate_normals()
# o3d.visualization.draw_geometries([mesh, mesh_r])
# o3d.visualization.draw_geometries([pc, sphere, mesh_face, pcd, pcd2, m])



poses, t = LoCoMo.sample_finger_poses_opposite(point, normal, finger_face=mesh_face, finger_mesh=mesh, amount_poses=2)
face = copy.deepcopy(mesh_face).transform(t[0])

o3d.visualization.draw_geometries([box_pcd, sphere, poses[0], face])


rotation = np.array([   [ 3.36565656e-01, -3.65493022e-01,  8.67835474e-01],
                        [-3.98180461e-01,  7.79917068e-01,  4.82888897e-01],
                        [-8.53332221e-01, -5.08078948e-01,  1.16961120e-01]])

print(rotation)

r = Rotation.from_matrix(rotation)
print('okay')
print(r.as_matrix())
angles = r.as_euler('xyz', degrees=False)

print('Rotation angles = ', angles)

mesh = UtilityOpen3d.read_mesh("Grasper_Locomo.STL")
mesh.paint_uniform_color(np.array([.5, .5, .9]))
# o3d.visualization.draw_geometries([mesh])


going_back = Rotation.from_euler('xyz', angles, degrees=False)

# r = R.from_euler('zyx', [[90, 45, 30]], degrees=True)
rotation = going_back.as_matrix()
print('rotation matrix')
print(rotation)


r = Rotation.from_matrix(rotation)
angles = r.as_euler('xyz', degrees=False)

print('Rotation angles = ', angles)

# def rotate_to_vector():
    
