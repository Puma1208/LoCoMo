# Euler's angles
import numpy as np
from scipy.spatial.transform import Rotation
import UtilityOpen3d
import open3d as o3d

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
