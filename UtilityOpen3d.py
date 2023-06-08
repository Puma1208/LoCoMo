
import itertools
import open3d as o3d
import numpy as np
import math
from typing import List
from itertools import combinations
from skspatial.objects import Point

def read_point_cloud(file):
    pcd = o3d.io.read_point_cloud(file)
    # o3d.visualization.draw_geometries([pcd],
    #                                 point_show_normal=True)
    pcd.estimate_normals(fast_normal_computation=True)
    # o3d.visualization.draw_geometries([pcd],
    #                             point_show_normal=True)
    return pcd

def read_mesh(file):
    mesh = o3d.io.read_triangle_mesh(file)
    mesh.compute_triangle_normals()
    # o3d.visualization.draw_geometries([mesh],
    #                         point_show_normal=True)
    return mesh
def get_tri_meshes(mesh):
    '''
    convert 1 mesh into multiple meshes of the faces
    Basically split the mesh

    '''
    tri_meshes =[]
    for triangle in mesh.triangles:
        triangle_vertices = [mesh.vertices[vertex] for vertex in triangle]
        triangle_vector = np.array([[0, 1, 2]]).astype(np.int32)
        triangles_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(triangle_vertices), 
                                                   triangles=o3d.utility.Vector3iVector(triangle_vector))
        tri_meshes.append(triangles_mesh)
    return tri_meshes

def visualise_tri_faces(mesh:o3d.geometry.TriangleMesh):
    entire_mesh = []
    for triangle in mesh.triangles:
        vertices = []
        for vertex in triangle:
            vertices.append(mesh.vertices[vertex])
        new_vert = o3d.utility.Vector3dVector(vertices)
        new_tri = np.array([[0, 1, 2]]).astype(np.int32)
        new_triangles = o3d.utility.Vector3iVector(new_tri)
        separate_mesh = o3d.geometry.TriangleMesh(new_vert, new_triangles)
        separate_mesh.paint_uniform_color(np.random.rand(3))
        entire_mesh.append(separate_mesh)
    print("Amount of triangles=", len(entire_mesh))

    o3d.visualization.draw_geometries(entire_mesh)

def visualise_different_meshes(meshes: List[o3d.geometry.TriangleMesh], change_color=True, colors: List[np.array]=[], index:int=0):


    if change_color:
        if(colors==[]):
            # colors = [np.random.rand(3) for _ in range(len(meshes))]
            colors = [np.array([.5, .5, .8]) for _ in range(len(meshes))]
        # new_color = np.array([.8, .2, .2])
        # colors[2:5] = np.repeat(new_color[np.newaxis, :], len(colors[2:5]), axis=0)
        for mesh, color in zip(meshes, colors):
            # mesh.paint_uniform_color(color)
            mesh.paint_uniform_color(color)

    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True, window_name=str(index))
    

def mesh_to_point_cloud(mesh:o3d.geometry.TriangleMesh, number_of_points=5000):
    # o3d.visualization.draw_geometries([mesh])
    mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_uniformly(number_of_points=number_of_points)
    # o3d.visualization.draw_geometries([point_cloud, mesh])
    # print(np.asarray(point_cloud.points))
    return point_cloud

def only_points_from_mesh(mesh):
    '''
    Different from mesh_to_point_cloud - here only the points from file are show
    '''
    points = o3d.geometry.PointCloud(mesh.vertices)
    return points

def simplify_mesh(mesh, simplify_amount=8, color=[.5, 0.706, 0.8]):
    mesh.remove_duplicated_vertices()

    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / simplify_amount
    # print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print(
        f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    )
    mesh_smp.paint_uniform_color(color)

    # o3d.visualization.draw_geometries([mesh_smp])
    return mesh_smp


def mesh_decimation(mesh):
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=6500)
    print(
        f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh_smp])

    mesh_smp2 = mesh.simplify_quadric_decimation(target_number_of_triangles=20)
    print(
        f'Simplified mesh has {len(mesh_smp2.vertices)} vertices and {len(mesh_smp2.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh_smp2])
    return mesh_smp, mesh_smp2


def distance_queries(mesh, point):
    # http://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html
    # Convert a mesh to an implicit representation
    # 1. Initialise a RaycastingScene
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    #2. compute distances and occupancy for a single point
    query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
    # Compute distance of the query point from the surface
    unsigned_distance = scene.compute_distance(query_point)
    signed_distance = scene.compute_signed_distance(query_point)
    occupancy = scene.compute_occupancy(query_point)

    print('unsigned distance = ', unsigned_distance.numpy())
    print('signed distance = ', signed_distance.numpy())
    print('occupancy = ', occupancy.numpy())

def distance_mesh_points(mesh, 
                         points:o3d.cpu.pybind.geometry.PointCloud):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()

    _ = scene.add_triangles(mesh)
    points = o3d.core.Tensor(np.asarray(points.points), dtype=o3d.core.Dtype.Float32)
    distance = scene.compute_signed_distance(points)
    return distance.numpy()

def distance_point_clouds(pc_1, pc_2):
    return pc_1.compute_point_cloud_distance(pc_2)

def get_adjacent_faces(m:o3d.geometry.TriangleMesh, face:o3d.geometry.TriangleMesh, visualise:bool=True):
    adjacent = []
    mesh_tri = get_tri_meshes(m)
    reference_mesh = np.array(face.vertices)

    for m in mesh_tri:
        mesh_matrix = np.array(m.vertices)

        if np.sum(np.sum(np.equal(mesh_matrix, reference_mesh[:, None]).all(axis=2), axis=1)) >= 2 :
            adjacent.append(m)
        #
        #   OPTION 2
        # 
        # conditions = [    np.all((np.array(mesh.vertices)==(np.array(mesh_tri[2].vertices[2]))), axis=1),
        #                 np.all((np.array(mesh.vertices)==(np.array(mesh_tri[2].vertices[1]))), axis=1),
        #                 np.all((np.array(mesh.vertices)==(np.array(mesh_tri[2].vertices[0]))), axis=1)]
        # comb = [np.all(np.any(c, axis=1)) for c in combinations(conditions, r=2)]
        # print(comb)
        # # At least 2 points are the same as the mesh vertex
        # if np.count_nonzero(comb)>=2:
        #     print('_____________')
        #     colors = [np.array([.8, .8, .8]) for _ in range(len(mesh_tri))]
        #     new_list = mesh_tri.copy()
        #     new_list.append(mesh)
        #     new_col = colors.copy()
        #     new_col[mesh_tri.index(mesh)] = np.random.rand(3)
        #     visualise_different_meshes(new_list, True, new_col)
        #     searching.append(mesh)


    # Get the indeces of the faces and add them some color
    colors = [np.array([.8, .9, .9]) for _ in range(len(mesh_tri))]
    for m in adjacent:
        print('Adjacent face : ', mesh_tri.index(m))
        colors[mesh_tri.index(m)] = np.random.rand(3)
    if visualise:
        visualise_different_meshes(meshes=mesh_tri, change_color=True, colors=colors)





# read_point_cloud('1_cone_hat.stl')
# mesh = read_mesh("Boxes STLs/Labeled Bin - 1x2x5 - pinballgeek.obj")
mesh2 = read_mesh("Grasper_Locomo.STL")

box_pcd = mesh_to_point_cloud(mesh=mesh2, number_of_points=10000)
mesh_simple = simplify_mesh(mesh2, simplify_amount=7)
mesh_tri = get_tri_meshes(mesh_simple)
# get_adjacent_faces(mesh_simple, mesh_tri[40])

for mesh in mesh_tri:
    colors = [np.array([.8, .9, .9]) for _ in range(len(mesh_tri))]
    colors[mesh_tri.index(mesh)] = np.random.rand(3)
    # visualise_different_meshes(meshes=mesh_tri, change_color=True, colors=colors, index=mesh_tri.index(mesh))

# get_adjacent_faces(mesh2, mesh2.triangles[2], )
# mesh2_simple = simplify_mesh(mesh2,simplify_amount=8, color=[.5, 0.6, 1])
# o3d.visualization.draw_geometries([ box_pcd, mesh_simple])
# print('points = ', np.asarray(box_pcd.points))
# pointsss = np.asarray(box_pcd.points)
# print(pointsss[0] , ' -> ', (math.sqrt(pointsss[0][0]**2 + pointsss[0][1]**2+pointsss[0][2]**2)))
# point = [[0, 0, 0]]
# point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
# d = distance_point_clouds(box_pcd, point)
# print(d)
# visualise_tri_faces(mesh=mesh_simple)

# dista

# distance_queries(mesh=mesh, point=[0, 0, 0])
# mesh_to_point_cloud(mesh)
# n = only_points_from_mesh(mesh)