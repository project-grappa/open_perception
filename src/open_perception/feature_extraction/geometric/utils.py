import numpy as np
import open3d as o3d
from typing import Tuple, List
from open3d.geometry import TriangleMesh

import itertools

def get_rotation_matrix_from_vector(v: Tuple[float, float, float]):
    v_unit = v / np.linalg.norm(v)
    z_axis = np.array([0, 0, 1])
    cross_vec = np.cross(z_axis, v_unit)
    cross_norm = np.linalg.norm(cross_vec)

    if cross_norm < 1e-8:
        # v is parallel or anti-parallel to the z-axis
        if np.dot(z_axis, v_unit) < 0:
            # Rotate 180 degrees around x (or any axis orthogonal to z)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0, np.pi])
        else:
            R = np.eye(3)
    else:
        angle = np.arccos(np.dot(z_axis, v_unit))
        axis = cross_vec / cross_norm
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    return R


def slice_mesh_around_pointcloud(mesh: o3d.geometry.TriangleMesh,pointcloud: np.ndarray,
                                 tolerance: float = 0.1) -> o3d.geometry.TriangleMesh:
    pointcloud_extedned = (pointcloud - np.mean(pointcloud, axis=0)) * (1 + tolerance) + np.mean(pointcloud, axis=0)
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pointcloud_extedned)

    hull = pc.compute_convex_hull()[0]
    hull_t = o3d.t.geometry.TriangleMesh.from_legacy(hull)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    intersection = hull_t.boolean_intersection(mesh_t)
    return intersection


# def slice_mesh_around_pointcloud_old(mesh: o3d.geometry.TriangleMesh,
#                                  pointcloud: np.ndarray,
#                                  tolerance: float = 0.1) -> o3d.geometry.TriangleMesh:

#     #fit an oriented bounding box around the pointcloud
#     pointcloud_o3d = o3d.geometry.PointCloud()
#     pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud)
#     bbox = pointcloud_o3d.get_oriented_bounding_box()
#     bbox.extent = bbox.extent * (1 + tolerance)
#     # for each face of of the oriened_bounding_box, slice the mesh and keep the part that is inside the bbox

#     vertices = np.asarray(mesh.vertices)
#     triangles = np.asarray(mesh.triangles)
#     new_vertices = list(vertices)
#     new_triangles = []

#     intersection_edges = dict()

#     def compute_intersection(obb, p1_index, p2_index):
#         edge_vector = vertices[p2_index] - vertices[p1_index]
#         edge_length = np.linalg.norm(edge_vector)
#         edge_vector /= edge_length
#         ray_points = np.array(np.linspace(0, edge_length, 10))[:,None] * edge_vector + vertices[p1_index]
#         ray_inside_bbox = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(ray_points))
#         if len(ray_inside_bbox) > 0:
#             new_vertices.append(ray_points[max(ray_inside_bbox)])
#             intersection_index = len(new_vertices) - 1
#             intersection_edges[(p1_index, p2_index)] = intersection_index
#             return intersection_index
#         else:
#             print("No intersection found")
#             return p2_index
        
#     for triangle in triangles:
#         v1, v2, v3 = triangle
#         # triangle = [mesh.vertices[p_idx] for p_idx in triangle]
#         points = o3d.utility.Vector3dVector([mesh.vertices[p_idx] for p_idx in triangle])
#         points_idx_inside_bbox = bbox.get_point_indices_within_bounding_box(points)
#         vertices_idx_inside_bbox = [i for i in range(3) if i in points_idx_inside_bbox]
#         vertices_idx_outside_bbox = [i for i in range(3) if i not in points_idx_inside_bbox]
#         if len(points_idx_inside_bbox) == 3:
#             # new_vertices.extend(triangle)
#             new_triangles.append(triangle)

#         elif len(points_idx_inside_bbox) == 1: # 2 points out
#             # all points are outside the obb

#             # add new triangle 
#             new_triangles.append([vertices_idx_inside_bbox[0],
#                                 compute_intersection(bbox, vertices_idx_inside_bbox[0], vertices_idx_outside_bbox[0]), 
#                                 compute_intersection(bbox, vertices_idx_inside_bbox[0], vertices_idx_outside_bbox[0])])

#         elif len(points_idx_inside_bbox) == 2:
#             # one vertice outside, add 2 triangles
#             intersection_idx_1 = compute_intersection(bbox, vertices_idx_inside_bbox[0], vertices_idx_outside_bbox[0])
#             new_triangles.append([vertices_idx_inside_bbox[0],
#                                   intersection_idx_1,
#                                   vertices_idx_inside_bbox[1]])
            
#             intersection_idx_2 = compute_intersection(bbox, vertices_idx_inside_bbox[1], vertices_idx_outside_bbox[0])
#             new_triangles.append([intersection_idx_1, 
#                                   intersection_idx_2,
#                                   vertices_idx_inside_bbox[1]])
#         else:
#             # handle the case where the triangle intersects the bounding box

#             pass

#     new_mesh = o3d.geometry.TriangleMesh()
#     new_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
#     new_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))

#     return new_mesh
