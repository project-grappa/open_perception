"""
This module contains geometric primitives for feature extraction from point clouds.

Classes:
    Plane: Represents a plane primitive.
    Cylinder: Represents a cylinder primitive.
    Sphere: Represents a sphere primitive.
    Cone: Represents a cone primitive.
    Torus: Represents a torus primitive.
    
Methods:
    - __init__(): Initializes the geometric primitive.
    - get_mesh(): Gets the mesh of the geometric primitive. 
"""
from typing import List, Tuple
import numpy as np
import open3d as o3d
import abc
import torch
from utils import get_rotation_matrix_from_vector, slice_mesh_around_pointcloud
import open3d.visualization as vis


class Primitive:
    def get_mesh(self):
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def equation(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_aux_meshes(self):
        return []
    
    def get_geometries(self, color=[1, 0, 0], aplha=0.5, pc_reference=None) -> list[dict]:
        mat = vis.rendering.MaterialRecord()
        mat.shader = 'defaultLitTransparency'
        mat.base_color = list(color) + [aplha/2]
        mesh = self.get_mesh()
        geometries = [{"name": self.name, "geometry": mesh, "material": mat}]
        if pc_reference is not None:
            mat_inter = vis.rendering.MaterialRecord()
            mat_inter.shader = 'defaultLitTransparency'
            mat_inter.base_color = list(color) + [aplha]
            mesh_inter = slice_mesh_around_pointcloud(mesh, pc_reference, 0.2)
            geometries.append({"name": f"{self.name}_intersection", "geometry": mesh_inter, "material": mat_inter})
        return geometries
    
    def get_aux_geometries(self, color=[0, 1, 0], aplha=1.0):
        meshes = self.get_aux_meshes()

        mat = vis.rendering.MaterialRecord()
        mat.shader = 'defaultLitTransparency'
        mat.base_color = list(color) + [aplha]
        return [{"name": f"{self.name}_{i}", "geometry": aux_mesh, "material": mat} for i, aux_mesh in enumerate(meshes)]
    
    def get_error(self, points):
        params = self.get_params()
        return self.residuals(params, points)
    
    def get_params(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @staticmethod
    def params_to_dict(params):
        raise NotImplementedError("Subclasses should implement this method")
    @staticmethod
    def residuals(params, points):
        raise NotImplementedError("Subclasses should implement this method")
    
# ==================================================================================================
class Plane(Primitive):
    name = "plane"
    def __init__(self, point: Tuple[float, float, float], normal: Tuple[float, float, float]):
        self.point = np.array(point)
        self.normal = np.array(normal)

    def get_aux_meshes(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(self.point - sphere.get_center())
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=0.5, cone_height=0.2)
        arrow.translate(self.point - arrow.get_center())
        arrow.rotate(get_rotation_matrix_from_vector(self.normal))
        return [sphere, arrow]
    
    def get_mesh(self):
        plane = o3d.geometry.TriangleMesh.create_box(width=20, height=20, depth=0.01)
        plane.translate(self.point - plane.get_center())
        plane.rotate(get_rotation_matrix_from_vector(self.normal))
        return plane
    
    def get_params(self):
        return np.hstack((self.point, self.normal))
    @staticmethod
    def params_to_dict(params):
        return {
            "point": params[:3],
            "normal": params[3:]
        }
    @staticmethod
    def residuals(params, points):
        point = params[:3]
        normal = params[3:]
        normal /= np.linalg.norm(normal)
        return np.dot(points - point, normal)

class Cylinder(Primitive):
    name = "cylinder"
    def __init__(self, center: Tuple[float, float, float], axis: Tuple[float, float, float], radius: float, height: float=20):
        self.center = np.array(center)
        self.axis = np.array(axis)
        self.radius = radius
        self.height = height

    def get_aux_meshes(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(self.center - sphere.get_center())
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=0.5, cone_height=0.2)
        arrow.translate(self.center - arrow.get_center())
        arrow.rotate(get_rotation_matrix_from_vector(self.axis))
        return [sphere, arrow]
    
    def get_mesh(self):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius, height=self.height)
        cylinder.translate(self.center - cylinder.get_center())
        cylinder = o3d.t.geometry.TriangleMesh.from_legacy(cylinder)
        small_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius*0.95, height=self.height*1.1)
        small_cylinder.translate(self.center - small_cylinder.get_center())
        small_cylinder = o3d.t.geometry.TriangleMesh.from_legacy(small_cylinder)
        cylinder = cylinder.boolean_difference(small_cylinder).to_legacy()

        R = get_rotation_matrix_from_vector(self.axis)
        cylinder.rotate(R)
        return cylinder
    
    def get_params(self):
        return np.hstack((self.center, self.axis, self.radius))
    @staticmethod
    def params_to_dict(params):
        return {
            "center": params[:3],
            "axis": params[3:6],
            "radius": params[6],
        }

    @staticmethod
    def residuals(params, points):
        center = params[:3]
        axis = params[3:6]
        radius = params[6]
        axis /= np.linalg.norm(axis)
        v = points - center
        v_proj = v - np.dot(v, axis)[:, None] * axis
        return np.linalg.norm(v_proj, axis=1) - radius

class Sphere(Primitive):
    name = "sphere"
    def __init__(self, center: Tuple[float, float, float], radius: float):
        self.center = np.array(center)
        self.radius = radius

    def get_aux_meshes(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(self.center - sphere.get_center())
        return [sphere]
    
    def get_mesh(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
        # sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
        # small_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius*0.95)
        # small_sphere = o3d.t.geometry.TriangleMesh.from_legacy(small_sphere)
        # sphere = sphere.boolean_difference(small_sphere).to_legacy()

        sphere.translate(self.center - sphere.get_center())
        return sphere
    
    def get_params(self):
        return np.hstack((self.center, self.radius))
    @staticmethod
    def params_to_dict(params):
        return {
            "center": params[:3],
            "radius": params[3]
        }

    @staticmethod
    def residuals(params, points):
        center = params[:3]
        radius = params[3]
        return np.linalg.norm(points - center, axis=1) - radius

class Cone(Primitive):
    name = "cone"
    def __init__(self, apex: Tuple[float, float, float], axis: Tuple[float, float, float], tan_angle: float):
        self.apex = np.array(apex)
        self.axis = np.array(axis)
        self.tan_angle = tan_angle

    def get_aux_meshes(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(self.apex - sphere.get_center())
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=0.5, cone_height=0.2)
        arrow.translate(self.apex - arrow.get_center())
        arrow.rotate(get_rotation_matrix_from_vector(self.axis))
        return [sphere, arrow]
    
    def get_mesh(self):
        height = 20
        r = self.tan_angle * height
        cone = o3d.geometry.TriangleMesh.create_cone(radius=r, height=height)
        cone = o3d.t.geometry.TriangleMesh.from_legacy(cone)
        shifted_cone = o3d.geometry.TriangleMesh.create_cone(radius=r, height=height) 
        shifted_cone.translate(np.array([0, 0, -height * 0.5]) - shifted_cone.get_center())
        shifted_cone = o3d.t.geometry.TriangleMesh.from_legacy(shifted_cone)
        cone = cone.boolean_difference(shifted_cone).to_legacy()

        cone.translate(self.apex - cone.get_center() + self.axis * height)
        cone.rotate(-get_rotation_matrix_from_vector(self.axis))
        return cone
    
    def get_params(self):
        return np.hstack((self.apex, self.axis, self.tan_angle))
    @staticmethod
    def params_to_dict(params):
        return {
            "apex": params[:3],
            "axis": params[3:6],
            "tan_angle": params[6]
        }

    @staticmethod
    def residuals(params, points):
        apex = params[:3]
        axis = params[3:6]
        tan_angle = params[6]
        axis /= np.linalg.norm(axis)
        v = points - apex
        height_proj = np.dot(v, axis)
        v_proj = v - height_proj[:, None] * axis
        dist_to_axis = np.linalg.norm(v_proj, axis=1) # distance from point to axis
        return dist_to_axis - tan_angle * height_proj

class Torus(Primitive):
    name = "torus"
    def __init__(self, center: Tuple[float, float, float], axis: Tuple[float, float, float], major_radius: float, minor_radius: float):
        self.center = np.array(center)
        self.axis = np.array(axis)
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def get_aux_meshes(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(self.center - sphere.get_center())
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=0.5, cone_height=0.2)
        arrow.translate(self.center - arrow.get_center())
        arrow.rotate(get_rotation_matrix_from_vector(self.axis))
        return [sphere, arrow]

    def get_mesh(self):
        torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=self.major_radius, tube_radius=self.minor_radius)
        torus.translate(self.center - torus.get_center())
        torus.rotate(get_rotation_matrix_from_vector(self.axis))
        return torus
    
    def get_params(self):
        return np.hstack((self.center, self.axis, self.major_radius, self.minor_radius))
    @staticmethod
    def params_to_dict(params):
        return {
            "center": params[:3],
            "axis": params[3:6],
            "major_radius": params[6],
            "minor_radius": params[7]
        }
    @staticmethod
    def residuals(params, points):
        center = params[:3]
        axis = params[3:6]
        major_radius = params[6]
        minor_radius = params[7]
        axis /= np.linalg.norm(axis)
        v = points - center
        height = np.dot(v, axis)
        r = np.linalg.norm(v - height[:, None] * axis, axis=1)
        return (r**2 + height**2 + major_radius**2 - minor_radius**2)**2 - 4 * major_radius**2 * r**2

class Box(Primitive):
    name = "box"
    def __init__(self, point: Tuple[float, float, float], size: Tuple[float, float, float], direction: Tuple[float, float, float]):
        self.point = np.array(point)
        self.size = np.array(size)
        self.direction = np.array(direction)

    def get_mesh(self):
        box = o3d.geometry.TriangleMesh.create_box(*self.size)
        box.translate(self.point - box.get_center())
        box.rotate(get_rotation_matrix_from_vector(self.direction))

        return box

    def get_params(self):
        return np.hstack((self.point, self.size, self.direction))
    @staticmethod
    def params_to_dict(params):
        return {
            "point": params[:3],
            "size": params[3:6],
            "direction": params[6:]
        }

    @staticmethod
    def residuals(params, points):
        point = params[:3]
        size = params[3:6]
        direction = params[6:]
        direction /= np.linalg.norm(direction)
        points = points - point
        points = np.abs(np.dot(points, direction))
        return np.abs(points - size)


# ==================================================================================================

class PrimitivePointsGenerator:
    def __init__(self,  noise=0.0):
        self.noise = noise
            
    def get_random_points(self, n=100, shape_type=None, partial=False, noise=None):
        
        if noise is None:
            noise = self.noise

        if shape_type is None:
            shape_type = np.random.choice(['plane', 'cylinder', 'sphere', 'cone', 'torus'])

        direction = np.random.rand(3)
        R = get_rotation_matrix_from_vector(direction)
        scale = np.random.rand(1) * 3 + 1
        t = np.random.rand(3)*scale

        if shape_type == 'plane':
            pointcloud = np.random.rand(n, 3) * np.array([0.1, 1.0, 1.0])
        elif shape_type == 'cylinder':
            theta = np.random.rand(n) * 2 * np.pi
            z = np.random.rand(n)
            x = np.cos(theta)*0.1
            y = np.sin(theta)*0.1
            pointcloud = np.vstack((x, y, z)).T
        elif shape_type == 'sphere':
            theta = np.random.rand(n) * 2 * np.pi
            phi = np.random.rand(n) * np.pi
            r = 1
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            pointcloud = np.vstack((x, y, z)).T

        elif shape_type == 'cone':
            theta = np.random.rand(n) * 2 * np.pi
            z = np.random.rand(n)
            x = np.cos(theta)*(1-z)
            y = np.sin(theta)*(1-z)
            pointcloud = np.vstack((x, y, z)).T
        elif shape_type == 'torus':
            theta = np.random.rand(n) * 2 * np.pi
            phi = np.random.rand(n) * 2 * np.pi
            r_max = 1
            r = 0.5
            x = (r_max + r * np.cos(theta)) * np.cos(phi)
            y = (r_max + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            pointcloud = np.vstack((x, y, z)).T

        if partial:
            # iteratively slice the pointcloud and keep only the points above the plane
            for i in range(3):
                # sample a point inside inside the coords range of the pointcloud [10%, 50%]
                min_margin = 0.1
                max_margin = 0.5
                point_range = (np.max(pointcloud, axis=0) - np.min(pointcloud, axis=0)) * ( max_margin- min_margin) 
                point = np.random.rand(3) * point_range + np.min(pointcloud, axis=0) + min_margin * point_range
                

                normal = np.random.rand(3)
                rotation_matrix = get_rotation_matrix_from_vector(normal)
                pointcloud = np.dot(pointcloud, rotation_matrix.T)
                
                #check if num of points is greater then 20% of the original
                if sum(pointcloud[:, 2] > point[2]) < n * 0.1:
                    continue
                pointcloud = pointcloud[pointcloud[:, 2] > point[2]]

        pointcloud *= scale
        pointcloud = np.dot(pointcloud, R.T)
        pointcloud += t
        pointcloud += np.random.randn(*pointcloud.shape) * noise * scale

        return pointcloud