"""
This module contains geometric regressors for fitting geometric primitives to data.

Classes:
    - GeometricRegressor: A base class for geometric regression models.

Methods:
    - fit(): Fits a geometric primitive to the given pointcloud data.
    - predict(): Predicts geometric configurations based on the fitted model.
"""

from primitives import Primitive, Sphere, Cylinder, Cone, Torus, Plane, PrimitivePointsGenerator
from typing import Tuple
import numpy as np
import open3d as o3d
import scipy.optimize as opt

class GeometricRegressor:
    def __init__(self):
        self.model = None

        self.type_to_primitive = {
            'plane': Plane,
            'cylinder': Cylinder,
            'sphere': Sphere,
            'cone': Cone,
            'torus': Torus,
        }
        self.type_to_primitive_fit = {
            'plane': self._fit_plane,
            'cylinder': self._fit_cylinder,
            'sphere': self._fit_sphere,
            'cone': self._fit_cone,
            'torus': self._fit_torus,
        }

    def fit(self, pointcloud: np.ndarray, primitive_type: str) -> Primitive:

        if self.type_to_primitive_fit.get(primitive_type) is None:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")
        else:
            self.model, result = self.type_to_primitive_fit[primitive_type](pointcloud)
        return self.model, result
    
    def predict(self):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model

    def _fit_plane(self, pointcloud: np.ndarray):
        plane_residuals = Plane.residuals

        centroid = np.mean(pointcloud, axis=0)
        normal = np.array([0, 0, 1])
        initial_guess = np.hstack((centroid, normal))
        result = opt.least_squares(plane_residuals, initial_guess, args=(pointcloud,))
        point, normal = result.x[:3], result.x[3:]
        point = centroid - np.dot(centroid - point, normal) * normal
        
        return Plane(point, normal), result

    def _fit_cylinder(self, pointcloud: np.ndarray):
        cylinder_residuals = Cylinder.residuals

        centroid = np.mean(pointcloud, axis=0)
        axis = np.array([0, 0, 1])
        radius = np.mean(np.linalg.norm(pointcloud - centroid, axis=1))
        initial_guess = np.hstack((centroid, axis, radius))
        result = opt.least_squares(cylinder_residuals, initial_guess, args=(pointcloud,))
        center, axis, radius = result.x[:3], result.x[3:6], result.x[6]
        center = center - np.dot(center - centroid, axis) * axis
        height = np.ptp(np.dot(pointcloud - center, axis))
        return Cylinder(center, axis, radius, height), result

    def _fit_sphere(self, pointcloud: np.ndarray):
        sphere_residuals = Sphere.residuals

        centroid = np.mean(pointcloud, axis=0)
        radius = np.mean(np.linalg.norm(pointcloud - centroid, axis=1))
        initial_guess = np.hstack((centroid, radius))
        result = opt.least_squares(sphere_residuals, initial_guess, args=(pointcloud,))
        center, radius = result.x[:3], result.x[3]
        return Sphere(center, radius), result

    def _fit_cone(self, pointcloud: np.ndarray):
        cone_residuals = Cone.residuals
        centroid = np.mean(pointcloud, axis=0)
        axis = np.array([0, 0, 1])
        angle = np.pi / 6
        initial_guess = np.hstack((centroid, axis, angle))
        result = opt.least_squares(cone_residuals, initial_guess, args=(pointcloud,))
        apex, axis, angle = result.x[:3], result.x[3:6], result.x[6]
        return Cone(apex, axis, angle), result

    def _fit_torus(self, pointcloud: np.ndarray):
        torus_residuals = Torus.residuals

        centroid = np.mean(pointcloud, axis=0)
        axis = np.array([0, 0, 1])
        major_radius = np.mean(np.linalg.norm(pointcloud - centroid, axis=1))
        minor_radius = major_radius / 2
        initial_guess = np.hstack((centroid, axis, major_radius, minor_radius))
        result = opt.least_squares(torus_residuals, initial_guess, args=(pointcloud,))
        center, axis, major_radius, minor_radius = result.x[:3], result.x[3:6], result.x[6], result.x[7]
        return Torus(center, axis, major_radius, minor_radius), result

if __name__ == "__main__":

    n = 1000
    for shape_type in ['plane', 'cylinder', 'sphere', 'cone', 'torus']:
        pts_gen = PrimitivePointsGenerator()
        pointcloud = pts_gen.get_random_points(n, partial=True, shape_type=shape_type)

        regressor = GeometricRegressor()
        shape, res = regressor.fit(pointcloud, primitive_type=shape_type)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        shape_geometries = shape.get_geometries(pc_reference=pointcloud)
        
        geometries = [{"name": "pcd", "geometry": pc},
                    {"name": "frame", "geometry": frame},
                    ]+ shape_geometries

        o3d.visualization.draw(geometries)