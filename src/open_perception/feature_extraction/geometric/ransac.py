"""
ransac.py

Implements ransac to fit multiple 3d primitive shapes into a point cloud

classes:
    RANSAC
"""

from regressor import GeometricRegressor
import numpy as np
import open3d as o3d

class RANSAC:
    def __init__(self, max_iter: int, min_support: int, inlier_threshold: float):
        self.max_iter = max_iter
        self.min_support = min_support
        self.inlier_threshold = inlier_threshold
        self.regressor = GeometricRegressor()

    def random_partition(self, n, n_data):
        """return n random rows of data (and also the other len(data)-n rows)"""
        all_idxs = np.arange(n_data)
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2
    
    def fit(self, pointcloud: np.ndarray, primitive_type: str, sample_points: int = 5):
        """
        Fits the primitive shape into the point cloud using RANSAC

        Args:
            pointcloud: np.ndarray: point cloud data
            primitive_type: str: type of primitive shape to fit
            sample_points: int: number of points to sample from the point cloud
            d: int: minimum number of inliers to accept the primitive shape

        Returns:
            list: list of primitive shapes fitted into the point cloud
        """

        
        data = pointcloud
        best_model = None
        best_inliers_idx = []
        best_err = np.inf
        # sample_points = pointcloud.shape[0]
        for i in range(self.max_iter):
            maybe_idxs, test_idxs = self.random_partition(sample_points,data.shape[0])
            model, res = self.regressor.fit(data[maybe_idxs], primitive_type=primitive_type)
            maybe_inliers = data[test_idxs,:]

            test_err = model.get_error(data[test_idxs])
            also_idxs = test_idxs[test_err < self.inlier_threshold] # select indices of rows with accepted points
            also_inliers = data[also_idxs,:]

            #debug
            # print(f"{i} test_err: {test_err} \t also_idxs: {also_idxs} \t also_inliers: {also_inliers}")

            if len(also_inliers) > self.min_support:
                better_data = np.concatenate( (maybe_inliers, also_inliers) )
                
                better_model, res = self.regressor.fit(better_data, primitive_type=primitive_type)
                better_err = better_model.get_error(better_data)
                this_err = np.mean(better_err)
                if this_err < best_err:
                    best_model = better_model
                    best_inliers_idx = np.concatenate((maybe_idxs, also_idxs))
                    best_err = this_err
                    
        return best_model, best_inliers_idx

    def predict(self):
        """
        Predicts the primitive shapes in the point cloud

        Returns:
            list: list of predicted primitive shapes
        """
        pass

class PrimitiveFinder:
    """uses RANSAC to iteratively fit multiple primitive shapes into a point cloud"""
    def __init__(self, max_iter: int, min_support: int, inlier_threshold: float):
        self.max_iter = max_iter
        self.min_support = min_support
        self.inlier_threshold = inlier_threshold
        self.ransac = RANSAC(max_iter, min_support, inlier_threshold)

    def predict(self, pointcloud: np.ndarray,
                primitive_types: list = ['plane', 'cylinder', 'sphere', 'cone', 'torus'],
                max_primitives: int = 10):
        """
        Fits multiple primitive shapes into a point cloud using RANSAC
        """
        shapes = []
        shape_inliers = []
        points_left = pointcloud.copy()
        
        while len(shapes) < max_primitives and len(points_left) > self.min_support:
            print(f"points_left: {len(points_left)}")
            best_shape = None
            best_inliers = []
            for primitive_type in primitive_types:
                shape, inliers = self.ransac.fit(points_left, primitive_type=primitive_type)
                if len(inliers) > self.min_support and len(inliers) > len(best_inliers):
                    best_shape = shape
                    best_inliers = inliers
                    print(f"best_shape: {best_shape} \t best_inliers: {len(best_inliers)}")
            
            if best_shape is not None:
                points_left = np.delete(points_left, best_inliers, axis=0)
                shapes.append(best_shape)
                shape_inliers.append(best_inliers)
            else:
                break
        return shapes, shape_inliers

if __name__ == "__main__":
    from primitives import Primitive, Sphere, Cylinder, Cone, Torus, Plane, PrimitivePointsGenerator

    # build random scene with multiple primitive shapes randomly placed
    noise_points = 30
    pc_scene = np.random.rand(noise_points, 3) 
    # for shape in ['plane', 'cylinder', 'sphere', 'cone', 'torus']:
    for shape in ['plane', 'cylinder', 'plane', 'cylinder', 'torus']:

        shape_pc_gen = PrimitivePointsGenerator(noise=0.01)
        shape_points = shape_pc_gen.get_random_points(n=1000, partial=True, shape_type=shape)
        pc_scene = np.vstack((pc_scene, shape_points))

    # display scene pointcloud
    pc_scene_o3d = o3d.geometry.PointCloud()
    pc_scene_o3d.points = o3d.utility.Vector3dVector(pc_scene)
    # o3d.visualization.draw_geometries([pc_scene_o3d])

    # fit primitive shapes using RANSAC, try each primitive type, the one with the most inliers is the best fit
    primitive_finder = PrimitiveFinder(max_iter=100, min_support=50, inlier_threshold=0.01)
    
    shapes, shape_inliers = primitive_finder.predict(pc_scene, primitive_types=['plane', 'cylinder', 'torus'])

    # display all shapes
    all_geometries = [{"name": "pcd", "geometry": pc_scene_o3d}]
    for shape, inliers in zip(shapes, shape_inliers):  # corrected from shape_inilers to shape_inliers
        shape_geometries = shape.get_geometries(pc_reference=pc_scene[inliers])
        all_geometries += shape_geometries

    o3d.visualization.draw(all_geometries)
