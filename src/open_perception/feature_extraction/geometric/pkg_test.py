import numpy as np
import open3d as o3d

np.random.seed(42)  # For reproducibility

def get_random_traj_between_points(p1, p2, num_points=100, k = 5):
    """
    Generate a random trajectory between two points in 3D space using a spline with noise.
    Args:
        p1 (np.ndarray): Starting point (3D vector).
        p2 (np.ndarray): Ending point (3D vector).
        num_points (int): Number of points in the trajectory.
    Returns:
        np.ndarray: Array of shape (num_points, 3) representing the trajectory.
    """
    # Generate a linear trajectory
    t = np.linspace(0, 1, k)

    linear_trajectory = np.outer(1 - t, p1) + np.outer(t, p2)
    # Generate random noise
    noise = np.random.normal(0, 0.1, linear_trajectory.shape)
    # Add noise to the trajectory
    noisy_trajectory = linear_trajectory + noise
    noisy_trajectory[0] = p1
    noisy_trajectory[-1] = p2
    # Interpolate the trajectory using a cubic spline passing by the points
    trajectory = np.zeros((num_points, 3))
    for i in range(3):
        trajectory[:, i] = np.interp(np.linspace(0, 1, num_points), t, noisy_trajectory[:, i])

    # Normalize the trajectory
    # trajectory = trajectory - np.mean(trajectory, axis=0)
    # trajectory = trajectory / np.linalg.norm(trajectory, axis=1, keepdims=True)
    # # Scale the trajectory
    return trajectory

def rotation_matrix_between_vectors(v1, v2):
    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Cross and dot products
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)

    # If vectors are nearly identical
    if np.isclose(dot_prod, 1.0):
        return np.eye(3)

    # If vectors are nearly opposite
    if np.isclose(dot_prod, -1.0):
        # Find any orthonormal vector to v1
        tmp = np.cross(v1, [1, 0, 0])
        if np.linalg.norm(tmp) < 1e-8:
            tmp = np.cross(v1, [0, 1, 0])
        tmp /= np.linalg.norm(tmp)
        # Rotation by pi around this vector
        K = np.array([[0, -tmp[2], tmp[1]],
                      [tmp[2], 0, -tmp[0]],
                      [-tmp[1], tmp[0], 0]])
        return np.eye(3) + 2 * K @ K

    # Rodrigues' rotation formula
    k = cross_prod / np.linalg.norm(cross_prod)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    angle = np.arccos(dot_prod)

    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def align_traj_to_vector(traj, v2):

    v1 = traj[-1] - traj[0]
    v1_scale = np.linalg.norm(v1)

    v2_scale = np.linalg.norm(v2)

    R = rotation_matrix_between_vectors(v1, v2)
    traj_rotated = np.dot(traj, R.T)
    # Scale the trajectory
    traj_rotated = traj_rotated * v2_scale / v1_scale + traj[0]

    return traj_rotated


# --- Validation Demo ---
# Generate random 3D vectors
v1 = np.random.rand(3)
v2 = np.random.rand(3)

v1_scale = np.linalg.norm(v1)
v2_scale = np.linalg.norm(v2)

# Compute rotation matrix
R = rotation_matrix_between_vectors(v1, v2)
v1_rot = R @ v1 * v2_scale / v1_scale  # Apply the rotation


# Numerical check: v1_rot should match the direction of v2
print("v1 original  =", v1 / np.linalg.norm(v1))
print("v2 target    =", v2 / np.linalg.norm(v2))
print("v1 rotated   =", v1_rot / np.linalg.norm(v1_rot))
angle_diff = np.arccos(np.clip(np.dot(v1_rot/np.linalg.norm(v1_rot),
                                      v2/np.linalg.norm(v2)), -1, 1))
print("Angle difference (radians) =", angle_diff)

# Visualization using Open3D
v1_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([[0, 0, 0], v1]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
v1_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red for v1

v2_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([[0, 0, 0], v2]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
v2_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green for v2

v1_rot_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([[0, 0, 0], v1_rot]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
v1_rot_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue for rotated v1
# add an offset to the v1_rot_line
offset = np.array([0, 0, 0.05])
v1_rot_line.points = o3d.utility.Vector3dVector(np.asarray(v1_rot_line.points) + offset)

goal = np.array([0, 0, 0])

# get a random trajectory between the two points
traj = get_random_traj_between_points(goal, v1, num_points=100, k=5)
# create a line set from the trajectory

traj_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(traj),
    lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
)
traj_line.colors = o3d.utility.Vector3dVector([[0, 1, 1]])  # Cyan for trajectory
# add an offset to the traj_line
# traj_line.points = o3d.utility.Vector3dVector(np.asarray(traj_line.points) + offset)


new_traj = align_traj_to_vector(traj, v2)
# create a line set from the new trajectory
new_traj_line = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(new_traj),
    lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(new_traj) - 1)])
)

new_traj_line.points = o3d.utility.Vector3dVector(np.asarray(new_traj_line.points) + offset)

o3d.visualization.draw_geometries([v1_line, v2_line, v1_rot_line, traj_line, new_traj_line], window_name="Rotation Matrix Validation")
