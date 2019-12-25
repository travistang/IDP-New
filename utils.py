from sklearn.decomposition import PCA
import numpy as np

def get_velocity_from_trajectories(trajectories):
    velocities = np.diff(trajectories, axis = 1).mean(axis = 1)
    return velocities


def remove_irrelevant_trajectories(trajectories_list):
    velocities = [
        get_velocity_from_trajectories(traj)
        for traj in trajectories_list
    ]
    
    # accumulate all velocities
    all_velocities = np.concatenate(velocities, axis = 0)
    mean_velocities = all_velocities.mean(axis = 0, keepdims = True).T
    mean_velocities /= np.linalg.norm(mean_velocities)
    # and drop those closer to the trajectoires    
    valid_indices  = [
        np.argwhere(
            np.arccos(
                np.divide(np.dot(vel, mean_velocities), np.linalg.norm(vel, axis = -1, keepdims = True))
            ) < np.deg2rad(20)
        )[:, 0]
        for vel in velocities
    ]
    return [
        traj[vi]
        for traj, vi in zip(trajectories_list, valid_indices)
        if len(vi) > 0
    ]

def as_individual_coordinates(trajectories_list):
    all_trajectories = np.concatenate(trajectories_list, axis = 0)
    n_points = all_trajectories.size // 2
    return all_trajectories.reshape(n_points, 2)

def normalize_trajectories(trajectories_list):
    # first flip and rotate components by performing PCA
    pca = PCA(n_components = 2)
    all_trajectories = as_individual_coordinates(trajectories_list)
    pca.fit(all_trajectories)

    rotated_trajectories = [traj @ pca.components_.T for traj in trajectories_list]
    all_trajectories = as_individual_coordinates(rotated_trajectories)
    # then renormalize all axis to -1, 1
    # examine axis and normalize each of them
    x_min, y_min = all_trajectories.min(axis = 0)
    x_max, y_max = all_trajectories.max(axis = 0)
    
    for i in range(len(rotated_trajectories)):
        xs, ys = rotated_trajectories[i][..., 0], rotated_trajectories[i][..., 1]

        rotated_trajectories[i][..., 0] = 2 / (x_max - x_min) * (xs - x_min) - 1
        rotated_trajectories[i][..., 1] = 2 / (y_max - y_min) * (ys - y_min) - 1
    
    return rotated_trajectories
