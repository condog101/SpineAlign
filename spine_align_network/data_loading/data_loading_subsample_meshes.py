import torch
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, Sampler
import numpy as np
import open3d as o3d
import pickle
import scipy
from pytorch3d.ops import ball_query, knn_gather, knn_points, points_alignment, sample_points_from_meshes
from pytorch3d.transforms import Rotate, random_rotations, Scale, euler_angles_to_matrix
from random import randint, uniform
import os
from pathlib import Path
from color_transforms import adjust_brightness, adjust_contrast, adjust_gamma, adjust_hue, adjust_saturation, expand_to_im_size, reduce_from_im_size
from pytorch3d.structures import Meshes
"""
source_paths = ['data/source/cloud1.npy', 'data/source/cloud2.npy']
target_paths = ['data/target/cloud1.npy', 'data/target/cloud2.npy']

train_loader = create_dataloader(source_paths, target_paths)

for xyz, rgb, target_xyz in train_loader:
    # xyz: [B, 921600, 3]
    # rgb: [B, 921600, 3]
    # target_xyz: [B, 4000, 3]
    break
"""


transformed_file = "../CT_DATA/CT_NIFTIS/NIFTI_SOLIDS_TRANSFORMED"


def open3d_to_pytorch3d(o3d_mesh):
    # Get vertices and faces
    vertices = torch.FloatTensor(np.asarray(o3d_mesh.vertices))
    faces = torch.LongTensor(np.asarray(o3d_mesh.triangles))

    # Create PyTorch3D mesh (add batch dimension)
    verts = vertices.unsqueeze(0)  # (1, V, 3)
    faces = faces.unsqueeze(0)     # (1, F, 3)

    return Meshes(verts=verts, faces=faces)


def normalize_pointcloud_tensor(points):
    # Center
    centroid = points.mean(dim=0)
    points = points - centroid

    # Scale to unit sphere
    scale = torch.max(torch.norm(points, dim=1))
    points = points / scale

    return points


def normalize_pointcloud_tensor_with_scale(points):
    # Center
    centroid = points.mean(dim=0)
    points = points - centroid

    # Scale to unit sphere
    scale = torch.max(torch.norm(points, dim=1))
    points = points / scale

    return points, scale, centroid


def load_transformed_meshes():
    prefix = "mesh_transformed_"
    meshes = {}
    for filename in os.listdir(transformed_file):
        if prefix in filename and "undeformed" not in filename:

            mesh_id = filename.split(
                '/')[-1].removeprefix(prefix).removesuffix(".ply")
            file_path = os.path.join(transformed_file, filename)

            # Process file

            mesh = o3d.io.read_triangle_mesh(file_path)
            meshes[mesh_id] = mesh

    return meshes


def center_pointcloud_numpy(pointcloud, centroid=None):
    """
    Center a point cloud so its centroid is at the origin.
    Args:
        pointcloud: NumPy array of shape (N, 3) or (B, N, 3) for a batch.
    Returns:
        Centered pointcloud of the same shape.
    """
    if centroid is None:
        centroid = pointcloud.mean(
            axis=0, keepdims=True)  # (B, 1, 3) or (1, 3)
    return pointcloud - centroid, centroid


def scale_pointcloud_to_unit_sphere_numpy(pointcloud, max_distance=None):
    """
    Scale a point cloud to fit within a unit sphere.
    Args:
        pointcloud: NumPy array of shape (N, 3) or (B, N, 3) for a batch.
    Returns:
        Scaled pointcloud of the same shape.
    """
    if max_distance is None:
        max_distance = np.sqrt(
            (pointcloud ** 2).sum(axis=-1)).max()   # (B, 1) or scalar
    return pointcloud / max_distance, max_distance


def get_intersection_indices(points1, points2, tolerance=1e-10):
    """
    Find indices of points in points1 that intersect with points2.

    Args:
        points1: numpy array of shape (N, D) representing first point cloud
        points2: numpy array of shape (M, D) representing second point cloud
        tolerance: float, maximum distance to consider points as identical

    Returns:
        numpy array of indices from points1 that have matching points in points2
    """
    # Using a vectorized approach for efficiency
    points2_set = set(map(tuple, points2))
    return np.array([tuple(p) in points2_set for p in points1]).astype(int)


def random_small_rotation(batch_size, max_degrees=15):
    max_radians = max_degrees * torch.pi / 180.0
    angles = (torch.rand(batch_size, 3) - 0.5) * 2 * max_radians
    return Rotate(euler_angles_to_matrix(angles, "XYZ"))


def Jitter(points, v):
    assert 0.0 <= v <= 10
    v = int(v)

    sigma = 0.1 * v
    n_idx = 50 * v

    idx = np.random.choice(points.size(0), n_idx)
    jitter = sigma * (np.random.random([n_idx, 3]) - 0.5)
    points[idx, 0:3] += torch.from_numpy(jitter).float()
    return points


def RandomFlipZ(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 2] *= -1
    return points


def RandomErase(points, v, target_clouds_inds):
    assert 0 <= v <= 0.2
    "v : the radius of erase ball"
    valid_indices = np.where(~target_clouds_inds.astype(bool))[0]
    random_idx = valid_indices[np.random.randint(len(valid_indices))]
    mask = torch.sum(
        (points[:, 0:3] - points[random_idx, 0:3]).pow(2), dim=1) < v ** 2
    mask = mask & ~torch.from_numpy(target_clouds_inds)
    masked = points[random_idx].clone()
    points[mask] = masked
    return points


def RandomDropout(points, v, exclude_mask):
    assert 0.01 <= v <= 0.4
    dropout_rate = v
    drop = torch.rand(points.size(0)) < dropout_rate
    valid_indices = np.where(~exclude_mask.astype(bool))[0]
    save_idx = valid_indices[np.random.randint(len(valid_indices))]
    drop = drop & ~torch.from_numpy(exclude_mask)
    masked = points[save_idx].clone()
    points[drop] = masked
    return points


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )

    return R.float()


def RotateX(points, v):
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([1., 0., 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points


def RotateY(points, v):  # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0., 1., 0.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points


def RandomFlipX(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 0] *= -1
    return points


def RandomFlipY(points, v):
    assert 0 <= v <= 1
    if np.random.random() < v:
        points[:, 1] *= -1
    return points


def RotateZ(points, v):  # ( 0 , 2 * pi)
    assert 0 <= v <= 2 * np.pi
    if np.random.random() > 0.5:
        v *= -1
    axis = np.array([0., 0., 1.])

    rotation_angle = np.random.uniform() * v
    rotation_matrix = angle_axis(rotation_angle, axis)

    normals = points.size(1) > 3
    if not normals:
        return points @ rotation_matrix.t()
    else:
        pc_xyz = points[:, 0:3]
        pc_normals = points[:, 3:]
        points[:, 0:3] = pc_xyz @ rotation_matrix.t()
        points[:, 3:] = pc_normals @ rotation_matrix.t()

        return points


def RotatePerturbation(points, v):
    assert 0 <= v <= 10
    v = int(v)
    angle_sigma = 0.1 * v
    angle_clip = 0.1 * v
    n_idx = 50 * v

    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
    Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
    Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

    rotation_matrix = Rz @ Ry @ Rx

    center = torch.mean(points[:, 0:3], dim=0)
    idx = np.random.choice(points.size(0), n_idx)

    perturbation = points[idx, 0:3] - center
    points[idx, :3] += (perturbation @ rotation_matrix.t()) - perturbation

    normals = points.size(1) > 3
    if normals:
        pc_normals = points[idx, 3:]
        points[idx, 3:] = pc_normals @ rotation_matrix.t()

    return points


def white_color_dropout(colors: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """
    Randomly sets color channels to white (1.0) with probability p.
    Works with PyTorch tensors.

    Args:
        colors (torch.Tensor): Color data of shape (N, 3) for RGB values in range [0, 1]
        p (float): Probability of dropping out (whitening) each color channel

    Returns:
        torch.Tensor: Colors with some channels set to white
    """
    # Generate random mask where True means keep original, False means set to white
    mask = torch.rand_like(colors) > p

    # Create output tensor starting with all white
    white_dropped = torch.ones_like(colors)

    # Keep original colors where mask is True
    white_dropped = torch.where(mask, colors, white_dropped)

    return white_dropped


def color_contrast(colors: torch.Tensor, contrast_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """
    Apply random contrast adjustment to RGB color channels.

    Args:
        colors (torch.Tensor): Color data of shape (N, 3) for RGB values in range [0, 1]
        contrast_range (tuple): Range for random contrast adjustment (min, max)

    Returns:
        torch.Tensor: Colors with adjusted contrast
    """
    # Generate random contrast factors for each color channel
    contrast_factors = torch.rand(3, device=colors.device) * (
        contrast_range[1] - contrast_range[0]
    ) + contrast_range[0]

    # Calculate mean color across points
    mean_color = colors.mean(dim=0, keepdim=True)

    # Apply contrast adjustment
    adjusted_colors = mean_color + contrast_factors * (colors - mean_color)

    # Clamp values to [0, 1]
    return torch.clamp(adjusted_colors, 0, 1)


def ScaleX(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1-v, high=1 + v)
    pts[:, 0] *= scaler
    return pts


def ScaleY(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1-v, high=1 + v)
    pts[:, 1] *= scaler
    return pts


def ScaleZ(pts, v):  # (0 , 2)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low=1-v, high=1 + v)
    pts[:, 2] *= scaler
    return pts


def NonUniformScale(pts, v, exclude_indices):  # Resize in [0.5 , 1.5]
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(
        low=1 - v, high=1 + v, size=3)
    pts[~exclude_indices.astype(bool), 0:3] *= torch.from_numpy(scaler).float()
    return pts


def color_jitter(colors: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """
    Apply random jittering to RGB color channels independently.

    Args:
        colors (torch.Tensor): Color data of shape (N, 3) for RGB values in range [0, 1]
        std (float): Standard deviation of the normal noise

    Returns:
        torch.Tensor: Colors with added jitter
    """
    # Generate random noise
    noise = torch.randn_like(colors) * std

    # Add noise and clamp to valid range
    jittered_colors = torch.clamp(colors + noise, 0, 1)

    return jittered_colors


def NonUniformTranslate(points, v, exclude_indices):
    assert 0 <= v <= 1
    translation = ((2 * np.random.random(3) - 1) *
                   v)
    points[~exclude_indices.astype(
        bool), 0:3] += torch.from_numpy(translation).float()
    return points


def sample_and_label(mesh, label_points, num_samples=10000):

    pytorchmesh = open3d_to_pytorch3d(mesh)
    sampled_points = sample_points_from_meshes(
        pytorchmesh, num_samples=num_samples).float()
    target_anchors = torch.from_numpy(label_points).unsqueeze(0).float()
    knn = knn_points(target_anchors, sampled_points, K=10)
    distances, idx = knn.dists, knn.idx  # idx shape: (1, M, 2)
    sampled_points = sampled_points.squeeze(0)
    idx = idx.squeeze(0)  #
    mask = torch.zeros(len(sampled_points), dtype=torch.int64)
    mask[idx.flatten()] = 1

    return sampled_points, mask


def rotate_outer_points(points, center_radius=0.35, rotation_angle=torch.pi/4):
    """
    Rotates points outside a central region around the z-axis.

    Args:
        points (torch.Tensor): Point cloud tensor of shape (N, 3)
        center_radius (float): Radius of the central region to keep unchanged
        rotation_angle (float): Angle in radians to rotate outer points around z-axis

    Returns:
        torch.Tensor: Modified point cloud with outer points rotated
    """
    # Calculate distance from origin for each point (using x,y coordinates only)
    distances = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)

    # Create mask for points outside center region
    outer_mask = distances > center_radius

    # Create modified points tensor
    modified_points = points.clone()

    # Create rotation matrix around z-axis
    cos_theta = torch.cos(rotation_angle)
    sin_theta = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], device=points.device)

    # Apply rotation to outer points
    modified_points[outer_mask] = torch.matmul(
        modified_points[outer_mask],
        rotation_matrix.T
    )

    return modified_points


def augment_and_scale_points(points):
    points = RotateX(points, uniform(0, 2))
    points = RotateY(points, uniform(0, 2))
    points = RotateZ(points, uniform(0, 2))
    points = normalize_pointcloud_tensor(
        points)
    return points


class PointCloudDataset(Dataset):
    def __init__(self, pickle_map, combined_labels, mesh_dict, num_points=50000, augmented=False):
        """
        source_paths: List of paths to source point cloud files
        target_paths: List of paths to target point cloud files (4000 points)
        """

        self.num_points = num_points
        self.pickle_map = pickle_map
        self.combined_labels = combined_labels
        self.mesh_dict = mesh_dict
        self.augmented = augmented
        self.pickle_key = {ind: key for ind,
                           (key, _) in enumerate(self.pickle_map.items())}

    def random_sample_pointcloud(self, pointcloud, colors, num_samples=50000):
        """
        Randomly sample a point cloud and its corresponding colors.
        Args:
            pointcloud: Tensor of shape (N, 3), representing the point cloud.
            colors: Tensor of shape (N, C), representing the colors or features.
            num_samples: Integer, number of points to sample.
        Returns:
            sampled_points: Tensor of shape (num_samples, 3), sampled points.
            sampled_colors: Tensor of shape (num_samples, C), sampled colors.
        """

        assert pointcloud.shape[0] == colors.shape[0], "Point cloud and colors must have the same number of points"
        new_point_cloud = pointcloud[pointcloud[:, 2]
                                     != 0]
        new_colors = colors[pointcloud[:, 2]
                            != 0]
        if len(new_point_cloud) == 0:
            new_point_cloud = pointcloud
            new_colors = colors
        N = new_point_cloud.shape[0]

        if N >= num_samples:
            # Undersample
            idx = np.random.choice(N, num_samples, replace=False)
        else:
            # Oversample with replacement
            idx = np.random.choice(N, num_samples, replace=True)

        sampled_points = new_point_cloud[idx]
        sampled_colors = new_colors[idx]
        return sampled_points, sampled_colors

    def __len__(self):
        if self.augmented:
            return len(self.pickle_map) * 7
        else:
            return len(self.pickle_map)

    def __getitem__(self, idx):
        augmented = False
        actual_idx = idx
        while idx >= len(self.pickle_map):
            augmented = True
            idx = idx - len(self.pickle_map)
        pickle_key = self.pickle_key[idx]

        # Load source point cloud (assumes .npy format)
        source_file = self.pickle_map[pickle_key][2]

        sampled_file = source_file
        pcd = o3d.io.read_point_cloud(sampled_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        points, colors = self.random_sample_pointcloud(
            points, colors, self.num_points)

        pickle_subset_path = self.pickle_map[pickle_key][0]
        pickle_subset_path = pickle_subset_path.replace(
            'SAMPLED_RESULTS', 'RESULTS')

        with open(pickle_subset_path, 'rb') as handle:
            b = pickle.load(handle)

        mesh_id = pickle_subset_path.split(
            '/')[-1].removeprefix("closest_points_").removesuffix(".pkl")

        transformed_mesh = self.mesh_dict[mesh_id]
        vert_inds = self.pickle_map[pickle_key][4]

        anchor_points = np.asarray(transformed_mesh.vertices)[vert_inds]

        mesh_points, mesh_labels = sample_and_label(
            transformed_mesh, anchor_points, num_samples=30000)

        mesh_points = augment_and_scale_points(mesh_points)
        mesh_points = RandomDropout(
            mesh_points, uniform(0.01, 0.3), mesh_labels.numpy())

        target = b[self.pickle_map[pickle_key][1]][0]

        target_clouds_inds = get_intersection_indices(points, target)

        points, centroid = center_pointcloud_numpy(points)

        points, max_distance = scale_pointcloud_to_unit_sphere_numpy(points)

        xyz = torch.from_numpy(points).float()
        rgb = torch.from_numpy(colors).float()

        if augmented:

            xyz = Jitter(xyz, randint(0, 3))
            # xyz = rotate_outer_points(
            #     xyz, rotation_angle=torch.Tensor([uniform(0.1, np.pi)]))
            xyz = RotateX(xyz, uniform(0, 0.7))
            xyz = RotateY(xyz, uniform(0, 0.7))
            xyz = RotateZ(xyz, uniform(0, 0.2))
            xyz = NonUniformScale(xyz, uniform(0, 0.3), target_clouds_inds)
            xyz = NonUniformTranslate(xyz, uniform(0, 1), target_clouds_inds)
            xyz = RandomFlipX(xyz, uniform(0, 1))
            xyz = RandomFlipY(xyz, uniform(0, 1))
            xyz = RandomFlipZ(xyz, uniform(0, 1))
            # xyz = NonUniformTranslate(xyz, uniform(0, 1))
            xyz = ScaleX(xyz, uniform(0, 0.3))
            xyz = ScaleY(xyz, uniform(0, 0.3))
            xyz = ScaleZ(xyz, uniform(0, 0.3))

            xyz = RandomErase(
                xyz, uniform(0, 0.2), target_clouds_inds)

            expanded_rgb = expand_to_im_size(rgb)
            expanded_rgb = adjust_brightness(expanded_rgb, uniform(0.3, 3))
            expanded_rgb = adjust_contrast(expanded_rgb, uniform(0.4, 3))
            expanded_rgb = adjust_gamma(expanded_rgb, uniform(0.01, 2))
            expanded_rgb = adjust_hue(expanded_rgb, uniform(-0.5, 0.5))
            expanded_rgb = adjust_saturation(expanded_rgb, uniform(0, 2))
            rgb = reduce_from_im_size(expanded_rgb)

            rgb = color_jitter(rgb)
            rgb = white_color_dropout(rgb)

        xyz = RandomDropout(
            xyz, uniform(0.01, 0.3), target_clouds_inds)

        return xyz, rgb, target_clouds_inds, mesh_points, mesh_labels


class RandomSampler(Sampler):
    def __init__(self, pickle_map, num_samples=20000, augmentation_factor=5):
        self.data_source = pickle_map
        self._num_samples = num_samples
        self.augmentation_factor = augmentation_factor
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source) * self.augmentation_factor
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def create_dataloader(pickle_map, combined_labels, mesh_dict, batch_size=1, num_workers=4, augmented=False, sample=False, sample_size=10000):
    dataset = PointCloudDataset(
        pickle_map, combined_labels, mesh_dict, num_points=60000, augmented=augmented)
    if not sample:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=RandomSampler(
                pickle_map, augmentation_factor=7, num_samples=sample_size)

        )

    return dataloader


def split_pickle_loader(pickle_map, train_sequences=17):

    target_files = set()
    target_files_ordered = []
    for key, value in pickle_map.items():
        if value[0] not in target_files:
            target_files.add(value[0])
            target_files_ordered.append(value[0])

    val_files = target_files_ordered[:len(target_files)-train_sequences]

    train_split = {}
    val_split = {}
    train_index = 0
    val_index = 0
    for key, value in pickle_map.items():
        if value[0] not in val_files:
            train_split[train_index] = value
            train_index += 1
        else:
            val_split[val_index] = value
            val_index += 1
    return train_split, val_split, val_files
