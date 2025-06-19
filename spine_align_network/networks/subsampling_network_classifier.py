import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather


class SAModule(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, dropout_rate=0.2):
        super(SAModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_gns = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate)

        last_channel = in_channel + 3

        for out_channel in mlp:
            # Add bottleneck layer
            bottleneck_dim = max(last_channel // 4, out_channel // 4)
            self.mlp_convs.append(nn.Sequential(
                nn.Conv2d(last_channel, bottleneck_dim, 1),
                nn.Conv2d(bottleneck_dim, out_channel, 1)
            ))
            num_groups = max(4, out_channel // 16)  # Reduced number of groups
            self.mlp_gns.append(nn.GroupNorm(num_groups, out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        """
        xyz: (B, N, 3)
        features: (B, N, C)
        """
        # Sample points using FPS
        fps_idx = sample_farthest_points(xyz, K=self.npoint)[1]  # (B, npoint)
        new_xyz = torch.gather(
            xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)

        # Find K nearest neighbors
        knn_result = knn_points(new_xyz, xyz, K=self.nsample)
        # (B, npoint, nsample, 3)
        grouped_xyz = knn_gather(xyz, knn_result.idx)
        grouped_xyz_norm = grouped_xyz - \
            new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)

        if features is not None:
            grouped_features = knn_gather(
                features, knn_result.idx)  # (B, npoint, nsample, C)
            # Concatenate relative coordinates with features
            # (B, npoint, nsample, C+3)
            grouped_features = torch.cat(
                [grouped_xyz_norm, grouped_features], dim=-1)
        else:
            grouped_features = grouped_xyz_norm

        # Apply MLP
        grouped_features = grouped_features.permute(
            0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        for i, conv in enumerate(self.mlp_convs):
            grouped_features = self.dropout(
                F.relu(self.mlp_gns[i](conv(grouped_features))))

        # Max pooling
        new_features = torch.max(grouped_features, 2)[0]  # (B, C', npoint)
        new_features = new_features.permute(0, 2, 1)  # (B, npoint, C')

        return new_xyz, new_features


class FPModule(nn.Module):
    def __init__(self, in_channel, mlp, dropout_rate=0.2):
        super(FPModule, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_gns = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate)
        # last_channel = in_channel
        # for out_channel in mlp:
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     num_groups = max(4, out_channel // 8)
        #     self.mlp_gns.append(nn.GroupNorm(num_groups, out_channel))
        #     last_channel = out_channel
        last_channel = in_channel
        for out_channel in mlp:
            # Add bottleneck structure
            bottleneck_dim = max(last_channel // 4, out_channel // 4)
            self.mlp_convs.append(nn.Sequential(
                nn.Conv2d(last_channel, bottleneck_dim, 1),
                nn.Conv2d(bottleneck_dim, out_channel, 1)
            ))
            num_groups = max(2, out_channel // 16)
            self.mlp_gns.append(nn.GroupNorm(num_groups, out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: (batch_size, ndataset1, 3) tensor of the first point cloud positions
            xyz2: (batch_size, ndataset2, 3) tensor of the second point cloud positions
            points1: (batch_size, ndataset1, channel1) tensor of the first point cloud features
            points2: (batch_size, ndataset2, channel2) tensor of the second point cloud features
        Returns:
            new_points: (batch_size, ndataset1, mlp[-1]) tensor of the new features
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.expand(B, N, points2.shape[-1])
        else:
            # Calculate squared distances
            dists = torch.cdist(xyz1, xyz2)  # (B, N, S)

            # Get k nearest neighbors
            k = 3
            dists, idx = torch.topk(
                dists, k=k, dim=2, largest=False)  # (B, N, k)

            # Calculate weights
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # (B, N, k)

            # Gather points2 features using idx
            # (B, N, k, C)
            idx_expand = idx.unsqueeze(-1).expand(-1, -
                                                  1, -1, points2.shape[-1])
            points2_neighbors = torch.gather(points2.unsqueeze(
                2).expand(-1, -1, k, -1), 1, idx_expand)  # (B, N, k, C)

            # Apply weights
            interpolated_points = torch.sum(
                points2_neighbors * weight.unsqueeze(-1), dim=2)  # (B, N, C)

        if points1 is not None:
            new_points = torch.cat(
                [interpolated_points, points1], dim=-1)  # (B, N, C1 + C2)
        else:
            new_points = interpolated_points

        # Apply MLPs
        new_points = new_points.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, N)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_gns[i]
            new_points = self.dropout(F.relu(bn(conv(new_points))))

        new_points = new_points.squeeze(2).permute(0, 2, 1)  # (B, N, C_out)

        return new_points


class BinaryPointNetPlusPlus(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(BinaryPointNetPlusPlus, self).__init__()

        self.register_buffer(
            'rgb_weights', torch.tensor([0.2126, 0.7152, 0.0722]))

        self.dropout = nn.Dropout(p=dropout_rate)
        # Simpler encoder
        # in_channel=6 (3 for xyz + 3 for rgb)
        # First layer: 1024 points (in_channel=6: 3 for xyz + 3 for rgb)
        self.sa1 = SAModule(1024, 0.1, 8, 6, [16, 32], dropout_rate=0)
        # Second layer: 256 points
        self.sa2 = SAModule(256, 0.2, 8, 32, [64, 96], dropout_rate=0)
        # Third layer: 64 points
        self.sa3 = SAModule(64, 0.4, 8, 96, [128, 256], dropout_rate=0)

        # Enhanced decoder with balanced FP layers
        # l3_points + l2_points channels
        self.fp3 = FPModule(256 + 96, [128, 96], dropout_rate=0.15)
        # l2_points + l1_points channels
        self.fp2 = FPModule(96 + 32, [64, 32], dropout_rate=0.15)
        # l1_points channels (since points1 is None)
        self.fp1 = FPModule(32, [16, 8], dropout_rate=0.15)

        # Final layers remain the same
        self.conv1 = nn.Conv1d(8, 4, 1)
        self.gn1 = nn.GroupNorm(2, 4)
        self.conv2 = nn.Conv1d(4, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def rgb_to_intensity(self, rgb):
        """Convert RGB values to intensity using standard coefficients"""
        # rgb shape: (batch_size, num_points, 3)
        # Return shape: (batch_size, num_points, 1)
        return torch.sum(rgb * self.rgb_weights, dim=-1, keepdim=True)

    def forward(self, xyz, rgb, return_features=False):

        # intensity = self.rgb_to_intensity(rgb)

        features = torch.cat([xyz, rgb], dim=-1)

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, features)
        # l1_points = self.dropout(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l2_points = self.dropout(l2_points)
        # print("After SA2:", l2_points.requires_grad)
        # Decoder
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points = self.dropout(l1_points)

        # print("After FP2:", l1_points.requires_grad)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        if return_features:
            return l0_points
        l0_points = self.dropout(l0_points)

        # print("After FP1:", l0_points.requires_grad)

        # Final layers
        x = l0_points.permute(0, 2, 1)
        x = self.dropout(F.relu(self.gn1(self.conv1(x))))
        x = self.conv2(x)

        return x
