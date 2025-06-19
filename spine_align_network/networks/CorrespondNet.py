import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from data_loading.data_loading_subsample_meshes import create_dataloader, split_pickle_loader, load_transformed_meshes
import pickle

from tqdm import tqdm
import numpy as np

from monai.losses import DiceFocalLoss

from pathlib import Path
from networks.subsampling_network_classifier import BinaryPointNetPlusPlus, FPModule


def init_weights(m):
    """Initialize weights using kaiming_normal for Conv layers with ReLU."""
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CombinedModel(nn.Module):
    def __init__(self, base_model, dropout_rate=0.4):
        super(CombinedModel, self).__init__()

        self.register_buffer(
            'rgb_weights', torch.tensor([0.2126, 0.7152, 0.0722]))

        self.dropout = nn.Dropout(p=dropout_rate)
        # Simpler encoder
        # in_channel=6 (3 for xyz + 3 for rgb)
        # First layer: 1024 points (in_channel=6: 3 for xyz + 3 for rgb)
        self.sa1 = base_model.sa1
        # Second layer: 256 points
        self.sa2 = base_model.sa2
        # Third layer: 64 points
        self.sa3 = base_model.sa3

        # Enhanced decoder with balanced FP layers
        # l3_points + l2_points channels
        self.fp3 = base_model.fp3
        # l2_points + l1_points channels
        self.fp2 = base_model.fp2
        # l1_points channels (since points1 is None)
        self.fp1 = base_model.fp1

        self.fp3_mesh = FPModule(256 + 96, [128, 96], dropout_rate=0.15)
        self.fp2_mesh = FPModule(96 + 32, [64, 32], dropout_rate=0.15)
        self.fp1_mesh = FPModule(32, [16, 8], dropout_rate=0.15)

        self.transfer_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(24, 8),  # 16 from transferred + 8 from mesh
            nn.ReLU()
        )

        # Final layers remain the same
        self.conv1 = base_model.conv1
        self.gn1 = base_model.gn1
        self.conv2 = base_model.conv2

        self.conv3 = nn.Conv1d(8, 4, 1)
        self.gn2 = nn.GroupNorm(2, 4)
        self.conv4 = nn.Conv1d(4, 1, 1)

        self.transfer_net.apply(init_weights)
        self.fusion_layer.apply(init_weights)

        self.fp3_mesh.apply(init_weights)
        self.fp2_mesh.apply(init_weights)
        self.fp1_mesh.apply(init_weights)

        self.conv3.apply(init_weights)
        self.conv4.apply(init_weights)

        self.gn2.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, rgb, mesh_xyz, return_features=False):

        # intensity = self.rgb_to_intensity(rgb)

        features = torch.cat([xyz, rgb], dim=-1)
        fake_rgb = torch.zeros_like(mesh_xyz)
        mesh_features = torch.cat([mesh_xyz, fake_rgb], dim=-1)

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, features)
        # l1_points = self.dropout(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_xyz_mesh, l1_points_mesh = self.sa1(mesh_xyz, mesh_features)
        # l1_points = self.dropout(l1_points)

        l2_xyz_mesh, l2_points_mesh = self.sa2(l1_xyz_mesh, l1_points_mesh)
        # l2_points = self.dropout(l2_points)
        # print("After SA2:", l2_points.requires_grad)
        # Decoder
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points = self.dropout(l1_points)

        # print("After FP2:", l1_points.requires_grad)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        l1_points_mesh = self.fp2_mesh(
            l1_xyz_mesh, l2_xyz_mesh, l1_points_mesh, l2_points_mesh)
        # l1_points = self.dropout(l1_points)

        # print("After FP2:", l1_points.requires_grad)
        l0_points_mesh = self.fp1_mesh(
            mesh_xyz, l1_xyz_mesh, None, l1_points_mesh)

        if return_features:
            return l0_points

        transformed_features = self.transfer_net(l0_points)

        # Global max pooling of transformed features
        global_features = torch.max(transformed_features, dim=1)[0]  # (B, C)

        # Expand global features to mesh size
        global_features = global_features.unsqueeze(
            1).expand(-1, mesh_xyz.shape[1], -1)

        combined_features = torch.cat(
            [global_features, l0_points_mesh], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        l0_points = self.dropout(l0_points)
        fused_features = self.dropout(fused_features)

        # Final layers
        x = l0_points.permute(0, 2, 1)
        x = self.dropout(F.relu(self.gn1(self.conv1(x))))
        x = self.conv2(x)

        x2 = fused_features.permute(0, 2, 1)
        x2 = self.dropout(F.relu(self.gn2(self.conv3(x2))))
        x2 = self.conv4(x2)

        return x, x2


def train(model, train_loader, val_loader, optimizer_state_dict=None, num_epochs=50, lr=0.01, device="cuda"):
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer.load_state_dict(optimizer_state_dict)

    with open('trainlog_adjusted_unified_fused.txt', 'w') as s1:
        s1.write("Epoch, train loss combined, train loss seg, train loss mesh, val loss combined, val loss seg, val loss mesh")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    model.train()

    loss_fn = DiceFocalLoss(sigmoid=True, softmax=False,
                            include_background=False,
                            squared_pred=False)

    for epoch in range(num_epochs):

        train_losses = []
        train_losses_seg = []
        train_losses_mesh = []
        model.train()
        for batch_idx, (xyz, rgb, target, mesh_points, mesh_target) in tqdm(enumerate(train_loader)):

            mesh_points = mesh_points.to(device)
            mesh_target = mesh_target.to(device)
            xyz = xyz.to(device)

            rgb = rgb.to(device)
            target = target.to(device)

            pred_xyz, pred_mesh = model(xyz, rgb, mesh_points)

            loss1 = loss_fn(pred_xyz, target.unsqueeze(1))
            loss2 = loss_fn(pred_mesh, mesh_target.unsqueeze(1))
            loss = loss1 + loss2

            if batch_idx % 15 == 0:
                loss1.backward()
            else:
                loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            if (batch_idx+1) % 16 == 0 or (batch_idx + 1 == len(train_loader)):
                # every 10 iterations of batches of size 10
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item())
            train_losses_seg.append(loss1.item())
            train_losses_mesh.append(loss2.item())

            if batch_idx % 1000 == 0:
                print(
                    f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.8f}, Loss Seg: {loss1:.8f}, Loss Mesh: {loss2:.8f}')

            if batch_idx % 5000 == 0:
                print("Post-backward gradients:")
                total_layers = sum(1 for _ in model.named_parameters())
                # First, middle, and last
                check_positions = [0, total_layers//2, total_layers-1]
                # check_positions = [i for i in range(0, total_layers)]

                for i, (name, param) in enumerate(model.named_parameters()):
                    if i in check_positions and param.grad is not None:
                        print(
                            f'{name}: gradient norm = {param.grad.norm().item()}')
        model.eval()
        val_losses = []
        val_losses_seg = []
        val_losses_mesh = []
        with torch.no_grad():
            for xyz, rgb, target_xyz, mesh_points, mesh_target in val_loader:
                xyz = xyz.to(device)
                rgb = rgb.to(device)
                mesh_points = mesh_points.to(device)
                mesh_target = mesh_target.to(device)
                target_xyz = target_xyz.to(device)

                pred_xyz, pred_mesh = model(xyz, rgb, mesh_points)

                loss1 = loss_fn(pred_xyz, target_xyz.unsqueeze(1))
                loss2 = loss_fn(pred_mesh, mesh_target.unsqueeze(1))
                loss = loss1 + loss2
                val_losses.append(loss.item())
                val_losses_seg.append(loss1.item())
                val_losses_mesh.append(loss2.item())

        avg_loss = np.mean(train_losses)
        avg_loss_seg = np.mean(train_losses_seg)
        avg_loss_mesh = np.mean(train_losses_mesh)
        val_loss = np.mean(val_losses)
        val_loss_seg = np.mean(val_losses_seg)
        val_loss_mesh = np.mean(val_losses_mesh)
        print(
            f'Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}')
        print(
            f'Epoch {epoch+1}: Train Loss Seg = {avg_loss_seg:.6f}, Val Loss Seg = {val_loss_seg:.6f}')
        print(
            f'Epoch {epoch+1}: Train Loss Mesh = {avg_loss_mesh:.6f}, Val Loss Mesh = {val_loss_mesh:.6f}')

        with open('trainlog_adjusted_unified_fused.txt', 'a') as s1:
            s1.write(
                f'\n {epoch+1}, {avg_loss:.6f}, {avg_loss_seg:.6f}, {avg_loss_mesh:.6f}, {val_loss:.6f}, {val_loss_seg:.6f}, {val_loss_mesh:.6f}')

        scheduler.step(val_loss)

        if val_loss_mesh < best_val_loss:
            best_val_loss = val_loss_mesh
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model_adjusted_unified_fusion.pt')


def main():
    torch.cuda.empty_cache()

    labels_path = "../CT_DATA/CT_PROXY_LABELS/combined_labels.npy"

    pickle_map = "../CT_DATA/index_data_map_with_meshes.pkl"
    with open(pickle_map, 'rb') as fp:
        pickle_map_dict = pickle.load(fp)
    labels = np.load(
        labels_path, allow_pickle=True)
    train_map, val_map, _ = split_pickle_loader(
        pickle_map_dict, train_sequences=18)
    mesh_dict = load_transformed_meshes()
    train_data_loader = create_dataloader(
        train_map, labels, mesh_dict, batch_size=1, augmented=True, sample=True, sample_size=20000)
    val_data_loader = create_dataloader(
        val_map, labels, mesh_dict, batch_size=1)

    base_model = BinaryPointNetPlusPlus()

    combined_model = CombinedModel(base_model)

    # Assuming train_loader is defined with data shape [B, C, N] and target shape [B, N]
    train(combined_model, train_data_loader, val_data_loader, num_epochs=200)


if __name__ == "__main__":
    main()
