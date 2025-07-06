import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from torch import nn
import torch.nn.functional as F
from medpy.metric.binary import hd95
from scipy.spatial import cKDTree


def dice_coefficient(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(-1)
    dice_loss = (2. * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)
    mean_dice = sum(dice_loss) / num
    return mean_dice


def batch_hausdorff_95(outputs, labels):

    batch_size, C, H, W = outputs.shape
    hausdorff_dists = []
    for i in range(batch_size):
        output_mask = (torch.sigmoid(outputs[i]) > 0.5).float().cpu().numpy()
        label_mask = labels[i].cpu().numpy()

        output_mask = output_mask.reshape(H,W)
        label_mask = label_mask.reshape(H,W)
        if(np.all(output_mask == 0)):
            hausdorff_dists.append(np.sqrt(H ** 2 + W ** 2))
            continue
        current_hd95 = hd95(output_mask, label_mask)
        hausdorff_dists.append(current_hd95 )
    return np.mean(hausdorff_dists) if hausdorff_dists else float('nan')


class BoundaryDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pred_edges = self._edges(preds)
        target_edges = self._edges(targets)

        pred_dt = torch.from_numpy(distance(1.0 - pred_edges.cpu().detach().numpy())).to(device)
        target_dt = torch.from_numpy(distance(1.0 - target_edges.cpu().detach().numpy())).to(device)

        pred_err = target_edges * pred_dt
        target_err = pred_edges * target_dt

        loss = (pred_err.mean() + target_err.mean()) / 2.0
        loss = torch.sigmoid(loss)
        return loss

    def _edges(self, seg):
        if seg.dim() == 2:
            seg = seg.unsqueeze(0).unsqueeze(0)
        elif seg.dim() == 3:
            seg = seg.unsqueeze(1)

        kernel = torch.ones((1, 1, 3, 3), device=seg.device, dtype=seg.dtype)

        padding = 1
        eroded = F.conv2d(seg, kernel, padding=padding, stride=1)
        eroded[eroded < kernel.numel()] = 0.0
        eroded = (eroded == kernel.numel()).float()

        edges = seg - eroded
        if edges.dim() == 4:
            edges = edges.squeeze(0).squeeze(0)

        return edges


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def binary_distance_transform(self, mask):
        """
        Compute a simple binary distance transform.
        Args:
            mask (torch.Tensor): Input binary mask, values should be 0 or 1.

        Returns:
            torch.Tensor: Distance transform of the input mask.
        """
        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Pretend this is a proper distance transform for demonstration purposes
        distance = 1.0 - mask
        return distance

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Model prediction with shape (N, 1, H, W)
            target (torch.Tensor): Ground truth with shape (N, 1, H, W), values should be 0 or 1.

        Returns:
            torch.Tensor: Calculated boundary loss.
        """
        assert pred.size() == target.size(), "Prediction and target size must be the same."

        # Get the distance map of the ground truth
        distance_map = self.binary_distance_transform(target)

        # Optionally make sure that your predictions are in [0, 1] range e.g., with sigmoid
        pred = torch.sigmoid(pred)

        # Calculate the boundary loss
        multipled = distance_map * pred

        # Normalize loss to be in 0-1
        epsilon = 1e-5
        loss = multipled.sum() / (distance_map.sum()+epsilon)

        # Extra normalization to definitely be between 0 and 1
        if loss > 1:
            loss = torch.tensor(1.0)
        elif loss < 0:
            loss = torch.tensor(0.0)

        return loss






def batch_average_hausdorff(outputs, labels):

    if outputs.ndim == 3:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)

    batch_size, C, H, W = outputs.shape
    if C != 1:
        raise ValueError("`Only supports single-channel binary classification tasks`")

    max_dist = np.sqrt(H**2 + W**2)
    avg_hausdorff_dists = []

    for i in range(batch_size):
        output_mask = (torch.sigmoid(outputs[i]) > 0.5).float().cpu().numpy().squeeze()
        label_mask = labels[i].cpu().numpy().squeeze()

        output_points = np.argwhere(output_mask > 0)
        label_points = np.argwhere(label_mask > 0)

        if len(output_points) == 0 or len(label_points) == 0:
            avg_hausdorff_dists.append(max_dist)
            continue

        def _kdtree_hausdorff(set1, set2):
            tree = cKDTree(set2)
            distances, _ = tree.query(set1, k=1)
            return np.mean(distances)

        d_AB = _kdtree_hausdorff(output_points, label_points)
        d_BA = _kdtree_hausdorff(label_points, output_points)
        ahd = (d_AB + d_BA) / 2

        avg_hausdorff_dists.append(ahd)

    return np.nanmean(avg_hausdorff_dists) if len(avg_hausdorff_dists) > 0 else float('nan')


