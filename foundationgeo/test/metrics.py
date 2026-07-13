from typing import *
from numbers import Number
import math

import torch
import torch.nn.functional as F
import numpy as np
import utils3d
import math

from ..utils.geometry_torch import (
    weighted_mean, 
    mask_aware_nearest_resize,
    intrinsics_to_fov
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale_xyz_shift, 
    align_points_xyz_shift,
    align_affine_lstsq, 
    align_depth_scale, 
    align_depth_affine, 
    align_points_scale,
)
from ..utils.tools import key_average, timeit


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def mae_depth(pred: torch.Tensor, gt: torch.Tensor):
    mae = torch.abs(pred - gt).mean()
    return mae.item()


def rmse_depth(pred: torch.Tensor, gt: torch.Tensor):
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    return rmse.item()


def irmse_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    pred_inv = 1.0 / pred.clamp_min(eps)
    gt_inv = 1.0 / gt.clamp_min(eps)
    irmse = torch.sqrt(torch.mean((pred_inv - gt_inv) ** 2))
    return irmse.item()


def silog_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    log_diff = torch.log(pred.clamp_min(eps)) - torch.log(gt.clamp_min(eps))
    silog = torch.sqrt(torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2).clamp_min(0)
    return (100.0 * silog).item()


def delta_threshold_depth(pred: torch.Tensor, gt: torch.Tensor, threshold: float, eps: float = 1e-6):
    delta = (torch.maximum(gt / pred.clamp_min(eps), pred / gt.clamp_min(eps)) < threshold).float().mean()
    return delta.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    delta1 = (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean()
    return delta1.item()

def delta05_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    delta05 = (torch.maximum(gt / pred, pred / gt) < 1.118).float().mean()
    return delta05.item()


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()


def delta1_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta1.item()

def delta05_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta05 = (dist_err < 0.118 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta05.item()


def rel_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    delta1 = (dist_err < 0.25 * diameter).float().mean()
    return delta1.item()


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, radius: int = 1):
    neighbor_x, neight_y = torch.meshgrid(
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        indexing='xy'
    )
    neighbor_mask = (neighbor_x ** 2 + neight_y ** 2) <= radius ** 2 + 1e-5

    pred_window = utils3d.torch.sliding_window_2d(pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                 # [H, W, 2*R+1, 2*R+1]
    gt_window = utils3d.torch.sliding_window_2d(gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                     # [H, W, 2*R+1, 2*R+1]
    mask_window = neighbor_mask & utils3d.torch.sliding_window_2d(mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)) # [H, W, 2*R+1, 2*R+1]

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window
    
    f1_list = []
    w_list = t_list = torch.linspace(0.05, 0.25, 10).tolist()

    for t in t_list:
        pred_label = pred_rel > 1 + t
        gt_label = gt_rel > 1 + t
        TP = (pred_label & gt_label & valid).float().sum()
        precision = TP / (gt_label & valid).float().sum().clamp_min(1e-12)
        recall = TP / (pred_label & valid).float().sum().clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_list.append(f1.item())
    
    f1_avg = sum(w * f1 for w, f1 in zip(w_list, f1_list)) / sum(w_list)
    return f1_avg

def ray_angle_error(
    pred_points: torch.Tensor,   # (H,W,3) or (B,H,W,3)
    gt_points: torch.Tensor,     # (H,W,3) or (B,H,W,3)
    mask: torch.Tensor,          # (H,W) or (B,H,W) bool
    deg_thresholds: Tuple[float, ...] = (3.0, 5.0),
    eps: float = 1e-6,
):
    """
    Calculate the angle error (degree) of unit direction vectors, return:
      - mae_deg: mean angle error
      - pct_Xdeg: the ratio of error <= X degrees
    """
    # Unify to (N,3)
    v_pred = pred_points[mask]      # (N,3)
    v_gt   = gt_points[mask]        # (N,3)
    if v_pred.numel() == 0:
        return {'mae_deg': float('nan'), **{f'pct_{int(t)}deg': float('nan') for t in deg_thresholds}}

    # Normalize
    v_pred = F.normalize(v_pred, dim=-1, eps=eps)
    v_gt   = F.normalize(v_gt,   dim=-1, eps=eps)

    cos = (v_pred * v_gt).sum(-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos) * 180.0 / math.pi  # (N,)

    out = {'mae_deg': angle.mean().item()}
    for t in deg_thresholds:
        out[f'pct_{int(t)}deg'] = (angle <= t).float().mean().item()
    return out


def ray_angle_error_map(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    pred_dirs = F.normalize(pred_points, dim=-1, eps=eps)
    gt_dirs = F.normalize(gt_points, dim=-1, eps=eps)
    cos = (pred_dirs * gt_dirs).sum(-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos) * 180.0 / math.pi
    angle = torch.where(mask, angle, torch.nan)
    return angle


def aligned_depth_variants(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    lr_index: torch.Tensor,
    lr_mask: torch.Tensor,
):
    pred_depth_lr_masked = pred_depth[lr_index][lr_mask]
    gt_depth_lr_masked = gt_depth[lr_index][lr_mask]

    scale = align_depth_scale(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
    pred_depth_scale = pred_depth * scale

    scale_affine, shift_affine = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
    pred_depth_affine = pred_depth * scale_affine + shift_affine

    return {
        'scale_invariant': depth_metrics_all(pred_depth_scale[mask], gt_depth[mask]),
        'affine_invariant': depth_metrics_all(pred_depth_affine[mask], gt_depth[mask]),
    }, pred_depth_scale, pred_depth_affine


def aligned_points_variants(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    lr_index: torch.Tensor,
    lr_mask: torch.Tensor,
):
    pred_points_lr_masked = pred_points[lr_index][lr_mask]
    gt_points_lr_masked = gt_points[lr_index][lr_mask]

    scale = align_points_scale(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
    pred_points_scale = pred_points * scale

    scale_affine, shift_affine = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
    pred_points_affine = pred_points * scale_affine + shift_affine

    return {
        'scale_invariant': {
            'rel': rel_point(pred_points_scale[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points_scale[mask], gt_points[mask]),
        },
        'affine_invariant': {
            'rel': rel_point(pred_points_affine[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points_affine[mask], gt_points[mask]),
        }
    }, pred_points_scale, pred_points_affine


def depth_metrics_all(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    return {
        'rel': rel_depth(pred, gt, eps),
        'absrel': rel_depth(pred, gt, eps),
        'silog': silog_depth(pred, gt, eps),
        'rmse': rmse_depth(pred, gt),
        'irmse': irmse_depth(pred, gt, eps),
        'mae': mae_depth(pred, gt),
        'delta1': delta1_depth(pred, gt, eps),
        'delta_1_025': delta_threshold_depth(pred, gt, 1.025, eps),
        'delta_1_05': delta_threshold_depth(pred, gt, 1.05, eps),
        'delta_1_10': delta_threshold_depth(pred, gt, 1.10, eps),
    }

def compute_metrics(
    pred: Dict[str, torch.Tensor], 
    gt: Dict[str, torch.Tensor], 
    vis: bool = False
) -> Tuple[Dict[str, Dict[str, Number]], Dict[str, torch.Tensor]]:
    """
    A unified function to compute metrics for different types of predictions and ground truths.
    
    #### Supported keys in pred:
        - `disparity_affine_invariant`: disparity map predicted by a depth estimator with scale and shift invariant. 
        - `depth_scale_invariant`: depth map predicted by a depth estimator with scale invariant. 
        - `depth_affine_invariant`: depth map predicted by a depth estimator with scale and shift invariant. 
        - `depth_metric`: depth map predicted by a depth estimator with no scale or shift. 
        - `points_scale_invariant`: point map predicted by a point estimator with scale invariant. 
        - `points_affine_invariant`: point map predicted by a point estimator with scale and xyz shift invariant. 
        - `points_metric`: point map predicted by a point estimator with no scale or shift. 
        - `intrinsics`: normalized camera intrinsics matrix.

    #### Required keys in gt:
        - `depth`: depth map ground truth (in metric units if `depth_metric` is used)
        - `points`: point map ground truth in camera coordinates.
        - `mask`: mask indicating valid pixels in the ground truth.
        - `intrinsics`: normalized ground-truth camera intrinsics matrix.
        - `is_metric`: whether the depth is in metric units.
    """
    metrics = {}
    misc = {}
    
    mask = gt['depth_mask']
    gt_depth = gt['depth']
    gt_points = gt['points']

    height, width = mask.shape[-2:]
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (64, 64), return_index=True)
    
    only_depth = not any('point' in k for k in pred)
    pred_depth_aligned, pred_points_aligned = None, None

    # Metric depth
    if 'depth_metric' in pred and gt['is_metric']:
        pred_depth, gt_depth = pred['depth_metric'], gt['depth']
        metrics['depth_metric'] = depth_metrics_all(pred_depth[mask], gt_depth[mask])

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale_affine, shift_affine = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth_metric_affine = pred_depth * scale_affine + shift_affine
        metrics['depth_metric_affine_invariant'] = depth_metrics_all(pred_depth_metric_affine[mask], gt_depth[mask])

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Scale-invariant depth
    if 'depth_scale_invariant' in pred:
        pred_depth_scale_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_scale_invariant = pred['depth_metric']
    else:
        pred_depth_scale_invariant = None

    if pred_depth_scale_invariant is not None:
        pred_depth = pred_depth_scale_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale = align_depth_scale(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale
    
        metrics['depth_scale_invariant'] = depth_metrics_all(pred_depth[mask], gt_depth[mask])

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant depth
    if 'depth_affine_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_affine_invariant = pred['depth_metric']
    else:
        pred_depth_affine_invariant = None

    if pred_depth_affine_invariant is not None:
        pred_depth = pred_depth_affine_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale + shift

        metrics['depth_affine_invariant'] = depth_metrics_all(pred_depth[mask], gt_depth[mask])

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant disparity
    if 'disparity_affine_invariant' in pred:
        pred_disparity_affine_invariant = pred['disparity_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_metric']
    else:
        pred_disparity_affine_invariant = None
        
    if pred_disparity_affine_invariant is not None:
        pred_disp = pred_disparity_affine_invariant
        
        scale, shift = align_affine_lstsq(pred_disp[mask], 1 / gt_depth[mask])
        pred_disp = pred_disp * scale + shift

        # Alignment in disparity space can create outliers near zero disparity.
        pred_depth = 1 / pred_disp.clamp_min(1 / gt_depth[mask].max().item())

        metrics['disparity_affine_invariant'] = depth_metrics_all(pred_depth[mask], gt_depth[mask])

        if pred_depth_aligned is None:
            pred_depth_aligned = 1 / pred_disp.clamp_min(1e-6)

    # Metric points
    if 'points_metric' in pred and gt['is_metric']:
        pred_points = pred['points_metric']

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        shift = align_points_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points + shift

        metrics['points_metric'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask]),
            'delta05': delta05_point(pred_points[mask], gt_points[mask])
        }

        scale_affine, shift_affine = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points_metric_affine = pred['points_metric'] * scale_affine + shift_affine
        metrics['points_metric_affine_invariant'] = {
            'rel': rel_point(pred_points_metric_affine[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points_metric_affine[mask], gt_points[mask])
        }

        if pred_points_aligned is None:
            pred_points_aligned = pred['points_metric']

    # Scale-invariant points (in camera space)
    if 'points_scale_invariant' in pred:
        pred_points_scale_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_scale_invariant = pred['points_metric']
    else:
        pred_points_scale_invariant = None
        
    if pred_points_scale_invariant is not None:
        pred_points = pred_points_scale_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points_scale_invariant[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale = align_points_scale(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale

        metrics['points_scale_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_scale_invariant'] * scale
    
    # Affine-invariant points
    if 'points_affine_invariant' in pred:
        pred_points_affine_invariant = pred['points_affine_invariant']
    elif 'points_scale_invariant' in pred:
        pred_points_affine_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_affine_invariant = pred['points_metric']
    else:
        pred_points_affine_invariant = None

    if pred_points_affine_invariant is not None:
        pred_points = pred_points_affine_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale + shift

        metrics['points_affine_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_affine_invariant'] * scale + shift

    # Local points
    if 'segmentation_mask' in gt and 'points' in gt and any('points' in k for k in pred.keys()):
        pred_points = next(pred[k] for k in pred.keys() if 'points' in k)
        gt_points = gt['points']
        segmentation_mask = gt['segmentation_mask']
        segmentation_labels = gt['segmentation_labels']
        segmentation_mask_lr =  segmentation_mask[lr_index]
        local_points_metrics = []
        for _, seg_id in segmentation_labels.items():
            valid_mask = (segmentation_mask == seg_id) & mask
            
            pred_points_masked = pred_points[valid_mask]
            gt_points_masked = gt_points[valid_mask]

            valid_mask_lr = (segmentation_mask_lr == seg_id) & lr_mask
            if valid_mask_lr.sum().item() < 10:
                continue
            pred_points_masked_lr = pred_points[lr_index][valid_mask_lr]
            gt_points_masked_lr = gt_points[lr_index][valid_mask_lr]
            diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()
            scale, shift = align_points_scale_xyz_shift(pred_points_masked_lr, gt_points_masked_lr, 1 / diameter.expand(gt_points_masked_lr.shape[0]))
            pred_points_masked = pred_points_masked * scale + shift

            local_points_metrics.append({
                'rel': rel_point_local(pred_points_masked, gt_points_masked, diameter),
                'delta1': delta1_point_local(pred_points_masked, gt_points_masked, diameter),
            })
        
        metrics['local_points'] = key_average(local_points_metrics)

    # FOV. NOTE: If there is no random augmentation applied to the input images, all GT FOV are generally the same.
    # Fair evaluation of FOV requires random augmentation.
    if 'intrinsics' in pred and 'intrinsics' in gt:
        pred_intrinsics = pred['intrinsics']
        gt_intrinsics = gt['intrinsics']
        pred_fov_x, pred_fov_y = intrinsics_to_fov(pred_intrinsics)
        gt_fov_x, gt_fov_y = intrinsics_to_fov(gt_intrinsics)
        metrics['fov_x'] = {
            'mae': torch.rad2deg(pred_fov_x - gt_fov_x).abs().mean().item(),
            'deviation': torch.rad2deg(pred_fov_x - gt_fov_x).item(),
        }

    # Boundary F1
    if pred_depth_aligned is not None and gt['has_sharp_boundary']:
        metrics['boundary'] = {
            'radius1_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=1),
            'radius2_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=2),
            'radius3_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=3),
        }

    if vis:
        if pred_points_aligned is not None:
            misc['pred_points'] = pred_points_aligned
        if only_depth:
            misc['pred_points'] = utils3d.torch.depth_to_points(pred_depth_aligned, intrinsics=gt['intrinsics'])
        if pred_depth_aligned is not None:
            misc['pred_depth'] = pred_depth_aligned
        if 'points_metric' in pred:
            misc['pred_points_metric'] = pred['points_metric']
            misc['ray_angle_error_metric'] = ray_angle_error_map(pred['points_metric'], gt_points, mask)
        if 'depth_metric' in pred:
            misc['pred_depth_metric'] = pred['depth_metric']
        if 'scalefield' in pred:
            misc['pred_scalefield'] = pred['scalefield']
        if 'delta' in pred:
            misc['pred_delta'] = pred['delta']

    # Ray angle error (pure direction; no scale/shift alignment)
    if gt['is_metric']:
        if 'points_metric' in pred:
            ray_metrics = ray_angle_error(pred['points_metric'], gt_points, mask)
            metrics['ray_direction_metric'] = ray_metrics
        elif 'points' in pred:
            ray_metrics = ray_angle_error(pred['points'], gt_points, mask)
            metrics['ray_direction_metric'] = ray_metrics

    return metrics, misc
