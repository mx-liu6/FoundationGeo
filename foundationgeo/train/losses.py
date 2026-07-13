from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    mask_aware_nearest_resize,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    mask: torch.Tensor, 
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device = pred_points.device

    # Align
    (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution))
    scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
    valid = scale > 0
    scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # Compute loss
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used now
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }

    return loss, misc, scale.detach()

def metric_l1_loss_xyz(
    pred_metric_points: torch.Tensor, # (B, H, W, 3)
    gt_points: torch.Tensor, # (B, H, W, 3)
    gt_mask: torch.Tensor, # (B, H, W)
    beta: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    device, dtype = pred_metric_points.device, pred_metric_points.dtype

    weight = gt_mask.float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, gt_mask, dim=(-2, -1), keepdim=True))  
    
    loss = _smooth((pred_metric_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    err = (pred_metric_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]

    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), gt_mask).item(),
        'delta': weighted_mean((err < 1).float(), gt_mask).item()
    }

    return loss, misc

def metric_ray_direction_loss(
    pred_metric_points: torch.Tensor,   # (B,H,W,3) points_metric
    gt_points: torch.Tensor,            # (B,H,W,3)
    gt_mask: torch.Tensor,              # (B,H,W) bool
    *,
    beta_rad: float = math.radians(3.0),
    min_angle: float = math.radians(0.01),
    max_angle: float = math.radians(30.0),
):
    """
    Encourage ray direction of predicted metric points to match GT ray direction.
    Uses angle_diff_vec3 on normalized rays.
    """
    device, dtype = pred_metric_points.device, pred_metric_points.dtype
    eps = 1e-8

    # normalize rays
    pred_r = pred_metric_points / pred_metric_points.norm(dim=-1, keepdim=True).clamp_min(eps)
    gt_r   = gt_points         / gt_points.norm(dim=-1, keepdim=True).clamp_min(eps)

    ang = angle_diff_vec3(pred_r, gt_r).clamp(min_angle, max_angle)  # (B,H,W)

    # weights: prefer near pixels (1/z) like your other losses
    w = gt_mask.float() / gt_points[..., 2].clamp_min(1e-3).sqrt()
    w_mean = weighted_mean(w, gt_mask, dim=(-2, -1), keepdim=True)
    w = w.clamp_max(10.0 * w_mean)

    huber = _smooth(ang, beta=beta_rad)

    nume = (huber * w).sum(dim=(-2, -1))
    denom = w.sum(dim=(-2, -1)).clamp_min(1e-6)
    loss = (nume / denom).to(dtype)

    misc = {}
    with torch.no_grad():
        misc["ray/ang_mean_deg"] = (ang[gt_mask].mean() * 180.0 / math.pi).item() if gt_mask.any() else 0.0

    return loss, misc

def metric_scalefield_range_loss(
    pred_points_rel: torch.Tensor,      # (B,H,W,3) points_rel (after remap, in your forward output)
    pred_scalefield: torch.Tensor,      # (B,H,W,1) or (B,H,W)
    gt_points: torch.Tensor,            # (B,H,W,3)
    gt_mask: torch.Tensor,              # (B,H,W)
    *,
    beta: float = math.log(1.25),       # ±25% in log domain
    s_min: float = 0.05,
    s_max: float = 20.0,
):
    """
    Supervise scalefield as ratio of GT range to REL range:
        s* = ||g|| / ||p_rel||
    (this matches your d_metric = s * d_rel parameterization)
    """
    device, dtype = pred_points_rel.device, pred_points_rel.dtype
    eps = 1e-6

    d_rel = pred_points_rel.norm(dim=-1).clamp_min(eps)   # (B,H,W)
    d_gt  = gt_points.norm(dim=-1).clamp_min(eps)         # (B,H,W)

    s_star = (d_gt / d_rel)
    finite = torch.isfinite(s_star)
    valid = gt_mask & finite & (s_star > 0)
    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    s_star = torch.where(valid, s_star, torch.ones_like(s_star)).clamp(s_min, s_max)
    log_s_star = torch.log(s_star.clamp_min(eps))

    # pred_scalefield is expected to be (..., H, W) or (..., H, W, 1).
    # In the training loop we pass pred_scalefield[i], so its shape is (H, W, 1),
    # which would broadcast incorrectly against (H, W) if we don't squeeze.
    if pred_scalefield.shape[-1] == 1:
        # Works for both (B, H, W, 1) and (H, W, 1)
        s_pred = pred_scalefield[..., 0]
    else:
        s_pred = pred_scalefield
    log_s_pred = torch.log(s_pred.clamp_min(eps))

    diff = (log_s_pred - log_s_star).abs()
    huber = _smooth(diff, beta=beta)

    # weights (same style)
    w = valid.float() / gt_points[..., 2].clamp_min(1e-5)
    w_mean = weighted_mean(w, valid, dim=(-2, -1), keepdim=True)
    w = w.clamp_max(10.0 * w_mean)

    nume = (huber * w).sum(dim=(-2, -1))
    denom = w.sum(dim=(-2, -1)).clamp_min(1e-6)
    loss = (nume / denom).to(dtype)

    misc = {}
    with torch.no_grad():
        misc["scale/log_abs_mean"] = diff[valid].mean().item()
        misc["scale/median_s"] = s_star[valid].median().item()

    return loss, misc

def delta_reg_loss(
    pred_delta: torch.Tensor,   # (B,H,W,2) raw delta1/delta2
    gt_mask: torch.Tensor,      # (B,H,W)
    *,
    p: float = 2.0,
):
    """
    Penalize bounded angular correction magnitude.
    """
    d = torch.tanh(pred_delta)
    valid = gt_mask[..., None].float()

    loss_map = d.abs().pow(p) * valid

    denom = valid.sum(dim=(-3, -2, -1)).clamp_min(1.0) * pred_delta.shape[-1]
    loss = loss_map.sum(dim=(-3, -2, -1)) / denom

    return loss, {}


def metric_scalefield_loss(
    pred_points: torch.Tensor,
    pred_scalefield: torch.Tensor,
    gt_points: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    beta: float = math.log(1.25), # ≈ 0.223143551 -> ±25% tolerance
    s_min: float = 0.05,
    s_max: float = 20.0,
    sparsity_aware: bool = False,
    align_resolution: int = 32
):
    """
    Per-pixel supervision: compute closed-form s_i* from pred_points & gt_points,
    then supervise pred_scalefield (in log domain) with huber-weighted loss.
    Returns (loss_scalar, misc_dict).
    """

    device, dtype = pred_points.device, pred_points.dtype
    eps_z, eps_den, eps_sum = 1e-5, 1e-6, 1e-6

    # 1) compute s_star (closed-form) using pred_points
    p = pred_points
    g = gt_points

    num = (p * g).sum(dim=-1) # [..., H, W]
    den = p.square().sum(dim=-1).clamp_min(eps_den) # [..., H, W]
    s = num / den # [..., H, W]

    finite = torch.isfinite(s)
    valid = gt_mask & finite & ( s > 0 )
    if not valid.any():
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # clamp target scales
    s_safe = torch.where(valid, s, torch.ones_like(s))
    s_safe = s_safe.clamp(s_min, s_max)
    log_s_star = torch.log(s_safe.clamp_min(eps_den))

    # 2) pred scalefield -> ensure linear -> to log
    if pred_scalefield.dim() == 3 and pred_scalefield.shape[-1] == 1:
        pred_s = pred_scalefield.squeeze(-1)
    else:
        pred_s = pred_scalefield
    
    pred_s = torch.log(pred_s.clamp_min(eps_den))
    
    # 3) huber loss

    diff = (pred_s - log_s_star).abs()
    huber = _smooth(diff, beta=beta)

    # 4) weights + winsor cutting edge
    w = (valid.float()) / gt_points[..., 2].clamp_min(eps_z)
    w_mean = weighted_mean(w, valid, dim=(-2, -1), keepdim=True)
    w = w.clamp_max(10 * w_mean)

    # 5) image-level weighted normalization
    nume = (huber * w).sum(dim=(-2, -1))
    denom = w.sum(dim=(-2, -1)).clamp_min(eps_sum)
    loss = (nume / denom).to(dtype)
    
    if sparsity_aware:
        (_, _), lr_mask = mask_aware_nearest_resize(
            (pred_points, gt_points), mask=gt_mask, size=(align_resolution, align_resolution)
        )
        sparsity = gt_mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    
    # misc logging
    with torch.no_grad():
        log_abs_mean = diff[valid].mean().item()
        median_s = s_safe[valid].median().item()
        num_valid = valid.sum().item()
    
    misc = {
        "scalefield_sup/loss_raw": loss.item(),
        "scalefield_sup/valid_pixels": num_valid,
        "scalefield_sup/median_s": median_s,
        "scalefield_sup/log_abs_mean": log_abs_mean,
    }

    return loss, misc
    

def scalefield_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    beta: float = math.log(1.25), # ≈ 0.223143551 -> ±25% tolerance
    s_min: float = 0.05,
    s_max: float = 20.0,
    sparsity_aware: bool = False,
    align_resolution: int = 32
):
    """
    Per-pixel invariant-scale-map loss:
      1) s_i = (p·g)/(||p||^2 + eps_den)  (per-pixel 3D least squares)
      2) clamp s_i to [s_min, s_max], residual r_i = log(s_i)
      3) robust penalty: Huber(|r_i|; beta)
      4) weights: w_i = 1/z_gt, winsor cap at 10× mean weight
      5) image-level weighted normalization: L = (Σ w_i * huber_i) / (Σ w_i)
      6) optional sparsity-aware re-scaling (same convention as global loss)
    """
    device, dtype = pred_points.device, pred_points.dtype
    eps_z, eps_den, eps_sum = 1e-5, 1e-6, 1e-6

    # (1) per-pixel 3D least squares
    num = (pred_points * gt_points).sum(dim=-1) # [..., H, W]
    den = pred_points.square().sum(dim=-1).clamp_min(eps_den) # [..., H, W]
    s = num / den # [..., H, W]

    finite = torch.isfinite(s)
    valid = mask & finite & ( s > 0 )
    if not valid.any():
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    s = torch.where(valid, s, torch.ones_like(s))
    s = s.clamp(s_min, s_max)

    # (2) log residual + Huber(Smooth-L1)
    r = torch.log(s.clamp_min(eps_den))
    err = r.abs()
    huber = _smooth(err, beta=beta)

    # (3) weights + winsor cutting edge
    w = (valid.float()) / gt_points[..., 2].clamp_min(eps_z)
    w_mean = weighted_mean(w, valid, dim=(-2, -1), keepdim=True)
    w = w.clamp_max(10 * w_mean)

    # (4) image-level weighted normalization
    nume = (huber * w).sum(dim=(-2, -1))
    denom = w.sum(dim=(-2, -1)).clamp_min(eps_sum)
    loss = (nume / denom).to(dtype)

    if sparsity_aware:
        (_, _), lr_mask = mask_aware_nearest_resize(
            (pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution)
        )
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    
    # log
    with torch.no_grad():
        log_abs_mean = err[valid].mean().item()
        median_s = s[valid].median().item()
        pct_25 = (err[valid] < math.log(1.25)).float().mean().item()
    
    misc = {
        "scale_loss/log_abs_mean": log_abs_mean,
        "scale_loss/median_s": median_s,
        "scale_loss/pct_25": pct_25,
    }
    
    return loss, misc


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor, 
    mask: torch.Tensor, 
    radius_2d: torch.Tensor, 
    radius_3d: torch.Tensor, 
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    gt_mask: torch.Tensor, 
    focal: torch.Tensor, 
    global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], 
    align_resolution: int = 32, 
    num_patches: int = 16, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None
    
    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    # Align patch points
    (pred_patch_points_lr, gt_patch_points_lr), patch_lr_mask = mask_aware_nearest_resize((pred_patch_points, gt_patch_points), mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
    else:
        patch_valid = local_scale > 0
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]
    
    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]
    
    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc

def normal_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss, {}



def edge_loss(points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}

def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}

