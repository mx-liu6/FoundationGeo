from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import utils3d
from huggingface_hub import hf_hub_download

from .dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitl16, dinov3_vits16, dinov3_vith16plus, _make_dinov3_vit, Weights
from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, gaussian_blur_2d, dilate_with_mask, apply_delta_to_ray, safe_norm
from .utils import wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit


class ResidualConvBlock(nn.Module):  
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, padding_mode: str = 'replicate', activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  


class Head(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512, # ori 512
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
    ):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])
    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )
            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 16, img_w // 16

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)
        
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output


class FoundationGeo(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        encoder: str = 'dinov3_vitl16', 
        intermediate_layers: Union[int, List[int]] = [4, 11, 17, 23],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        remap_output: Literal[False, True, 'linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        num_tokens_range: Tuple[Number, Number] = [1200, 2500],
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        mask_threshold: float = 0.5,
        dinov3_ckpt: Optional[str] = None,
        freeze_encoder: bool = False,
        freeze_relative: bool = False,
        **deprecated_kwargs
    ):
        super(FoundationGeo, self).__init__()

        if deprecated_kwargs:
            # Process legacy arguments
            if 'trained_area_range' in deprecated_kwargs:
                num_tokens_range = [deprecated_kwargs['trained_area_range'][0] // 16 ** 2, deprecated_kwargs['trained_area_range'][1] // 16 ** 2]
                del deprecated_kwargs['trained_area_range']
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold
        self.dinov3_ckpt = dinov3_ckpt
        self.freeze_encoder = freeze_encoder
        self.freeze_relative = freeze_relative

        # Load DINOv3 backbone according to encoder config
        backbone_factories = {
            'dinov3_vit7b16': dinov3_vit7b16,
            'dinov3_vitl16': dinov3_vitl16,
            'dinov3_vits16': dinov3_vits16,
            'dinov3_vith16plus': dinov3_vith16plus,
        }
        if self.encoder not in backbone_factories:
            raise ValueError(f"Unsupported encoder: {self.encoder}. "
                             f"Supported encoders are: {list(backbone_factories.keys())}")
        self.backbone = backbone_factories[self.encoder](
            pretrained=False,
            weights=Weights.LVD1689M,
            device='cuda',
        )
        dim_feature = self.backbone.embed_dim

        self.head = Head(
            num_features=intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers), 
            dim_in=dim_feature, 
            dim_out=[3, 1, 1, 2], # pointmap, mask, scalefield, delta
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size 
        )

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self._apply_freeze()

    def _set_module_trainable(self, module: nn.Module, trainable: bool):
        for param in module.parameters():
            param.requires_grad = trainable
        if not trainable:
            module.eval()

    def _apply_freeze(self):
        if self.freeze_relative:
            self._set_module_trainable(self.backbone, False)
            self._set_module_trainable(self.head.projects, False)
            self._set_module_trainable(self.head.upsample_blocks, False)
            self._set_module_trainable(self.head.output_block[0], False)  # relative points
            self._set_module_trainable(self.head.output_block[1], False)  # mask
        elif self.freeze_encoder:
            self._set_module_trainable(self.backbone, False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._apply_freeze()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'FoundationGeo':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `FoundationGeo` with the parameters loaded from the checkpoint.
        """
        pretrained_model_name_or_path = os.path.expanduser(os.path.expandvars(str(pretrained_model_name_or_path)))
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'])
        return model

    # DINOv3
    def init_weights(self, ckpt_path: Optional[str] = None, device: str = "cuda"):
        "Load the backbone with pretrained dinov3 weights from ckpt"
        if ckpt_path is None:
            raise ValueError("A DINOv3 checkpoint path is required to initialize backbone weights.")
        ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        res = self.backbone.load_state_dict(sd, strict=False)
        print("[DINOv3] load - missing:", len(res.missing_keys), " unexpected:", len(res.unexpected_keys))
    
    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for the backbone and head
        """
        if any(param.requires_grad for param in self.backbone.parameters()):
            for i in range(len(self.backbone.blocks)):
                self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])
    
    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        original_height, original_width = image.shape[-2:]
        
        # Resize to expected resolution defined by num_tokens
        resize_factor = ((num_tokens * 16 ** 2) / (original_height * original_width)) ** 0.5
        resized_width, resized_height = int(original_width * resize_factor), int(original_height * resize_factor)
        image = F.interpolate(image, (resized_height, resized_width), mode="bicubic", align_corners=False, antialias=True)
    
        # Apply image transformation for DINOv3
        image = (image - self.image_mean) / self.image_std
        image_16 = F.interpolate(image, (resized_height // 16 * 16, resized_width // 16 * 16), mode="bilinear", align_corners=False, antialias=True)

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(image_16, n=self.intermediate_layers, return_class_token=True)

        points_rel, mask, logS, delta = self.head(features, image)

        delta1, delta2 = delta.split([1, 1], dim=1) # (B, 1, H, W), (B, 1, H, W)
        
        # --- Back to Original Resolution in fp32 ---
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # resize rel outputs
            points_rel = F.interpolate(points_rel, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            mask = F.interpolate(mask, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            
            # resize metric params to original
            logS = F.interpolate(logS, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            delta1 = F.interpolate(delta1, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            delta2 = F.interpolate(delta2, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)

            # Post-process
            points_rel = points_rel.permute(0, 2, 3, 1).contiguous()
            mask = mask.squeeze(1).sigmoid().contiguous()
            points_rel = self._remap_points(points_rel)
            points_pre_delta = points_rel

            # Metric Calibration
            scalefield = torch.exp(logS.permute(0, 2, 3, 1).contiguous())
            delta1 = delta1.permute(0, 2, 3, 1).contiguous()
            delta2 = delta2.permute(0, 2, 3, 1).contiguous()

            # --- Route A fusion: ray direction bias + range scale ---
            # r = p_rel / ||p_rel||
            d_rel = safe_norm(points_rel, dim=-1, eps=1e-8) # (B, H, W, 1)
            r = points_rel / d_rel # (B, H, W, 3)

            # apply delat to ray
            r_prime = apply_delta_to_ray(r, delta1, delta2, delta_max_rad=0.0523598) # 3 deg

            points_rel = d_rel * r_prime

            delta = torch.cat([delta1, delta2], dim=-1) # (B, H, W, 2)

        return {
            'points': points_rel,
            'points_pre_delta': points_pre_delta,
            'mask': mask,
            'scalefield': scalefield,
            'delta': delta, # (B, H, W, 2)
        }

    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        fov_x: Union[Number, torch.Tensor] = None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = True,
        force_projection: bool = False,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)\
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `resolution_level`: An integer [0-9] for the resolution level for inference. 
            The higher, the finer details will be captured, but slower. Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size.
            `resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.
        - `num_tokens`: number of tokens used for inference. A integer in the (suggested) range of `[1200, 3600]`.
            `resolution_level` will be ignored if `num_tokens` is provided. Default: None
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `force_projection`: if True, the output point map will be recomputed to match the projection constraint. Default: True
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens)
        
        points = output['points']
        points_pre_delta = output.get('points_pre_delta', points)
        mask = output['mask']
        scalefield = output.get('scalefield', None)
        delta = output.get('delta', None)

        # if scalefield is not None:
        #     points_metric = points * scalefield
        #     depth_metric = depth * scalefield.squeeze(-1)

        #     if apply_mask:
        #         points_metric = torch.where(mask_binary[..., None], points_metric, torch.inf)
        #         depth_metric = torch.where(mask_binary, depth_metric, torch.inf)
            
        #     # return_dict.update({
        #     #     'scalefield': scalefield,
        #     #     'points_metric': points_metric,
        #     #     'depth_metric': depth_metric,
        #     # })

            # return_dict = {
            #     'scalefield': scalefield,
            #     'points_metric': points_metric,
            #     'depth_metric': depth_metric,
            # }

        # Always process the output in fp32 precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points, mask, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, mask, fov_x])

            mask_binary = mask > self.mask_threshold

            def postprocess_points(points_in: torch.Tensor):
                """
                Estimate Intrinsics + shift from pointmap and optionally enforce projection constraint
                Returns: points_out, depth_out, intrinsics_out, mask_binary
                """
                depth = points_in[..., 2]

                # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
                if fov_x is None:
                    focal, shift = recover_focal_shift(points_in, mask_binary)
                else:
                    focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points_in.device, dtype=points_in.dtype) / 2))
                    if focal.ndim == 0:
                        focal = focal[None].expand(points_in.shape[0])
                    _, shift = recover_focal_shift(points_in, mask_binary, focal=focal)
                fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
                fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
                intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                depth_out = depth + shift[..., None, None]
                
                # If projection constraint is forced, recompute the point map using the actual depth map
                if force_projection:
                    points_out = utils3d.torch.depth_to_points(depth_out, intrinsics=intrinsics)
                else:
                    points_out = points_in + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

                return points_out, depth_out, intrinsics

            # Relative postprocess
            points_pre_field, depth_pre_field, intrinsics_pre_field = postprocess_points(points_pre_delta)
            points_rel, depth_rel, intrinsics_rel = postprocess_points(points)
            points_metric = points_rel * scalefield
            depth_metric = points_metric[..., 2]
            # depth_metric = depth_rel * scalefield.squeeze(-1)

            if apply_mask:
                points_pre_field = torch.where(mask_binary[..., None], points_pre_field, torch.inf)
                depth_pre_field = torch.where(mask_binary, depth_pre_field, torch.inf)
                points_rel = torch.where(mask_binary[..., None], points_rel, torch.inf)
                depth_rel = torch.where(mask_binary, depth_rel, torch.inf)
                points_metric = torch.where(mask_binary[..., None], points_metric, torch.inf)
                depth_metric = torch.where(mask_binary, depth_metric, torch.inf)

        # return_dict.update({
        #     'points': points,
        #     'intrinsics': intrinsics,
        #     'depth': depth,
        #     'mask': mask_binary,
        # })

        return_dict = {
            'points_pre_field': points_pre_field,
            'depth_pre_field': depth_pre_field,
            'intrinsics_pre_field': intrinsics_pre_field,
            'points': points_rel,
            'points_post_delta': points_rel,
            'intrinsics': intrinsics_rel,
            'depth': depth_rel,
            'depth_post_delta': depth_rel,
            'mask': mask_binary,
            'points_metric': points_metric,
            'depth_metric': depth_metric,
            'scalefield': scalefield,
            'delta': delta,
        }

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
