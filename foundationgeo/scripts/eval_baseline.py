import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, oracle_mode: bool, output_path: Union[str, Path], dump_pred: bool, dump_gt: bool):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from foundationgeo.test.baseline import MGEBaselineInterface
    from foundationgeo.test.dataloader import EvalDataLoaderPipeline
    from foundationgeo.test.metrics import compute_metrics
    from foundationgeo.utils.geometry_torch import intrinsics_to_fov
    from foundationgeo.utils.vis import colorize_depth, colorize_error_map, colorize_normal, colorize_scalar_field, colorize_vector_field
    from foundationgeo.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    
    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    # Iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            # Iterate over the samples in the dataset
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                image = sample['image']
                gt_intrinsics = sample['intrinsics']

                # Inference
                torch.cuda.synchronize()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image)
                    torch.cuda.synchronize()

                # Compute metrics
                metrics, misc = compute_metrics(pred, sample, vis=dump_pred or dump_gt)
                metrics['inference_time'] = timer.time
                metrics_list.append(metrics)

                # Dump results
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)

                    pred_mask = pred['mask'].cpu().numpy() if 'mask' in pred else None

                    def save_depth_vis(filename: str, depth_map: np.ndarray):
                        cv2.imwrite(str(dump_path / 'pred' / filename), cv2.cvtColor(colorize_depth(depth_map, pred_mask), cv2.COLOR_RGB2BGR))

                    def save_error_vis(filename: str, error_map: np.ndarray, value_range: Tuple[float, float] = (0.0, 10.0)):
                        cv2.imwrite(
                            str(dump_path / 'pred' / filename),
                            cv2.cvtColor(colorize_error_map(error_map, pred_mask, cmap='coolwarm', value_range=value_range), cv2.COLOR_RGB2BGR)
                        )

                    def save_signed_error_vis(filename: str, error_map: np.ndarray, value_range: Optional[Tuple[float, float]] = (-0.2, 0.2)):
                        if value_range is None:
                            valid = np.isfinite(error_map)
                            if np.any(valid):
                                vmin = float(np.nanquantile(error_map, 0.01))
                                vmax = float(np.nanquantile(error_map, 0.99))
                                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                                    value_range = (-0.2, 0.2)
                                else:
                                    value_range = (vmin, vmax)
                            else:
                                value_range = (-0.2, 0.2)
                        cv2.imwrite(
                            str(dump_path / 'pred' / filename),
                            cv2.cvtColor(colorize_error_map(error_map, pred_mask, cmap='coolwarm', value_range=value_range), cv2.COLOR_RGB2BGR)
                        )

                    if 'pred_points' in misc:
                        points = misc['pred_points'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    
                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if pred_mask is not None:
                            depth = np.where(pred_mask, depth, np.inf)
                        save_depth_vis('depth.png', depth)

                    if 'mask' in pred:
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (pred_mask * 255).astype(np.uint8))

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                    if 'pred_depth_metric' in misc:
                        save_depth_vis('depth_metric.png', misc['pred_depth_metric'].cpu().numpy())

                    if 'pred_scalefield' in misc:
                        scalefield = misc['pred_scalefield'].cpu().numpy()
                        cv2.imwrite(
                            str(dump_path / 'pred' / 'scalefield_vis.png'),
                            cv2.cvtColor(colorize_scalar_field(scalefield[..., 0], pred_mask), cv2.COLOR_RGB2BGR)
                        )

                    if 'pred_delta' in misc:
                        delta = misc['pred_delta'].cpu().numpy()
                        delta_magnitude = np.linalg.norm(delta, axis=-1)
                        cv2.imwrite(
                            str(dump_path / 'pred' / 'delta_mag.png'),
                            cv2.cvtColor(colorize_scalar_field(delta_magnitude, pred_mask, cmap='magma'), cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            str(dump_path / 'pred' / 'delta_vis.png'),
                            cv2.cvtColor(colorize_vector_field(delta, pred_mask), cv2.COLOR_RGB2BGR)
                        )

                    if 'ray_angle_error_metric' in misc:
                        save_error_vis('ray_angle_error_metric.png', misc['ray_angle_error_metric'].cpu().numpy())

                    if 'intrinsics' in pred:
                        intrinsics = pred['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)
                
                if dump_gt:
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if 'points' in sample:
                        points = sample['points']
                        cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'gt' / 'info.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    Path(output_path).write_text(
                        json.dumps({
                            **all_metrics, 
                            benchmark_name: key_average(metrics_list)
                        }, indent=4)
                    )
                pbar.update(1)

            all_metrics[benchmark_name] = key_average(metrics_list)

    # Save final results
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == '__main__':
    main()
