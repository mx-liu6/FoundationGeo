<h1 align="center">FoundationGeo: Learning Spatial Pixel-Wise Fields for Monocular Metric Geometry</h1>

<p align="center">
  <a href="https://mx-liu6.github.io/FoundationGeo-web/" target="_blank">Muxin Liu</a><sup>1,2,*</sup>,
  <a href="https://shawlyu.github.io" target="_blank">Xiaoyang Lyu</a><sup>1,2,*</sup>,
  <a href="https://rentainhe.github.io" target="_blank">Tianhe Ren</a><sup>1</sup>,
  <a href="https://daipengwa.github.io" target="_blank">Peng Dai</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=9nsSKpsAAAAJ&hl=en" target="_blank">Xiaoshan Wu</a><sup>1,2</sup>,
  <a href="https://mx-liu6.github.io/FoundationGeo-web/" target="_blank">Zhiyue Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=V1zSNIYAAAAJ&hl=en" target="_blank">Jiaqi Zhang</a><sup>1</sup>,
  <a href="https://jiehonglin.github.io" target="_blank">Jiehong Lin</a><sup>1</sup>,
  <a href="https://shishaoshuai.com" target="_blank">Shaoshuai Shi</a><sup>2,✉</sup>,
  <a href="https://xjqi.github.io" target="_blank">Xiaojuan Qi</a><sup>1,✉</sup>
</p>

<p align="center">
  <sup>1</sup>The University of Hong Kong &nbsp;&nbsp;
  <sup>2</sup>Voyager Research, Didi Chuxing
  <br>
  <sup>*</sup>Equal Contribution &nbsp;
  <sup>✉</sup>Corresponding Author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.20251">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
  <a href="https://mx-liu6.github.io/FoundationGeo-web/">
    <img src="https://img.shields.io/badge/Project%20Page-Website-green" alt="Project Page">
  </a>
  <a href="https://github.com/mx-liu6/FoundationGeo">
    <img src="https://img.shields.io/badge/Code-GitHub-blue" alt="Code">
  </a>
  <a href="https://huggingface.co/mxliu-hku/FoundationGeo-1.1">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face">
  </a>
</p>

<p align="center">
  <img src="assets/Teaser.png" alt="Teaser">
</p>

## TODO

- [x] Release paper and project page
- [x] Release Stage-I Base model and code
- [x] Release Stage-II FoundationGeo model and code
- [x] Release training code and config details
- [ ] Release full model zoo

## ⚙️ Installation

```bash
git clone https://github.com/mx-liu6/FoundationGeo.git
cd FoundationGeo
conda create -n foundationgeo python=3.10 -y
conda activate foundationgeo
pip install -r requirements.txt
```

## 📦 Pretrained Models

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Hugging Face Model</th>
      <th>Version</th>
      <th>Release Type</th>
      <th>Metric Scale</th>
      <th># Params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Stage-I Base Model</td>
      <td><a href="https://huggingface.co/mxliu-hku/FoundationGeo-Base">FoundationGeo-Base-v1</a></td>
      <td>v1</td>
      <td>Submission version</td>
      <td>-</td>
      <td>313M</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/mxliu-hku/FoundationGeo-Base-1.1">FoundationGeo-Base-v1.1</a></td>
      <td>v1.1</td>
      <td><b>Recommended version</b></td>
      <td>-</td>
      <td>313M</td>
    </tr>
    <tr>
      <td rowspan="2">Stage-II FoundationGeo</td>
      <td><a href="https://huggingface.co/mxliu-hku/FoundationGeo">FoundationGeo-v1</a></td>
      <td>v1</td>
      <td>Submission version</td>
      <td>✅</td>
      <td>314M</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/mxliu-hku/FoundationGeo-1.1">FoundationGeo-v1.1</a></td>
      <td>v1.1</td>
      <td><b>Recommended version</b></td>
      <td>✅</td>
      <td>314M</td>
    </tr>
  </tbody>
</table>


## 🚀 Training

We provide training configurations and launch scripts for both stages:

- Stage-I Base model: [`configs/train/BaseModel.json`](configs/train/BaseModel.json), [`start_scripts/BaseModel.sh`](start_scripts/BaseModel.sh)
- Stage-II FoundationGeo model: [`configs/train/FoundationGeo.json`](configs/train/FoundationGeo.json), [`start_scripts/FoundationGeo.sh`](start_scripts/FoundationGeo.sh)

### Additional Requirements

Training uses [`accelerate`](https://github.com/huggingface/accelerate) for distributed training and `mlflow` for optional logging:

```bash
pip install accelerate mlflow sympy tqdm
```

### Data Preparation

Training datasets are expected under `data/train`. Each dataset should contain an index file and per-sample folders:

```text
data/train/somedataset
├── index.txt
├── sample_000001
│   ├── image.jpg
│   ├── depth.png
│   └── meta.json
└── ...
```

`index.txt` stores one sample folder per line. `meta.json` should include normalized camera intrinsics:

```json
{
  "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
}
```

Depth maps can be read and written with the helpers in [`foundationgeo/utils/io.py`](foundationgeo/utils/io.py). You can inspect prepared samples with:

```bash
python foundationgeo/scripts/vis_data.py data/train/somedataset --ply --depth_vis
```

### Run Training

The provided launch scripts are designed for multi-GPU or multi-node training environments. They read distributed settings from environment variables such as `RESOURCE_NUM_GPU`, `DISTRIBUTED_NODE_COUNT`, `DISTRIBUTED_NODE_RANK`, and `DISTRIBUTED_MASTER_HOSTS`.

```bash
# Train Stage-I Base model
bash start_scripts/BaseModel.sh

# Train Stage-II FoundationGeo model
bash start_scripts/FoundationGeo.sh
```

For a single-machine launch, you can call `accelerate` directly and adjust `--num_processes`, batch size, workspace, and checkpoint path as needed:

```bash
accelerate launch --multi_gpu --num_processes 8 \
  foundationgeo/scripts/train.py \
  --config configs/train/FoundationGeo.json \
  --workspace workspace/FoundationGeo \
  --gradient_accumulation_steps 1 \
  --batch_size_forward 8 \
  --checkpoint latest \
  --enable_gradient_checkpointing False \
  --enable_mlflow True
```

To train the Stage-I Base model directly, replace the script and config with `foundationgeo/scripts/train_base.py` and `configs/train/BaseModel.json`.

## 📏 Evaluation

We provide a unified evaluation pipeline that wraps a baseline model, evaluates it on multiple benchmarks, and writes metrics to a JSON file.

### Benchmarks

Download the processed evaluation datasets from [Hugging Face Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and place them under `data/eval`:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation \
  --repo-type dataset \
  --local-dir data/eval \
  --local-dir-use-symlinks False
```

Then unzip the downloaded benchmark files:

```bash
cd data/eval
unzip '*.zip'
```

### Configuration

See [`configs/eval/all_benchmarks.json`](configs/eval/all_benchmarks.json) for the default benchmark configuration. You can edit this file to change dataset paths, image sizes, or benchmark subsets.

### Run Evaluation

The provided scripts evaluate the Stage-I Base model and Stage-II FoundationGeo model from local checkpoints:

```bash
# Evaluate Stage-I Base model
CHECKPOINT_PATH=checkpoints/FoundationGeo_BaseModel.pt \
bash start_scripts/Eval/FoundationGeo_Base_Eval.sh

# Evaluate Stage-II FoundationGeo model
CHECKPOINT_PATH=checkpoints/FoundationGeo_v1.pt \
bash start_scripts/Eval/FoundationGeo_Eval.sh
```

You can also run the evaluation script directly. Extra arguments after `--output` are passed to the baseline loader:

```bash
# Evaluate Stage-II FoundationGeo from a Hugging Face repo or local checkpoint
python foundationgeo/scripts/eval_baseline.py \
  --baseline baselines/foundationgeo.py \
  --config configs/eval/all_benchmarks.json \
  --output eval_output/FoundationGeo.json \
  --pretrained mxliu-hku/FoundationGeo-1.1 \
  --resolution_level 9 \
  --version v1

# Evaluate Stage-I Base model
python foundationgeo/scripts/eval_baseline.py \
  --baseline baselines/foundationgeo.py \
  --config configs/eval/all_benchmarks.json \
  --output eval_output/FoundationGeo_Base.json \
  --pretrained mxliu-hku/FoundationGeo-Base-1.1 \
  --resolution_level 9 \
  --version base
```

Useful evaluation options include `--oracle` for GT intrinsics, `--dump_pred` for prediction dumps, and `--dump_gt` for ground-truth dumps. To evaluate a customized method, implement the interface in [`foundationgeo/test/baseline.py`](foundationgeo/test/baseline.py); see [`baselines/foundationgeo.py`](baselines/foundationgeo.py) for an example.


## 🏗️ Architecture

![Structure](assets/Structure.png)

## 📊 Main Results

![Performance](assets/Performance.png)

Quantitative results for metric and relative depth estimation. *AbsRel* and *delta1* are in percentage. The best values are highlighted in **bold**, and the second-best ones are <u>underlined</u>. `*` indicates the model requires GT intrinsics as input. <span style="color: gray;">Gray numbers</span> denote models trained on respective benchmarks or requiring GT intrinsics, and are therefore excluded from ranking.

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@article{liu2025foundationgeo,
  title={FoundationGeo: Learning Pixel-Wise Scale Fields for Monocular Metric Geometry},
  author={Liu, Muxin and Lyu, Xiaoyang and Ren, Tianhe and Wu, Xiaoshan and Zhang, Jiaqi and Lin, Jiehong and Shi, Shaoshuai and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2509.20251},
  year={2025}
}
```

## 🔗 Links

- [Paper](https://arxiv.org/abs/2509.20251)
- [Project Page](https://mx-liu6.github.io/FoundationGeo-web/)
- [Code](https://github.com/mx-liu6/FoundationGeo)
- [Hugging Face](https://huggingface.co/mxliu-hku/FoundationGeo-1.1)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

We thank the MoGe series of works and DINOv3.
