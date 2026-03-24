<h1 align="center">FoundationGeo: Learning Pixel-Wise Spatial Fields for Monocular Metric Geometry</h1>

<p align="center">
  <a href="" target="_blank">Muxin Liu</a><sup>1,2,*</sup>,
  <a href="https://shawlyu.github.io" target="_blank">Xiaoyang Lyu</a><sup>1,2,*</sup>,
  <a href="https://rentainhe.github.io" target="_blank">Tianhe Ren</a><sup>1</sup>,
  <a href="https://daipengwa.github.io" target="_blank">Peng Dai</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=9nsSKpsAAAAJ&hl=en" target="_blank">Xiaoshan Wu</a><sup>1,2</sup>,
  <a href="" target="_blank">Zhiyue Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=V1zSNIYAAAAJ&hl=en" target="_blank">Jiaqi Zhang</a><sup>1</sup>,
  <a href="https://jiehonglin.github.io" target="_blank">Jiehong Lin</a><sup>1</sup>,
  <a href="https://shishaoshuai.com" target="_blank">Shaoshuai Shi</a><sup>2,†</sup>,
  <a href="https://xjqi.github.io" target="_blank">Xiaojuan Qi</a><sup>1,✉</sup>
</p>

<p align="center">
  <sup>1</sup>HKU CVMI &nbsp;&nbsp;
  <sup>2</sup>Voyager Research, Didi Chuxing
  <br>
  <sup>*</sup>Equal Contribution &nbsp;
  <sup>†</sup>Project Lead &nbsp;
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
  <a href="https://huggingface.co/your-model">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face">
  </a>
</p>

<p align="center">
  <img src="assets/Teaser.png" alt="Teaser">
</p>

## TODO

- [ ] Release paper and project page
- [ ] Release Stage-I ViT-L base model and code
- [ ] Release Stage-II FoundationGeo model and code
- [ ] Release model zoo
- [ ] Release training code and config details

## ⚙️ Installation

```bash
git clone https://github.com/mx-liu6/FoundationGeo.git
cd FoundationGeo
pip install -r requirments.txt
```

## 📦 Pretrained Models

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Hugging Face Model</th>
      <th>Metric Scale</th>
      <th># Params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Stage-I Base Model</td>
      <td><a href="">ViT-S</a></td>
      <td>-</td>
      <td>30M</td>
    </tr>
    <tr>
      <td><a href="">ViT-L</a></td>
      <td>-</td>
      <td>313M</td>
    </tr>
    <tr>
      <td><a href="">ViT-Hplus</a></td>
      <td>-</td>
      <td>948M</td>
    </tr>
    <tr>
      <td rowspan="2">Stage-II FoundationGeo</td>
      <td><a href="">ViT-L</a></td>
      <td>✅</td>
      <td>314M</td>
    </tr>
    <tr>
      <td><a href="">ViT-Hplus</a></td>
      <td>✅</td>
      <td></td>
    </tr>
  </tbody>
</table>



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
- [Hugging Face](https://huggingface.co/your-model)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

We thank the MoGe series of works and DINOv3.
