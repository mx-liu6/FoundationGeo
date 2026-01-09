# FoundationGeo: Learning Pixel-Wise Scale Fields for Monocular Metric Geometry

<div align="center">

![Teaser](assets/Teaser.png)

**FoundationGeo: Learning Pixel-Wise Scale Fields for Monocular Metric Geometry**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/2509.20251)
[![Project Page](https://img.shields.io/badge/Project%20Page-Website-green)](https://mx-liu6.github.io/FoundationGeo-web/)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/mx-liu6/FoundationGeo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/your-model)

</div>

## 📝 Abstract

Estimating metric 3D geometry from a single RGB image remains an open challenge, as perspective projection removes absolute scale information. Recent foundation models have achieved highly accurate relative depth through large-scale pretraining, while metric estimation, which recovering depth in real-world units, still lags far behind due to camera ambiguities and data scarcity. We revisit this problem by asking: can the strong structural priors of relative models be effectively repurposed for accurate metric prediction?

We present **FoundationGeo**, with two-stage training strategy that explicitly bridges the relative–metric divide. In the first stage, a DINOv3-vitl backbone is trained with global and multi-scale geometric constraints over a curated 9.1M sample dataset spanning synthetic, indoor, and outdoor scenes to achieve high-fidelity relative geometry. In the second stage, a pixel-level scale field is introduced to learn spatially varying scale factors, jointly optimized through coupled metric–relative objectives with a decoupled scale field loss for stable training.

FoundationGeo attains state-of-the-art zero-shot performance across nine datasets on both metric depth and metric point map benchmarks, while maintaining strong relative consistency. Further analyses show that incorporating sparse metric cues, known intrinsics or domain-specific fine-tuning yields additional gains. In particular, with 3000 sparse metric priors, our predictions can almost match the performance of relative results.


## 🏗️ Architecture

![Structure](assets/Structure.png)

**FoundationGeo architecture.** A ViT encoder with a lightweight up-sampling convolutional decoder first learns a high-fidelity *relative geometry* branch that predicts a validity mask **M̂** and an affine-invariant point map **P̂**. In the second stage, a pixel-level scale field **ŝ** converts these relative predictions into metric geometry through detail-aware rescaling, yielding the metric-scale point **P̃**. Metric depth and surface normals are subsequently derived from **P̃**.

## 📊 Main Results

![Performance](assets/Performance.png)

Quantitative results for metric depth and metric point map estimation. *AbsRel* and δ₁ are in percentage. The best values are highlighted in **bold**, and the second-best ones are <u>underlined</u>. * means we use GT intrinsic as input. <span style="color: #808080;">Gray numbers</span> denote models trained on respective benchmarks or need GT intrinsics, thus excluded from ranking.

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
