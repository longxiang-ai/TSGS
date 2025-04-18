# TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors

[![arXiv](https://img.shields.io/badge/arXiv-2504.12799-b31b1b.svg)](https://arxiv.org/abs/2504.12799)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://longxiang-ai.github.io/TSGS/)
[![GitHub](https://img.shields.io/badge/Code-Coming%20Soon-lightgrey)](https://github.com/longxiang-ai/TSGS)
[![Data](https://img.shields.io/badge/Data-Coming%20Soon-lightgrey)](https://longxiang-ai.github.io/TSGS/)

Official code release for the paper: **TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors**.

[Mingwei Li<sup>1,2</sup>](https://github.com/longxiang-ai), [Pu Pang<sup>3,2</sup>](https://github.com/fankewen), [Hehe Fan<sup>1</sup>](https://hehefan.github.io/), [Hua Huang<sup>4</sup>](https://ai.bnu.edu.cn/xygk/szdw/zgj/194482e0996d4044806ac39019896e9c.htm), [Yi Yang<sup>1,&#9993;</sup>](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en)

*<sup>1</sup>Zhejiang University, <sup>2</sup>Zhongguancun Academy, Beijing, <sup>3</sup>Xi'an Jiaotong University, <sup>4</sup>Beijing Normal University*

## News
*   **[2025-04-18]**: 🎉 Our arXiv paper is released! You can find it [here](https://arxiv.org/abs/2504.12799). Project page is also live!


![Teaser Image](./static/images/teaser_02_00.jpg)
*We present TSGS, a framework for high-fidelity transparent surface reconstruction from multi-views. (a) We introduce TransLab, a novel dataset for evaluating transparent object reconstruction. (b) Comparative results on TransLab demonstrate the superior capability of TSGS.*

## Abstract

Reconstructing transparent surfaces is essential for tasks such as robotic manipulation in labs, yet it poses a significant challenge for 3D reconstruction techniques like 3D Gaussian Splatting (3DGS). These methods often encounter a transparency-depth dilemma, where the pursuit of photorealistic rendering through standard alpha-blending undermines geometric precision, resulting in considerable depth estimation errors for transparent materials. To address this issue, we introduce Transparent Surface Gaussian Splatting (TSGS), a new framework that separates geometry learning from appearance refinement. In the geometry learning stage, TSGS focuses on geometry by using specular-suppressed inputs to accurately represent surfaces. In the second stage, TSGS improves visual fidelity through anisotropic specular modeling, crucially maintaining the established opacity to ensure geometric accuracy. To enhance depth inference, TSGS employs a first-surface depth extraction method. This technique uses a sliding window over alpha-blending weights to pinpoint the most likely surface location and calculates a robust weighted average depth. To evaluate the transparent surface reconstruction task under realistic conditions, we collect a TransLab dataset that includes complex transparent laboratory glassware. Extensive experiments on TransLab show that TSGS achieves accurate geometric reconstruction and realistic rendering of transparent objects simultaneously within the efficient 3DGS framework. Specifically, TSGS significantly surpasses current leading methods, achieving a 37.3% reduction in chamfer distance and an 8.0% improvement in F1 score compared to the top baseline. Additionally, TSGS maintains high-quality novel view synthesis, evidenced by a 0.41dB gain in PSNR, demonstrating that TSGS overcomes the transparency-depth dilemma.

## Method Overview

![Pipeline Image](./static/images/pipeline.jpg)
*(a) The two-stage training process. Stage 1 optimizes 3D Gaussians using geometric priors and de-lighted inputs. Stage 2 refines appearance while fixing opacity. (b) Inference extracts the first-surface depth map for mesh reconstruction. (c) The first-surface depth extraction module uses a sliding window for robust depth calculation.*

## TransLab Dataset

We introduce **TransLab**, a novel dataset specifically designed for evaluating transparent object reconstruction in laboratory settings. It features 8 diverse, high-resolution 360° scenes with challenging transparent glassware.

*(Link to download the dataset - Coming Soon)*

## Results

TSGS significantly improves geometric accuracy and maintains high rendering quality on the TransLab dataset compared to state-of-the-art methods.

*   **Geometry:** 37.3% reduction in Chamfer Distance, 8.0% improvement in F1 Score.
*   **Appearance:** 0.41dB gain in PSNR for novel view synthesis.

## TODO
*   [x] Release Arxiv paper link.
*   [ ] Release source code.
*   [ ] Release TransLab dataset and download link.
*   [ ] Provide detailed installation and usage instructions.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{li2025tsgs,
  title={TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors}, 
  author={Mingwei Li and Pu Pang and Hehe Fan and Hua Huang and Yi Yang},
  year={2025},
  eprint={2504.12799},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.12799}, 
}
```