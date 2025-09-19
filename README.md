<div align="center">
  <img src="documents/icon_medquanbench.png" alt="MedQuanBench icon" width=40%>
</div>

<h2><p align="center">
  MedQuanBench: Quantization-Aware Analysis for Efficient Medical Imaging Models
</p></h2>


<div align="center">

[![MedQuanBench Models](https://img.shields.io/badge/MedQuanBench-Models-59A9B0.svg)](#medquanbench-models)
[![MedQuanBench Results](https://img.shields.io/badge/MedQuanBench-Results-59A9B0.svg)](#medquanbench-results)
[![MedQuanBench Paper](https://img.shields.io/badge/MedQuanBench-Paper-59A9B0.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![GitHub Repo stars](https://img.shields.io/github/stars/Chongyu1117/PTQ4MedSeg?style=social)](https://github.com/Chongyu1117/PTQ4MedSeg/stargazers)

</div>

We introduce **MedQuanBench**, a large-scale and diverse benchmark designed to rigorously evaluate quantization techniques for 3D medical imaging models. Our benchmark spans a wide range of modern architectures (e.g., CNNs and Transformers). We systematically evaluate representative post-training quantization (PTQ) strategies **across model scales and dataset sizes**. Additionally, we perform detailed **sensitivity analyses** to identify which model components are most vulnerable to quantization, including layer-wise degradation and activation distribution shifts. 


## News

- **[2025-09-18]** 🔥 Released — 4-bit simulated quantization code and the [MedFormer checkpoint](#medquanbench-models) are now available, ready to be used in benchmarking.
- **[2025-09-17]** 🚀 Released — 8-bit real quantization code is now available. Check out [MedPTQ](https://github.com/hrlblab/PTQ).  

## Getting Started

- Installation Guide
- Usage Tutorial

## Quantization Granularity 

<div align="center">
  <img src="documents/fig_granularity.png" alt="MedQuanBench Quantization Granularity" width=80%>
</div>

**Quantization granularity in MedQuanBench**
- **(a) Quantization schemes for linear layers:**  
  Activation per-tensor and weight per-tensor quantization (top), activation per-token and weight per-channel quantization (bottom). Vector-wise quantization schemes (per-token, per-channel) efficiently utilize low-bit kernels when scaling factors align with outer tensor dimensions (token dimension *T* and output channel dimension *C<sub>o</sub>*).

- **(b) Quantization schemes for convolutional layers:**  
  Activation per-tensor and weight per-tensor quantization (top), activation per-channel and weight per-channel quantization (bottom). Outer tensor dimension alignment (output channel dimension *C<sub>o</sub>*) facilitates efficient low-bit convolutional implementations.

- **(c) Quantization schemes for spatial dimension:**  
  Per-voxel quantization assigns unique scaling factors for each voxel. For kernel size = 1 (top), one scaling factor per voxel is sufficient; for larger kernels (bottom, shown as *2×2×2*), each position within the kernel uses a separate scaling factor.


## MedQuanBench Models

<table>
  <tr>
    <th>Model</th>
    <th>Download</th>
    <th>Dataset</th>
  </tr>

  <tr>
    <td>MedFormer</td>
    <td>
      <a href="https://www.dropbox.com/scl/fi/xxxxxx/medformer_checkpoint.pth?dl=1">
        <img alt="Checkpoint" src="https://img.shields.io/badge/Checkpoint-59A9B0.svg">
      </a>
    </td>
    <td><a href="https://www.synapse.org/Synapse:syn3193805/wiki/89480">BTCV</a></td>
  </tr>
</table>

> [!NOTE]  
> INT8 quantized **U-Net** and **TransUNet** have already been released and are ready for deployment on real GPUs — see [MedPTQ](https://github.com/hrlblab/PTQ).



## MedQuanBench Results
<details>
<summary><b>MedQuanBench results on BTCV test-set</b></summary>

<p>We evaluate multiple 3D medical segmentation models under <b>FP32</b>, <b>INT8</b> (INT W8A8), and <b>INT4</b> (INT W4A4) post-training quantization.<br/>
Three granularity schemes are compared:<br/>
1. <i>Per-tensor</i> — a single scale for activations and weights in each layer<br/>
2. <i>Per-channel/token</i> — separate scales for each convolutional input channel or transformer token, with all weights quantized per channel<br/>
3. <i>Adaptive stratification</i> — per-voxel scaling for 1×1×1 convolutions, and per-channel/token scaling elsewhere
</p>

<p>
INT8 quantization consistently preserves full-precision accuracy. In contrast, INT4 performance varies depending on model architecture and quantization granularity: <b>hybrid models are particularly sensitive under per-tensor quantization, while CNNs degrade more gradually</b>. Reported DSC and NSD values are shown along with relative drop (↓Δ%) from the FP32 baseline.
</p>

<table>
  <thead>
    <tr>
      <th>Framework</th>
      <th>Architecture</th>
      <th>Category</th>
      <th>Param</th>
      <th>Precision</th>
      <th>Quant-Granularity</th>
      <th>DSC (↓Δ%)</th>
      <th>NSD (↓Δ%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="28">nnUNet</td>
      <td rowspan="7">nnU-Net</td>
      <td rowspan="7">CNN</td>
      <td rowspan="7">26.9&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.872 (--)</td><td>0.888 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel</td><td>0.870 (0.2%)</td><td>0.887 (0.1%)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.870 (0.2%)</td><td>0.888 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.870 (0.2%)</td><td>0.887 (0.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel</td><td>0.387 (55.6%)</td><td>0.354 (60.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.170 (80.5%)</td><td>0.169 (80.9%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.393 (54.9%)</td><td>0.358 (59.7%)</td></tr>
    <tr>
      <td rowspan="7">STU-Net-B</td>
      <td rowspan="7">CNN</td>
      <td rowspan="7">58.3&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.881 (--)</td><td>0.903 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel</td><td>0.881 (0)</td><td>0.901 (0.2%)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.881 (0)</td><td>0.902 (0.1%)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.881 (0)</td><td>0.902 (0.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel</td><td>0.647 (26.6%)</td><td>0.619 (31.5%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.654 (25.8%)</td><td>0.636 (29.6%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.829 (5.9%)</td><td>0.833 (7.8%)</td></tr>
    <tr>
      <td rowspan="7">STU-Net-L</td>
      <td rowspan="7">CNN</td>
      <td rowspan="7">440.3&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.880 (--)</td><td>0.903 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel</td><td>0.880 (0)</td><td>0.902 (0.1%)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.880 (0)</td><td>0.903 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.880 (0)</td><td>0.902 (0.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel</td><td>0.701 (20.3%)</td><td>0.695 (23.0%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.466 (47.0%)</td><td>0.460 (49.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.857 (2.6%)</td><td>0.870 (3.7%)</td></tr>
    <tr>
      <td rowspan="7">STU-Net-H</td>
      <td rowspan="7">CNN</td>
      <td rowspan="7">1,457.3&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.873 (--)</td><td>0.889 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel</td><td>0.873 (0)</td><td>0.889 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.872 (0.1%)</td><td>0.889 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.872 (0.1%)</td><td>0.889 (0)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel</td><td>0.700 (19.8%)</td><td>0.681 (23.4%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.734 (15.9%)</td><td>0.716 (19.5%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.840 (3.8%)</td><td>0.848 (4.6%)</td></tr>
    <tr>
      <td rowspan="14">MONAI</td>
      <td rowspan="7">SwinUNETR</td>
      <td rowspan="7">Hybrid</td>
      <td rowspan="7">58.5&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.849 (--)</td><td>0.760 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel/token</td><td>0.849 (0)</td><td>0.761 (1%↑)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.849 (0)</td><td>0.761 (1%↑)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.849 (0)</td><td>0.761 (1%↑)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel/token</td><td>0.565 (33.5%)</td><td>0.446 (41.3%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.059 (93.1%)</td><td>0.054 (92.9%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.571 (32.7%)</td><td>0.447 (41.2%)</td></tr>
    <tr>
      <td rowspan="7">UNETR</td>
      <td rowspan="7">Hybrid</td>
      <td rowspan="7">92.4&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.824 (--)</td><td>0.714 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel/token</td><td>0.824 (0)</td><td>0.714 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.802 (2.7%)</td><td>0.669 (6.3%)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.809 (1.8%)</td><td>0.676 (5.3%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel/token</td><td>0.553 (35.3%)</td><td>0.366 (48.7%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.004 (99.5%)</td><td>0.004 (94.4%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.590 (28.4%)</td><td>0.386 (45.9%)</td></tr>
    <tr>
      <td rowspan="7">MedFormer</td>
      <td rowspan="7">MedFormer</td>
      <td rowspan="7">Hybrid</td>
      <td rowspan="7">36.6&nbsp;M</td>
      <td>FP32</td><td>--</td><td>0.882 (--)</td><td>0.826 (--)</td>
    </tr>
    <tr><td>INT W8A8</td><td>Per-channel/token</td><td>0.882 (0)</td><td>0.826 (0)</td></tr>
    <tr><td>INT W8A8</td><td>Per-tensor</td><td>0.880 (0.2%)</td><td>0.823 (0.3%)</td></tr>
    <tr><td>INT W8A8</td><td>Adaptive stratification</td><td>0.882 (0)</td><td>0.826 (0)</td></tr>
    <tr><td>INT W4A4</td><td>Per-channel/token</td><td>0.654 (25.9%)</td><td>0.462 (44.1%)</td></tr>
    <tr><td>INT W4A4</td><td>Per-tensor</td><td>0.000 (100%)</td><td>0.000 (100%)</td></tr>
    <tr><td>INT W4A4</td><td>Adaptive stratification</td><td>0.719 (18.5%)</td><td>0.610 (26.3%)</td></tr>

  </tbody>
</table>
</details>
