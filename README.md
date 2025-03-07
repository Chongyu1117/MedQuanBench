<h1 align="center">PTQ for 3D Medical Image Segmentation</h1>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/hrlblab/PTQ?style=social)](https://github.com/hrlblab/PTQ/stargazers)

</div>

we introduce a real post-training quantization (PTQ) framework that successfully implements **true 8-bit quantization** on state-of-the-art (SOTA) 3D medical segmentation models. First, we use [TensorRT](https://github.com/NVIDIA/TensorRT-Model-Optimizer) to perform fake quantization for both weights and activations with unlabeled calibration dataset. Second, we convert this fake quantization into real quantization via TensorRT engine on real GPUs, resulting in **real-world reductions** in model size and inference latency.

<p align="center"><img width="100%" src="documents/fig_workflow.png" /></p>

## Paper
---

<b>Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines </b> <br/>
[Chongyu Qu](https://scholar.google.com/citations?user=5irnqbkAAAAJ&hl=en)<sup>1</sup>, [Ritchie Zhao](https://scholar.google.com/citations?user=8dswaWgAAAAJ&hl=en)<sup>2</sup>, [Ye Yu](https://scholar.google.com/citations?user=x4IFIuYAAAAJ&hl=en)<sup>2</sup>, [Bin Liu](https://scholar.google.com/citations?user=fG2BCO8AAAAJ&hl=en)<sup>2</sup>,[Tianyuan Yao](https://scholar.google.com/citations?user=DeADZl0AAAAJ&hl=en)<sup>1</sup>, [Junchao Zhu](https://scholar.google.com/citations?user=1RyYo7gAAAAJ&hl=en)<sup>1</sup>, [Bennett A. Landman](https://scholar.google.com/citations?user=tmTcH0QAAAAJ&hl=en)<sup>1</sup>, [Yucheng Tang](https://scholar.google.com/citations?user=0xheliUAAAAJ&hl=en)<sup>2</sup>, [Yuankai Huo](https://scholar.google.com/citations?user=WRxmxNgAAAAJ&hl=en)<sup>1*</sup> <br/>
<sup>1 </sup> Vanderbilt University, <br/>
<sup>2 </sup> NVIDIA <br/>
[![arXiv](https://img.shields.io/badge/arXiv-2501.17343-b31b1b.svg)](https://arxiv.org/pdf/2501.17343)


## 0. Installation
---
```bash
git clone https://github.com/hrlblab/PTQ.git
```
See [installation instructions](documents/INSTALL.md) to download TensorRT and create environment.
## 1. Prepare models and data
---

<table>
<tr>
<th>Models</th>
<th>Download</th>
<th>Dataset</th>
</tr>
    
<tr>
<td>U-Net</td>
<td><a href='https://www.dropbox.com/scl/fi/2ym99l4gaf6umow9c6z57/unet_fp32.onnx?rlkey=42j06jicadpaw8qfe4gx1liik&st=cga8r0gw&dl=1'><img src='https://img.shields.io/badge/FP32.onnx-DDA14A'></a> <a href='https://www.dropbox.com/scl/fi/ux2jp2sd8t2g74y190v2a/unet_int8.onnx?rlkey=x6gl2yyd1xa0moc73r22i99yn&st=j84pmxkb&dl=1'><img src='https://img.shields.io/badge/INT8.onnx-479E78'></a></td>
<td rowspan="2"><a href="https://www.synapse.org/Synapse:syn3193805/wiki/89480">BTCV</a></td>
</tr>
    
<tr>
<td>TransUNet</td>
<td><a href='https://www.dropbox.com/scl/fi/i47fjndx3mmgdx0sseds2/transunet_fp32.onnx?rlkey=sv9hocvxiae4zr8cnwrv16fen&st=r0obt4in&dl=1'><img src='https://img.shields.io/badge/FP32.onnx-DDA14A'></a> <a             href='https://www.dropbox.com/scl/fi/qcfk8tl0gehy3dilkml4v/transunet_int8.onnx?rlkey=k87xj51wrw6vevouq1a4fggcp&st=683ydc2c&dl=1'><img src='https://img.shields.io/badge/INT8.onnx-479E78'></a></td>
</tr>

<tr>
<td>UNesT</td>
<td>🔥coming soon</td>
<td>Whole Brain</td>
</tr>
    
<tr>
<td>VISTA3D</td>
<td>🔥coming soon</td>
<td rowspan="4"><a href="https://github.com/wasserth/TotalSegmentator">TotalSegmentator V2</a></td>
</tr>

<tr>
<td>SegResNet</td>
<td>🔥coming soon</td>
</tr>

<tr>
<td>SwinUNETR</td>
<td>🔥coming soon</td>
</tr>

<tr>
<td>nnU-Net</td>
<td>🔥coming soon</td>
</tr>

</table>


## 2. Deploy model
---
##### INT8
```
python deploy.py \
    --onnx_path unet_int8.onnx \
    --data_path ./BTCV/data/test/images \
    --label_path ./BTCV/data/test/labels \
    --net unet \
    --compute_dice
```
##### Compare with FP32 counterpart
```
python deploy.py \
    --onnx_path unet_fp32.onnx \
    --data_path ./BTCV/data/test/images \
    --label_path ./BTCV/data/test/labels \
    --net unet \
    --compute_dice
```


## 3. Try entire PTQ framework
---
##### 3.1 Train models on your own dataset
For example, train U-Net on BTCV
```
python train.py \
    --model unet3d \
    --data_path ./BTCV/data/train/images\
    --label_path ./BTCV/data/train/labels
    --output_dir ./output_unet \
    --batch_size 2
```
##### 3.2 Export pre-trained PyTorch model to ONNX format

```
python export.py \
    --onnx_file_name unet_fp32.onnx \
    --net unet \
    --pretrained_weights_path ./output_unet/checkpoint_epoch_460.pth
```
##### 3.3 Prepare Calibration Data
```
python image_prep.py \
    --output_path calib_btcv.npy \
    --data_path ./BTCV/data/test/images \
    --label_path ./BTCV/data/test/labels
```
##### 3.4 Run INT8 Quantization
```
python -m modelopt.onnx.quantization \
    --onnx_path unet_fp32.onnx \
    --quantize_mode int8 \
    --calibration_data calib_btcv.npy \
    --calibration_method max \
    --output_path unet_int8.onnx \
    --high_precision_dtype fp32
```
##### 3.5 Deploy quantized model
```
python deploy.py \
    --onnx_path unet_int8.onnx \
    --data_path ./BTCV/data/test/images \
    --label_path ./BTCV/data/test/labels \
    --net unet \
    --compute_dice
```

## Citation
---
```
@misc{qu2025posttrainingquantization3dmedical,
      title={Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines}, 
      author={Chongyu Qu and Ritchie Zhao and Ye Yu and Bin Liu and Tianyuan Yao and Junchao Zhu and Bennett A. Landman and Yucheng Tang and Yuankai Huo},
      year={2025},
      eprint={2501.17343},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.17343}, 
}
```
## Acknowledgements
This research was supported by NIH R01DK135597 (Huo) and KPMP Glue Grant. This work was also supported by Vanderbilt Seed Success Grant, Vanderbilt Discovery Grant, and VISE Seed Grant. This project was supported by The Leona M. and Harry B. Helmsley Charitable Trust grant G-1903-03793 and G-2103-05128. This research was also supported by NIH grants R01EB033385, R01DK132338, REB017230, R01MH125931, and NSF 2040462. We extend gratitude to NVIDIA for their support by means of the NVIDIA hardware grant. This work was also supported by NSF NAIRR Pilot Award NAIRR240055.

