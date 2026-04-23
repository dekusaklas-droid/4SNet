# 4SNet

# 4SNet: Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-Identification

**Mingfu Xiong**<sup>a</sup>, **Feiyang Luo**<sup>a</sup>, **Junjie Huang**<sup>a,\*</sup>, **Yifei Guo**<sup>b</sup>, **Aziz Alotaibi**<sup>c</sup>, **Sambit Bakshi**<sup>d</sup>, **Javier Del Ser**<sup>e,f</sup>, **Khan Muhammad**<sup>g,\*</sup>

<sup>a</sup> Affiliation A  
<sup>b</sup> Affiliation B  
<sup>c</sup> Affiliation C  
<sup>d</sup> Affiliation D  
<sup>e</sup> Affiliation E  
<sup>f</sup> Affiliation F  
<sup>g</sup> Affiliation G  

\* Corresponding authors
<img width="1026" height="536" alt="image" src="https://github.com/user-attachments/assets/b7e370be-4eeb-4da8-802e-7e723bac490d" />

## Introduction
4SNet is designed for visible-infrared person re-identification (VI-ReID).  
The framework introduces two key modules:

- **Adaptive Frequency Filter (AFF):** performs modality-specific and input-adaptive early frequency filtering.
- **Spectrum Synergy Module (SSM):** decomposes features into low-, middle-, and high-frequency bands and exploits their synergy for robust cross-modality alignment.

## Environment
- Python 3.10
- PyTorch
- Ubuntu 22.04
- NVIDIA RTX 4090

## Datasets
Please prepare the following datasets before training and testing:

- SYSU-MM01
- RegDB
- LLCM

Organize the datasets under your local dataset directory.

## Training
Example command:

```bash
python train.py# 4SNet



@article{4snet,
  title={4SNet: Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-Identification},
  author={Xiong, Mingfu and Luo, Feiyang and Huang, Junjie and Guo, Yifei and Alotaibi, Aziz and Bakshi, Sambit and Del Ser, Javier and Muhammad, Khan},
  journal={under review},
  year={2025}
}
