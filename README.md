# 4SNet

# 4SNet: Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-Identification
by Mingfu Xiong, Feiyang Luo, Junjie Huang*, Yifei Guo, Aziz Alotaibi, Sambit Bakshi, Javier Del Ser, and Khan Muhammad*

<img width="1026" height="536" alt="image" src="https://github.com/user-attachments/assets/b7e370be-4eeb-4da8-802e-7e723bac490d" />

## Introduction
4SNet is designed for visible-infrared person re-identification (VI-ReID).  
The framework introduces two key modules:

- **Adaptive Frequency Filter (AFF):** performs modality-specific and input-adaptive early frequency filtering.
- **Spectrum Synergy Module (SSM):** decomposes features into low-, middle-, and high-frequency bands and exploits their synergy for robust cross-modality alignment.

## Installation
Example command:

```bash
git clone https://github.com/dekusaklas-droid/4SNet.git
cd code
```

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


## Training
Train a model by:

```bash
python train.py --dataset llcm --gpu 0
```
--dataset: which dataset "llcm", "sysu" or "regdb".

--gpu: which gpu to run.
## Test
Test a model on LLCM, SYSU-MM01 or RegDB dataset by
```bash
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 0 --dataset llcm
```
--dataset: which dataset "llcm", "sysu" or "regdb".

--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).

--tvsearch: whether thermal to visible search (only for RegDB dataset).

--resume: the saved model path.

--gpu: which gpu to run.

## Results
<img width="1050" height="1000" alt="image" src="https://github.com/user-attachments/assets/16d452a9-70a3-4440-a3ba-1476e94ac1cb" />

<img width="964" height="514" alt="image" src="https://github.com/user-attachments/assets/8e788a22-e3cf-4690-abb5-12bbfec7ae6d" />


## Citation

```bibtex
@misc{4snet,
  title={4SNet: Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-Identification},
  author={Xiong, Mingfu and Luo, Feiyang and Huang, Junjie and Guo, Yifei and Alotaibi, Aziz and Bakshi, Sambit and Del Ser, Javier and Muhammad, Khan},
  year={2026},
  note={Under review}
}
