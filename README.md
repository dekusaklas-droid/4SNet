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

```bash
git clone https://github.com/dekusaklas-droid/4SNet.git
cd 4SNet
```

## Environment

- Python 3.10
- PyTorch
- Ubuntu 22.04
- NVIDIA RTX 4090

## Datasets

Please create a `data/` folder under the root directory and place the downloaded datasets in this folder.

### RegDB Dataset

The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

On the website, it is named:

```text
Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)
```

After downloading the RegDB dataset, please place it under:

```text
data/RegDB/
```

### SYSU-MM01 Dataset

The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

After downloading the SYSU-MM01 dataset, please place it under:

```text
data/SYSU-MM01/
```

Then run the following command to prepare the dataset:

```bash
python pre_process_sysu.py
```

The processed training data will be stored in `.npy` format.

### LLCM Dataset

The LLCM dataset can be downloaded by sending a signed [dataset release agreement](https://github.com/ZYK100/LLCM/blob/main/Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) copy to:

```text
zhangyk@stu.xmu.edu.cn
```

After downloading the LLCM dataset, please place it under:

```text
data/LLCM/
```

## Training

Train a model by:

```bash
python train.py --dataset llcm --gpu 0
```

Arguments:

- `--dataset`: which dataset to use, including `llcm`, `sysu`, or `regdb`.
- `--gpu`: which GPU to use.

## Test

Test a model on LLCM, SYSU-MM01, or RegDB dataset by:

```bash
python test.py --mode all --tvsearch True --resume 'model_path' --gpu 0 --dataset llcm
```

Arguments:

- `--dataset`: which dataset to use, including `llcm`, `sysu`, or `regdb`.
- `--mode`: `all` or `indoor`, where `indoor` is only used for the SYSU-MM01 dataset.
- `--tvsearch`: whether to perform thermal-to-visible search, only used for the RegDB dataset.
- `--resume`: the saved model path.
- `--gpu`: which GPU to use.

## Results

<img width="865" height="841" alt="image" src="https://github.com/user-attachments/assets/dd6a496f-ac88-46d1-99e3-23b1c7aed291" />

<img width="808" height="434" alt="image" src="https://github.com/user-attachments/assets/d301cf44-6dd7-4e7d-aba4-3c399dba2e42" />

## Citation

```bibtex
@misc{4snet,
  title={4SNet: Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-Identification},
  author={Xiong, Mingfu and Luo, Feiyang and Huang, Junjie and Guo, Yifei and Alotaibi, Aziz and Bakshi, Sambit and Del Ser, Javier and Muhammad, Khan},
  year={2026},
  note={Under review}
}
```
