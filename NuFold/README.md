# NuFold: 3D RNA Structure Prediction Method
![nufold](nufold/NuFoldLogo.png)

NuFold is a state-of-the-art method designed for predicting 3D RNA structures, leveraging deep learning for high accuracy and reliability. This tool is particularly useful for biologists and bioinformatics researchers focusing on RNA function and structure.

License: GPL v3 (Please contact us for different licensing)

Contact: Daisuke Kihara (dkihara@purdue.edu)



Online Platform:
1. [Google Colab](https://colab.research.google.com/github/kiharalab/nufold/blob/master/ColabNuFold.ipynb)
    * This implements quick rMSA due to a hardware limitation.


## Environment Setup and Installation

### 0. System requirements
The code was tested on the following environment.
* OS: Ubuntu 22.04 LTS
* CPU: modern x86_64 CPUs
* GPU: NVIDIA GeForce RTX 3090 24GB
    * VRAM requirement depends on the length of input RNA

### 1. Clone the repository

```bash
git clone https://github.com/kiharalab/NuFold.git
cd NuFold/
```

### 2. Conda Environment
Start by setting up a dedicated Conda environment:

```bash
conda create -n nufold_P python=3.10
conda activate nufold_P
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install ml-collections dm-tree deepspeed protobuf scipy biopython numpy matplotlib
```
* NuFold was trained with PyTorch `1.10.0+cu111`, and tested with PyTorch `2.0.1+cu117`.


### 3. third party software
#### 3.1. rMSA
Clone rMSA and set up the database by following the instruction of rMSA:
```bash
git clone https://github.com/pylelab/rMSA
cd rMSA/
./database/script/update.sh
cd ../
```
* This may takes ~2 TB of disks, and a few hours to process. Please be patient.

#### 3.2. IPknot
IPknot is used for RNA secondary structure prediction. Download and set it up with the following commands:

```bash
wget https://github.com/satoken/ipknot/releases/download/v1.1.0/ipknot-1.1.0-x86_64-linux.zip
unzip ipknot-1.1.0-x86_64-linux.zip && rm ipknot-1.1.0-x86_64-linux.zip
chmod +x ipknot-1.1.0-x86_64-linux/ipknot
```

## Model Checkpoint
Download the NuFold model checkpoint to a designated directory:

```bash
mkdir -p checkpoints
wget -O checkpoints/global_step145245.pt http://kiharalab.org/nufold/global_step145245.pt
```

## Running NuFold with the End-to-End Script
To run NuFold, you need to prepare the directory structure as following:
```
input_dir/
    [TARGET_ID]/
        [TARGET_ID].fasta
        [TARGET_ID].a3m
        [TARGET_ID].ipknot.ss
```

You can take a look at `test_input` directory, which have following structure.
```
test_input/
    2DER_C/
        2DER_C.fasta
        2DER_C.a3m
        2DER_C.ipknot.ss
```
To generate those files, you can follow the steps below

### 1. Run rMSA
First, generate the MSA with following command:
```
rMSA/rMSA.pl test_input/2DER_C/2DER_C.fasta -cpu=32
cp test_input/2DER_C/2DER_C.afa test_input/2DER_C/2DER_C.a3m
```
This step may take a couple of hours to a half day, depends on the target sequence.

### 2. Run IPknot
```
ipknot-1.1.0-x86_64-linux/ipknot test_input/2DER_C/2DER_C.fasta > test_input/2DER_C/2DER_C.ipknot.ss
```
This step typically takes a few seconds.

### 3. Run NuFold
To run NuFold inference, you can use following command:
```
python3 run_nufold.py \
  --ckpt_path checkpoints/global_step145245.pt \
  --input_fasta test_input/2DER_C/2DER_C.fasta \
  --input_dir test_input/ \
  --output_dir test_output \
  --config_preset initial_training
```
This step will take up to ~5 minutes.
Your result will be found at `test_output/2DER_C/2DER_C_rank_1.pdb`.


## Cite
Yuki Kagaya, Zicong Zhang, Nabil Ibtehaz, Xiao Wang, Tsukasa Nakamura, Pranav Deep Punuru & Daisuke Kihara. "NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation." Nature Communications, 2025 [https://www.nature.com/articles/s41467-025-56261-7](https://www.nature.com/articles/s41467-025-56261-7)
```
@article{Kagaya2025NuFold,
  title={NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation},
  author={Yuki Kagaya, Zicong Zhang, Nabil Ibtehaz, Xiao Wang, Tsukasa Nakamura, Pranav Deep Punuru, and Daisuke Kihara},
  journal={Nature Communications},
  year={2025}
}
```
