# Patch Reco

## How to set up
1. create conda env
```bash
conda create -n patch_reco python==3.9
conda activate patch_reco
```
2. install pytorch
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
3. install transformers and related package
```bash
conda install -c huggingface -c conda-forge datasets 
pip install evaluate transformers scikit-learn tensorboardX
```

## training
```bash
bash train.sh
```
