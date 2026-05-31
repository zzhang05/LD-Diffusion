# Official codes of PAR module for TPAMI submission Advancing Diffusion Models with Limited Data via Pixel-Aware Refinement (LD-Diffusion+)

We provide the Training Codes and Evaluation codes for the PAR. The FID cacukation codes can be found in the evaluation parts of LD-Diffusion.

# Dataset

The low-shot datasets can be found in [[link]](https://drive.google.com/file/d/1rWqaVlms55604jrP5t9ShacL6mZKWL8f/view?usp=sharing).

# Requirement: 
Use the following commands with Miniconda3 to create and activate your par Python environment.

```
conda env create -f environment.yml
```


```
conda activate par
```

```
pip install "numpy<2"
```

# Training
To train your own PAR model on the 100-shot Obama dataset, please run the following command on eight GPU:

```
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=100-shot-obama.zip --eff-attn=True --cond=0 --batch=64 --lr=1e-4 --augment=0.12 --dropout=0.1 --fp16=True --xflip=True --ls=100 --train_on_latents=0 --arch=ncsnpp --precond=blur --block-scale=0.15 --prob-length=0.93 --blur-sigma-max=3.0
```

# Evaluation

To generate the image by LD-Diffusion+ on the 100-shot Obama dataset, please run the following command on eight GPU:

```
torchrun --standalone --nproc_per_node=8 generate.py --sampler_stages=second --seeds=0-4999 --indir=out --outdir=fid-tmp --network_second=PAR_obama.pt --num_steps_second=40
```

where ''out'' file is the 5000 generated samples by LD-Diffusion and should be copy to this folder by yourself manully. The generated images by LD-Diffusion+ is in folder ''fid-tmp''. You should then use evaluation modules of LD-Diffusion to caculate the matrics of this ''fid-tmp''.

# Important notes

1. The codes of this module is built upon the codes of the Relay Diffusion [[link]](https://github.com/zai-org/RelayDiffusion) and Patch Diffusion [[link]](https://github.com/Zhendong-Wang/Patch-Diffusion). We thank them a lot for their great work.

2. Feel free to contact me at zzhang55@qub.ac.uk if you have any questions.

3. Our paper is now under review by TPAMI, if you use our proposed PAR module, please cite our original LD-Diffusion paper. Thanks!


