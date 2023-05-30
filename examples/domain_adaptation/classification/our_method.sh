#!/usr/bin/env bash
# Office31
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_amazon2dslr_cfg_SC.yaml --log logs/our_method/Office31_A2D
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_dslr2amazon_cfg_SC.yaml --log logs/our_method/Office31_D2A
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_webcam2amazon_cfg_SC.yaml --log logs/our_method/Office31_W2A
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_amazon2webcam_cfg_SC.yaml --log logs/our_method/Office31_A2W
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_dslr2webcam_cfg_SC.yaml --log logs/our_method/Office31_D2W
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 1024 --seed 1 --module semantic --cfg ../../../experiments/configs/Office31/office31_train_webcam2dslr_cfg_SC.yaml --log logs/our_method/Office31_W2D

## Office-Home
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_A2C__cfg_SC.yaml --log logs/our_method/OfficeHome_Ar2Cl_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_A2P__cfg_SC.yaml --log logs/our_method/OfficeHome_Ar2Pr_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_A2R__cfg_SC.yaml --log logs/our_method/OfficeHome_Ar2Rw_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_C2A__cfg_SC.yaml --log logs/our_method/OfficeHome_Cl2Ar_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_C2P__cfg_SC.yaml --log logs/our_method/OfficeHome_Cl2Pr_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_C2R__cfg_SC.yaml --log logs/our_method/OfficeHome_Cl2Rw_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_P2A__cfg_SC.yaml --log logs/our_method/OfficeHome_Pr2Ar_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_P2C__cfg_SC.yaml --log logs/our_method/OfficeHome_Pr2Cl_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_P2R__cfg_SC.yaml --log logs/our_method/OfficeHome_Pr2Rw_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_R2A__cfg_SC.yaml --log logs/our_method/OfficeHome_Rw2Ar_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_R2C__cfg_SC.yaml --log logs/our_method/OfficeHome_Rw2Cl_semantic
# python our_method.py -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0  --module semantic --cfg ../../../experiments/configs/OfficeHome/home_train_R2P__cfg_SC.yaml --log logs/our_method/OfficeHome_Rw2Pr_semantic

 # # VisDA-2017
python our_method.py -a resnet101 --epochs 30 --bottleneck-dim 1024 --seed 0 --module semantic --cfg ../../../experiments/configs/VisDA/visda17_train_train2val_cfg_res101.yaml --per-class-eval --log logs/our_method/VisDA2017_semantic
