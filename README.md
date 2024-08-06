# Target Structure Learning Framework for Unsupervised Multi-Class Domain Adaptation
Update: 

This is the Pytorch implementation of [Target Structure Learning Framework for Unsupervised Multi-Class Domain Adaptation] publish in TOMM.

## Dataset
The structure of the dataset should be like

```
Office-31
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```

## Training


For example, for the VisDA-2017 dataset, you need to change the current directory to './examples/domain_adaptation/classification'
```
python our_method.py -a resnet101 --epochs 30 --bottleneck-dim 1024 --seed 0 --module semantic --cfg ../../../experiments/configs/VisDA/visda17_train_train2val_cfg_res101.yaml --per-class-eval --log logs/our_method/VisDA2017_semantic
```
The experiment log file and the saved checkpoints will be stored at ./examples/domain_adaptation/classification/logs/ckpt/${experiment_name}

Training model for all tranfer tasks on Office31, Office-Home, and VisDA-2017 datasets
```
./examples/domain_adaptation/classification/our_method.sh
```

## Test

For example, for the VisDA-2017 dataset,: 
```
python our_method.py -a resnet101 --epochs 30 --bottleneck-dim 1024 --seed 0 --module semantic --cfg ../../../experiments/configs/VisDA/visda17_train_train2val_cfg_res101.yaml --per-class-eval --log logs/our_method/VisDA2017_semantic --phase test
```

## License

[MIT](LICENSE)
