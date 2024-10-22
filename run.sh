#!/bin/bash
# ps -ef | grep python
# nohup bash run.sh >results/run.log
python train.py --gpu 0 --data_path data/WN18RR/ --task_mode transductive --max_epoch 20 --hidden_dim 64 --n_layer 4 --lr 5e-3
python train.py --gpu 0 --data_path data/fb15k-237/ --task_mode transductive --max_epoch 20 --hidden_dim 64 --n_layer 2 --lr 5e-3
python train.py --gpu 0 --data_path data/WN18RR_v1/ --task_mode inductive --max_epoch 20 --hidden_dim 32 --n_layer 4 --lr 5e-4
python train.py --gpu 0 --data_path data/fb237_v1/ --task_mode inductive --max_epoch 20 --hidden_dim 32 --n_layer 3 --lr 5e-4
python train.py --gpu 0 --data_path data/nell_v1/ --task_mode inductive --max_epoch 20 --hidden_dim 32 --n_layer 4 --lr 5e-4
