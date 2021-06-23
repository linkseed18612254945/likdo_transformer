#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python3 -u ./train.py \
    --train_data HuffPost_train_new \
    --val_data HuffPost_val_new \
    --test_data HuffPost_test_new \
    -N 5 \
    -K 5 \
    -Q 10 \
    --encoder bert-base \
    --model att-induction \
    --optim adamw \
    --train_episodes 5000 \
    --val_steps 100 \
    --val_episodes 3000 \
    --test_episodes 5000 \
    --max_length 52 \
    --hidden_size 768 \
    --induction_iters 3 \
    --n_heads 4 \
    --dropout 0.4 \
    -H 160 \
    -B 4 \
    --grad_steps 1 \
    --lr 1e-5 \
    --warmup 0.06 \
    --weight_decay 0.01 \
    --pretrain_path ../resource/pretrain/ \
    --output_path ../log/ &