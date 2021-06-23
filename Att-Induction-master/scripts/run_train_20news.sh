#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python3 -u ./train.py \
    --train_data 20news_train_new \
    --val_data 20news_val_new \
    --test_data 20news_test_new \
    -N 5 \
    -K 5 \
    -Q 1 \
    --encoder bert-base \
    --model att-induction \
    --optim adamw \
    --train_episodes 15000 \
    --val_steps 200 \
    --val_episodes 2000 \
    --test_episodes 5000 \
    --max_length 500 \
    --hidden_size 768 \
    --induction_iters 3 \
    --n_heads 4 \
    --dropout 0.4 \
    -H 130 \
    -B 1 \
    --grad_steps 32 \
    --lr 6e-5 \
    --warmup 0.06 \
    --weight_decay 0.01 \
    --pretrain_path ../resource/pretrain/ \
    --output_path ../log/ &