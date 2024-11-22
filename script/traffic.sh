#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_1440_168 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 1440 \
    --pred_len 168 \
    --e_layers 1 \
    --enc_in 862 \
    --d_model 256 \
    --des 'Exp' \
    --itr 1 \
    --stride 48 \
    --patch_len 96 \
    --num_heads 8 \
    --train_epochs 100 \
    --batch_size 16 \
    --patience 20 \
    --skip_dropout 0.4 \
    --dropout 0.0 \
    --s_steps 2 \
    --skip_type "time_uniform" \
    --method "multistep" \
    --order 2 \
    --lower_order_final "true" \
    --coss 5.0 \
    --is_diff 1\
    --loss_type 'MAE'\
    --rmom 15 \
    --n_b 3 \
    --vs_times 6 \
    --sample_times 50\
    --use_mom 1 \
    --new_norm 1 