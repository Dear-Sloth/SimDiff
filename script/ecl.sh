#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,2

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_720_168 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 168 \
    --e_layers 1 \
    --enc_in 321 \
    --d_model 256 \
    --des 'Exp' \
    --itr 1 \
    --stride 24 \
    --patch_len 48 \
    --num_heads 64 \
    --train_epochs 100 \
    --batch_size 32 \
    --patience 10 \
    --skip_dropout 0.2 \
    --dropout 0.0 \
    --s_steps 2 \
    --skip_type "time_uniform" \
    --method "multistep" \
    --order 2 \
    --lower_order_final "true" \
    --coss 5.0 \
    --is_diff 1\
    --loss_type 'MAE'\
    --rmom 1 \
    --n_b 1 \
    --vs_times 5 \
    --sample_times 50\
    --use_mom 1 \
    --new_norm 1 

