#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path caiso_20130101_20210630.csv \
    --model_id caiso_1440_720 \
    --model $model_name \
    --data caiso \
    --features M \
    --seq_len 1440 \
    --pred_len 720 \
    --e_layers 1 \
    --enc_in 10 \
    --d_model 256 \
    --des 'Exp' \
    --itr 1 \
    --stride 48 \
    --patch_len 96 \
    --num_heads 32 \
    --train_epochs 100 \
    --batch_size 128 \
    --patience 20 \
    --skip_dropout 0.3 \
    --dropout 0.0 \
    --s_steps 2 \
    --skip_type "time_uniform" \
    --method "multistep" \
    --order 2 \
    --lower_order_final "true" \
    --coss 5.0 \
    --is_diff 1\
    --loss_type 'MAE'\
    --rmom 5 \
    --n_b 3 \
    --vs_times 6 \
    --sample_times 50 \
    --use_mom 1 \
    --new_norm 1
