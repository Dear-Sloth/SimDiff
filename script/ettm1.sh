export CUDA_VISIBLE_DEVICES=0

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_1440_192 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 1440 \
    --pred_len 192 \
    --e_layers 1 \
    --enc_in 7 \
    --d_model 32 \
    --des 'Exp' \
    --itr 1 \
    --stride 8 \
    --patch_len 16 \
    --num_heads 2 \
    --train_epochs 100 \
    --batch_size 128 \
    --patience 20 \
    --skip_dropout 0.5 \
    --dropout 0.0 \
    --s_steps 3 \
    --skip_type "time_quadratic" \
    --method "multistep" \
    --order 2 \
    --lower_order_final "true" \
    --coss 5.0 \
    --is_diff 1\
    --loss_type 'MAE'\
    --rmom 20 \
    --n_b 5 \
    --vs_times 10 \
    --sample_times 50\
    --use_mom 1 \
    --new_norm 1 