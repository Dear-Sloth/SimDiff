export CUDA_VISIBLE_DEVICES=2

model_name=SimDiff

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path production.csv \
    --model_id norpool_1440_720 \
    --model $model_name \
    --data norpool \
    --features M \
    --seq_len 1440 \
    --pred_len 720 \
    --e_layers 1 \
    --enc_in 18 \
    --d_model 64 \
    --des 'Exp' \
    --itr 1 \
    --stride 32 \
    --patch_len 32 \
    --num_heads 2 \
    --train_epochs 100 \
    --batch_size 128 \
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
    --rmom 1 \
    --n_b 1 \
    --vs_times 5 \
    --sample_times 50 \
    --use_mom 1\
    --new_norm 1 \
    --learning_rate 0.00005
