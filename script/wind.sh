export CUDA_VISIBLE_DEVICES=1

model_name=SimDiff
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path wind.csv \
    --model_id wind_1440_192 \
    --model $model_name \
    --data wind \
    --features M \
    --seq_len 1440 \
    --pred_len 192 \
    --e_layers 1 \
    --enc_in 7 \
    --d_model 64\
    --des 'Exp' \
    --itr 1 \
    --stride 16 \
    --patch_len 16 \
    --num_heads 4 \
    --train_epochs 150 \
    --batch_size 128 \
    --patience 10 \
    --skip_dropout 0.4 \
    --dropout 0.0 \
    --s_steps 2 \
    --skip_type "time_quadratic" \
    --method "multistep" \
    --order 2 \
    --lower_order_final "true" \
    --coss 5.0 \
    --is_diff 1\
    --loss_type 'MAE'\
    --n_b 1 \
    --rmom 15 \
    --vs_times 5 \
    --sample_times 50\
    --use_mom 1 \
    --new_norm 1 \
    --learning_rate 0.000004
