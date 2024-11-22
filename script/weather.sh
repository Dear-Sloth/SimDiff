export CUDA_VISIBLE_DEVICES=0

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_1440_672 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 1440 \
    --pred_len 672 \
    --e_layers 1 \
    --enc_in 21 \
    --d_model 32 \
    --des 'Exp' \
    --itr 1 \
    --stride 16 \
    --patch_len 16 \
    --num_heads 4 \
    --train_epochs 100 \
    --batch_size 128 \
    --patience 10 \
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
    --rmom 15 \
    --n_b 5 \
    --sample_times 50 \
    --vs_times 10 \
    --use_mom 1 \
    --new_norm 1 