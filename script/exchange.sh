export CUDA_VISIBLE_DEVICES=0

model_name=SimDiff

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_96_14 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 14 \
    --e_layers 1 \
    --enc_in 8 \
    --d_model 128 \
    --des 'Exp' \
    --itr 1 \
    --stride 1 \
    --patch_len 2 \
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
    --rmom 13 \
    --n_b 3 \
    --vs_times 6 \
    --sample_times 50\
    --use_mom 1 \
    --new_norm 1 