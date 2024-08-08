export CUDA_VISIBLE_DEVICES=3

model_name=SimDiff


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_168 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --pred_len 168 \
    --e_layers 1 \
    --enc_in 7 \
    --d_model 128 \
    --des 'Exp' \
    --itr 1 \
    --stride 8 \
    --patch_len 16 \
    --num_heads 8 \
    --train_epochs 100 \
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
    --rmom 5 \
    --n_b 5 \
    --sample_times 100 \
    --vs_times 20 \
    --use_mom 1 \
    --new_norm 1 