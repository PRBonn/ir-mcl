python train.py \
  --root_dir ./data/intel --N_samples 1024 --perturb 1 \
  --noise_std 0 --L_pos 10 --feature_size 256 --use_skip --seed 42 \
  --batch_size 512 --chunk 262144 --num_epochs 128 --loss_type smoothl1 \
  --optimizer adam --weight_decay 1e-3 --lr 1e-4 --decay_step 64 --decay_gamma 0.1 \
  --lambda_opacity 1e-5 --exp_name nof_intel


python eval.py \
    --root_dir ./data/intel \
    --ckpt_path ./logs/nof_intel/version_0/checkpoints/best.ckpt \
    --N_samples 1024 --chunk 184320 --perturb 0 --noise_std 0 --L_pos 10 \
    --feature_size 256 --use_skip
