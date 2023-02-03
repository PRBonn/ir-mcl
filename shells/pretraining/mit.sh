cd ~/ir-mcl
python train.py \
  --root_dir ./data/mit --N_samples 1024 --perturb 1 --noise_std 0 --L_pos 10 \
  --feature_size 256 --use_skip --seed 42 --batch_size 512 --chunk 262144 \
  --num_epochs 32 --loss_type smoothl1 --optimizer adam --weight_decay 1e-3 \
  --lr 1e-4 --decay_step 4 8 --decay_gamma 0.5 --lambda_opacity 1e-5 \
  --exp_name nof_mit

python eval.py \
  --root_dir ./data/mit \
  --ckpt_path ./logs/nof_mit/version_2/checkpoints/best.ckpt \
  --N_samples 1024 --chunk 92160 --perturb 0 --noise_std 0 --L_pos 10 \
  --feature_size 256 --use_skip
