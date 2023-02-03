cd ~/ir-mcl
python train.py \
  --root_dir ./data/fr079 --N_samples 256 --perturb 1 --noise_std 0 --L_pos 10 \
  --feature_size 256 --use_skip --seed 42 --batch_size 1024 --chunk 262144 \
  --num_epochs 32 --loss_type smoothl1 --optimizer adam --weight_decay 1e-3 \
  --lr 1e-4 --decay_step 4 8 --decay_gamma 0.5 --lambda_opacity 1e-5 \
  --exp_name nof_fr079

python eval.py \
  --root_dir ./data/fr079 \
  --ckpt_path ./logs/nof_fr079/version_0/checkpoints/best.ckpt \
  --N_samples 256 --chunk 92160 --perturb 0 --noise_std 0 --L_pos 10 \
  --feature_size 256 --use_skip
