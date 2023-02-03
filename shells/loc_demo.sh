cd ~/ir-mcl
python loc_demo.py \
  --loc_results ./results/ipblab/loc_test/test1/loc_results.npz \
  --occ_map ./data/ipblab/occmap.npy \
  --output_gif ./results/ipblab/loc_test/test1/loc_demo.gif \
  --map_size -15 17.5 -12.5 5
