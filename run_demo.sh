python train_swin.py --source_path data/DyNeRF/cut_roasted_beef --scale_reg 1e-2 --opacity_reg 2e-2 --noise_lr 5e4 \
--cap_max 200000 --swin_size 5 --max_frame 300 --init_type sfm \
--sh_degree 0 --eval -m cb_300_10k --iterations 10010 --genesis_iterations 30010 --resolution 2

python train_swin.py --source_path data/DyNeRF/cook_spinach --scale_reg 1e-2 --opacity_reg 2e-2 --noise_lr 5e4 \
--cap_max 200000 --swin_size 5 --max_frame 300 --init_type sfm \
--sh_degree 0 --eval -m cs_300_10k --iterations 10010 --genesis_iterations 30010 --resolution 2

python train_swin.py --source_path data/DyNeRF/sear_steak --scale_reg 1e-2 --opacity_reg 2e-2 --noise_lr 5e4 \
--cap_max 200000 --swin_size 5 --max_frame 300 --init_type sfm \
--sh_degree 0 --eval -m ss_300_10k --iterations 10010 --genesis_iterations 30010 --resolution 2