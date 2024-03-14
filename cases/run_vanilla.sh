
data_dir=${1}
back_iterations=${2}
hidden_dim=${3}

python src/smd/main.py \
    --in_memory \
    --data_dir ${data_dir} \
    --save_dir save/ddpm_iter${back_iterations}_dim${hidden_dim} \
    --data_type image \
    --img_size 32 \
    --batch_size 128 \
    --training_steps 500000 \
    --save_every 50000 \
    --test_bsz 1000 \
    --demo_samples 64 \
    --test_samples 5000 \
    --denoising_iters ${back_iterations} \
    --hidden_dim ${hidden_dim} \
    --multi_nums 1,2,2,2 \
    --resnet_groups 4 \
    --time_dim 32 \
    --attn_heads 4 \
    --head_dim 32
