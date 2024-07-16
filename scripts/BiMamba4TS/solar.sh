work_space="/home/Bi-Mamba4TS/main.py"
# model params
model="BiMamba4TS"
seq_len=96
loss="mse"
patch_len=24
stride=12
random_seed=2024

e_layers=2

d_model=128
d_ff=256
d_state=32
batch_size=32

# dataset
dataset_name="solar"
root_path=data/Solar
data_path="solar_AL.txt"
dataset_type=custom
enc_in=137

gpu=0
ch_ind=0
residual=1
bi_dir=1

for dropout in 0.1 0.2 0.0
do
    for pred_len in 96 192 336 720
    do
            log_file="${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})[${patch_len}]_dp${dropout}_el${e_layers}_e0_c${ch_ind}_r${residual}_b${bi_dir}.log"
            python $work_space $model --gpu=$gpu \
            --embed_type=0 --num_workers=4 --seed=$random_seed --batch_size=$batch_size \
            --seq_len=$seq_len --pred_len=$pred_len \
            --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir \
            --loss=$loss \
            --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
            --enc_in=$enc_in \
            --e_layers=$e_layers \
            --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
            --learning_rate=1e-04 --dropout=$dropout \
            > $log_file 2>&1
            # gpu=$(($gpu+1))
            # if [ $gpu -eq 2 ]; then
            #     gpu=0
            # fi
    done    
done