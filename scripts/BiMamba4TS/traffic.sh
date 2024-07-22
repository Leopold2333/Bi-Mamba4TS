project_path=$(pwd)
work_space="$project_path/main.py"
# model params
model="BiMamba4TS"
seq_len=96
loss="mse"
patch_len=48
stride=24
random_seed=2024
e_layers=4

d_model=256
d_ff=512
d_state=32
batch_size=32

# dataset
dataset_name="traffic"
root_path=data/traffic
data_path="${dataset_name}.csv"
dataset_type=custom
enc_in=862

gpu=0
ch_ind=0
residual=1
bi_dir=1
embed_type=2

lr="1e-4"
dropout=0.1

is_training=$1

for pred_len in 96 192 336 720
do
    if [ $pred_len -eq 96 ]; then
        lr="4e-3"
        dropout=0.2
    elif [ $pred_len -eq 192 ]; then
        lr="2e-3"
        dropout=0.3
    elif [ $pred_len -eq 336 ]; then
        lr="2e-3"
        dropout=0.2
    elif [ $pred_len -eq 720 ]; then
        lr="1e-3"
        dropout=0.1
    fi
    log_file="${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})[${patch_len}]_dp${dropout}_el${e_layers}_em${embed_type}_r${residual}_b${bi_dir}.log"
    python $work_space $model --is_training=$is_training --gpu=$gpu \
    --embed_type=$embed_type --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
    --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in \
    --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir --SRA \
    --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
    --e_layers=$e_layers \
    --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
    --learning_rate=$lr --dropout=$dropout \
    > $log_file 2>&1
done
