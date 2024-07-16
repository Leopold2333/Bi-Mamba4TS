project_path=$(pwd)
work_space="$project_path/main.py"
# model params
model="BiMamba4TS"
seq_len=96
loss="mse"
patch_len=4
stride=2

e_layers=3

d_model=64
d_ff=128
d_state=16
batch_size=32

# dataset
dataset_name="weather"

if [ "$dataset_name" = "weather" ]; then
    root_path=data/weather
    data_path="${dataset_name}.csv"
    dataset_type=custom
    enc_in=21
fi

gpu=0
ch_ind=0
residual=1
bi_dir=1
embed_type=2
dropout=0.1

for dropout in 0.1 0.2
do
for pred_len in 96 192 336 720
do
    for random_seed in 2024
    do
        log_file="${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})[${patch_len}]_dp${dropout}_el${e_layers}_em${embed_type}_c${ch_ind}_r${residual}_b${bi_dir}.log"
        python $work_space $model --is_training=1 --gpu=$gpu \
        --embed_type=$embed_type --num_workers=4 --seed=$random_seed --batch_size=$batch_size --loss=$loss \
        --seq_len=$seq_len --pred_len=$pred_len --enc_in=$enc_in \
        --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
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
done