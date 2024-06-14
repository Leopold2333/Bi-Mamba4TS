work_space="/mnt/data/lab/Bi-Mamba4TS/main.py"
# model params
model="BiMamba4TS"
seq_len=96
loss="mse"
patch_len=12
stride=6

e_layers=3

d_model=64
d_ff=128
d_state=16
batch_size=32

# dataset
dataset_name="solar"

if [ "$dataset_name" = "weather" ]; then
    root_path=data/weather
    data_path="${dataset_name}.csv"
    dataset_type=custom
    enc_in=21
elif [ "$dataset_name" = "solar" ]; then
    root_path=data/Solar
    data_path="solar.txt"
    dataset_type=custom
    enc_in=137
fi
gpu=1
ch_ind=0
residual=1
bi_dir=0

for patch_len in 24
do
stride=$(echo "$patch_len / 2" | bc)
for pred_len in 96 720
do
    for random_seed in 2023
    do
        log_file="${dataset_name}(${random_seed})_${model}(${seq_len}-${pred_len})(${patch_len}-${stride})_emb0_el${e_layers}_ch${ch_ind}_res${residual}_bi${bi_dir}.log"
        python $work_space $model \
        --gpu=$gpu \
        --embed_type=0 --num_workers=8 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=$pred_len \
        --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
        --enc_in=$enc_in \
        --e_layers=$e_layers \
        --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
        --learning_rate=1e-04 --dropout=0.1 \
        > $log_file 2>&1
        # gpu=$(($gpu+1))
        # if [ $gpu -eq 2 ]; then
        #     gpu=0
        # fi
    done
done
done