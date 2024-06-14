work_space="/mnt/data/lab/Bi-Mamba4TS/main.py"
# model params
gpu=0
model="DMamba"
seq_len=96
loss="mse"

e_layers=2

d_model=512
d_ff=512
d_state=2
batch_size=32

# dataset
dataset_name="ETTm2"

if [ "$dataset_name" = "weather" ]; then
    root_path=data/weather
    data_path="${dataset_name}.csv"
    dataset_type=custom
    enc_in=21
elif [ "$dataset_name" = "electricity" ]; then
    root_path=data/electricity
    data_path="${dataset_name}.csv"
    dataset_type=custom
    enc_in=321
elif [ "$dataset_name" = "traffic" ]; then
    root_path=data/traffic
    data_path="${dataset_name}.csv"
    dataset_type=custom
    enc_in=862
elif [ "$dataset_name" = "ETTh1" ] || [ "$dataset_name" = "ETTh2" ]; then
    root_path=data/ETT-small
    data_path="${dataset_name}.csv"
    dataset_type=ETTh1
    enc_in=7
elif [ "$dataset_name" = "ETTm1" ] || [ "$dataset_name" = "ETTm2" ]; then
    root_path=data/ETT-small
    data_path="${dataset_name}.csv"
    dataset_type=ETTm1
    enc_in=7
elif [ "$dataset_name" = "ILI" ]; then
    root_path=data/illness
    data_path="national_illness.csv"
    dataset_type=custom
    enc_in=7
fi

for seq_len in 48 192 336 720
do
    for random_seed in 2000
    do
        log_file="${dataset_name}(${random_seed})_${model}(${seq_len}-${pred_len})_ch${ch_ind}_res${residual}_bi${bi_dir}.log"
        nohup python $work_space $model \
        --gpu=$gpu \
        --embed_type=0 --num_workers=4 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=96 \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type --SRA \
        --enc_in=$enc_in \
        --e_layers=$e_layers \
        --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
        --learning_rate=1e-04 --dropout=0.1 \
        > $log_file 2>&1 &
        gpu=$(($gpu+1))
        if [ $gpu -eq 2 ]; then
            gpu=0
        fi
    done
done