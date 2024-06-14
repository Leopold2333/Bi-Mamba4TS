work_space="/mnt/data/lab/Bi-Mamba4TS/main.py"
# model params
gpu=1
model="WITRAN"
seq_len=96
loss="mse"

batch_size=32

e_layers=3
d_model=64
WITRAN_deal="standard"
WITRAN_grid_cols=24

# dataset
dataset_name="solar"

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
elif [ "$dataset_name" = "solar" ]; then
    root_path=data/Solar
    data_path="solar.txt"
    dataset_type=custom
    enc_in=137
fi

for pred_len in 96 192 720 336
do
    for random_seed in 2023
    do
        log_file="${dataset_name}(${random_seed})_${seq_len}_${pred_len}_${model}_${loss}-bs${batch_size}.log"
        python $work_space $model \
        --gpu=$gpu \
        --embed_type=1 --num_workers=4 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=$pred_len \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
        --enc_in=$enc_in --c_out=$enc_in \
        --e_layers=$e_layers --d_model=$d_model \
        --WITRAN_deal=$WITRAN_deal --WITRAN_grid_cols=$WITRAN_grid_cols \
        --learning_rate=1e-04 \
        > $log_file 2>&1
        # gpu=$(($gpu+1))
        # if [ $gpu -eq 2 ]; then
        #     gpu=0
        # fi
    done
done