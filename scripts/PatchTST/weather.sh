work_space="/home/Bi-Mamba4TS/main.py"
# model params
gpu=0
model="PatchTST"
seq_len=96
loss="mse"

e_layers=3

d_model=128
d_ff=256
n_heads=16
batch_size=32

# dataset
dataset_name="weather"

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

for pred_len in 96
do
    for random_seed in 2023
    do
        log_file="${dataset_name}(${random_seed})_${seq_len}_${pred_len}_${model}_${loss}-bs${batch_size}_nh${n_heads}.log"
        python $work_space $model --is_training=1 \
        --gpu=$gpu \
        --embed_type=3 --num_workers=8 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=$pred_len --patch_len=16 --stride=8 \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
        --enc_in=$enc_in --dec_in=$enc_in --c_out=$enc_in \
        --e_layers=$e_layers \
        --d_model=$d_model --d_ff=$d_ff --n_heads=$n_heads \
        --pos_embed_type=sincos \
        --learning_rate=1e-04 --dropout=0.2 \
        > $log_file 2>&1
        # gpu=$(($gpu+1))
        # if [ $gpu -eq 2 ]; then
        #     gpu=0
        # fi
    done
done