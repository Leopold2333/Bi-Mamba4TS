work_space="/mnt/data/lab/Bi-Mamba4TS/main.py"
# model params
gpu=1
model="BiMamba4TS"
seq_len=96
loss="mse"
patch_len=24
stride=12

e_layers=3

d_model=128
d_ff=256
d_state=32
batch_size=32

# dataset
dataset_name="traffic"

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

ch_ind=0
residual=1
bi_dir=1
embed_type=0

for seq_len in 96
# for patch_len in 48
do
patch_len=$(echo "$seq_len / 2" | bc)
stride=$(echo "$patch_len / 2" | bc)
for pred_len in 96
do
    for random_seed in 2023
    do
        log_file="${dataset_name}(${random_seed})_${model}(${seq_len}-${pred_len})(${patch_len}-${stride})_el${e_layers}_em${emb_type}_ch${ch_ind}_res${residual}_bi${bi_dir}.log"
        python $work_space $model --is_training=0 \
        --gpu=$gpu \
        --embed_type=$embed_type --num_workers=4 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=$pred_len \
        --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
        --enc_in=$enc_in \
        --e_layers=$e_layers \
        --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
        --learning_rate=1e-04 --dropout=0.1
        # > $log_file 2>&1
        # gpu=$(($gpu+1))
        # if [ $gpu -eq 2 ]; then
        #     gpu=0
        # fi
    done
done
done
