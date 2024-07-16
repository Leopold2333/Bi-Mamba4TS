project_path=$(pwd)
work_space="$project_path/main.py"
# model params
model="BiMamba4TS"
seq_len=96
loss="mse"
random_seed=2024
e_layers=3
patch_len=4
stride=2

d_model=64
d_ff=128
d_state=2
batch_size=32
gpu=0

ch_ind=1
residual=1
bi_dir=1

# dataset
for patch_len in 4 12 24
do
    stride=$(echo "$patch_len / 2" | bc)
    for dataset_name in "ETTh2" "ETTm2"
    # for dataset_name in "ETTh1" "ETTm1"
    do
        if [ "$dataset_name" = "ETTh1" ] || [ "$dataset_name" = "ETTh2" ]; then
            root_path=data/ETT-small
            data_path="${dataset_name}.csv"
            dataset_type=ETTh1
            enc_in=7
        elif [ "$dataset_name" = "ETTm1" ] || [ "$dataset_name" = "ETTm2" ]; then
            root_path=data/ETT-small
            data_path="${dataset_name}.csv"
            dataset_type=ETTm1
            enc_in=7
        fi
        for e_layers in 1 2
        do
            for pred_len in 96 192 336 720
            do
                log_file="${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})[${patch_len}]_el${e_layers}_c${ch_ind}_r${residual}_b${bi_dir}.log"
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
done