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
d_state=8
batch_size=32
gpu=0

ch_ind=1
residual=1
bi_dir=1

is_training=$1

# dataset
for dataset_name in "ETTh2" "ETTm2" "ETTh1" "ETTm1"
do
    for pred_len in 96 192 336 720
    do
        if [ "$dataset_name" = "ETTh1" ] || [ "$dataset_name" = "ETTh2" ]; then
            root_path=data/ETT-small
            data_path="${dataset_name}.csv"
            dataset_type=ETTh1
            enc_in=7
            if [ "$dataset_name" = "ETTh1" ]; then
                e_layers=1
                patch_len=4
                if [ "$pred_len" = 96 ]; then
                    lr="4e-5"
                    dropout=0.2
                elif [ "$pred_len" = 192 ]; then
                    lr="1e-4"
                    droppout=0.2
                elif [ "$pred_len" = 336 ]; then
                    lr="2e-4"
                    dropout=0.2
                else
                    lr="1e-4"
                    dropout=0.1
                fi
            elif [ "$dataset_name" = "ETTh2" ]; then
                e_layers=1
                patch_len=24
                if [ "$pred_len" = 96 ]; then
                    lr="2e-4"
                    dropout=0.1
                elif [ "$pred_len" = 192 ]; then
                    lr="2e-4"
                    dropout=0.1
                elif [ "$pred_len" = 336 ]; then
                    lr="1e-4"
                    dropout=0.1
                else
                    lr="4e-4"
                    dropout=0.1
                fi
            fi
        elif [ "$dataset_name" = "ETTm1" ] || [ "$dataset_name" = "ETTm2" ]; then
            root_path=data/ETT-small
            data_path="${dataset_name}.csv"
            dataset_type=ETTm1
            enc_in=7
            if [ "$dataset_name" = "ETTm1" ]; then
                e_layers=1
                patch_len=4
                if [ "$pred_len" = 96 ]; then
                    lr="4e-4"
                    dropout=0.2
                elif [ "$pred_len" = 192 ]; then
                    lr="1e-4"
                    droppout=0.2
                elif [ "$pred_len" = 336 ]; then
                    lr="4e-4"
                    dropout=0.0
                else
                    lr="4e-4"
                    dropout=0.2
                fi
            elif [ "$dataset_name" = "ETTm2" ]; then
                e_layers=1
                patch_len=4
                if [ "$pred_len" = 96 ]; then
                    lr="4e-5"
                    dropout=0.0
                elif [ "$pred_len" = 192 ]; then
                    lr="4e-5"
                    dropout=0.0
                elif [ "$pred_len" = 336 ]; then
                    lr="4e-5"
                    dropout=0.0
                else
                    lr="4e-5"
                    dropout=0.0
                fi
            fi
        fi
        stride=$(echo "$patch_len / 2" | bc)
        log_file="${random_seed}(${dataset_name})_${model}(${seq_len}-${pred_len})[${patch_len}]_el${e_layers}_c${ch_ind}_r${residual}_b${bi_dir}.log"
        python $work_space $model --gpu=$gpu --is_training=$is_training \
        --embed_type=0 --num_workers=8 --seed=$random_seed --batch_size=$batch_size \
        --seq_len=$seq_len --pred_len=$pred_len \
        --patch_len=$patch_len --stride=$stride --ch_ind=$ch_ind --residual=$residual --bi_dir=$bi_dir --SRA \
        --loss=$loss \
        --dataset_name=$dataset_name --data_path=$data_path --root_path=$root_path --dataset_type=$dataset_type \
        --enc_in=$enc_in \
        --e_layers=$e_layers \
        --d_model=$d_model --d_ff=$d_ff --d_state=$d_state \
        --learning_rate=$lr --dropout=$dropout \
        > $log_file 2>&1
    done
done