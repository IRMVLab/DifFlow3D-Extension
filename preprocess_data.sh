#! /bin/bash

# --- original single sequence processing commands ---

# DATASET="kitti360"
# SEQ_ID="3353"

# python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_id $SEQ_ID

# python -m data.preprocess.kitti360_to_nerf --sequence_id $SEQ_ID

# python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID

DATASET="kitti360"



# --- batch processing multiple sequences ---

SEQ_IDS=("3353" "4950" "8120")
# eg:
# ./preprocess_data.sh 2350 4950 8120 10200 10750 11400 1538 1728 1908 3353
# ./preprocess_data.sh 1538 1728 1908 3353



# 如果命令行传入了参数，则使用命令行参数
if [ $# -gt 0 ]; then
    SEQ_IDS=("$@")
fi

echo "将要处理的序列ID: ${SEQ_IDS[@]}"

# 循环处理每个SEQ_ID
for SEQ_ID in "${SEQ_IDS[@]}"; do
    echo "正在处理序列ID: $SEQ_ID"
    
    echo "步骤1: 生成range view..."
    python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_id $SEQ_ID
    
    if [ $? -ne 0 ]; then
        echo "错误: 序列 $SEQ_ID 的range view生成失败"
        continue
    fi
    
    echo "步骤2: 转换为nerf格式..."
    python -m data.preprocess.kitti360_to_nerf --sequence_id $SEQ_ID
    
    if [ $? -ne 0 ]; then
        echo "错误: 序列 $SEQ_ID 的nerf转换失败"
        continue
    fi
    
    echo "步骤3: 计算序列配置..."
    python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID
    
    if [ $? -ne 0 ]; then
        echo "错误: 序列 $SEQ_ID 的配置计算失败"
        continue
    fi
    
    echo "序列 $SEQ_ID 处理完成!"
    echo "----------------------------------------"
    
done

echo "所有序列处理完成!"
