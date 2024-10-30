#!/bin/bash
root_path=/home/yxueat/course@ust/elce6910c/hw2/Multi-View-Image-Classification
export PYTHONPATH=$root_path:$PYTHONPATH

task_name=train_quant
log_dir=${root_path}/log

generate_unique_logfile() {
    local logfile="$1"
    local base="${logfile%.*}"
    local ext="${logfile##*.}"
    local counter=1

    logfile="${base}_$counter.$ext"
    while [[ -e "$logfile" ]]; do
        ((counter++))
        logfile="${base}_$counter.$ext"
    done

    echo "$logfile"
}

# 创建日志目录
mkdir -p "$log_dir"

# 生成唯一的日志文件名
logfile=$(generate_unique_logfile "$log_dir/$task_name.log")

config=${root_path}/config/main.yml
nohup python ${root_path}/src/__main__.py \
    --config "$config" \
    > "$logfile" 2>&1 &
