#!/bin/bash
#source activate pt2
ulimit -n 100000
input=$1
output=$2
prefix=$3
parallel=${4:-2}

start_time=$(date +%s)

WATCH_DIR=${prefix}_model

if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

python split_data.py ${input} -n ${parallel}

pids=()
# 顺序启动后续任务，每个等待前一个模型文件生成
for ((i=0;i<parallel;i++)); do
    {
        # i=0不需要等待，i>0需要等待前一个模型文件生成
        if [ $i -gt 0 ]; then
            while [ ! -f "${WATCH_DIR}/${prefix}.$((i-1)).pth" ]; do
                sleep 1
            done 
        fi
        
        # 根据i的值决定参数
        if [ $i -eq 0 ]; then
            # 第一个任务只保存模型
            python compress.py ${prefix}.${i} ${prefix}.${i}.mz --save -i ${i} --prefix ${prefix} --sp
        elif [ $i -eq $((parallel-1)) ]; then
            # 最后一个任务只加载模型
            python compress.py ${prefix}.${i} ${prefix}.${i}.mz --load -i ${i} --prefix ${prefix} --sp
        else
            # 中间任务既保存又加载模型
            python compress.py ${prefix}.${i} ${prefix}.${i}.mz --save --load -i ${i} --prefix ${prefix} --sp
        fi
    } &
    pids+=($!)
done


for pid in "${pids[@]}"; do
    wait $pid
done

mz_files=""
for ((j=0;j<parallel;j++)); do
    mz_files="${mz_files} ${prefix}.${j}.mz"
done
tar -czf ${output} $mz_files

rm -rf ${WATCH_DIR}
for ((k=0;k<parallel;k++)); do
    rm -rf ${prefix}.${k}
    rm -rf ${prefix}.${k}.mz
done

input_size=$(stat -c %s "$input")
output_size=$(stat -c %s "$output")
ratio=$(awk "BEGIN{printf \"%.3f\", (${output_size}/${input_size})*8}")
end_time=$(date +%s)
elapsed=$((end_time-start_time))
echo "Compression Ratio (bits/base): $ratio"
echo "Compression Time (secs): ${elapsed}"