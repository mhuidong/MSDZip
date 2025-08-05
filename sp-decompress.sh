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

tar -xvf ${input}

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
            python decompress.py ${prefix}.${i}.mz ${prefix}.${i}.out --save -i ${i} --prefix ${prefix} --sp
        elif [ $i -eq $((parallel-1)) ]; then
            # 最后一个任务只加载模型
            python decompress.py ${prefix}.${i}.mz ${prefix}.${i}.out --load -i ${i} --prefix ${prefix} --sp
        else
            # 中间任务既保存又加载模型
            python decompress.py ${prefix}.${i}.mz ${prefix}.${i}.out --save --load -i ${i} --prefix ${prefix} --sp
        fi
    } &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

cat_out=""
for ((j=0;j<parallel;j++)); do
    cat_out="${cat_out} ${prefix}.${j}.out"
done
cat ${cat_out} > ${output}

rm -rf "$WATCH_DIR"
for ((k=0;k<parallel;k++)); do
    rm -rf ${prefix}.${k}.mz
    rm -rf ${prefix}.${k}.out
done

end_time=$(date +%s)
elapsed=$((end_time-start_time))
echo "Decompression Time (secs): ${elapsed}"