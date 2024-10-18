#!/bin/bash
#source activate pt2
ulimit -n 100000
data=$1
output=$2
prefix=$3

WATCH_DIR=${prefix}_model
FileBaseName=$(basename ${data})   # data name
echo $FileBaseName

## 解压缩
if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

tar -xvf ${FileBaseName}.mz

pids=()
START_TIME=$(date +%s)
if [ -z "$(ls -A "$WATCH_DIR")" ]; then
    python decompress.py ${FileBaseName}.0.mz ${FileBaseName}.0.out --save -i 0 --prefix ${prefix} --gpu 0 & # 并行执行 X 脚本 >
    pids+=($!)
fi
#
for i in 1; do
    inotifywait -e create --format '%f' "$WATCH_DIR" | while read file; do
        if [ "$file" == ${prefix}".$((i - 1)).pth" ]; then  #"model.$((i - 1)).pth"
            python decompress.py ${FileBaseName}.${i}.mz ${FileBaseName}.${i}.out --load -i ${i} --prefix ${prefix} --gpu 1
        fi
    done
done
for pid in "${pids[@]}"; do
    wait $pid
done


cat ${FileBaseName}.0.out ${FileBaseName}.1.out > ${output}

rm -rf "$WATCH_DIR"
rm -rf ${FileBaseName}.0.mz
rm -rf ${FileBaseName}.1.mz
rm -rf ${FileBaseName}.0.out
rm -rf ${FileBaseName}.1.out