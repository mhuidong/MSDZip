#!/bin/bash
#source activate pt2
ulimit -n 100000
input=$1
output=$2

prefix=$(echo "$input" | awk -F/ '{print $NF}' | awk -F. '{print $1}')

WATCH_DIR=${prefix}_model

if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

python split_data.py ${input} -n 2

pids=()
if [ -z "$(ls -A "$WATCH_DIR")" ]; then
    python compress.py ${prefix}.0 ${prefix}.0.mz --save -i 0 --prefix ${prefix} --gpu 0 --sp &
    pids+=($!)
fi

for i in 1; do
    inotifywait -e create --format '%f' "$WATCH_DIR" | while read file; do
        if [ "$file" == ${prefix}".$((i - 1)).pth" ]; then  #"model.$((i - 1)).pth"
            python compress.py ${prefix}.${i} ${prefix}.${i}.mz --load -i ${i} --prefix ${prefix} --gpu 1 --sp
        fi
    done
done

for pid in "${pids[@]}"; do
    wait $pid
done

tar -czf ${output} ${prefix}.0.mz ${prefix}.1.mz

rm -rf ${WATCH_DIR}
rm -rf ${prefix}.0
rm -rf ${prefix}.1
rm -rf ${prefix}.0.mz
rm -rf ${prefix}.1.mz

