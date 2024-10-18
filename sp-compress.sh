#!/bin/bash
#source activate pt2
ulimit -n 100000
data=$1
prefix=$2

WATCH_DIR=${prefix}_model
FileBaseName=$(basename ${data})   # data name
echo $FileBaseName


if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

python split_data.py ${FileBaseName} -n 2

pids=()
if [ -z "$(ls -A "$WATCH_DIR")" ]; then
    python compress.py ${FileBaseName}.0 ${FileBaseName}.0.msdzip --save -i 0 --prefix ${prefix} --gpu 0 &
    pids+=($!)
fi


for i in 1; do
    inotifywait -e create --format '%f' "$WATCH_DIR" | while read file; do
        if [ "$file" == ${prefix}".$((i - 1)).pth" ]; then  #"model.$((i - 1)).pth"
            python compress.py ${FileBaseName}.${i} ${FileBaseName}.${i}.msdzip --load -i ${i} --prefix ${prefix} --gpu 1
        fi
    done
done

for pid in "${pids[@]}"; do
    wait $pid
done

tar -czf ${FileBaseName}.msdzip ${FileBaseName}.0.msdzip ${FileBaseName}.1.msdzip

rm -rf ${WATCH_DIR}
rm -rf ${FileBaseName}.0
rm -rf ${FileBaseName}.1
rm -rf ${FileBaseName}.0.msdzip
rm -rf ${FileBaseName}.1.msdzip

