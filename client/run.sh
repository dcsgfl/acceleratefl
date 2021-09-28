#!/bin/bash
END=$1
i=0
while [ "$i" -le "$END" ]; do
    python3 run_client.py --dataset "$2" --summary "$3" --centralip "$4" &
    i=$((i + 1))
done
wait
