#!/bin/bash
END = $1
for ((i = 0; i < $END; i++)); do
    python3 run_client.py --dataset "$2" --summary "$3" --centralip "$4" &
done
wait
