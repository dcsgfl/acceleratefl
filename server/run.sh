DATASET="MNISTROTOWN"
SCHEDULER="PXYSched"
DROP=0
EPOCHS=400
CLTHRES=10

python3 run_server.py --dataset "${1:-$DATASET}" --scheduler "${2:-$SCHEDULER}" --drop "${3:-$DROP}" --epochs "${4:-$EPOCHS}" --threshold "${5:-$CLTHRES}"

# Sample command
# sh run.sh dataset scheduler
# sh run.sh FEMNIST RNDSched 10 200 10
