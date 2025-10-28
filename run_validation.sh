#!/bin/bash
set -euo pipefail
trap 'echo "Stopping..."; kill 0' SIGINT

# ------------------------
# Configuration
# ------------------------
datasets=("davis" "kiba")
seed=0
validation_folds=(0 1 2 3 4)
GPUS=(0 1)
MAX_PER_GPU=5  # max concurrent jobs per GPU

# ------------------------
# GPU scheduling state
# ------------------------
declare -A gpu_count    # gpu_count[gpu]=number of jobs running on GPU
declare -A pid_to_gpu   # pid_to_gpu[pid]=gpu
for g in "${GPUS[@]}"; do gpu_count[$g]=0; done

# ensure logs dir
mkdir -p logs

# ------------------------
# Helpers
# ------------------------
log() {
    # use stderr for scheduler log so it doesn't interfere with anything else
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $*" >&2
}

# returns a free gpu index, blocks until one becomes available
choose_free_gpu() {
    while true; do
        for g in "${GPUS[@]}"; do
            local count=${gpu_count[$g]:-0}
            if [ "$count" -lt "$MAX_PER_GPU" ]; then
                echo "$g"
                return
            fi
        done

        # all GPUs saturated -> wait until a tracked pid frees a GPU
        while true; do
            # wait for any child to exit (bash builtin)
            wait -n || true

            freed_any=false
            for pid in "${!pid_to_gpu[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    freed_gpu=${pid_to_gpu[$pid]}
                    if [ -n "${gpu_count[$freed_gpu]:-}" ]; then
                        gpu_count[$freed_gpu]=$(( gpu_count[$freed_gpu]-1 ))
                        log "freed GPU $freed_gpu from pid $pid (now ${gpu_count[$freed_gpu]})"
                    fi
                    unset pid_to_gpu[$pid]
                    freed_any=true
                fi
            done

            if [ "$freed_any" = true ]; then
                break
            fi

            sleep 0.05
        done
    done
}


# launch a job on a given GPU: args are the python script + args (without --cuda)
launch_job() {
    local gpu="$1"; shift
    local args=( "$@" )
    # build tidy log filename based on model/dataset/fold if available
    local mdl=""
    local dset=""
    local fold=""
    for (( i=0; i<${#args[@]}; i++ )); do
        case "${args[i]}" in
            --model) mdl="${args[i+1]}";;
            --dataset) dset="${args[i+1]}";;
            --validation_fold) fold="${args[i+1]}";;
        esac
    done
    local logfile="logs/${mdl:-job}_${dset:-dataset}_fold_${fold:-unk}_$$.log"

    gpu_count[$gpu]=$(( gpu_count[$gpu]+1 ))

    (
        # print start line and run job; overwrite logfile (single >)
        echo "$(date +'%Y-%m-%d %H:%M:%S') - START GPU ${gpu} CMD: python ${args[*]} --cuda ${gpu}" >&2
        python "${args[@]}" --cuda "$gpu" >"$logfile" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "$(date +'%Y-%m-%d %H:%M:%S') - DONE  GPU ${gpu} (rc=0)" >&2
        else
            echo "$(date +'%Y-%m-%d %H:%M:%S') - FAIL  GPU ${gpu} (rc=${rc}) -- see ${logfile}" >&2
        fi
        exit $rc
    ) &

    local pid=$!
    pid_to_gpu[$pid]="$gpu"
    log "Launched pid $pid on GPU $gpu (now ${gpu_count[$gpu]}) -> logfile: ${logfile}"
}

# cleanup on exit
on_exit() {
    log "Cleaning up, killing children..."
    kill 0 2>/dev/null || true
}
trap on_exit EXIT

# ------------------------
# Common args
# ------------------------
base_common_args=( --seed "$seed" --wandb )

# ------------------------
# Launch jobs
# ------------------------

# ------------------------
# Models configuration
# ------------------------

# mv data/processed/davis_deepfri.pt data/processed/davis_deepfri_cc.pt
# mv data/processed/kiba_deepfri.pt data/processed/kiba_deepfri_cc.pt

# mv data/davis/proteins_deepfri.json data/davis/proteins_deepfri_cc.json
# mv data/davis/proteins_deepfri_mf.json data/davis/proteins_deepfri.json

# mv data/kiba/proteins_deepfri.json data/kiba/proteins_deepfri_cc.json
# mv data/kiba/proteins_deepfri_mf.json data/kiba/proteins_deepfri.json

models=("PLM_Vnoc_GINConvNet")
datasets=("davis_mutation")
protein_embedding_type="deepfri_ec"
kernels=(16)
conv_layers=("32 64 96")
plm_layers=("128")
validation_folds=(0 1 2 3 4)


for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for plm in "${plm_layers[@]}"; do
            for conv in "${conv_layers[@]}"; do
                for kernel in "${kernels[@]}"; do
                    for fold in "${validation_folds[@]}"; do

                        gpu=$(choose_free_gpu)
                        launch_job "$gpu" training_validation.py \
                            "${base_common_args[@]}" \
                            --dataset "$dataset" \
                            --model "$model" \
                            --plm_layers $plm \
                            --conv_layers $conv \
                            --kernel_size "$kernel" \
                            --validation_fold "$fold" \
                            --description "$protein_embedding_type" \
                            --protein_embedding_type "$protein_embedding_type"

                    done
                done
            done
        done
    done
done

models=("PLM_Vnoc_GINConvNet")
datasets=("kiba")
protein_embedding_type="deepfri_mf"
kernels=(16)
conv_layers=("32 64 96")
plm_layers=("128")
validation_folds=(0 1 2 3 4)


for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for plm in "${plm_layers[@]}"; do
            for conv in "${conv_layers[@]}"; do
                for kernel in "${kernels[@]}"; do
                    for fold in "${validation_folds[@]}"; do

                        gpu=$(choose_free_gpu)
                        launch_job "$gpu" training_validation.py \
                            "${base_common_args[@]}" \
                            --dataset "$dataset" \
                            --model "$model" \
                            --plm_layers $plm \
                            --conv_layers $conv \
                            --kernel_size "$kernel" \
                            --validation_fold "$fold" \
                            --description "$protein_embedding_type" \
                            --protein_embedding_type "$protein_embedding_type"

                    done
                done
            done
        done
    done
done

# ------------------------
# Wait for remaining jobs to finish
# ------------------------
log "All jobs launched; waiting for remaining ${#pid_to_gpu[@]} jobs..."
# keep waiting until pid_to_gpu map is empty
while [ "${#pid_to_gpu[@]}" -gt 0 ]; do
    wait -n || true
    # free finished pids
    for pid in "${!pid_to_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            freed_gpu=${pid_to_gpu[$pid]}
            gpu_count[$freed_gpu]=$(( gpu_count[$freed_gpu]-1 ))
            log "freed GPU $freed_gpu from pid $pid (now ${gpu_count[$freed_gpu]})"
            unset pid_to_gpu[$pid]
        fi
    done
done

log "All validation jobs completed!"
