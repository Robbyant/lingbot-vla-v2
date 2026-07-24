#!/bin/bash
# ============================================================
# Inference + simulation launcher (queue-scheduled).
# num_per_gpu inference servers stay resident per GPU; sim tasks match
# inference slots from a queue. Finishing a task frees its slot and starts the next.
#
# Usage: bash start_inference_and_eval.sh [options]
#   --model_path        inference model path (default: <path/to/your/checkpoint>)
#   --inference_script  inference-side module path (default: deploy/lingbot_vla_v2_policy.py)
#   --inference_workdir inference-side working dir (default: current working dir)
#   --start_port        starting port (default: 9330)
#   --pid_name          PID file prefix (default: test_pid)
#   --num_tasks         number of sim tasks, taken in order from the task list (default: 50, max: 50)
#   --num_gpus          total GPUs (default: 8)
#   --num_per_gpu       inference servers per GPU (default: 3, i.e. 8*3=24 slots)
#   --use_length        chunk length (default: 50)
#   --use_bf16          use bfloat16 inference (default: True)
#   --use_fp32          use float32 inference (default: False)
#   --use_compile       enable model compile in policy inference (default: True)
#   --keep_inference    keep inference servers resident after simulation
#
# Examples:
#   bash start_inference_and_eval.sh --num_tasks 50
#   bash start_inference_and_eval.sh --num_tasks 20 --num_per_gpu 2
# ============================================================

# ===== Parse keyword arguments =====
project_root="$(pwd)"
inference_workdir="${project_root}/"
inference_script="deploy/lingbot_vla_v2_policy.py"
# Override MODEL_PATH / OUTPUT_BASE / QWEN3VL_PATH / EVAL_WORKDIR via env or flags.
model_path="${MODEL_PATH:-/path/to/your/checkpoint}"
output_base="${OUTPUT_BASE:-/path/to/your/eval_output}"
start_port=9330
pid_name="test_pid"
num_tasks=50
num_gpus=8
num_per_gpu=3
use_length=50
use_bf16=True
use_fp32=False
use_compile=True
robo_name="robotwin"
video_fps=10
enable_video=False
task_config="demo_clean"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)      model_path="$2";      shift 2 ;;
        --inference_script) inference_script="$2"; shift 2 ;;
        --inference_workdir) inference_workdir="$2"; shift 2 ;;
        --output_base)     output_base="$2";     shift 2 ;;
        --start_port)      start_port="$2";      shift 2 ;;
        --pid_name)        pid_name="$2";        shift 2 ;;
        --num_tasks)       num_tasks="$2";       shift 2 ;;
        --num_gpus)        num_gpus="$2";        shift 2 ;;
        --num_per_gpu)     num_per_gpu="$2";     shift 2 ;;
        --use_length)      use_length="$2";      shift 2 ;;
        --use_bf16)        use_bf16="$2";        shift 2 ;;
        --use_fp32)        use_fp32="$2";        shift 2 ;;
        --use_compile)     use_compile="$2";     shift 2 ;;
        --robo_name)       robo_name="$2";       shift 2 ;;
        --task_config)     task_config="$2";     shift 2 ;;
        --video_fps)       video_fps="$2";       shift 2 ;;
        --no_video)        enable_video=False;   shift ;;
        -h|--help)
            echo "Usage: bash $0 [options]"
            echo "  --model_path        inference model path"
            echo "  --inference_script  inference-side module path"
            echo "  --inference_workdir inference-side working dir"
            echo "  --output_base       result output path"
            echo "  --start_port        starting port (default: 9330)"
            echo "  --pid_name          PID file prefix (default: test_pid)"
            echo "  --num_tasks         number of sim tasks (default: 50, max: 50)"
            echo "  --num_gpus          total GPUs (default: 8)"
            echo "  --num_per_gpu       inference servers per GPU (default: 3)"
            echo "  --use_length        chunk length (default: 50)"
            echo "  --use_bf16          use bfloat16 inference (default: True)"
            echo "  --use_fp32          use float32 inference (default: False)"
            echo "  --use_compile       enable model compile in policy inference (default: True)"
            echo "  --robo_name         robot config name (default: robotwin_clean_and_aug)"
            echo "  --task_config       config file in task_config/（default: demo_clean）"
            echo "  --video_fps         video recording fps (default: 10)"
            echo "  --no_video          disable video recording to speed up simulation"
            exit 0 ;;
        *)
            echo -e "\033[31mUnknown argument: $1\033[0m"; exit 1 ;;
    esac
done

# ===== Fix Vulkan ICD (required by Sapien rendering) =====
# nvidia_icd="/etc/vulkan/icd.d/nvidia_icd.json"
# echo -e "\033[33mWriting NVIDIA Vulkan ICD config: ${nvidia_icd}\033[0m"
# mkdir -p "$(dirname "$nvidia_icd")"
# cat > "$nvidia_icd" << 'VULKAN_EOF'
# {
#     "file_format_version" : "1.0.0",
#     "ICD": {
#         "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
#         "api_version" : "1.3"
#     }
# }
# VULKAN_EOF
# export VK_ICD_FILENAMES="$nvidia_icd"
# export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# ===== Common environment =====
# Cleanup: kill all child processes on exit / Ctrl-C / kill
cleanup() {
    echo ""
    echo -e "\033[33m=== Cleaning up all child processes ===\033[0m"
    # Kill inference servers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${inference_pids[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    # Kill eval workers
    for slot in $(seq 0 $((num_slots-1))); do
        local_pid=${slot_pid[$slot]:-0}
        if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
            kill -TERM -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                || kill -TERM "$local_pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Force kill survivors
    for slot in $(seq 0 $((num_slots-1))); do
        for local_pid in ${inference_pids[$slot]:-0} ${slot_pid[$slot]:-0}; do
            if [ "$local_pid" != "0" ] && kill -0 "$local_pid" 2>/dev/null; then
                kill -KILL -- -"$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
                    || kill -KILL "$local_pid" 2>/dev/null || true
            fi
        done
    done
    echo -e "\033[33m=== Cleanup done ===\033[0m"
}
trap cleanup EXIT INT TERM

export QWEN3VL_PATH="${QWEN3VL_PATH:-/path/to/your/checkpoints/Qwen3-VL-4B-Instruct}"

# ===== Working dir =====
eval_workdir="${EVAL_WORKDIR:-/path/to/Robotwin}"

# ===== Sim-side python =====
# The sim side runs with the current shell's Python environment.
if ! command -v python >/dev/null 2>&1; then
    echo -e "\033[31mError: python command not found for sim side\033[0m"
    exit 1
fi
echo -e "\033[36mSim-side python: $(command -v python) ($(python --version 2>&1))\033[0m"

# ===== Task count validation =====
if [ "$num_tasks" -gt 50 ]; then
    echo -e "\033[31mError: num_tasks ${num_tasks} exceeds max 50; use 1~50\033[0m"
    exit 1
fi

# ===== Sim-side args =====
policy_name=ACT
train_config_name=0
seed=0

# ===== Compute inference slot count =====
# actual slots = min(num_tasks, num_gpus * num_per_gpu)
# When tasks < max concurrency, spin up only as many servers as tasks to avoid
# the waste of starting then shutting down extras.
num_slots=$(( num_gpus * num_per_gpu ))
if [ "$num_tasks" -lt "$num_slots" ]; then
    echo -e "\033[36mnum_tasks ${num_tasks} < max slots ${num_slots}; starting servers by task count\033[0m"
    num_slots=$num_tasks
fi

# ===== Full task list (50) =====
task_list_all=("lift_pot" "hanging_mug" "stack_bowls_three" "scan_object" "handover_block" "click_bell" "put_object_cabinet" "open_microwave" "stack_blocks_three" "place_shoe" "adjust_bottle" "beat_block_hammer" "blocks_ranking_rgb" "blocks_ranking_size" "click_alarmclock" "dump_bin_bigbin" "grab_roller" "handover_mic" "move_can_pot" "move_pillbottle_pad" "move_playingcard_away" "place_cans_plasticbox" "place_container_plate" "place_dual_shoes" "place_empty_cup" "place_fan" "place_mouse_pad" "place_object_basket" "place_object_scale" "place_object_stand" "place_phone_stand" "move_stapler_pad" "open_laptop" "pick_diverse_bottles" "pick_dual_bottles" "place_a2b_left" "place_a2b_right" "place_bread_basket" "place_bread_skillet" "place_burger_fries" "place_can_basket" "press_stapler" "rotate_qrcode" "shake_bottle_horizontally" "shake_bottle" "stack_blocks_two" "stack_bowls_two" "stamp_seal" "turn_switch" "put_bottles_dustbin")

# Build the sim task queue
task_queue=()
for i in $(seq 0 $((num_tasks-1))); do
    task_queue+=("${task_list_all[$i]}")
done
echo -e "\033[36mTasks this run (${num_tasks}): ${task_queue[*]}\033[0m"
echo -e "\033[36mInference config: ${num_gpus} GPU x ${num_per_gpu} servers/GPU = ${num_slots} slots\033[0m"

# ===== Common variables =====
batch_time=$(date +%Y%m%d_%H%M%S)
# Extract experiment name and step from model_path (e.g. qwen35_robotwin_forward_no_mask_30k)
_exp_name=$(echo "$model_path" | grep -oP '[^/]+(?=/checkpoints)')
_step_num=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
_step_k=$(( _step_num / 1000 ))k
run_dir="${output_base}/${_exp_name}_${_step_k}_${task_config}_${batch_time}"
mkdir -p "${run_dir}/inference_logs" "${run_dir}/eval_logs"
log_dir="${run_dir}"
inference_pid_file="${run_dir}/inference_pids.txt"
eval_pid_file="${run_dir}/eval_pids.txt"
> "$inference_pid_file"
> "$eval_pid_file"
stats_file="${run_dir}/stats.txt"
echo -e "\033[36mRun directory: ${run_dir}\033[0m"

# Result collection arrays
declare -a result_task_names=()
declare -a result_durations=()
declare -a result_logs=()
declare -a result_status=()

# ============================================================
# Phase 1: start inference-side QwenPi servers (resident)
# ============================================================
echo -e "\033[32m========== Starting inference side: ${num_slots} QwenPi servers, ports ${start_port}~$((start_port + num_slots - 1)) ==========\033[0m"

cd "$inference_workdir" || { echo -e "\033[31mError: inference workdir ${inference_workdir} missing\033[0m"; exit 1; }

for slot in $(seq 0 $((num_slots-1))); do
    gpu_id=$(( slot % num_gpus ))
    port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${run_dir}/inference_logs/qwenpi_slot${slot}_gpu${gpu_id}_port${port}.log"

    echo -e "\033[33m[inf slot $slot] GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    # deploy/qwen3_5vl_dit_policy.py -> deploy.qwen3_5vl_dit_policy
    inference_script_for_module="${inference_script}"
    if [[ "${inference_script_for_module}" = /* ]]; then
        inference_script_for_module="${inference_script_for_module#${inference_workdir%/}/}"
    fi
    inference_module=$(echo "${inference_script_for_module}" | sed 's|/|.|g; s|\.py$||')
    setsid python -m ${inference_module} \
        --model_path "${model_path}" \
        --use_length "${use_length}" \
        --use_bf16 "${use_bf16}" \
        --use_fp32 "${use_fp32}" \
        --use_compile "${use_compile}" \
        --port "${port}" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" >> "$inference_pid_file"
done

echo -e "\033[32m${num_slots} inference servers started, PIDs saved to ${inference_pid_file}\033[0m"

# Record each slot's inference server PID for on-demand shutdown
declare -a inference_pids=()
for slot in $(seq 0 $((num_slots-1))); do
    inference_pids[$slot]=0
done
# Re-read PIDs (in launch order, mapping 1:1 to slots)
mapfile -t pid_list < "$inference_pid_file"
for slot in $(seq 0 $((num_slots-1))); do
    if [ $slot -lt ${#pid_list[@]} ]; then
        inference_pids[$slot]=${pid_list[$slot]}
    fi
done
active_inference=$num_slots

# ============================================================
# Phase 2: queue-scheduled sim tasks
# ============================================================
echo -e "\033[32m========== Starting sim side (queue-scheduled, ${num_slots} concurrent slots) ==========\033[0m"

cd "$eval_workdir" || { echo -e "\033[31mError: sim workdir ${eval_workdir} missing\033[0m"; exit 1; }

total_start_time=$(date +%s)

# Slot state: free or busy
# slot_pid[slot]   = sim process PID (0 = free)
# slot_task[slot]  = current task name
# slot_start[slot] = task start time
# slot_log[slot]   = log file path
declare -a slot_pid=()
declare -a slot_task=()
declare -a slot_start=()
declare -a slot_log=()
for slot in $(seq 0 $((num_slots-1))); do
    slot_pid[$slot]=0
    slot_task[$slot]=""
    slot_start[$slot]=0
    slot_log[$slot]=""
done

# Queue pointer and retry counters
queue_idx=0
max_retries=3
declare -A task_retries=()
for t in "${task_queue[@]}"; do
    task_retries[$t]=0
done

# Launch a sim task onto a given slot
launch_task() {
    local slot=$1
    local task_name=$2
    local gpu_id=$(( slot % num_gpus ))
    local port=$(( start_port + slot ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    local log_file="${run_dir}/eval_logs/${task_name}.log"

    local attempt=$((task_retries[$task_name]+1))
    echo -e "\033[33m  [launch] ${task_name} -> slot $slot (GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}) [attempt ${attempt}/${max_retries}]\033[0m"

    # On retry, append instead of overwriting the failed log; mark this attempt with a banner
    {
        echo ""
        echo "================================================================"
        echo "  [attempt ${attempt}/${max_retries}] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  task=${task_name}  slot=${slot}  gpu=${gpu_id}  port=${port}"
        echo "================================================================"
    } >> "$log_file"

    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    setsid python -u script/eval_polict_client_openpi.py --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --port ${port} \
        --robo_name ${robo_name} \
        --video_fps ${video_fps} \
        --eval_video_log ${enable_video} \
        --output_dir "${run_dir}/eval_results" >> "$log_file" 2>&1 &

    local pid=$!
    echo "${pid}" >> "$eval_pid_file"
    slot_pid[$slot]=$pid
    slot_task[$slot]="$task_name"
    slot_start[$slot]=$(date +%s)
    slot_log[$slot]="$log_file"
}

# Initial fill: assign a task to each free slot
for slot in $(seq 0 $((num_slots-1))); do
    if [ $queue_idx -lt ${#task_queue[@]} ]; then
        launch_task $slot "${task_queue[$queue_idx]}"
        queue_idx=$((queue_idx + 1))
    fi
done

# Shut down the inference server on a given slot
shutdown_inference_slot() {
    local slot=$1
    local inf_pid=${inference_pids[$slot]}
    if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
        kill "$inf_pid" 2>/dev/null
        echo -e "\033[36m  [release] slot $slot inference server (PID: ${inf_pid}) stopped, remaining: $((active_inference - 1))\033[0m"
    fi
    inference_pids[$slot]=0
    active_inference=$((active_inference - 1))
}

# ============================================================
# Phase 3: scheduling loop - detect finished slots, reclaim and dispatch
# ============================================================
completed=0
skipped=0
total_tasks=${#task_queue[@]}

# Exit when all tasks done/skipped and nothing running
has_running() {
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && return 0
    done
    return 1
}

# Count remaining tasks (running + queued)
remaining_tasks() {
    local running=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${slot_pid[$slot]}" != "0" ] && running=$((running + 1))
    done
    local queued=$(( ${#task_queue[@]} - queue_idx ))
    echo $((running + queued))
}

while [ $((completed + skipped)) -lt $total_tasks ] && has_running; do
    for slot in $(seq 0 $((num_slots-1))); do
        pid=${slot_pid[$slot]}

        # Skip free slots
        [ "$pid" = "0" ] && continue

        # Check whether the process has exited
        if ! kill -0 "$pid" 2>/dev/null; then
            task_end_time=$(date +%s)
            task_duration=$(( task_end_time - slot_start[$slot] ))
            wait "$pid"
            exit_code=$?

            task_name="${slot_task[$slot]}"
            log_file="${slot_log[$slot]}"

            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
                echo -e "\033[32m  [done] ${task_name} slot $slot took ${task_duration}s (progress ${completed}/${total_tasks})\033[0m"
                result_status+=("done")
                result_task_names+=("$task_name")
                result_durations+=("$task_duration")
                result_logs+=("$log_file")

                # Free the slot
                slot_pid[$slot]=0
                slot_task[$slot]=""
                slot_log[$slot]=""

                # Dispatch next task from the queue
                if [ $queue_idx -lt ${#task_queue[@]} ]; then
                    launch_task $slot "${task_queue[$queue_idx]}"
                    queue_idx=$((queue_idx + 1))
                fi
            else
                task_retries[$task_name]=$((task_retries[$task_name] + 1))
                if [ ${task_retries[$task_name]} -lt $max_retries ]; then
                    echo -e "\033[31m  [fail] ${task_name} slot $slot took ${task_duration}s exit=${exit_code}, retrying on this slot (${task_retries[$task_name]}/${max_retries})\033[0m"
                    # Retry immediately on the same slot
                    launch_task $slot "$task_name"
                else
                    skipped=$((skipped + 1))
                    echo -e "\033[31m  [skip] ${task_name} failed ${max_retries} times, no more retries\033[0m"
                    result_status+=("skip:${exit_code}")
                    result_task_names+=("$task_name")
                    result_durations+=("$task_duration")
                    result_logs+=("$log_file")

                    # Free the slot
                    slot_pid[$slot]=0
                    slot_task[$slot]=""
                    slot_log[$slot]=""

                    # Dispatch next task from the queue
                    if [ $queue_idx -lt ${#task_queue[@]} ]; then
                        launch_task $slot "${task_queue[$queue_idx]}"
                        queue_idx=$((queue_idx + 1))
                    fi
                fi
            fi
        fi
    done

    # Adaptive shutdown: when remaining tasks < active inference servers, stop idle-slot servers
    remaining=$(remaining_tasks)
    for slot in $(seq 0 $((num_slots-1))); do
        [ $remaining -lt $active_inference ] || break
        if [ "${slot_pid[$slot]}" = "0" ] && [ "${inference_pids[$slot]}" != "0" ]; then
            shutdown_inference_slot $slot
        fi
    done

    sleep 5
done

total_end_time=$(date +%s)
total_duration=$(( total_end_time - total_start_time ))

if [ $skipped -gt 0 ]; then
    echo -e "\033[31m========== Sim finished: ${completed} done, ${skipped} skipped, total ${total_duration}s ==========\033[0m"
else
    echo -e "\033[32m========== All sim tasks done, total ${total_duration}s ==========\033[0m"
fi

# ============================================================
# Phase 4: generate the stats file
# ============================================================
echo -e "\033[36mGenerating stats file: ${stats_file}\033[0m"

{
    echo "============================================"
    echo "  Eval Result Stats"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Model: ${_exp_name}_${_step_k}"
    echo "  Model path: ${model_path}"
    echo "  Tasks: ${num_tasks}"
    echo "  Task Config: ${task_config}"
    echo "  Inference: ${num_gpus} GPU x ${num_per_gpu}/GPU = ${num_slots} slots"
    echo "  Result: ${completed} done, ${skipped} skipped"
    echo "============================================"
    echo ""
    printf "%-30s %-10s %-12s %-10s %-10s\n" "Task" "Time(s)" "Done(100)" "Success/Total" "Rate"
    echo "--------------------------------------------------------------------------------"

    total_success=0
    total_episodes=0
    all_complete=true

    for i in "${!result_task_names[@]}"; do
        task_name="${result_task_names[$i]}"
        duration="${result_durations[$i]}"
        log_file="${result_logs[$i]}"
        status="${result_status[$i]}"

        # Extract success rate from the log (strip ANSI color codes)
        success_rate="-"
        episodes_done="-"
        if [ -f "$log_file" ]; then
            last_rate_line=$(sed 's/\x1b\[[0-9;]*m//g' "$log_file" | grep -oP 'Success rate: \K.*' | tail -1)
            if [ -n "$last_rate_line" ]; then
                suc_num=$(echo "$last_rate_line" | grep -oP '^\d+' | head -1)
                total_num=$(echo "$last_rate_line" | grep -oP '/\K\d+' | head -1)
                rate_pct=$(echo "$last_rate_line" | grep -oP '=> \K[\d.]+')
                if [ -n "$suc_num" ] && [ -n "$total_num" ]; then
                    success_rate="${rate_pct}%"
                    episodes_done="${suc_num}/${total_num}"
                    total_success=$((total_success + suc_num))
                    total_episodes=$((total_episodes + total_num))
                fi
            fi
        fi

        # Check whether all 100 episodes ran
        if [ "$episodes_done" != "-" ]; then
            total_ep=$(echo "$episodes_done" | cut -d'/' -f2)
            if [ "$total_ep" = "100" ]; then
                complete_mark="YES"
            else
                complete_mark="NO(${total_ep}/100)"
                all_complete=false
            fi
        else
            complete_mark="NO(0/100)"
            all_complete=false
        fi

        printf "%-30s %-10s %-12s %-10s %-10s\n" "$task_name" "$duration" "$complete_mark" "$episodes_done" "$success_rate"
    done

    echo "--------------------------------------------------------------------------------"
    if [ $total_episodes -gt 0 ]; then
        overall_rate=$(awk "BEGIN {printf \"%.1f\", $total_success/$total_episodes*100}")
    else
        overall_rate="0.0"
    fi
    printf "Summary: total %ds, success %d/%d, overall rate %s%%\n" "$total_duration" "$total_success" "$total_episodes" "$overall_rate"
    if $all_complete; then
        echo "All tasks fully executed 100 episodes"
    else
        echo "Warning: some tasks did not complete 100 episodes; check logs"
    fi
    echo "============================================"
} | tee "$stats_file"

# ============================================================
# Phase 5: handle inference servers (stop the remaining ones)
# ============================================================
if $keep_inference; then
    alive_count=0
    for slot in $(seq 0 $((num_slots-1))); do
        [ "${inference_pids[$slot]}" != "0" ] && alive_count=$((alive_count + 1))
    done
    echo -e "\033[36m========== Sim done, ${alive_count} inference servers kept resident ==========\033[0m"
    echo -e "\033[36mStop inference manually: kill \$(cat ${inference_pid_file})\033[0m"
else
    echo -e "\033[36m========== Stopping remaining inference servers ==========\033[0m"
    for slot in $(seq 0 $((num_slots-1))); do
        inf_pid=${inference_pids[$slot]}
        if [ "$inf_pid" != "0" ] && kill -0 "$inf_pid" 2>/dev/null; then
            kill "$inf_pid" 2>/dev/null
            echo -e "\033[36m  slot $slot inference server (PID: ${inf_pid}) stopped\033[0m"
        fi
    done
    echo -e "\033[32mAll inference servers stopped\033[0m"
fi

echo ""
echo -e "\033[32m========== All done ==========\033[0m"
echo -e "Run dir: ${run_dir}"
echo -e "Stats file: ${stats_file}"
echo -e "Inference logs: ${run_dir}/inference_logs/"
echo -e "Sim logs: ${run_dir}/eval_logs/"
echo -e "Eval results & videos: ${run_dir}/eval_results/"
if $keep_inference; then
    echo -e "Inference PIDs: ${inference_pid_file} (resident)"
else
    echo -e "Inference PIDs: ${inference_pid_file} (stopped)"
fi
echo -e "Sim PIDs: ${eval_pid_file}"
