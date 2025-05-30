#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=bench_output_weak_1proc_baselines_%j.txt
#SBATCH --ntasks=1 # Only 1 process
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # Increased time limit for larger runs
#SBATCH -n 1

# This script measures time_per_iteration for various problem sizes
# on a single process. This helps in selecting a "local problem size"
# for weak scaling tests.

TEST_CASE=1 # Using test case 1 for simplicity
TEND=0.5 # Simulation time in hours (enough for multiple steps)
OUTPUT_N=0 # No output files for performance benchmarking

# Different global grid sizes for a single process
GLOBAL_SIZES=(100 200 300 400 500 600 700 800 900 1000)
REPEAT=3

mkdir -p bench_results_weak_scaling_1proc

echo "test_case,global_nx,global_ny,procs,run,total_time_seconds,num_iterations,time_per_iteration" > bench_results_weak_scaling_1proc/timings_1proc_baselines.csv

for SIZE in "${GLOBAL_SIZES[@]}"; do
  for ((i = 1; i <= REPEAT; i++)); do
    TEMP_OUTPUT_FILE="tmp_output_1proc_baseline_${SIZE}_${i}_$$_${SLURM_JOB_ID}.txt"
    
    echo "[RUNNING] 1-proc baseline: Size ${SIZE}x${SIZE} (run $i)"
    
    export OMP_NUM_THREADS=1

    srun -n 1 ./swe \
      --test-case $TEST_CASE \
      --Tend $TEND \
      --global-nx $SIZE \
      --global-ny $SIZE \
      --output-n $OUTPUT_N \
      --output-prefix "1proc_baseline_case${TEST_CASE}_${SIZE}x${SIZE}" \
      --full-log \
      2>&1 | tee "$TEMP_OUTPUT_FILE"

    TOTAL_TIME=$(grep "\[TIMING\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')
    NUM_ITERATIONS=$(grep "\[ITERATIONS\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF)}')
    TIME_PER_ITERATION=$(grep "\[TIME_PER_ITER\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')

    if [ -z "$TOTAL_TIME" ] || [ -z "$NUM_ITERATIONS" ] || [ -z "$TIME_PER_ITERATION" ]; then
        echo "WARNING: Could not extract all metrics for 1-proc baseline: Size ${SIZE}x${SIZE} (run $i). Output below:"
        cat "$TEMP_OUTPUT_FILE"
        TOTAL_TIME="ERROR"
        NUM_ITERATIONS="ERROR"
        TIME_PER_ITERATION="ERROR"
    fi

    echo "$TEST_CASE,$SIZE,$SIZE,1,$i,$TOTAL_TIME,$NUM_ITERATIONS,$TIME_PER_ITERATION" >> bench_results_weak_scaling_1proc/timings_1proc_baselines.csv
    rm "$TEMP_OUTPUT_FILE"
  done
done

echo "1-Process baseline runs completed. Results in bench_results_weak_scaling_1proc/timings_1proc_baselines.csv"