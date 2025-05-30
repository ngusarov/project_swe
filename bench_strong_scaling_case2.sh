#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=bench_output_case2_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # Increased time limit for larger runs
#SBATCH -n 72

GLOBAL_SIZES=(100 200 500 750 1000) # Global NX and NY
PROCS=(1 2 4 8 12 16 20 24 28 32 40 48 56 64 72) # Varying number of processes
REPEAT=3
TEST_CASE=2 # Test Case ID for analytical tsunami
TEND=0.5 # Simulation time in hours
OUTPUT_N=0

mkdir -p bench_results_case2

echo "test_case,global_nx,global_ny,procs,run,total_time_seconds,num_iterations,time_per_iteration" > bench_results_case2/timings_case2.csv

for SIZE in "${GLOBAL_SIZES[@]}"; do
  for NPROCS in "${PROCS[@]}"; do
    if (( NPROCS > SIZE * SIZE / 4 && NPROCS > 1 )); then
        echo "Skipping NPROCS=$NPROCS for SIZE=$SIZE (too many processes for small problem)"
        continue
    fi

    for ((i = 1; i <= REPEAT; i++)); do
      TEMP_OUTPUT_FILE="tmp_output_case2_${SIZE}_${NPROCS}_${i}_$$_${SLURM_JOB_ID}.txt"
      
      echo "[RUNNING] Test Case $TEST_CASE, Size ${SIZE}x${SIZE}, NPROCS=$NPROCS (run $i)"
      
      export OMP_NUM_THREADS=1

      srun -n $NPROCS ./swe \
        --test-case $TEST_CASE \
        --Tend $TEND \
        --global-nx $SIZE \
        --global-ny $SIZE \
        --output-n $OUTPUT_N \
        --output-prefix "strong_scaling_case${TEST_CASE}_${SIZE}x${SIZE}_P${NPROCS}" \
        2>&1 | tee "$TEMP_OUTPUT_FILE"

      TOTAL_TIME=$(grep "\[TIMING\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')
      NUM_ITERATIONS=$(grep "\[ITERATIONS\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF)}')
      TIME_PER_ITERATION=$(grep "\[TIME_PER_ITER\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')

      if [ -z "$TOTAL_TIME" ] || [ -z "$NUM_ITERATIONS" ] || [ -z "$TIME_PER_ITERATION" ]; then
          echo "WARNING: Could not extract all metrics for Test Case $TEST_CASE, Size ${SIZE}x${SIZE}, NPROCS=$NPROCS (run $i). Output below:"
          cat "$TEMP_OUTPUT_FILE"
          TOTAL_TIME="ERROR"
          NUM_ITERATIONS="ERROR"
          TIME_PER_ITERATION="ERROR"
      fi

      echo "$TEST_CASE,$SIZE,$SIZE,$NPROCS,$i,$TOTAL_TIME,$NUM_ITERATIONS,$TIME_PER_ITERATION" >> bench_results_case2/timings_case2.csv
      rm "$TEMP_OUTPUT_FILE"
    done
  done
done

echo "Benchmarking for Test Case $TEST_CASE completed. Results in bench_results_case2/timings_case2.csv"