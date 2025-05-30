#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=bench_output_case1_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # Increased time limit for larger runs
#SBATCH -n 72

# Ensure your executable is compiled and named 'swe'
# Example: make swe

GLOBAL_SIZES=(1000) # Global NX and NY 100 200 500 750 
PROCS=(1 2 4 8 12 16 20 24 28 32 40 48 56 64 72) # Varying number of processes
REPEAT=3  # Number of repetitions per config
TEST_CASE=1 # Test Case ID for water drops in a box
TEND=0.5 # Simulation time in hours
OUTPUT_N=0 # No output files for performance benchmarking

# Ensure the output directory exists
mkdir -p bench_results_case1

# echo "test_case,global_nx,global_ny,procs,run,total_time_seconds,num_iterations,time_per_iteration" > bench_results_case1/timings_case1.csv

for SIZE in "${GLOBAL_SIZES[@]}"; do
  for NPROCS in "${PROCS[@]}"; do
    # Skip if NPROCS is greater than the total number of cells for a reasonable decomposition
    # This is a heuristic, adjust if needed based on how MPI_Dims_create behaves for small sizes
    if (( NPROCS > SIZE * SIZE / 4 && NPROCS > 1 )); then # Assuming min 2x2 cells per proc or so
        echo "Skipping NPROCS=$NPROCS for SIZE=$SIZE (too many processes for small problem)"
        continue
    fi

    for ((i = 1; i <= REPEAT; i++)); do
      # Generate a unique temp file name for each run
      TEMP_OUTPUT_FILE="tmp_output_case1_${SIZE}_${NPROCS}_${i}_$$_${SLURM_JOB_ID}.txt"
      
      echo "[RUNNING] Test Case $TEST_CASE, Size ${SIZE}x${SIZE}, NPROCS=$NPROCS (run $i)"
      
      export OMP_NUM_THREADS=1 # Ensure single thread per MPI process

      srun -n $NPROCS ./swe \
        --test-case $TEST_CASE \
        --Tend $TEND \
        --global-nx $SIZE \
        --global-ny $SIZE \
        --output-n $OUTPUT_N \
        --output-prefix "strong_scaling_case${TEST_CASE}_${SIZE}x${SIZE}_P${NPROCS}" \
        2>&1 | tee "$TEMP_OUTPUT_FILE" # Redirect stderr to stdout for tee

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

      echo "$TEST_CASE,$SIZE,$SIZE,$NPROCS,$i,$TOTAL_TIME,$NUM_ITERATIONS,$TIME_PER_ITERATION" >> bench_results_case1/timings_case1.csv
      rm "$TEMP_OUTPUT_FILE" # Clean up the temp file
    done
  done
done

echo "Benchmarking for Test Case $TEST_CASE completed. Results in bench_results_case1/timings_case1.csv"