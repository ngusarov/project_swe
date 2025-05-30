#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=bench_output_overhead_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # Increased time limit for larger runs
#SBATCH -n 32

# This script benchmarks the overhead of writing solution files (HDF5/XDMF)
# by comparing runs with and without output for specific problem sizes and process counts.

# Ensure your executable is compiled and named 'swe'
# Example: make swe

TEST_CASE=1 # Using test case 1 for this benchmark
TEND=0.5    # Simulation time in hours (fixed for all runs)

# Global grid sizes to test
GLOBAL_SIZES=(1000) # Test with 100x100 and 1000x1000 global grids

# Number of processes to test
PROCS=(1 2 4 8 16 32) # Varying number of processes

# Output frequencies:
# 0: No output files written during the simulation.
# 10: Output files written every 10 simulation steps.
OUTPUT_FREQS=(0 10) 

REPEAT=3 # Number of repetitions for each configuration to get more reliable averages

# Create a directory for benchmark results if it doesn't exist
mkdir -p bench_results_output_overhead

# Define the header for the CSV output file

# echo "test_case,global_nx,global_ny,procs,output_n,run,total_time_seconds,num_iterations,time_per_iteration" > bench_results_output_overhead/timings_output_overhead.csv

# Loop through each global grid size
for SIZE in "${GLOBAL_SIZES[@]}"; do
  # Loop through each number of processes
  for NPROCS in "${PROCS[@]}"; do
    # Skip configurations where the number of processes is excessively high
    # for the given problem size, which might lead to very small local domains
    # and inefficient MPI decomposition. This is a heuristic.
    if (( NPROCS > SIZE * SIZE / 4 && NPROCS > 1 )); then
        echo "Skipping NPROCS=$NPROCS for SIZE=$SIZE (too many processes for small problem)"
        continue
    fi

    # Loop through each output frequency setting
    for OUTPUT_N_VAL in "${OUTPUT_FREQS[@]}"; do
      # Loop for repetitions
      for ((i = 1; i <= REPEAT; i++)); do
        # Generate a unique temporary output file name for each run
        # This prevents conflicts when multiple jobs or runs are executed in parallel.
        TEMP_OUTPUT_FILE="tmp_output_overhead_case1_${SIZE}_${NPROCS}_${OUTPUT_N_VAL}_${i}_$$_${SLURM_JOB_ID}.txt"
        
        echo "[RUNNING] Output Overhead: Size ${SIZE}x${SIZE}, P=${NPROCS}, Output N=${OUTPUT_N_VAL} (run $i)"
        
        # Ensure OMP_NUM_THREADS is set to 1 for pure MPI benchmarks
        export OMP_NUM_THREADS=1

        # Execute the SWE solver with the specified parameters
        # 2>&1 | tee "$TEMP_OUTPUT_FILE" redirects both stdout and stderr to the tee command,
        # which prints to the console and saves to the temporary file.
        srun -n $NPROCS ./swe \
          --test-case $TEST_CASE \
          --Tend $TEND \
          --global-nx $SIZE \
          --global-ny $SIZE \
          --output-n $OUTPUT_N_VAL \
          --output-prefix "output_overhead_case${TEST_CASE}_${SIZE}x${SIZE}_P${NPROCS}_OutN${OUTPUT_N_VAL}" \
          2>&1 | tee "$TEMP_OUTPUT_FILE"

        # Extract the total simulation time from the temporary output file
        TOTAL_TIME=$(grep "\[TIMING\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')
        # Extract the total number of iterations
        NUM_ITERATIONS=$(grep "\[ITERATIONS\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF)}')
        # Extract the time per iteration
        TIME_PER_ITERATION=$(grep "\[TIME_PER_ITER\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')

        # Check if all metrics were successfully extracted
        if [ -z "$TOTAL_TIME" ] || [ -z "$NUM_ITERATIONS" ] || [ -z "$TIME_PER_ITERATION" ]; then
            echo "WARNING: Could not extract all metrics for Output Overhead: Size ${SIZE}x${SIZE}, P=${NPROCS}, Output N=${OUTPUT_N_VAL} (run $i). Output below:"
            cat "$TEMP_OUTPUT_FILE" # Print the content of the temp file for debugging
            TOTAL_TIME="ERROR"       # Mark as ERROR in CSV if extraction failed
            NUM_ITERATIONS="ERROR"
            TIME_PER_ITERATION="ERROR"
        fi

        # Append the results to the CSV file
        echo "$TEST_CASE,$SIZE,$SIZE,$NPROCS,$OUTPUT_N_VAL,$i,$TOTAL_TIME,$NUM_ITERATIONS,$TIME_PER_ITERATION" >> bench_results_output_overhead/timings_output_overhead.csv
        
        # Clean up the temporary output file
        rm "$TEMP_OUTPUT_FILE"
      done
    done
  done
done

echo "Output overhead benchmarking for Test Case $TEST_CASE completed. Results in bench_results_output_overhead/timings_output_overhead.csv"
