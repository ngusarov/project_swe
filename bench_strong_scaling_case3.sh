#!/bin/bash
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=bench_output_case3_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # Increased time limit for larger runs
#SBATCH -n 72

# Note: This script assumes you have implemented parallel HDF5 data distribution
# in the SWESolver constructor that reads from file.
# The current swe.cc explicitly states: "Warning: HDF5 data only read by rank 0."
# This script will not produce meaningful parallel scaling results for case 3
# unless that warning is addressed and true parallel data loading is implemented.

# HDF5 file names and their corresponding global_nx/ny (actual grid size is NX by NY, so size+1)
# You will need to make sure these files exist and are accessible.
declare -A HDF5_FILES
HDF5_FILES["100"]="Data_nx101_500km.h5" # Global grid is 100x100
HDF5_FILES["200"]="Data_nx201_500km.h5" # Global grid is 200x200
HDF5_FILES["500"]="Data_nx501_500km.h5"
HDF5_FILES["1000"]="Data_nx1001_500km.h5"

GLOBAL_SIZES=(100 200 500 750 1000) # Global NX and NY
PROCS=(1 2 4 8 12 16 20 24 28 32 40 48 56 64 72) # Varying number of processes
REPEAT=3
TEST_CASE=3 # Use test case 3 to trigger HDF5 constructor in main.cc
TEND=0.2 # Simulation time in hours for HDF5 cases
OUTPUT_N=0

mkdir -p bench_results_case3

echo "test_case,hdf5_file,global_nx,global_ny,procs,run,total_time_seconds,num_iterations,time_per_iteration" > bench_results_case3/timings_case3.csv

for SIZE in "${GLOBAL_SIZES[@]}"; do
  H5_FILE="${HDF5_FILES[$SIZE]}"
  if [ -z "$H5_FILE" ]; then
    echo "Error: HDF5 file not defined for size $SIZE. Skipping."
    continue
  fi

  for NPROCS in "${PROCS[@]}"; do
    if (( NPROCS > SIZE * SIZE / 4 && NPROCS > 1 )); then
        echo "Skipping NPROCS=$NPROCS for SIZE=$SIZE (too many processes for small problem)"
        continue
    fi

    for ((i = 1; i <= REPEAT; i++)); do
      TEMP_OUTPUT_FILE="tmp_output_case3_${SIZE}_${NPROCS}_${i}_$$_${SLURM_JOB_ID}.txt"
      
      echo "[RUNNING] Test Case $TEST_CASE, HDF5 File $H5_FILE, Size ${SIZE}x${SIZE}, NPROCS=$NPROCS (run $i)"
      
      export OMP_NUM_THREADS=1

      srun -n $NPROCS ./swe \
        --test-case $TEST_CASE \
        --Tend $TEND \
        --global-nx $SIZE \
        --global-ny $SIZE \
        --output-n $OUTPUT_N \
        --output-prefix "strong_scaling_case${TEST_CASE}_${SIZE}x${SIZE}_P${NPROCS}" \
        --h5-file "$H5_FILE" \
        2>&1 | tee "$TEMP_OUTPUT_FILE"

      TOTAL_TIME=$(grep "\[TIMING\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')
      NUM_ITERATIONS=$(grep "\[ITERATIONS\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF)}')
      TIME_PER_ITERATION=$(grep "\[TIME_PER_ITER\]" "$TEMP_OUTPUT_FILE" | awk '{print $(NF-1)}')

      if [ -z "$TOTAL_TIME" ] || [ -z "$NUM_ITERATIONS" ] || [ -z "$TIME_PER_ITERATION" ]; then
          echo "WARNING: Could not extract all metrics for Test Case $TEST_CASE, HDF5 File $H5_FILE, Size ${SIZE}x${SIZE}, NPROCS=$NPROCS (run $i). Output below:"
          cat "$TEMP_OUTPUT_FILE"
          TOTAL_TIME="ERROR"
          NUM_ITERATIONS="ERROR"
          TIME_PER_ITERATION="ERROR"
      fi

      echo "$TEST_CASE,$H5_FILE,$SIZE,$SIZE,$NPROCS,$i,$TOTAL_TIME,$NUM_ITERATIONS,$TIME_PER_ITERATION" >> bench_results_case3/timings_case3.csv
      rm "$TEMP_OUTPUT_FILE"
    done
  done
done

echo "Benchmarking for Test Case $TEST_CASE (HDF5) completed. Results in bench_results_case3/timings_case3.csv"