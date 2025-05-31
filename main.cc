#include <mpi.h>
#include "swe.hh"

#include <string>
#include <cstddef>
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <chrono> // For high_resolution_clock

int
main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Parameters for the Cartesian grid
  int dims[2] = {0, 0}; // To be determined by MPI_Dims_create
  int periods[2] = {0, 0}; // Not periodic for this problem
  int reorder = 0;         // No reordering
  MPI_Comm cart_comm;      // Cartesian communicator

  // Let MPI determine the best 2D decomposition
  MPI_Dims_create(num_procs, 2, dims);

  // Create the Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  // Determine neighbors (rank_source, rank_dest)
  int neighbors[4];
  MPI_Cart_shift(cart_comm, 1, 1, &neighbors[0], &neighbors[1]); // UP (y-1), DOWN (y+1)
  MPI_Cart_shift(cart_comm, 0, 1, &neighbors[2], &neighbors[3]); // LEFT (x-1), RIGHT (x+1)

  // Default values
  int test_case_id = 1;
  double Tend = 0.1; // hours
  std::size_t global_nx = 100;
  std::size_t global_ny = 100;
  std::size_t output_n = 0; // No output by default
  std::string output_fname_prefix = "output";
  bool full_log = false; // Only rank 0 logs fully if set
  std::string h5_file_path = ""; // For test_case_id 3

  // Parse command line arguments
  try {
      for (int i = 1; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--test-case" && i + 1 < argc) {
              test_case_id = std::stoi(argv[++i]);
          } else if (arg == "--Tend" && i + 1 < argc) {
              Tend = std::stod(argv[++i]);
          } else if (arg == "--global-nx" && i + 1 < argc) {
              global_nx = std::stoul(argv[++i]);
          } else if (arg == "--global-ny" && i + 1 < argc) {
              global_ny = std::stoul(argv[++i]);
          } else if (arg == "--output-n" && i + 1 < argc) {
              output_n = std::stoul(argv[++i]);
          } else if (arg == "--output-prefix" && i + 1 < argc) {
              output_fname_prefix = argv[++i];
          } else if (arg == "--full-log") {
              full_log = true;
          } else if (arg == "--h5-file" && i + 1 < argc) { // argument for HDF5 file
              h5_file_path = argv[++i];
          } else {
              if (rank == 0) {
                  std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
                  std::cerr << "Usage: " << argv[0] << " [--test-case ID] [--Tend TIME] [--global-nx NX] [--global-ny NY] [--output-n N] [--output-prefix PREFIX] [--full-log] [--h5-file PATH]" << std::endl;
              }
              MPI_Finalize();
              return 1;
          }
      }
  } catch (const std::invalid_argument& e) {
      if (rank == 0) {
          std::cerr << "Invalid argument value: " << e.what() << std::endl;
      }
      MPI_Finalize();
      return 1;
  } catch (const std::out_of_range& e) {
      if (rank == 0) {
          std::cerr << "Argument value out of range: " << e.what() << std::endl;
      }
      MPI_Finalize();
      return 1;
  }

  if (rank == 0) {
      std::cout << "[CONFIG] Test Case: " << test_case_id
                << ", Tend: " << Tend
                << ", Global NX: " << global_nx
                << ", Global NY: " << global_ny
                << ", Output N: " << output_n
                << ", Output Prefix: " << output_fname_prefix
                << ", Full Log: " << (full_log ? "Yes" : "No");
      if (!h5_file_path.empty()) {
          std::cout << ", HDF5 File: " << h5_file_path;
      }
      std::cout << std::endl;
  }

  // Initialize SWESolver based on test_case_id
  SWESolver* solver = nullptr;
  if (test_case_id == 1 || test_case_id == 2) {
    solver = new SWESolver(test_case_id, global_nx, global_ny,
                           cart_comm, rank, num_procs, dims, coords, neighbors);
  } else if (test_case_id == 3) {
    // For test_case_id 3, we need an HDF5 file.
    // If --h5-file was provided, use it. Otherwise, construct a default name.
    if (h5_file_path.empty()) {
      // Assuming a naming convention like "Data_nx<size+1>_500km.h5" for simplicity
      // and a fixed domain size of 500km.
      // This implies global_nx (from cmd line) is the 'nx' in "Data_nx<nx+1>_500km.h5"
      // If the HDF5 file defines its own dimensions, these global_nx/ny values
      // passed here might be overridden or used as initial guesses.
      h5_file_path = "Data_nx" + std::to_string(global_nx + 1) + "_500km.h5";
      if (rank == 0) {
          std::cout << "[INFO] HDF5 file not explicitly provided. Using default: " << h5_file_path << std::endl;
      }
    }
    double domain_size = 500.0; // Assuming 500km for file-based cases
    solver = new SWESolver(h5_file_path, domain_size, domain_size,
                           cart_comm, rank, num_procs, dims, coords, neighbors);
  } else {
    if (rank == 0) {
        std::cerr << "Invalid test_case_id: " << test_case_id << std::endl;
    }
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 1;
  }

  // Measure execution time
  double start_time = MPI_Wtime();
  std::size_t total_iterations = solver->solve(Tend, full_log, output_n, output_fname_prefix);
  double end_time = MPI_Wtime();

  if (rank == 0) {
      double total_time = end_time - start_time;
      std::cout << "[TIMING] Total simulation time: " << total_time << " seconds" << std::endl;
      std::cout << "[ITERATIONS] Total iterations: " << total_iterations << std::endl;
      if (total_iterations > 0) {
          std::cout << "[TIME_PER_ITER] " << total_time / total_iterations << " seconds/iteration" << std::endl;
      } else {
          std::cout << "[TIME_PER_ITER] Not applicable (0 iterations)" << std::endl;
      }
  }

  delete solver; // Clean up allocated solver object

  MPI_Comm_free(&cart_comm);
  MPI_Finalize();

  return 0;
}