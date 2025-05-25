#include <mpi.h>
#include "swe.hh"

#include <string>
#include <cstddef>
#include <vector> // For MPI_Dims_create
#include <numeric> // For std::accumulate in MPI_Dims_create if needed
#include <cmath>   // For sqrt in MPI_Dims_create if needed
#include <iostream> // For error messages or rank-specific output

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
  // Neighbors: 0=UP (j-1), 1=DOWN (j+1), 2=LEFT (i-1), 3=RIGHT (i+1)
  int neighbors[4];
  MPI_Cart_shift(cart_comm, 0, -1, &neighbors[2], &neighbors[3]); // Shift along x-axis (dim 0) for LEFT/RIGHT
  MPI_Cart_shift(cart_comm, 1, -1, &neighbors[0], &neighbors[1]); // Shift along y-axis (dim 1) for UP/DOWN
                                                                // Note: Y-axis direction might depend on your convention.
                                                                // If UP is decreasing y-index, then shift by -1 for UP.
                                                                // Let's assume:
                                                                // neighbors[0] = UP (y-1)
                                                                // neighbors[1] = DOWN (y+1)
                                                                // neighbors[2] = LEFT (x-1)
                                                                // neighbors[3] = RIGHT (x+1)
                                                                // MPI_Cart_shift for dim 1 (y-dir): disp= -1 -> source=rank_UP, dest=rank_DOWN
                                                                // MPI_Cart_shift for dim 0 (x-dir): disp= -1 -> source=rank_LEFT, dest=rank_RIGHT

  // The project uses (i,j) where i is x and j is y.
  // Let cart_comm dim 0 be x, and dim 1 be y.
  // Shift in y direction (dimension 1 of cart_comm)
  MPI_Cart_shift(cart_comm, 1,  1, &neighbors[0], &neighbors[1]); // UP (coords[1]-1), DOWN (coords[1]+1)
                                                                // source for disp=1 is rank at coords[1]-1 (UP)
                                                                // dest for disp=1 is rank at coords[1]+1 (DOWN)
  // Shift in x direction (dimension 0 of cart_comm)
  MPI_Cart_shift(cart_comm, 0,  1, &neighbors[2], &neighbors[3]); // LEFT (coords[0]-1), RIGHT (coords[0]+1)
                                                                // source for disp=1 is rank at coords[0]-1 (LEFT)
                                                                // dest for disp=1 is rank at coords[0]+1 (RIGHT)


  // Uncomment the option you want to run.

  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = 1;  // Water drops in a box
  // const double Tend = 0.03;     // Simulation time in hours
  const double Tend = 0.05;
  // const std::size_t global_nx = 1000; // Global number of cells
  // const std::size_t global_ny = 1000; // Global number of cells
  const std::size_t global_nx = 100; // Global number of cells
  const std::size_t global_ny = 100; // Global number of cells
  const std::size_t output_n = 10; // Every rank manages output
  const std::string output_fname = "water_drops_mpi_small";
  const bool full_log = false;//(rank == 0); // Only rank 0 does full logging

  // Create SWESolver with MPI info
  // Note: Constructor signature will need to be updated in swe.hh and swe.cc
  SWESolver solver(test_case_id, global_nx, global_ny,
                   cart_comm, rank, num_procs, dims, coords, neighbors);
  solver.solve(Tend, full_log, output_n, output_fname);



  // // Option 2 - Solving analytical (dummy) tsunami example.
  // const int test_case_id = 2;  // Analytical tsunami test case
  // const double Tend = 1.0;     // Simulation time in hours
  // const std::size_t global_nx = 1000; // Global number of cells
  // const std::size_t global_ny = 1000; // Global number of cells
  // const std::size_t output_n = (rank == 0) ? 10 : 0;
  // const std::string output_fname = "analytical_tsunami_mpi";
  // const bool full_log = (rank == 0);

  // SWESolver solver(test_case_id, global_nx, global_ny,
  //                  cart_comm, rank, num_procs, dims, coords, neighbors);
  // solver.solve(Tend, full_log, output_n, output_fname);



  // // Option 3 - Solving tsunami problem with data loaded from file.
  // // This will require parallel HDF5 or a gather/scatter strategy later.
  // // For now, let's focus on Option 1 or 2 for MPI development.
  // if (rank == 0) { // Only rank 0 might handle file names initially
  //   const double Tend_h5 = 0.2;   // Simulation time in hours
  //   const double size_h5 = 500.0; // Size of the domain in km

  // // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
  // const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
  // // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
  // // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
  // // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

  //   const std::size_t output_n_h5 = 0; // Output handling needs to be parallelized or centralized
  //   const std::string output_fname_h5 = "tsunami_mpi";
  //   const bool full_log_h5 = true;

  //   // The HDF5 constructor will also need MPI info and a strategy for data distribution
  //   // SWESolver solver(fname_h5, size_h5, size_h5,
  //   //                  cart_comm, rank, num_procs, dims, coords, neighbors);
  //   // solver.solve(Tend_h5, full_log_h5, output_n_h5, output_fname_h5);
  // }

  MPI_Finalize();

  return 0;
}
