// xdmf_writer.hh
#pragma once // Use pragma once for header guards

#include <vector>
#include <string>
#include <cstddef> // For std::size_t

#include <mpi.h> // Added for MPI
#include <hdf5.h>  // Added for HDF5

/**
 * @brief This class is used to write the XDMF file that references HDF5 files.
 * It is now designed to work with parallel HDF5 outputs where a single HDF5
 * file contains the global data.
 */
class XDMFWriter
{
public:
  /**
   * @brief Constructor for the XDMFWriter class.
   * @param filename_prefix The prefix of the filename to write.
   * @param global_nx The global number of cells in the x direction.
   * @param global_ny The global number of cells in the y direction.
   * @param size_x The total physical size of the domain in the x direction.
   * @param size_y The total physical size of the domain in the y direction.
   */
  XDMFWriter(const std::string& filename_prefix,
             const std::size_t global_nx,
             const std::size_t global_ny,
             const double size_x,
             const double size_y);

  /**
   * @brief Adds a new solution step to the XDMF file.
   *
   * It updates the XDMF file to include the new solution step.
   * The actual HDF5 data writing is handled externally (e.g., by SWESolver).
   *
   * @param t The solution's instant.
   */
  void add_h_timestep(const double t);

  // Public access to time_steps_ for SWESolver to know the next HDF5 file index
  std::vector<double> time_steps_;

  /**
   * @brief Creates the block of global vertices this rank is responsible for.
   * @param rank The MPI rank of the current process.
   * @param num_procs The total number of MPI processes.
   * @param local_block_vertices Output vector to store (x,y) pairs for the local block of global vertices.
   */
  void create_vertices_parallel(int rank, int num_procs, std::vector<double>& local_block_vertices) const;

  /**
   * @brief Creates the block of global cells this rank is responsible for.
   * Vertex indices are global.
   * @param rank The MPI rank of the current process.
   * @param num_procs The total number of MPI processes.
   * @param local_block_cells Output vector to store (v0,v1,v2,v3) indices for the local block of global cells.
   */
  void create_cells_parallel(int rank, int num_procs, std::vector<int>& local_block_cells) const;

  /**
   * @brief Writes the mesh (vertices and cells) to an HDF5 file in parallel.
   * @param comm The MPI communicator.
   * @param rank The MPI rank of the current process.
   * @param num_procs The total number of MPI processes.
   */
  void write_mesh_hdf5_parallel(MPI_Comm comm, int rank, int num_procs) const;

private:
  /**
   * @brief Write the root XDMF file that calls the HDF5 files for the
   * mesh and the solution at the time steps.
   */
  void write_root_xdmf() const;

  /// @brief The prefix of the filename to write.
  const std::string filename_prefix_;
  /// @brief The global number of cells in the x direction.
  const std::size_t global_nx_;
  /// @brief The global number of cells in the y direction.
  const std::size_t global_ny_;
  /// @brief The total physical size of the domain in the x direction.
  const double size_x_;
  /// @brief The total physical size of the domain in the y direction.
  const double size_y_;
};