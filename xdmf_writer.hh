// xdmf_writer.hh
#pragma once // Use pragma once for header guards

#include <vector>
#include <string>
#include <cstddef> // For std::size_t

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
  // This is a pragmatic choice to avoid passing the index explicitly.
  // A cleaner design might involve the XDMFWriter returning the index or
  // SWESolver managing the index and passing it to add_h_timestep.
  // For this refactor, direct access is simpler.
  std::vector<double> time_steps_;


  void create_vertices(std::vector<double>& vertices) const;
  void create_cells(std::vector<int>& cells) const;
  void write_mesh_hdf5() const;

private:
  /**
   * @brief Write the root XDMF file that calls the HDF5 files for the
   * mesh and the solution at the time steps.
   *
   * This function may be called after each time step to update the XDMF file
   * for including the last generated solution.
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

