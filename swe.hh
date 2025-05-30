// swe.hh
#pragma once

#define H5_HAVE_PARALLEL_H5

#include <mpi.h>

#include <vector>
#include <string>
#include <cstddef>
#include <memory>

// HDF5 includes for parallel I/O
#include <hdf5.h>
#include <hdf5_hl.h>

// Forward declaration of XDMFWriter
class XDMFWriter;

class SWESolver
{
public:
  SWESolver() = delete;
  ~SWESolver();

  static constexpr double g = 127267.20000000;
  static const std::size_t halo_width_ = 1;

  SWESolver(const int test_case_id,
            std::size_t global_nx, std::size_t global_ny,
            MPI_Comm cart_comm, int rank, int num_procs,
            const int* dims, const int* coords, const int* neighbors);

  SWESolver(const std::string &h5_file, const double size_x, const double size_y,
            MPI_Comm cart_comm, int rank, int num_procs,
            const int* dims, const int* coords, const int* neighbors);

  // Modified to return total_iterations
  std::size_t solve(const double Tend,
                    const bool full_log = false,
                    const std::size_t output_n_param = 0,
                    const std::string &fname_prefix = "test");

private:
  // ... (rest of the class definition remains the same)
  void init_from_HDF5_file(const std::string &h5_file); // This method is declared but not defined/used
                                                          // Its functionality is largely absorbed into the HDF5 constructor
  void init_gaussian();
  void init_dummy_tsunami();
  void init_dx_dy();
  void exchange_halos_for_field(std::vector<double>& data_field);
  void print_debug_perimeter_and_halos(const std::vector<double>& data_field, const std::string& label) const;

  void write_field_to_hdf5_parallel(const std::string& full_h5_filepath,
                                    const std::string& dataset_name,
                                    const std::vector<double>& data_field_padded);
  void write_topography_to_hdf5_parallel(const std::vector<double>& topography_padded);


  // MPI-related members
  MPI_Comm cart_comm_;
  int rank_;
  int num_procs_;
  int dims_[2];
  int coords_[2];
  int neighbors_[4];

  // Grid and domain properties
  std::size_t global_nx_;
  std::size_t global_ny_;
  std::size_t nx_;
  std::size_t ny_;

  std::size_t nx_padded_;
  std::size_t ny_padded_;

  double size_x_;
  double size_y_;

  bool reflective_;

  // Data vectors
  std::vector<double> h0_;
  std::vector<double> h1_;
  std::vector<double> hu0_;
  std::vector<double> hu1_;
  std::vector<double> hv0_;
  std::vector<double> hv1_;
  std::vector<double> z_;
  std::vector<double> zdx_;
  std::vector<double> zdy_;

  std::size_t G_start_i_;
  std::size_t G_start_j_;

  // Output related members
  std::string filename_prefix_;
  std::unique_ptr<XDMFWriter> writer_ptr_;
  std::size_t current_output_idx_;

  inline double &at(std::vector<double> &vec, const std::size_t padded_i, const std::size_t padded_j) const
  {
    return vec[padded_j * nx_padded_ + padded_i];
  }

  inline const double &at(const std::vector<double> &vec, const std::size_t padded_i, const std::size_t padded_j) const
  {
    return vec[padded_j * nx_padded_ + padded_i];
  }

  void compute_kernel(const std::size_t padded_i,
                      const std::size_t padded_j,
                      const double dt,
                      const std::vector<double> &h0_local,
                      const std::vector<double> &hu0_local,
                      const std::vector<double> &hv0_local,
                      std::vector<double> &h_local,
                      std::vector<double> &hu_local,
                      std::vector<double> &hv_local) const;

  double compute_time_step(const std::vector<double> &h_local,
                           const std::vector<double> &hu_local,
                           const std::vector<double> &hv_local,
                           const double T,
                           const double Tend) const;

  void solve_step(const double dt,
                  std::vector<double> &h0_local,
                  std::vector<double> &hu0_local,
                  std::vector<double> &hv0_local,
                  std::vector<double> &h_local,
                  std::vector<double> &hu_local,
                  std::vector<double> &hv_local);

  void update_bcs(std::vector<double> &h_target, std::vector<double> &hu_target, std::vector<double> &hv_target);
};