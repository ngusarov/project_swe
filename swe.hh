#include <mpi.h>
#include <cstddef>
#include <vector>
#include <string>

class SWESolver
{
public:
  SWESolver() = delete;

  static constexpr double g = 127267.20000000;
  // Define halo width (for 1-stencil methods like Lax-Friedrichs)
  static const std::size_t halo_width_ = 1;

  SWESolver(const int test_case_id,
            std::size_t global_nx, std::size_t global_ny,
            MPI_Comm cart_comm, int rank, int num_procs,
            const int* dims, const int* coords, const int* neighbors);

  SWESolver(const std::string &h5_file, const double size_x, const double size_y,
            MPI_Comm cart_comm, int rank, int num_procs,
            const int* dims, const int* coords, const int* neighbors);

  void solve(const double Tend,
             const bool full_log = false,
             const std::size_t output_n = 0,
             const std::string &fname_prefix = "test");

  static void write_local_h5_data(const std::string& filename,
                                const std::string& dataset_name,
                                const std::vector<double>& local_owned_data,
                                std::size_t local_dim_nx, // Number of cells in x for this local data
                                std::size_t local_dim_ny  // Number of cells in y for this local data
                                );

private:
  void init_from_HDF5_file(const std::string &h5_file);
  void init_gaussian();
  void init_dummy_tsunami();
  void init_dummy_slope(); // This was unused, will remain so for now
  void init_dx_dy();
  void exchange_halos_for_field(std::vector<double>& data_field);
  void print_debug_perimeter_and_halos(const std::vector<double>& data_field, const std::string& label) const;

  // MPI-related members
  MPI_Comm cart_comm_;
  int rank_;
  int num_procs_;
  int dims_[2];      // Dimensions of the process grid {px, py}
  int coords_[2];    // Coordinates of this process in the grid {cx, cy}
  int neighbors_[4]; // Ranks: [UP, DOWN, LEFT, RIGHT] (Indices: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)

  // Grid and domain properties
  std::size_t global_nx_; // Global number of cells in x
  std::size_t global_ny_; // Global number of cells in y
  std::size_t nx_;        // Local number of OWNED cells in x
  std::size_t ny_;        // Local number of OWNED cells in y

  std::size_t nx_padded_; // Local dimensions INCLUDING halo cells (nx_ + 2*halo_width_)
  std::size_t ny_padded_; // Local dimensions INCLUDING halo cells (ny_ + 2*halo_width_)

  // Global physical size of domain
  double size_x_;
  double size_y_;

  bool reflective_;

  // Data vectors will store local subgrid + halo regions
  std::vector<double> h0_;
  std::vector<double> h1_;
  std::vector<double> hu0_;
  std::vector<double> hu1_;
  std::vector<double> hv0_;
  std::vector<double> hv1_;
  std::vector<double> z_;
  std::vector<double> zdx_;
  std::vector<double> zdy_;

  // Global starting indices for this process's owned subgrid
  std::size_t G_start_i_;
  std::size_t G_start_j_;


  /**
   * @brief Accessor for 2D vector elements using PADDED indices.
   * (padded_i, padded_j) are indices in the local array that includes halos.
   * E.g., owned cell (0,0) is at (halo_width_, halo_width_) in padded array.
   */
  inline double &at(std::vector<double> &vec, const std::size_t padded_i, const std::size_t padded_j) const
  {
    return vec[padded_j * nx_padded_ + padded_i];
  }

  inline const double &at(const std::vector<double> &vec, const std::size_t padded_i, const std::size_t padded_j) const
  {
    return vec[padded_j * nx_padded_ + padded_i];
  }

  // Forward declarations for methods to be modified/implemented later
  void exchange_halos_generic(std::vector<double>& data_field_0,
                              std::vector<double>& data_field_1,
                              std::vector<double>& data_field_2); // Example, might need separate for h, hu, hv
  void exchange_halos_h();
  void exchange_halos_hu();
  void exchange_halos_hv();
  void exchange_halos_z();


  void compute_kernel(const std::size_t padded_i, // Kernel will operate using padded indices
                      const std::size_t padded_j,
                      const double dt,
                      const std::vector<double> &h0_local, // Pass local refs for clarity
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
                  std::vector<double> &h0_local, // Pass local refs
                  std::vector<double> &hu0_local,
                  std::vector<double> &hv0_local,
                  std::vector<double> &h_local,
                  std::vector<double> &hu_local,
                  std::vector<double> &hv_local); // Removed const as h0, hu0, hv0 might be swapped or directly modified via BCs in future

  void update_bcs(std::vector<double> &h_target, std::vector<double> &hu_target, std::vector<double> &hv_target);
};