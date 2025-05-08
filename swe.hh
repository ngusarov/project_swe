#include <cstddef>
#include <vector>
#include <string>

class SWESolver
{
public:
  /**
   * @brief Constructor for the SWESolver class.
   * @warning Not allowed to be used.
   */
  SWESolver() = delete;

  /// Gravity 9.82 * (3.6)^2 * 1000 in[km / hour^2]
  static constexpr double g = 127267.20000000;

  /**
   * @brief Construtor.
   * @param test_case_id It can be 1 (water drops in a box) or 2 (analytical tsunami).
   * @param nx  Number of cells along the x direction.
   * @param ny  Number of cells along the y direction.
   */
  SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny);

  /**
   * @brief Constructor for the SWESolver class.
   *
   * This constructor corresponds to the case in which the initial conditions
   * and topography are read from a HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   * @param size_x  Size in km along the x direction.
   * @param size_y  Size in km along the y direction.
   */
  SWESolver(const std::string &h5_file, const double size_x, const double size_y);

  /**
   * @brief Solve the shallow water equations.
   * @brief Tend Total simulation time.
   * @brief full_log If true, the simulation will log the time step
   * and the time step size at each time step. Otherwise, only
   * the progress of the simulation will be logged.
   * @brief output_n If different from 0, the simulation will write
   * a solution each output_n time steps. E.g., if set to 10,
   * a solution file will be written each 10 steps.
   * @brief fname_prefix If @p output_n is different from 0, the generated
   * files will use this file name prefix.
   */
  void solve(const double Tend,
             const bool full_log = false,
             const std::size_t output_n = 0,
             const std::string &fname_prefix = "test");

private:
  /**
   * @brief Initializes the initial conditions and topography using
   * the provided HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   */
  void init_from_HDF5_file(const std::string &h5_file);

  /**
   * @brief Initializes the initial conditions and topography using
   * a Gaussian function.
   *
   * The water height is initialized with two separated Gaussian peaks.
   * The initial water velocity is set to zero and the topography is set to zero.
   */
  void init_gaussian();

  /**
   * @brief Initializes the initial conditions and topography using
   * a dummy tsunami function.
   */
  void init_dummy_tsunami();

  /**
   * @brief Initializes the initial conditions and topography using
   * a slope function.
   */
  void init_dummy_slope();

  /**
   * @brief Initializes the derivatives dx and dy from the topography.
   */
  void init_dx_dy();

  std::size_t nx_;
  std::size_t ny_;
  double size_x_;
  double size_y_;
  bool reflective_;
  std::vector<double> h0_;
  std::vector<double> h1_;
  std::vector<double> hu0_;
  std::vector<double> hu1_;
  std::vector<double> hv0_;
  std::vector<double> hv1_;
  std::vector<double> z_;
  std::vector<double> zdx_;
  std::vector<double> zdy_;

  /**
   * @brief Accessor for 2D vector elements.
   */
  inline double &at(std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Accessor for 2D vector elements.
   * @note Constant vector version.
   */
  inline const double &at(const std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Updates the water height and velocities using the SWE kernel at a given cell.
   * @param i x index of the cell.
   * @param j y index of the cell.
   * @param dt Time step.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  void compute_kernel(const std::size_t i,
                      const std::size_t j,
                      const double dt,
                      const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const;

  /**
   * @brief Computes the time step size that satisfied the CFL condition.
   *
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   * @param T Current time.
   * @param Tend Final time.
   * @return Compute time step.
   */
  double compute_time_step(const std::vector<double> &h,
                           const std::vector<double> &hu,
                           const std::vector<double> &hv,
                           const double T,
                           const double Tend) const;

  /**
   * @brief Solve one step of the SWE.
   * @param dt The time step size.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  void solve_step(const double dt,
                  const std::vector<double> &h0,
                  const std::vector<double> &hu0,
                  const std::vector<double> &hv0,
                  std::vector<double> &h,
                  std::vector<double> &hu,
                  std::vector<double> &hv) const;

  /**
   * @brief Update boundary conditions.
   * @note This function updates the boundary conditions for the SWE solver.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  void update_bcs(const std::vector<double> &h0,
                  const std::vector<double> &hu0,
                  const std::vector<double> &hv0,
                  std::vector<double> &h,
                  std::vector<double> &hu,
                  std::vector<double> &hv) const;
};