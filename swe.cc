// swe.cc

#include <mpi.h> // <--- MOVED TO TOP

#include "swe.hh"
#include "xdmf_writer.hh" // Now include XDMFWriter header

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>
#include <algorithm> // For std::copy, std::fill
#include <numeric>   // For std::iota (if needed for debugging)

const std::size_t SWESolver::halo_width_;

namespace
{

void
read_2d_array_from_DF5(const std::string &filename,
                       const std::string &dataset_name,
                       std::vector<double> &data,
                       std::size_t &nx,
                       std::size_t &ny)
{
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t dims[2];
  herr_t status;

  // Open the HDF5 file
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return;
  }

  // Open the dataset
  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error opening dataset: " << dataset_name << std::endl;
    H5Fclose(file_id);
    return;
  }

  // Get the dataspace
  dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
  {
    std::cerr << "Error getting dataspace" << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }

  // Get the dimensions of the dataset
  status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  if (status < 0)
  {
    std::cerr << "Error getting dimensions" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    data.clear();
    return;
  }
  nx = dims[0];
  ny = dims[1];

  // Resize the data vector
  data.resize(nx * ny);

  // Read the data
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0)
  {
    std::cerr << "Error reading data" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    data.clear();
    return;
  }

  // Close resources
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
}

} // namespace

// --- New Parallel HDF5 Write Functions ---

SWESolver::~SWESolver() = default; // <--- ADD THIS LINE


void SWESolver::write_field_to_hdf5_parallel(const std::string& full_h5_filepath,
                                             const std::string& dataset_name,
                                             const std::vector<double>& data_field_padded)
{
    // Create file access property list for MPI-IO
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, cart_comm_, MPI_INFO_NULL);

    // Create or open the HDF5 file collectively
    // Use H5F_ACC_TRUNC for creating a new file (e.g., for each time step's 'h' data)
    hid_t file_id = H5Fcreate(full_h5_filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    if (file_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating HDF5 file: " << full_h5_filepath << std::endl;
        H5Pclose(fapl_id);
        return;
    }

    // Define global dataset dimensions
    // HDF5 stores row-major (y then x), so global_ny_ is the number of rows, global_nx_ is number of columns
    hsize_t global_dims[1] = {global_ny_ * global_nx_}; // Flattened 1D representation for cell-centered data
    hid_t filespace_id = H5Screate_simple(1, global_dims, NULL);

    // Define memory dataspace for the local owned data
    hsize_t local_dims[1] = {ny_ * nx_}; // Local owned dimensions (flattened)
    hid_t memspace_id = H5Screate_simple(1, local_dims, NULL);

    // Define hyperslab in the file dataspace for this process's owned region
    // The offset needs to be calculated in the flattened 1D global array.
    // G_start_j_ * global_nx_ + G_start_i_ gives the 1D index of the first owned cell.
    hsize_t offset[1] = {G_start_j_ * global_nx_ + G_start_i_};
    hsize_t count[1] = {ny_ * nx_}; // Number of elements to select

    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Create the dataset collectively
    hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_IEEE_F64LE, filespace_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating dataset: " << dataset_name << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Fclose(file_id);
        H5Pclose(fapl_id);
        return;
    }

    // Create property list for collective I/O
    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    // Extract the owned data from the padded field into a contiguous buffer for writing
    std::vector<double> local_owned_data(nx_ * ny_);
    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
        for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
            local_owned_data[j_local * nx_ + i_local] = at(data_field_padded, i_local + halo_width_, j_local + halo_width_);
        }
    }

    // Write the data collectively
    herr_t status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, memspace_id, filespace_id, xfer_plist_id, local_owned_data.data());
    if (status < 0) {
        std::cerr << "Rank " << rank_ << ": Error writing data for dataset: " << dataset_name << std::endl;
    }

    // Close resources
    H5Pclose(xfer_plist_id);
    H5Dclose(dataset_id);
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Fclose(file_id);
    H5Pclose(fapl_id);

    if (rank_ == 0) {
        std::cout << "Successfully wrote HDF5 file: " << full_h5_filepath << ", dataset: " << dataset_name << std::endl;
    }
}

void SWESolver::write_mesh_to_hdf5_parallel()
{
    const std::string h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_mesh.h5";

    // Create file access property list for MPI-IO
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, cart_comm_, MPI_INFO_NULL);

    // Create the HDF5 file collectively
    hid_t file_id = H5Fcreate(h5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    if (file_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating HDF5 mesh file: " << h5_filename << std::endl;
        H5Pclose(fapl_id);
        return;
    }

    // --- Vertices ---
    // Each process generates its local portion of vertices.
    // The global grid has (global_nx_ + 1) * (global_ny_ + 1) vertices.
    // Each process is responsible for vertices from its G_start_i_ to G_start_i_ + nx_
    // and G_start_j_ to G_start_j_ + ny_.
    // The last process in a dimension also includes the final edge.

    std::vector<double> local_vertices_data;

    const double dx_global = size_x_ / global_nx_;
    const double dy_global = size_y_ / global_ny_;

    // Determine local vertex range (global indices)
    std::size_t local_vert_start_i = G_start_i_;
    std::size_t local_vert_end_i = G_start_i_ + nx_; // Inclusive, so nx_ + 1 vertices in x
    std::size_t local_vert_start_j = G_start_j_;
    std::size_t local_vert_end_j = G_start_j_ + ny_; // Inclusive, so ny_ + 1 vertices in y

    // Adjust for the last row/column if this is the last process in a dimension
    // This ensures that the rightmost and bottommost global vertices are written exactly once.
    if (coords_[0] == dims_[0] - 1) local_vert_end_i = global_nx_;
    if (coords_[1] == dims_[1] - 1) local_vert_end_j = global_ny_;

    // Calculate local dimensions of the vertex array (number of vertices in x and y for this process)
    std::size_t num_local_verts_x = local_vert_end_i - local_vert_start_i + 1;
    std::size_t num_local_verts_y = local_vert_end_j - local_vert_start_j + 1;

    local_vertices_data.reserve(num_local_verts_x * num_local_verts_y * 2);

    for (std::size_t j_global_vert = local_vert_start_j; j_global_vert <= local_vert_end_j; ++j_global_vert)
    {
        for (std::size_t i_global_vert = local_vert_start_i; i_global_vert <= local_vert_end_i; ++i_global_vert)
        {
            local_vertices_data.push_back(i_global_vert * dx_global);
            local_vertices_data.push_back(j_global_vert * dy_global);
        }
    }

    // Global dimensions for vertices dataset: (global_ny+1) * (global_nx+1) by 2
    // HDF5 stores row-major, so total_rows = (global_ny+1)*(global_nx+1), total_cols = 2
    hsize_t global_vert_dims[2] = {(global_ny_ + 1) * (global_nx_ + 1), 2};
    hid_t vert_filespace_id = H5Screate_simple(2, global_vert_dims, NULL);

    // Local dimensions for vertices memory space: (num_local_verts_y * num_local_verts_x) by 2
    hsize_t local_vert_dims[2] = {num_local_verts_y * num_local_verts_x, 2};
    hid_t vert_memspace_id = H5Screate_simple(2, local_vert_dims, NULL);

    // Hyperslab for vertices: needs to map local_vertices_data to global_vert_dims
    // The offset for each process's vertices needs to be calculated in this flattened 1D index.
    // First vertex in global array for this process: (global_j_vertex_start * (global_nx_ + 1) + global_i_vertex_start) * 2
    hsize_t vert_offset[2] = {(local_vert_start_j * (global_nx_ + 1) + local_vert_start_i), 0};
    hsize_t vert_count[2] = {num_local_verts_y * num_local_verts_x, 2}; // Number of elements to select

    H5Sselect_hyperslab(vert_filespace_id, H5S_SELECT_SET, vert_offset, NULL, vert_count, NULL);

    hid_t vert_dataset_id = H5Dcreate2(file_id, "/vertices", H5T_IEEE_F64LE, vert_filespace_id,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (vert_dataset_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating vertices dataset" << std::endl;
        H5Sclose(vert_memspace_id); H5Sclose(vert_filespace_id); H5Fclose(file_id); H5Pclose(fapl_id);
        return;
    }

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    herr_t status = H5Dwrite(vert_dataset_id, H5T_IEEE_F64LE, vert_memspace_id, vert_filespace_id,
                             xfer_plist_id, local_vertices_data.data());
    if (status < 0) {
        std::cerr << "Rank " << rank_ << ": Error writing vertices data" << std::endl;
    }
    H5Pclose(xfer_plist_id);
    H5Dclose(vert_dataset_id);
    H5Sclose(vert_memspace_id);
    H5Sclose(vert_filespace_id);


    // --- Cells ---
    // Each process generates its local portion of cells.
    // Cells are (i,j) where i in [0,nx_-1], j in [0,ny_-1] local.
    // Global cells are (G_start_i_ + i_local, G_start_j_ + j_local).
    // Each cell is 4 vertex indices.
    std::vector<int> local_cells_data;
    local_cells_data.reserve(nx_ * ny_ * 4);

    for (std::size_t j_local = 0; j_local < ny_; ++j_local)
    {
        for (std::size_t i_local = 0; i_local < nx_; ++i_local)
        {
            // Global indices of the bottom-left vertex of the current cell
            std::size_t global_i_bl = G_start_i_ + i_local;
            std::size_t global_j_bl = G_start_j_ + j_local;

            // Calculate global 1D indices of the four vertices
            // Vertex (i,j) is at 1D index j * (global_nx_ + 1) + i
            local_cells_data.push_back(global_j_bl * (global_nx_ + 1) + global_i_bl);             // BL
            local_cells_data.push_back(global_j_bl * (global_nx_ + 1) + global_i_bl + 1);         // BR
            local_cells_data.push_back((global_j_bl + 1) * (global_nx_ + 1) + global_i_bl + 1);   // TR
            local_cells_data.push_back((global_j_bl + 1) * (global_nx_ + 1) + global_i_bl);       // TL
        }
    }

    // Global dimensions for cells dataset: (global_nx * global_ny) by 4
    hsize_t global_cell_dims[2] = {global_nx_ * global_ny_, 4};
    hid_t cell_filespace_id = H5Screate_simple(2, global_cell_dims, NULL);

    // Local dimensions for cells memory space: (nx_ * ny_) by 4
    hsize_t local_cell_dims[2] = {nx_ * ny_, 4};
    hid_t cell_memspace_id = H5Screate_simple(2, local_cell_dims, NULL);

    // Hyperslab for cells: needs to map local_cells_data to global_cell_dims
    // First cell in global array for this process: (G_start_j_ * global_nx_ + G_start_i_)
    hsize_t cell_offset[2] = {(G_start_j_ * global_nx_ + G_start_i_), 0};
    hsize_t cell_count[2] = {nx_ * ny_, 4};

    H5Sselect_hyperslab(cell_filespace_id, H5S_SELECT_SET, cell_offset, NULL, cell_count, NULL);

    hid_t cell_dataset_id = H5Dcreate2(file_id, "/cells", H5T_STD_I32LE, cell_filespace_id,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (cell_dataset_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating cells dataset" << std::endl;
        H5Sclose(cell_memspace_id); H5Sclose(cell_filespace_id); H5Fclose(file_id); H5Pclose(fapl_id);
        return;
    }

    xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    status = H5Dwrite(cell_dataset_id, H5T_STD_I32LE, cell_memspace_id, cell_filespace_id,
                      xfer_plist_id, local_cells_data.data());
    if (status < 0) {
        std::cerr << "Rank " << rank_ << ": Error writing cells data" << std::endl;
    }
    H5Pclose(xfer_plist_id);
    H5Dclose(cell_dataset_id);
    H5Sclose(cell_memspace_id);
    H5Sclose(cell_filespace_id);


    // Close the file and FAPL
    H5Fclose(file_id);
    H5Pclose(fapl_id);

    if (rank_ == 0) {
        std::cout << "Successfully wrote HDF5 mesh file: " << h5_filename << std::endl;
    }
}

void SWESolver::write_topography_to_hdf5_parallel(const std::vector<double>& topography_padded)
{
    // Construct HDF5 filename within the directory
    const std::string h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_topography.h5";
    write_field_to_hdf5_parallel(h5_filename, "/topography", topography_padded);
}


/////////////////////////////
// New constructor for analytical test cases with MPI
// --- CONSTRUCTORS ---
SWESolver::SWESolver(const int test_case_id,
                     std::size_t global_nx_param, std::size_t global_ny_param,
                     MPI_Comm cart_comm_param, int rank_param, int num_procs_param,
                     const int* dims_param, const int* coords_param, const int* neighbors_param) :
  cart_comm_(cart_comm_param), rank_(rank_param), num_procs_(num_procs_param),
  global_nx_(global_nx_param), global_ny_(global_ny_param),
  size_x_(500.0), size_y_(500.0) // Default global physical sizes
{
  std::copy(dims_param, dims_param + 2, dims_);
  std::copy(coords_param, coords_param + 2, coords_);
  std::copy(neighbors_param, neighbors_param + 4, neighbors_);

  // Calculate local OWNED dimensions (nx_, ny_)
  std::size_t base_nx = global_nx_ / dims_[0];
  std::size_t remainder_nx = global_nx_ % dims_[0];
  nx_ = base_nx + (coords_[0] < remainder_nx ? 1 : 0);

  std::size_t base_ny = global_ny_ / dims_[1];
  std::size_t remainder_ny = global_ny_ % dims_[1];
  ny_ = base_ny + (coords_[1] < remainder_ny ? 1 : 0);

  // Calculate padded dimensions
  nx_padded_ = nx_ + 2 * halo_width_;
  ny_padded_ = ny_ + 2 * halo_width_;

  // Calculate global starting indices for this process's owned region
  G_start_i_ = 0;
  for (int px = 0; px < coords_[0]; ++px) {
    G_start_i_ += (global_nx_ / dims_[0]) + (px < remainder_nx ? 1 : 0);
  }
  G_start_j_ = 0;
  for (int py = 0; py < coords_[1]; ++py) {
    G_start_j_ += (global_ny_ / dims_[1]) + (py < remainder_ny ? 1 : 0);
  }

  if (rank_ == 0) {
      std::cout << "MPI Initialized: " << num_procs_ << " processes. Grid: "
                << dims_[0] << "x" << dims_[1] << std::endl;
      std::cout << "Global domain: " << global_nx_ << "x" << global_ny_ << std::endl;
  }
  printf("Rank %d (%d,%d): local_owned %lux%lu, local_padded %lux%lu, G_start (%lu,%lu)\n",
         rank_, coords_[0], coords_[1], nx_, ny_, nx_padded_, ny_padded_, G_start_i_, G_start_j_);
  fflush(stdout); // Ensure output is flushed


  // Resize data vectors to padded local dimensions
  h0_.resize(nx_padded_ * ny_padded_, 0.0); // Initialize with 0.0
  h1_.resize(nx_padded_ * ny_padded_, 0.0);
  hu0_.resize(nx_padded_ * ny_padded_, 0.0);
  hu1_.resize(nx_padded_ * ny_padded_, 0.0);
  hv0_.resize(nx_padded_ * ny_padded_, 0.0);
  hv1_.resize(nx_padded_ * ny_padded_, 0.0);
  z_.resize(nx_padded_ * ny_padded_, 0.0);
  zdx_.resize(nx_padded_ * ny_padded_, 0.0);
  zdy_.resize(nx_padded_ * ny_padded_, 0.0);

  assert(test_case_id == 1 || test_case_id == 2);
  if (test_case_id == 1) {
    this->reflective_ = true;
    this->init_gaussian();
  } else if (test_case_id == 2) {
    this->reflective_ = false;
    this->init_dummy_tsunami();
  }

  exchange_halos_for_field(z_); // Exchange halos for z before computing derivatives
  // Common initialization after specific test case setup
  this->init_dx_dy(); // Must be called after z_ is initialized for the local (owned) part
}

SWESolver::SWESolver(const std::string &h5_file_param, const double size_x_param, const double size_y_param,
                     MPI_Comm cart_comm_param, int rank_param, int num_procs_param,
                     const int* dims_param, const int* coords_param, const int* neighbors_param) :
  cart_comm_(cart_comm_param), rank_(rank_param), num_procs_(num_procs_param),
  size_x_(size_x_param), size_y_(size_y_param), reflective_(false)
{
  std::copy(dims_param, dims_param + 2, dims_);
  std::copy(coords_param, coords_param + 2, coords_);
  std::copy(neighbors_param, neighbors_param + 4, neighbors_);

  // For HDF5, global_nx_ and global_ny_ are determined by rank 0 reading the file.
  // Then these values are broadcasted.
  std::vector<double> h0_global, hu0_global, hv0_global, z_global;
  std::size_t file_global_nx = 0;
  std::size_t file_global_ny = 0;

  if (rank_ == 0) {
    std::cout << "HDF5 Constructor: Rank 0 reading global data from " << h5_file_param << std::endl;
    // Temporary vectors to hold global data read by rank 0
    read_2d_array_from_DF5(h5_file_param, "h0", h0_global, file_global_nx, file_global_ny);
    global_nx_ = file_global_nx; // Update member from file read
    global_ny_ = file_global_ny;
    // Assuming hu0, hv0, topography datasets have the same dimensions
    read_2d_array_from_DF5(h5_file_param, "hu0", hu0_global, file_global_nx, file_global_ny);
    read_2d_array_from_DF5(h5_file_param, "hv0", hv0_global, file_global_nx, file_global_ny);
    read_2d_array_from_DF5(h5_file_param, "topography", z_global, file_global_nx, file_global_ny);
  }

  // Broadcast global dimensions from rank 0 to all other processes
  MPI_Bcast(&global_nx_, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_); // Or MPI_SIZE_T if available and appropriate
  MPI_Bcast(&global_ny_, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_);


  // All processes now know global_nx_ and global_ny_
  // Calculate local OWNED dimensions (nx_, ny_)
  std::size_t base_nx = global_nx_ / dims_[0];
  std::size_t remainder_nx = global_nx_ % dims_[0];
  nx_ = base_nx + (coords_[0] < remainder_nx ? 1 : 0);

  std::size_t base_ny = global_ny_ / dims_[1];
  std::size_t remainder_ny = global_ny_ % dims_[1];
  ny_ = base_ny + (coords_[1] < remainder_ny ? 1 : 0);

  // Calculate padded dimensions
  nx_padded_ = nx_ + 2 * halo_width_;
  ny_padded_ = ny_ + 2 * halo_width_;

  // Calculate global starting indices for this process's owned region
  G_start_i_ = 0;
  for (int px = 0; px < coords_[0]; ++px) {
    G_start_i_ += (global_nx_ / dims_[0]) + (px < remainder_nx ? 1 : 0);
  }
  G_start_j_ = 0;
  for (int py = 0; py < coords_[1]; ++py) {
    G_start_j_ += (global_ny_ / dims_[1]) + (py < remainder_ny ? 1 : 0);
  }
  printf("Rank %d (%d,%d) HDF5: local_owned %lux%lu, local_padded %lux%lu, G_start (%lu,%lu)\n",
         rank_, coords_[0], coords_[1], nx_, ny_, nx_padded_, ny_padded_, G_start_i_, G_start_j_);

  fflush(stdout); // Ensure output is flushed


  // Resize local data vectors to padded local dimensions
  h0_.resize(nx_padded_ * ny_padded_, 0.0);
  h1_.resize(nx_padded_ * ny_padded_, 0.0);
  hu0_.resize(nx_padded_ * ny_padded_, 0.0);
  hu1_.resize(nx_padded_ * ny_padded_, 0.0);
  hv0_.resize(nx_padded_ * ny_padded_, 0.0);
  hv1_.resize(nx_padded_ * ny_padded_, 0.0);
  z_.resize(nx_padded_ * ny_padded_, 0.0);
  zdx_.resize(nx_padded_ * ny_padded_, 0.0);
  zdy_.resize(nx_padded_ * ny_padded_, 0.0);

  // Distribute data from rank 0's global vectors to each process's local (owned) region
  // This is a simplified scatter. A more robust MPI_Scatterv or custom loop would be better.
  // For now, each process (including rank 0) copies its relevant slice.
  // Rank 0 needs h0_global, hu0_global etc. Other ranks will use these if they are rank 0.
  // This is a placeholder for proper data distribution.
  // The following copy logic is conceptual for now.
  // Proper MPI_Scatterv or point-to-point would be required for true distribution from rank 0.
  // Here, we'll just have each process populate its part IF it were rank 0 and had the global data.
  // For ranks != 0, their local data remains 0.0 unless they are rank 0.
  // THIS IS A MAJOR SIMPLIFICATION AND PLACEHOLDER FOR HDF5 DATA DISTRIBUTION.
  if (rank_ == 0) {
      for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
          for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
              std::size_t g_i = G_start_i_ + i_local;
              std::size_t g_j = G_start_j_ + j_local;
              std::size_t padded_i = i_local + halo_width_;
              std::size_t padded_j = j_local + halo_width_;

              if (g_i < global_nx_ && g_j < global_ny_) { // Check bounds for safety
                  at(h0_, padded_i, padded_j) = h0_global[g_j * global_nx_ + g_i];
                  at(hu0_, padded_i, padded_j) = hu0_global[g_j * global_nx_ + g_i];
                  at(hv0_, padded_i, padded_j) = hv0_global[g_j * global_nx_ + g_i];
                  at(z_, padded_i, padded_j) = z_global[g_j * global_nx_ + g_i];
              }
          }
      }
  } else {
      // Other ranks currently don't receive data from HDF5 in this simplified model.
      // Their fields h0, hu0, hv0, z remain 0.0 in the owned region.
      // This needs proper MPI communication (e.g. Scatterv)
      if (rank_ == 1 && num_procs_ > 1) { // Example: let rank 1 print a warning once
          std::cout << "Warning: HDF5 data only read by rank 0. Other ranks have zeroed initial fields." << std::endl;
          std::cout << "         Proper MPI data distribution from HDF5 is required." << std::endl;
      }
  }

  exchange_halos_for_field(z_); // Exchange halos for z before computing derivatives
  this->init_dx_dy(); // Must be called after z_ is initialized for the local (owned) part
}

///////////////////////////----------------------------------------------

//*
void SWESolver::init_gaussian() {
  // Initialize hu0_, hv0_ to 0.0 in the PADDED local grid
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);
  // h0_ and z_ will be set next for owned cells. Halos remain 0.0 for now.

  const double dx_global = size_x_ / global_nx_; // Cell size based on GLOBAL grid
  const double dy_global = size_y_ / global_ny_;

  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  // Iterate over OWNED local cells and initialize them
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      // Global coordinates of the current local cell's center
      const double x_global = dx_global * (static_cast<double>(G_start_i_ + i_local) + 0.5);
      const double y_global = dy_global * (static_cast<double>(G_start_j_ + j_local) + 0.5);

      const double gauss_0 = 10.0 * std::exp(-((x_global - x0_0) * (x_global - x0_0) + (y_global - y0_0) * (y_global - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x_global - x0_1) * (x_global - x0_1) + (y_global - y0_1) * (y_global - y0_1)) / 1000.0);

      // Padded indices for storing in the local grid
      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      at(h0_, pi, pj) = 10.0 + gauss_0 + gauss_1;
      at(z_, pi, pj) = 0.0; // Topography is flat for this case
    }
  }
  // Note: h1_, hu1_, hv1_ are already zeroed from resize. zdx_, zdy_ will be set by init_dx_dy().

    // Debug: Print some initial values from owned cells for specific ranks
  if (rank_ < 2 || rank_ == num_procs_ - 1) { // Example: print for rank 0, 1 and the last rank
    printf("[Rank %d (%d,%d) init_gaussian] Checking owned cells (h0, z):\n", rank_, coords_[0], coords_[1]);

    // Top-left owned cell (local index 0,0)
    std::size_t li = 0, lj = 0;
    std::size_t pi = li + halo_width_;
    std::size_t pj = lj + halo_width_;
    if (nx_ > 0 && ny_ > 0) { // Ensure there's at least one cell
        printf("  Local (0,0) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }

    // Center-ish owned cell
    if (nx_ > 0 && ny_ > 0) {
        li = nx_ / 2; lj = ny_ / 2;
        pi = li + halo_width_; pj = lj + halo_width_;
        printf("  Local (%lu,%lu) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }

    // Bottom-right owned cell
    if (nx_ > 0 && ny_ > 0) {
        li = nx_ - 1; lj = ny_ - 1;
        pi = li + halo_width_; pj = lj + halo_width_;
        printf("  Local (%lu,%lu) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }
    fflush(stdout);
  }
  // Optional: MPI_Barrier(cart_comm_);
}
//*/

/*
void SWESolver::init_gaussian() {
  // Initialize hu0_, hv0_ to 0.0 in the PADDED local grid
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);
  // Also ensure h1, hu1, hv1 are zeroed if not already
  std::fill(h1_.begin(), h1_.end(), 0.0);
  std::fill(hu1_.begin(), hu1_.end(), 0.0);
  std::fill(hv1_.begin(), hv1_.end(), 0.0);


  // Iterate over OWNED local cells and initialize them
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      std::size_t g_i = G_start_i_ + i_local;
      std::size_t g_j = G_start_j_ + j_local;

      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      // Debug value: Rank * 10000 + global_j * 100 + global_i
      // This makes it easy to see where the data came from after exchange.
      at(h0_, pi, pj) = static_cast<double>(rank_ * 10000 + g_j * 100 + g_i);
      // Give z_ a slightly different pattern for distinction
      at(z_, pi, pj) = static_cast<double>(rank_ * 10000 + g_j * 100 + g_i + 0.5);
    }
  }
  // Halo regions of h0_ and z_ are still 0.0 (or whatever they were resized with from constructor)
  // unless physical BCs in update_bcs set them before the first exchange.
}
*/

void SWESolver::init_dummy_tsunami() {
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);
  // h1_, hu1_, hv1_ were already zeroed during resize in constructor

  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;

  const double x0_0 = 0.6 * size_x_;
  const double y0_0 = 0.6 * size_y_;
  const double x0_1 = 0.4 * size_x_;
  const double y0_1 = 0.4 * size_y_;
  const double x0_2 = 0.7 * size_x_;
  const double y0_2 = 0.3 * size_y_;

  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const double x_global = dx_global * (static_cast<double>(G_start_i_ + i_local) + 0.5);
      const double y_global = dy_global * (static_cast<double>(G_start_j_ + j_local) + 0.5);

      const double gauss_0 = 2.0 * std::exp(-((x_global - x0_0) * (x_global - x0_0) + (y_global - y0_0) * (y_global - y0_0)) / 3000.0);
      const double gauss_1 = 3.0 * std::exp(-((x_global - x0_1) * (x_global - x0_1) + (y_global - y0_1) * (y_global - y0_1)) / 10000.0);
      const double gauss_2 = 5.0 * std::exp(-((x_global - x0_2) * (x_global - x0_2) + (y_global - y0_2) * (y_global - y0_2)) / 100.0);

      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      const double z_val = -1.0 + gauss_0 + gauss_1;
      at(z_, pi, pj) = z_val;

      double h0_val = z_val < 0.0 ? -z_val + gauss_2 : 0.00001;
      at(h0_, pi, pj) = h0_val;
    }
  }

    // Debug: Print some initial values from owned cells for specific ranks
  if (rank_ < 2 || rank_ == num_procs_ - 1) { // Example: print for rank 0, 1 and the last rank
    printf("[Rank %d (%d,%d) init_dummy_tsunami] Checking owned cells (h0, z):\n", rank_, coords_[0], coords_[1]);

    // Top-left owned cell (local index 0,0)
    std::size_t li = 0, lj = 0;
    std::size_t pi = li + halo_width_;
    std::size_t pj = lj + halo_width_;
     if (nx_ > 0 && ny_ > 0) {
        printf("  Local (0,0) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }

    // Center-ish owned cell
    if (nx_ > 0 && ny_ > 0) {
        li = nx_ / 2; lj = ny_ / 2;
        pi = li + halo_width_; pj = lj + halo_width_;
        printf("  Local (%lu,%lu) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               nx_/2, ny_/2, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }

    // Bottom-right owned cell
    if (nx_ > 0 && ny_ > 0) {
        li = nx_ - 1; lj = ny_ - 1;
        pi = li + halo_width_; pj = lj + halo_width_;
        printf("  Local (%lu,%lu) [Padded (%lu,%lu)]: h0=%.2f, z=%.2f\n",
               li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
    }
    fflush(stdout);
  }
  // Optional: MPI_Barrier(cart_comm_);
}

void SWESolver::init_dx_dy() {
  // z_ now has its halo regions filled (or should have, from exchange_halos_for_field(z_))
  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;

  // Iterate over OWNED cells to compute derivatives for them.
  // The stencil will now access the filled halo regions of z_ when near the boundary.
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi = i_local + halo_width_; // Padded index for current cell
      const std::size_t pj = j_local + halo_width_; // Padded index for current cell

      // at() accesses the padded z_ vector, which now includes exchanged halo data
      double z_ip1 = at(z_, pi + 1, pj);
      double z_im1 = at(z_, pi - 1, pj);
      double z_jp1 = at(z_, pi, pj + 1);
      double z_jm1 = at(z_, pi, pj - 1);

      at(zdx_, pi, pj) = 0.5 * (z_ip1 - z_im1) / dx_global;
      at(zdy_, pi, pj) = 0.5 * (z_jp1 - z_jm1) / dy_global;
    }
  }
}

// --- solve(), compute_time_step(), compute_kernel(), solve_step(), update_bcs() ---

void SWESolver::solve(const double Tend, const bool full_log,
                      const std::size_t output_n_param,
                      const std::string &fname_prefix_param)
{
    // Initialize filename_prefix_ member
    filename_prefix_ = fname_prefix_param;

    if (rank_ == 0) {
        std::cout << "Starting SWE parallel simulation (Rank 0 output, with first-step debug prints)..." << std::endl;
        std::cout << "Global Grid: " << global_nx_ << "x" << global_ny_
                  << ", Processes: " << num_procs_ << " (" << dims_[0] << "x" << dims_[1] << ")" << std::endl;
        printf("Rank 0 local owned grid: %lux%lu\n", nx_, ny_);
        if (output_n_param > 0) {
            std::cout << "Outputting data every " << output_n_param
                      << " steps. Output dir/prefix: " << filename_prefix_ << std::endl;
        }
    }

    // All ranks create the XDMFWriter (it handles directory creation on rank 0)
    // The XDMFWriter needs global dimensions
    if (output_n_param > 0) {
        writer_ptr_.reset(new XDMFWriter(filename_prefix_, global_nx_, global_ny_, size_x_, size_y_));
    }

    // Write initial mesh and topography HDF5 files collectively
    if (output_n_param > 0) {
        write_mesh_to_hdf5_parallel();
        write_topography_to_hdf5_parallel(z_); // Pass the padded z_ field
    }
    MPI_Barrier(cart_comm_); // Ensure mesh and topography are written before first h output

    double T = 0.0;
    std::size_t nt_step = 0;
    const int DEBUG_PRINT_RANK_LIMIT = 4; // Print debug for ranks < this
    const bool FIRST_STEP_DEBUG_PRINTS = true;
    const int MAX_COMPUTE_STEPS_FOR_DEBUG = 1; // Limit computation for this specific debug

    // --- Debug prints for nt_step == 0 (Initial state and first boundary/halo processing) ---
    if (FIRST_STEP_DEBUG_PRINTS) {
        if (rank_ < DEBUG_PRINT_RANK_LIMIT) {
            print_debug_perimeter_and_halos(h0_, "h0 - Initial (after all init_*, before 1st BC/Exch)");
        }
        MPI_Barrier(cart_comm_);

        if (rank_ == 0) printf("\n--- [Step 0] Applying initial Physical BCs to h0_, hu0_, hv0_ ---\n");
        update_bcs(h0_, hu0_, hv0_);
        if (rank_ < DEBUG_PRINT_RANK_LIMIT) {
            print_debug_perimeter_and_halos(h0_, "h0 - After 1st update_bcs");
        }
        MPI_Barrier(cart_comm_);

        if (rank_ == 0) printf("--- [Step 0] Performing 1st Halo Exchange for h0_, hu0_, hv0_ ---\n");
        exchange_halos_for_field(h0_);
        exchange_halos_for_field(hu0_);
        exchange_halos_for_field(hv0_);
        MPI_Barrier(cart_comm_);

        if (rank_ < DEBUG_PRINT_RANK_LIMIT) {
            print_debug_perimeter_and_halos(h0_, "h0 - After 1st exchange_halos");
        }
        MPI_Barrier(cart_comm_);
        if (rank_ == 0) printf("--- [Step 0] End of initial boundary/halo processing ---\n\n");
    } else { // If not doing detailed first step debug, still do initial BCs and exchange
        update_bcs(h0_, hu0_, hv0_);
        exchange_halos_for_field(h0_);
        exchange_halos_for_field(hu0_);
        exchange_halos_for_field(hv0_);
    }

    // Initial output for h0
    if (output_n_param > 0) {
        // Construct HDF5 filename for initial h0
        const std::string h_h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_h_" + std::to_string(0) + ".h5";
        write_field_to_hdf5_parallel(h_h5_filename, "/h", h0_);
        if (rank_ == 0 && writer_ptr_) { // Only rank 0 updates the XDMF file
            writer_ptr_->add_h_timestep(0.0); // Add 0.0 time step to XDMF
        }
    }
    MPI_Barrier(cart_comm_); // Ensure initial h0 is written before loop starts

    // --- Main Time Loop ---
    while (T < Tend) {
        // For steps *after* the initial setup (nt_step > 0), halos of h0_ need updates
        if (nt_step > 0) {
            update_bcs(h0_, hu0_, hv0_);
            exchange_halos_for_field(h0_);
            exchange_halos_for_field(hu0_);
            exchange_halos_for_field(hv0_);
        }

        const double dt = this->compute_time_step(h0_, hu0_, hv0_, T, Tend);
        if (T + dt <= T + 1e-9 && (Tend - T) > 1e-9) { /* ... break ... */ if(rank_==0) printf("\nDT too small.\n"); break;}
        if (dt <= 0.0) { /* ... break ... */ if(rank_==0 && (Tend-T)>1e-9) printf("\nDT zero/negative.\n"); break;}
        const double T1 = T + dt;

        if (rank_ == 0 && (full_log || nt_step % 1 == 0 || (T1 >= Tend && T < Tend))) {
            printf("Step %zu: T = %.6f hr (dt = %.3e s), Progress: %.2f%%\r",
                   nt_step, T1, dt * 3600.0, std::min(100.0, T1 / Tend * 100.0));
            if (T1 >= Tend || full_log) printf("\n");
            fflush(stdout);
        }

        this->solve_step(dt, h0_, hu0_, hv0_, h1_, hu1_, hv1_);

        // Check if output is due for the *next* time step (h1_)
        // The condition (nt_step + 1) % output_n_param == 0 means output every 'output_n_param' steps.
        // The second part (T1 >= Tend ...) ensures the final step is always outputted if output_n_param > 0.
        if (output_n_param > 0 && (((nt_step + 1) % output_n_param == 0) || (T1 >= Tend && (nt_step +1)%output_n_param !=0 && T < Tend ) )) {
            // All ranks write their portion of h1_ to a new HDF5 file
            // The index for the HDF5 file is based on the number of time steps already recorded in XDMF.
            const std::string h_h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_h_" + std::to_string(writer_ptr_->time_steps_.size()) + ".h5";
            write_field_to_hdf5_parallel(h_h5_filename, "/h", h1_);
            MPI_Barrier(cart_comm_); // Ensure HDF5 write is complete before rank 0 updates XDMF

            if (rank_ == 0 && writer_ptr_) {
                writer_ptr_->add_h_timestep(T1); // Only rank 0 updates the XDMF file
            }
        }

        std::swap(h0_, h1_); std::swap(hu0_, hu1_); std::swap(hv0_, hv1_);
        T = T1;
        nt_step++;
        if (T >= Tend) break;

        // // For focused debugging of first step computation
        // if (FIRST_STEP_DEBUG_PRINTS && nt_step >= MAX_COMPUTE_STEPS_FOR_DEBUG) {
        //     if(rank_==0) printf("Stopping after %d compute step(s) for debug.\n", MAX_COMPUTE_STEPS_FOR_DEBUG);
        //     break;
        // }
    }

    if (rank_ == 0) {
        if (T < Tend && !(full_log || (nt_step > 0 && (nt_step-1) % 10 == 0))) { printf("\n"); }
        else if (T>= Tend && !(full_log || (nt_step > 0 && (nt_step-1)%10==0) ) ){
             printf("Step %zu: T = %.6f hr (dt = --- s), Progress: 100.00%%\n", nt_step > 0 ? nt_step-1:0, T);
        }
        std::cout << "Simulation finished after " << nt_step << " steps. Final time T = " << T << " hours." << std::endl;
    }
}



double SWESolver::compute_time_step(const std::vector<double> &h_local, // h0_
                                     const std::vector<double> &hu_local,// hu0_
                                     const std::vector<double> &hv_local,// hv0_
                                     const double T,
                                     const double Tend) const
{
  double local_max_nu_sqr = 1e-12; // Initialize to a very small positive number

  // Iterate over OWNED cells only
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi = i_local + halo_width_; // Padded index for current cell
      const std::size_t pj = j_local + halo_width_;

      const double h_val = at(h_local, pi, pj);
      if (h_val < 1e-6) continue; // Skip dry or very shallow cells for stability in nu calculation

      const double hu_val = at(hu_local, pi, pj);
      const double hv_val = at(hv_local, pi, pj);

      // Velocities u and v
      const double u_val = hu_val / h_val;
      const double v_val = hv_val / h_val;

      const double wave_speed_h = sqrt(g * h_val); // sqrt(g*h)
      const double nu_u = std::fabs(u_val) + wave_speed_h;
      const double nu_v = std::fabs(v_val) + wave_speed_h;
      local_max_nu_sqr = std::max(local_max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    }
  }

  double global_max_nu_sqr = 0.0;
  MPI_Allreduce(&local_max_nu_sqr, &global_max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, cart_comm_);

  if (global_max_nu_sqr < 1e-9) { // Avoid division by zero / very large dt if waves are tiny
      global_max_nu_sqr = 1e-9; // Or handle as simulation end / stable state
  }

  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;
  // CFL condition from PDF: dt <= min(dx,dy) / (sqrt(2) * max_wave_speed_magnitude)
  // where wave_speed_magnitude is sqrt(nu_u^2 + nu_v^2). So max_wave_speed_magnitude is sqrt(global_max_nu_sqr)
  double dt = std::min(dx_global, dy_global) / (sqrt(2.0 * global_max_nu_sqr));

  // Ensure dt doesn't overshoot Tend and isn't negative/zero
  if (T + dt > Tend) {
    dt = Tend - T;
  }
  if (dt <= 0 && Tend > T) { // If Tend not yet reached but dt became non-positive
      if (rank_ == 0) std::cerr << "Warning: dt calculation resulted in <=0 value (" << dt << ") before Tend." << std::endl;
      dt = (Tend - T) > 1e-7 ? 1e-7 : (Tend-T)*0.5; // small placeholder or error
      if (dt <=0 ) return 0; // force stop
  }
  return dt;
}

void SWESolver::compute_kernel(const std::size_t pi, // pi, pj are PADDED indices of the cell to update
                               const std::size_t pj,
                               const double dt,
                               const std::vector<double> &h0_local,  // Input field (n)
                               const std::vector<double> &hu0_local, // Input field (n)
                               const std::vector<double> &hv0_local, // Input field (n)
                               std::vector<double> &h_local,   // Output field (n+1)
                               std::vector<double> &hu_local,  // Output field (n+1)
                               std::vector<double> &hv_local) const { // Output field (n+1)
  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;
  const double C1x = 0.5 * dt / dx_global;
  const double C1y = 0.5 * dt / dy_global;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g; // g/2

  // Read from h0_local, hu0_local, hv0_local using stencil around (pi, pj)
  // Example for h_local calculation:
  double hij = 0.25 * (at(h0_local, pi, pj - 1) + at(h0_local, pi, pj + 1) + at(h0_local, pi - 1, pj) + at(h0_local, pi + 1, pj))
               - C1x * (at(hu0_local, pi + 1, pj) - at(hu0_local, pi - 1, pj))
               - C1y * (at(hv0_local, pi, pj + 1) - at(hv0_local, pi, pj - 1));

  if (hij < 0.0) {
      hij = 1.0e-5;
  }
  at(h_local, pi, pj) = hij; // Write to the current OWNED cell in the output field

  if (hij > 0.0001) {
      double h0_im1 = at(h0_local, pi - 1, pj); if (h0_im1 < 1e-6) h0_im1 = 1e-6;
      double h0_ip1 = at(h0_local, pi + 1, pj); if (h0_ip1 < 1e-6) h0_ip1 = 1e-6;
      double h0_jm1 = at(h0_local, pi, pj - 1); if (h0_jm1 < 1e-6) h0_jm1 = 1e-6;
      double h0_jp1 = at(h0_local, pi, pj + 1); if (h0_jp1 < 1e-6) h0_jp1 = 1e-6;

      at(hu_local, pi, pj) =
          0.25 * (at(hu0_local, pi, pj - 1) + at(hu0_local, pi, pj + 1) + at(hu0_local, pi - 1, pj) + at(hu0_local, pi + 1, pj))
          - C2 * hij * at(zdx_, pi, pj)
          - C1x * ( (at(hu0_local, pi + 1, pj) * at(hu0_local, pi + 1, pj) / h0_ip1 + C3 * h0_ip1 * h0_ip1)
                  - (at(hu0_local, pi - 1, pj) * at(hu0_local, pi - 1, pj) / h0_im1 + C3 * h0_im1 * h0_im1) )
          - C1y * ( (at(hu0_local, pi, pj + 1) * at(hv0_local, pi, pj + 1) / h0_jp1)
                  - (at(hu0_local, pi, pj - 1) * at(hv0_local, pi, pj - 1) / h0_jm1) );

      at(hv_local, pi, pj) =
          0.25 * (at(hv0_local, pi, pj - 1) + at(hv0_local, pi, pj + 1) + at(hv0_local, pi - 1, pj) + at(hv0_local, pi + 1, pj))
          - C2 * hij * at(zdy_, pi, pj)
          - C1x * ( (at(hu0_local, pi + 1, pj) * at(hv0_local, pi + 1, pj) / h0_ip1)
                  - (at(hu0_local, pi - 1, pj) * at(hv0_local, pi - 1, pj) / h0_im1) )
          - C1y * ( (at(hv0_local, pi, pj + 1) * at(hv0_local, pi, pj + 1) / h0_jp1 + C3 * h0_jp1 * h0_jp1)
                  - (at(hv0_local, pi, pj - 1) * at(hv0_local, pi, pj - 1) / h0_jm1 + C3 * h0_jm1 * h0_jm1) );
  } else {
      at(hu_local, pi, pj) = 0.0;
      at(hv_local, pi, pj) = 0.0;
  }
}


void SWESolver::solve_step(const double dt,
                           std::vector<double> &h0_local,  // Actually h0_ (current state)
                           std::vector<double> &hu0_local, // Actually hu0_
                           std::vector<double> &hv0_local, // Actually hv0_
                           std::vector<double> &h_local,   // Actually h1_ (next state)
                           std::vector<double> &hu_local,  // Actually hu1_
                           std::vector<double> &hv_local)  // Actually hv1_
{
  // Loop over OWNED local cells
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) { // ny_ is local owned height
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) { // nx_ is local owned width
      // Call compute_kernel with PADDED indices for the current OWNED cell
      this->compute_kernel(i_local + halo_width_, j_local + halo_width_, dt,
                           h0_local, hu0_local, hv0_local,
                           h_local, hu_local, hv_local);
    }
  }
}

void SWESolver::update_bcs(std::vector<double> &h_target, std::vector<double> &hu_target, std::vector<double> &hv_target)
{
  // This function applies physical BCs by setting values in the HALO regions
  // of h_target, hu_target, hv_target. These are the fields for time 'n'.
  // It uses values from the adjacent OWNED cells within the same 'n' fields.

  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Top global boundary (current process has no UP neighbor: neighbors_[0] == MPI_PROC_NULL)
  if (neighbors_[0] == MPI_PROC_NULL) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) { // Loop over all x-cells in owned domain
      const std::size_t pi_owned = i_local + halo_width_; // x-index in padded grid for owned cells
      const std::size_t pj_halo_top = halo_width_ - 1;    // y-index of the top halo cell
      const std::size_t pj_owned_adjacent = halo_width_; // y-index of the top-most owned cell

      // Copy from adjacent owned cell to halo (zero-gradient for h and parallel velocity hu)
      at(h_target, pi_owned, pj_halo_top) = at(h_target, pi_owned, pj_owned_adjacent);
      at(hu_target, pi_owned, pj_halo_top) = at(hu_target, pi_owned, pj_owned_adjacent); // hu is parallel to top boundary
      // Reflect normal velocity hv
      at(hv_target, pi_owned, pj_halo_top) = coef * at(hv_target, pi_owned, pj_owned_adjacent);
    }
  }

  // Bottom global boundary (neighbors_[1] == MPI_PROC_NULL)
  if (neighbors_[1] == MPI_PROC_NULL) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi_owned = i_local + halo_width_;
      const std::size_t pj_halo_bottom = ny_ + halo_width_; // y-index of bottom halo (ny_ is owned height)
      const std::size_t pj_owned_adjacent = ny_ + halo_width_ - 1; // y-index of bottom-most owned

      at(h_target, pi_owned, pj_halo_bottom) = at(h_target, pi_owned, pj_owned_adjacent);
      at(hu_target, pi_owned, pj_halo_bottom) = at(hu_target, pi_owned, pj_owned_adjacent);
      at(hv_target, pi_owned, pj_halo_bottom) = coef * at(hv_target, pi_owned, pj_owned_adjacent);
    }
  }

  // Left global boundary (neighbors_[2] == MPI_PROC_NULL)
  if (neighbors_[2] == MPI_PROC_NULL) {
    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
      const std::size_t pj_owned = j_local + halo_width_;
      const std::size_t pi_halo_left = halo_width_ - 1;
      const std::size_t pi_owned_adjacent = halo_width_;

      at(h_target, pi_halo_left, pj_owned) = at(h_target, pi_owned_adjacent, pj_owned);
      at(hu_target, pi_halo_left, pj_owned) = coef * at(hu_target, pi_owned_adjacent, pj_owned); // hu is normal
      at(hv_target, pi_halo_left, pj_owned) = at(hv_target, pi_owned_adjacent, pj_owned); // hv is parallel
    }
  }

  // Right global boundary (neighbors_[3] == MPI_PROC_NULL)
  if (neighbors_[3] == MPI_PROC_NULL) {
    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
      const std::size_t pj_owned = j_local + halo_width_;
      const std::size_t pi_halo_right = nx_ + halo_width_;
      const std::size_t pi_owned_adjacent = nx_ + halo_width_ - 1;

      at(h_target, pi_halo_right, pj_owned) = at(h_target, pi_owned_adjacent, pj_owned);
      at(hu_target, pi_halo_right, pj_owned) = coef * at(hu_target, pi_owned_adjacent, pj_owned);
      at(hv_target, pi_halo_right, pj_owned) = at(hv_target, pi_owned_adjacent, pj_owned);
    }
  }
}

void SWESolver::exchange_halos_for_field(std::vector<double>& data_field) {
    MPI_Request reqs[8];
    MPI_Status stats[8];
    int req_count = 0;

    std::vector<double> send_buffer_to_left(ny_);
    std::vector<double> send_buffer_to_right(ny_);
    std::vector<double> recv_buffer_from_left(ny_);
    std::vector<double> recv_buffer_from_right(ny_);

    constexpr int TAG_SEND_U_RECV_D = 10; // Data I send TO MY UP neighbor, which it receives as data FROM ITS DOWN direction
    constexpr int TAG_SEND_D_RECV_U = 11; // Data I send TO MY DOWN neighbor, which it receives as data FROM ITS UP direction
    constexpr int TAG_SEND_L_RECV_R = 12;
    constexpr int TAG_SEND_R_RECV_L = 13;

    // --- Post non-blocking receives ---
    // Receive from UP neighbor (neighbors_[0]) -> fills my top halo (data they sent "down" to me)
    if (neighbors_[0] != MPI_PROC_NULL) {
        MPI_Irecv(&at(data_field, halo_width_, 0), nx_, MPI_DOUBLE, neighbors_[0], TAG_SEND_D_RECV_U, cart_comm_, &reqs[req_count++]);
    }
    // Receive from DOWN neighbor (neighbors_[1]) -> fills my bottom halo (data they sent "up" to me)
    if (neighbors_[1] != MPI_PROC_NULL) {
        MPI_Irecv(&at(data_field, halo_width_, ny_ + halo_width_), nx_, MPI_DOUBLE, neighbors_[1], TAG_SEND_U_RECV_D, cart_comm_, &reqs[req_count++]);
    }
    // Receive from LEFT neighbor (neighbors_[2]) -> fills my left halo
    if (neighbors_[2] != MPI_PROC_NULL) {
        MPI_Irecv(recv_buffer_from_left.data(), ny_, MPI_DOUBLE, neighbors_[2], TAG_SEND_R_RECV_L, cart_comm_, &reqs[req_count++]);
    }
    // Receive from RIGHT neighbor (neighbors_[3]) -> fills my right halo
    if (neighbors_[3] != MPI_PROC_NULL) {
        MPI_Irecv(recv_buffer_from_right.data(), ny_, MPI_DOUBLE, neighbors_[3], TAG_SEND_L_RECV_R, cart_comm_, &reqs[req_count++]);
    }

    // --- Prepare and post non-blocking sends ---
    // Send my *TOP* owned row TO my UP neighbor (neighbors_[0])
    if (neighbors_[0] != MPI_PROC_NULL) {
        MPI_Isend(&at(data_field, halo_width_, halo_width_), /* My top owned row */
                  nx_, MPI_DOUBLE, neighbors_[0], TAG_SEND_U_RECV_D, cart_comm_, &reqs[req_count++]);
    }
    // Send my *BOTTOM* owned row TO my DOWN neighbor (neighbors_[1])
    if (neighbors_[1] != MPI_PROC_NULL) {
        MPI_Isend(&at(data_field, halo_width_, ny_ - 1 + halo_width_), /* My bottom owned row */
                  nx_, MPI_DOUBLE, neighbors_[1], TAG_SEND_D_RECV_U, cart_comm_, &reqs[req_count++]);
    }
    // Pack and send my *LEFTMOST* owned column TO my LEFT neighbor (neighbors_[2])
    if (neighbors_[2] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            send_buffer_to_left[j_local] = at(data_field, halo_width_, j_local + halo_width_);
        }
        MPI_Isend(send_buffer_to_left.data(), ny_, MPI_DOUBLE, neighbors_[2], TAG_SEND_L_RECV_R, cart_comm_, &reqs[req_count++]);
    }
    // Pack and send my *RIGHTMOST* owned column TO my RIGHT neighbor (neighbors_[3])
    if (neighbors_[3] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            send_buffer_to_right[j_local] = at(data_field, nx_ - 1 + halo_width_, j_local + halo_width_);
        }
        MPI_Isend(send_buffer_to_right.data(), ny_, MPI_DOUBLE, neighbors_[3], TAG_SEND_R_RECV_L, cart_comm_, &reqs[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, stats);
    }

    // Unpack received column data
    if (neighbors_[2] != MPI_PROC_NULL) { // Unpack from LEFT
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            at(data_field, 0, j_local + halo_width_) = recv_buffer_from_left[j_local];
        }
    }
    if (neighbors_[3] != MPI_PROC_NULL) { // Unpack from RIGHT
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            at(data_field, nx_ + halo_width_, j_local + halo_width_) = recv_buffer_from_right[j_local];
        }
    }
}

void SWESolver::print_debug_perimeter_and_halos(const std::vector<double>& data_field, const std::string& label) const {
    // Only print if local dimensions are valid for halo + some owned cells
    // This check isn't strictly necessary if print_extent_i/j handles bounds, but good for clarity.
    if (nx_padded_ == 0 || ny_padded_ == 0) {
        printf("[Rank %d (%d,%d)] %s: Padded grid is empty (nx_padded_=%zu, ny_padded_=%zu).\n",
               rank_, coords_[0], coords_[1], label.c_str(), nx_padded_, ny_padded_);
        fflush(stdout);
        return;
    }

    printf("[Rank %d (%d,%d)] %s (Printing top-left corner):\n", rank_, coords_[0], coords_[1], label.c_str());

    // Internal constant to control how many OWNED cells to print next to the halo.
    // For example, if corner_print_owned_size = 2 and halo_width_ = 1,
    // we'll print 1 halo row/col + 2 owned rows/cols = 3 cells in total from the edge.
    const std::size_t corner_print_owned_size = 10; // Print halo + this many owned cells

    // Determine the extent of printing in padded indices
    // Max padded index to print = halo_width_ (to cover halo) + corner_print_owned_size - 1 (for owned part)
    // Total cells to print = halo_width_ + corner_print_owned_size
    std::size_t print_extent_i = std::min(nx_padded_, halo_width_ + corner_print_owned_size);
    std::size_t print_extent_j = std::min(ny_padded_, halo_width_ + corner_print_owned_size);

    // Print header for i-indices (padded)
    printf("  pj\\pi |");
    for (std::size_t pi = 0; pi < print_extent_i; ++pi) {
        printf("%8zu ", pi);
    }
    printf("\n");
    printf("  -------+");
    for (std::size_t pi = 0; pi < print_extent_i; ++pi) {
        printf("---------");
    }
    printf("\n");

    // Print the selected corner region
    for (std::size_t pj = 0; pj < print_extent_j; ++pj) {
        printf("  %5zu |", pj); // Print current padded j-index
        for (std::size_t pi = 0; pi < print_extent_i; ++pi) {
            // Visual separator for halo vs owned (simple version)
            // if (pi == halo_width_ || pj == halo_width_) {
            //    // This gets messy with alignment. A simple space is fine.
            // }
            printf("%8.1f ", at(data_field, pi, pj));
        }
        printf("\n");
        // Visual separator row after halo rows
        if (pj == halo_width_ - 1 && halo_width_ > 0) {
            printf("  -------+");
            for (std::size_t pi_sep = 0; pi_sep < print_extent_i; ++pi_sep) {
                 printf("---------");
            }
            printf(" (halo boundary)\n");
        }
    }
    fflush(stdout);
}
