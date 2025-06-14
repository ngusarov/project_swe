// swe.cc
#define H5_HAVE_PARALLEL_H5

#include <mpi.h>

#include "swe.hh"
#include "xdmf_writer.hh"

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
#include <algorithm>
#include <numeric>

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

  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return;
  }

  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error opening dataset: " << dataset_name << std::endl;
    H5Fclose(file_id);
    return;
  }

  dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
  {
    std::cerr << "Error getting dataspace" << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }

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

  data.resize(nx * ny);

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

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
}

} // namespace

SWESolver::~SWESolver() = default;

void SWESolver::write_field_to_hdf5_parallel(const std::string& full_h5_filepath,
                                             const std::string& dataset_name,
                                             const std::vector<double>& data_field_padded)
{
    // printf("Rank %d: Entering write_field_to_hdf5_parallel for %s. nx_=%zu, ny_=%zu\n", rank_, dataset_name.c_str(), nx_, ny_); fflush(stdout);

    hsize_t global_dims[2] = {global_ny_, global_nx_};
    hsize_t local_dims[2] = {ny_, nx_};
    hsize_t offset[2] = {G_start_j_, G_start_i_};
    hsize_t count[2] = {ny_, nx_};

    // printf("Rank %d: %s - Global: %zu x %zu, Local: %zu x %zu, Offset: %zu x %zu, Count: %zu x %zu\n",
    //        rank_, dataset_name.c_str(), global_dims[0], global_dims[1], local_dims[0], local_dims[1],
    //        offset[0], offset[1], count[0], count[1]);
    fflush(stdout);

    // printf("Rank %d: %s - data_field_padded size: %zu\n", rank_, dataset_name.c_str(), data_field_padded.size()); fflush(stdout);


    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    // printf("Rank %d: %s - After H5Pcreate(H5P_FILE_ACCESS)\n", rank_, dataset_name.c_str()); fflush(stdout);
    H5Pset_fapl_mpio(fapl_id, cart_comm_, MPI_INFO_NULL);
    // printf("Rank %d: %s - After H5Pset_fapl_mpio\n", rank_, dataset_name.c_str()); fflush(stdout);


    hid_t file_id = H5Fcreate(full_h5_filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    if (file_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating HDF5 file: " << full_h5_filepath << std::endl;
        H5Pclose(fapl_id);
        return;
    }
    // printf("Rank %d: %s - After H5Fcreate\n", rank_, dataset_name.c_str()); fflush(stdout);

    hid_t filespace_id = H5Screate_simple(2, global_dims, NULL);
    // printf("Rank %d: %s - After H5Screate_simple(filespace)\n", rank_, dataset_name.c_str()); fflush(stdout);
    hid_t memspace_id = H5Screate_simple(2, local_dims, NULL);
    // printf("Rank %d: %s - After H5Screate_simple(memspace)\n", rank_, dataset_name.c_str()); fflush(stdout);

    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    // printf("Rank %d: %s - After H5Sselect_hyperslab\n", rank_, dataset_name.c_str()); fflush(stdout);

    hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, filespace_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating dataset: " << dataset_name << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Fclose(file_id);
        H5Pclose(fapl_id);
        return;
    }
    // printf("Rank %d: %s - After H5Dcreate2\n", rank_, dataset_name.c_str()); fflush(stdout);


    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    // printf("Rank %d: %s - After H5Pcreate(H5P_DATASET_XFER)\n", rank_, dataset_name.c_str()); fflush(stdout);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    // printf("Rank %d: %s - After H5Pset_dxpl_mpio\n", rank_, dataset_name.c_str()); fflush(stdout);


    std::vector<double> local_owned_data(nx_ * ny_);
    // printf("Rank %d: %s - local_owned_data allocated with size %zu\n", rank_, dataset_name.c_str(), local_owned_data.size()); fflush(stdout);

    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
        for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
            std::size_t padded_i = i_local + halo_width_;
            std::size_t padded_j = j_local + halo_width_;

            std::size_t padded_data_idx = padded_j * nx_padded_ + padded_i;
            if (padded_data_idx >= data_field_padded.size()) {
                fprintf(stderr, "Rank %d: ERROR: Out-of-bounds read from data_field_padded! Index %zu, max size %zu\n",
                        rank_, padded_data_idx, data_field_padded.size());
                fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::size_t local_owned_idx = j_local * nx_ + i_local;
            if (local_owned_idx >= local_owned_data.size()) {
                fprintf(stderr, "Rank %d: ERROR: Out-of-bounds write to local_owned_data! Index %zu, max size %zu\n",
                        rank_, local_owned_idx, local_owned_data.size());
                fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            local_owned_data[local_owned_idx] = at(data_field_padded, padded_i, padded_j);
        }
    }
    if (!local_owned_data.empty()) { // Add check to prevent accessing element 0 of empty vector
      // printf("Rank %d: %s - local_owned_data filled. First element: %.2f\n", rank_, dataset_name.c_str(), local_owned_data[0]); fflush(stdout);
    } else {
      // printf("Rank %d: %s - local_owned_data is empty.\n", rank_, dataset_name.c_str()); fflush(stdout);
    }

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id, xfer_plist_id, local_owned_data.data());
    if (status < 0) {
        std::cerr << "Rank " << rank_ << ": Error writing data for dataset: " << dataset_name << std::endl;
    }
    // printf("Rank %d: %s - After H5Dwrite. Status: %d\n", rank_, dataset_name.c_str(), status); fflush(stdout);


    H5Pclose(xfer_plist_id);
    H5Dclose(dataset_id);
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Fclose(file_id);
    H5Pclose(fapl_id);
    // printf("Rank %d: %s - All HDF5 resources closed.\n", rank_, dataset_name.c_str()); fflush(stdout);


    // if (rank_ == 0) {
    //     std::cout << "Successfully wrote HDF5 file: " << full_h5_filepath << ", dataset: " << dataset_name << std::endl;
    // }
    // printf("Rank %d: Exiting write_field_to_hdf5_parallel for %s\n", rank_, dataset_name.c_str()); fflush(stdout);
}

void SWESolver::write_topography_to_hdf5_parallel(const std::vector<double>& topography_padded)
{
    const std::string h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_topography.h5";
    // printf("Rank %d: Entering write_topography_to_hdf5_parallel() for %s\n", rank_, h5_filename.c_str()); fflush(stdout);
    write_field_to_hdf5_parallel(h5_filename, "/topography", topography_padded);
    // printf("Rank %d: Exiting write_topography_to_hdf5_parallel()\n", rank_); fflush(stdout);
}


SWESolver::SWESolver(const int test_case_id,
                     std::size_t global_nx_param, std::size_t global_ny_param,
                     MPI_Comm cart_comm_param, int rank_param, int num_procs_param,
                     const int* dims_param, const int* coords_param, const int* neighbors_param) :
  cart_comm_(cart_comm_param), rank_(rank_param), num_procs_(num_procs_param),
  global_nx_(global_nx_param), global_ny_(global_ny_param),
  size_x_(500.0), size_y_(500.0)
{
  std::copy(dims_param, dims_param + 2, dims_);
  std::copy(coords_param, coords_param + 2, coords_);
  std::copy(neighbors_param, neighbors_param + 4, neighbors_);

  std::size_t base_nx = global_nx_ / dims_[0];
  std::size_t remainder_nx = global_nx_ % dims_[0];
  nx_ = base_nx + (coords_[0] < remainder_nx ? 1 : 0);

  std::size_t base_ny = global_ny_ / dims_[1];
  std::size_t remainder_ny = global_ny_ % dims_[1];
  ny_ = base_ny + (coords_[1] < remainder_ny ? 1 : 0);

  nx_padded_ = nx_ + 2 * halo_width_;
  ny_padded_ = ny_ + 2 * halo_width_;

  G_start_i_ = 0;
  for (int px = 0; px < coords_[0]; ++px) {
    G_start_i_ += (global_nx_ / dims_[0]) + (px < remainder_nx ? 1 : 0);
  }
  G_start_j_ = 0;
  for (int py = 0; py < coords_[1]; ++py) {
    G_start_j_ += (global_ny_ / dims_[1]) + (py < remainder_ny ? 1 : 0);
  }

  // if (rank_ == 0) {
  //     std::cout << "MPI Initialized: " << num_procs_ << " processes. Grid: "
  //               << dims_[0] << "x" << dims_[1] << std::endl;
  //     std::cout << "Global domain: " << global_nx_ << "x" << global_ny_ << std::endl;
  // }
  // printf("Rank %d (%d,%d): local_owned %zu x %zu, local_padded %zu x %zu, G_start (%zu,%zu)\n",
  //        rank_, coords_[0], coords_[1], nx_, ny_, nx_padded_, ny_padded_, G_start_i_, G_start_j_);
  fflush(stdout);


  h0_.resize(nx_padded_ * ny_padded_, 0.0);
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

  exchange_halos_for_field(z_);
  this->init_dx_dy();
}


void SWESolver::read_field_from_hdf5_parallel(const std::string& full_h5_filepath,
                                              const std::string& dataset_name,
                                              std::vector<double>& data_field_padded)
{
    printf("Rank %d: Entering read_field_from_hdf5_parallel for %s. nx_=%zu, ny_=%zu\n", rank_, dataset_name.c_str(), nx_, ny_); fflush(stdout);

    hsize_t global_dims[2] = {global_ny_, global_nx_};
    hsize_t local_dims[2] = {ny_, nx_}; // Local dimensions of the owned data block
    hsize_t offset[2] = {G_start_j_, G_start_i_}; // Global starting index of this rank's block
    hsize_t count[2] = {ny_, nx_}; // Size of the block this rank will read

    printf("Rank %d: %s - Global: %zu x %zu, Local: %zu x %zu, Offset: %zu x %zu, Count: %zu x %zu\n",
           rank_, dataset_name.c_str(), global_dims[0], global_dims[1], local_dims[0], local_dims[1],
           offset[0], offset[1], count[0], count[1]);
    fflush(stdout);

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating file access property list." << std::endl;
        return;
    }
    H5Pset_fapl_mpio(fapl_id, cart_comm_, MPI_INFO_NULL);

    hid_t file_id = H5Fopen(full_h5_filepath.c_str(), H5F_ACC_RDONLY, fapl_id);
    H5Pclose(fapl_id); // Close fapl_id after use
    if (file_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error opening HDF5 file for reading: " << full_h5_filepath << std::endl;
        return;
    }

    hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error opening dataset for reading: " << dataset_name << std::endl;
        H5Fclose(file_id);
        return;
    }

    hid_t filespace_id = H5Dget_space(dataset_id);
    if (filespace_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error getting filespace for dataset: " << dataset_name << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Select the hyperslab in the file representing this rank's portion
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    hid_t memspace_id = H5Screate_simple(2, local_dims, NULL);
    if (memspace_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating memory space for dataset: " << dataset_name << std::endl;
        H5Sclose(filespace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    if (xfer_plist_id < 0) {
        std::cerr << "Rank " << rank_ << ": Error creating transfer property list." << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    // Create a temporary vector to hold the data read by this rank into its owned block
    std::vector<double> local_owned_data(nx_ * ny_);
    if (local_owned_data.empty() && (nx_ > 0 || ny_ > 0) ) {
        // This case indicates nx_*ny_ is 0 even if dimensions are non-zero,
        // which can happen if num_cells_this_rank_writes is 0 in create_cells_parallel.
        // It's mostly for debugging.
        printf("Rank %d: %s - local_owned_data is empty for reading. This rank likely owns no data.\n", rank_, dataset_name.c_str());
        // No need to abort, just don't try to read.
    } else {
        herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id, xfer_plist_id, local_owned_data.data());
        if (status < 0) {
            std::cerr << "Rank " << rank_ << ": Error reading data for dataset: " << dataset_name << std::endl;
        }
    }

    // Now, copy the read data into the correct padded locations in data_field_padded
    data_field_padded.assign(nx_padded_ * ny_padded_, 0.0); // Ensure padded vector is correctly sized and zeroed first

    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
        for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
            std::size_t padded_i = i_local + halo_width_;
            std::size_t padded_j = j_local + halo_width_;
            std::size_t local_owned_idx = j_local * nx_ + i_local;

            if (local_owned_idx < local_owned_data.size()) { // Safety check
                at(data_field_padded, padded_i, padded_j) = local_owned_data[local_owned_idx];
            }
        }
    }

    H5Pclose(xfer_plist_id);
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    printf("Rank %d: Exiting read_field_from_hdf5_parallel for %s\n", rank_, dataset_name.c_str()); fflush(stdout);
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

  // First, rank 0 needs to read global dimensions from ONE of the HDF5 files
  // (e.g., h0) and then broadcast them to all other ranks.
  // This is crucial because all ranks need to know the global_nx_ and global_ny_
  // before they can calculate their local sizes and offsets.
  std::size_t file_global_nx_temp = 0;
  std::size_t file_global_ny_temp = 0;

  if (rank_ == 0) {
    // Temporarily open the file just to get global dimensions from a dataset
    hid_t file_id = H5Fopen(h5_file_param.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      std::cerr << "Rank " << rank_ << ": Error opening HDF5 file to get dimensions: " << h5_file_param << std::endl;
      // Handle error, maybe MPI_Abort
    }
    hid_t dataset_id = H5Dopen2(file_id, "h0", H5P_DEFAULT); // Assume 'h0' exists and has the global dims
    if (dataset_id < 0) {
      std::cerr << "Rank " << rank_ << ": Error opening 'h0' dataset to get dimensions from " << h5_file_param << std::endl;
      H5Fclose(file_id);
      // Handle error, maybe MPI_Abort
    }
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    file_global_nx_temp = dims[1]; // Assuming HDF5 stores as (NY, NX)
    file_global_ny_temp = dims[0]; // So, dims[0] is NY, dims[1] is NX

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
  }

  // Broadcast global dimensions to all ranks
  MPI_Bcast(&file_global_nx_temp, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_);
  MPI_Bcast(&file_global_ny_temp, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_);

  global_nx_ = file_global_nx_temp;
  global_ny_ = file_global_ny_temp;


  std::size_t base_nx = global_nx_ / dims_[0];
  std::size_t remainder_nx = global_nx_ % dims_[0];
  nx_ = base_nx + (coords_[0] < remainder_nx ? 1 : 0);

  std::size_t base_ny = global_ny_ / dims_[1];
  std::size_t remainder_ny = global_ny_ % dims_[1];
  ny_ = base_ny + (coords_[1] < remainder_ny ? 1 : 0);

  nx_padded_ = nx_ + 2 * halo_width_;
  ny_padded_ = ny_ + 2 * halo_width_;

  G_start_i_ = 0;
  for (int px = 0; px < coords_[0]; ++px) {
    G_start_i_ += (global_nx_ / dims_[0]) + (px < remainder_nx ? 1 : 0);
  }
  G_start_j_ = 0;
  for (int py = 0; py < coords_[1]; ++py) {
    G_start_j_ += (global_ny_ / dims_[1]) + (py < remainder_ny ? 1 : 0);
  }
  printf("Rank %d (%d,%d) HDF5: global %zu x %zu, local_owned %zu x %zu, local_padded %zu x %zu, G_start (%zu,%zu)\n",
         rank_, coords_[0], coords_[1], global_nx_, global_ny_, nx_, ny_, nx_padded_, ny_padded_, G_start_i_, G_start_j_);

  fflush(stdout);

  // Resize vectors with padded dimensions
  h0_.resize(nx_padded_ * ny_padded_, 0.0);
  h1_.resize(nx_padded_ * ny_padded_, 0.0);
  hu0_.resize(nx_padded_ * ny_padded_, 0.0);
  hu1_.resize(nx_padded_ * ny_padded_, 0.0);
  hv0_.resize(nx_padded_ * ny_padded_, 0.0);
  hv1_.resize(nx_padded_ * ny_padded_, 0.0);
  z_.resize(nx_padded_ * ny_padded_, 0.0);
  zdx_.resize(nx_padded_ * ny_padded_, 0.0);
  zdy_.resize(nx_padded_ * ny_padded_, 0.0);

  // Now, use the new parallel read function to load data directly into the padded vectors
  read_field_from_hdf5_parallel(h5_file_param, "/h0", h0_);
  read_field_from_hdf5_parallel(h5_file_param, "/hu0", hu0_);
  read_field_from_hdf5_parallel(h5_file_param, "/hv0", hv0_);
  read_field_from_hdf5_parallel(h5_file_param, "/topography", z_);

  // Exchange halos for topography (z_) immediately after loading
  exchange_halos_for_field(z_);
  this->init_dx_dy(); // Recalculate zdx and zdy after z_ is finalized (including halos)

  // Set reflective to false by default for HDF5 loaded cases
  this->reflective_ = false;
}

void SWESolver::init_gaussian() {
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;

  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const double x_global = dx_global * (static_cast<double>(G_start_i_ + i_local) + 0.5);
      const double y_global = dy_global * (static_cast<double>(G_start_j_ + j_local) + 0.5);

      const double gauss_0 = 10.0 * std::exp(-((x_global - x0_0) * (x_global - x0_0) + (y_global - y0_0) * (y_global - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x_global - x0_1) * (x_global - x0_1) + (y_global - y0_1) * (y_global - y0_1)) / 1000.0);

      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      at(h0_, pi, pj) = 10.0 + gauss_0 + gauss_1;
      at(z_, pi, pj) = 0.0;
    }
  }

  // if (rank_ < 2 || rank_ == num_procs_ - 1) {
  //   // printf("[Rank %d (%d,%d) init_gaussian] Checking owned cells (h0, z):\n", rank_, coords_[0], coords_[1]);

  //   std::size_t li = 0, lj = 0;
  //   std::size_t pi = li + halo_width_;
  //   std::size_t pj = lj + halo_width_;
  //   // if (nx_ > 0 && ny_ > 0) {
  //   //     printf("  Local (0,0) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //   //            pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   // }

  //   // if (nx_ > 0 && ny_ > 0) {
  //   //     li = nx_ / 2; lj = ny_ / 2;
  //   //     pi = li + halo_width_; pj = lj + halo_width_;
  //   //     printf("  Local (%zu,%zu) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //   //            li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   // }

  //   // if (nx_ > 0 && ny_ > 0) {
  //   //     li = nx_ - 1; lj = ny_ - 1;
  //   //     pi = li + halo_width_; pj = lj + halo_width_;
  //   //     printf("  Local (%zu,%zu) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //   //            li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   // }
  //   // fflush(stdout);
  // }
}

void SWESolver::init_dummy_tsunami() {
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

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

  // if (rank_ < 2 || rank_ == num_procs_ - 1) {
  //   printf("[Rank %d (%d,%d) init_dummy_tsunami] Checking owned cells (h0, z):\n", rank_, coords_[0], coords_[1]);

  //   std::size_t li = 0, lj = 0;
  //   std::size_t pi = li + halo_width_;
  //   std::size_t pj = lj + halo_width_;
  //    if (nx_ > 0 && ny_ > 0) {
  //       printf("  Local (0,0) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //              pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   }

  //   if (nx_ > 0 && ny_ > 0) {
  //       li = nx_ / 2; lj = ny_ / 2;
  //       pi = li + halo_width_; pj = lj + halo_width_;
  //       printf("  Local (%zu,%zu) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //              nx_/2, ny_/2, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   }

  //   if (nx_ > 0 && ny_ > 0) {
  //       li = nx_ - 1; lj = ny_ - 1;
  //       pi = li + halo_width_; pj = lj + halo_width_;
  //       printf("  Local (%zu,%zu) [Padded (%zu,%zu)]: h0=%.2f, z=%.2f\n",
  //              li, lj, pi, pj, at(h0_, pi, pj), at(z_, pi, pj));
  //   }
  //   fflush(stdout);
  // }
}

void SWESolver::init_dx_dy() {
  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;

  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      double z_ip1 = at(z_, pi + 1, pj);
      double z_im1 = at(z_, pi - 1, pj);
      double z_jp1 = at(z_, pi, pj + 1);
      double z_jm1 = at(z_, pi, pj - 1);

      at(zdx_, pi, pj) = 0.5 * (z_ip1 - z_im1) / dx_global;
      at(zdy_, pi, pj) = 0.5 * (z_jp1 - z_jm1) / dy_global;
    }
  }
}

std::size_t SWESolver::solve(const double Tend, const bool full_log,
                      const std::size_t output_n_param,
                      const std::string &fname_prefix_param)
{
    filename_prefix_ = fname_prefix_param;

    // if (rank_ == 0) {
    //     std::cout << "Starting SWE parallel simulation (Rank 0 output, with first-step debug prints)..." << std::endl;
    //     std::cout << "Global Grid: " << global_nx_ << "x" << global_ny_
    //               << ", Processes: " << num_procs_ << " (" << dims_[0] << "x" << dims_[1] << ")" << std::endl;
    //     printf("Rank 0 local owned grid: %zu x %zu\n", nx_, ny_);
    //     if (output_n_param > 0) {
    //         std::cout << "Outputting data every " << output_n_param
    //                   << " steps. Output dir/prefix: " << filename_prefix_ << std::endl;
    //     }
    // }

    if (output_n_param > 0) {
        writer_ptr_.reset(new XDMFWriter(filename_prefix_, global_nx_, global_ny_, size_x_, size_y_));
    }

    // printf("Rank %d: Before initial mesh/topography writes.\n", rank_); fflush(stdout);
    if (output_n_param > 0) {

        writer_ptr_->write_mesh_hdf5_parallel(cart_comm_, rank_, num_procs_);
        // printf("Rank %d: After mesh (implicit) setup, before topography write.\n", rank_); fflush(stdout);
        write_topography_to_hdf5_parallel(z_);
        // printf("Rank %d: After topography write, before first barrier.\n", rank_); fflush(stdout);
    }
    MPI_Barrier(cart_comm_);
    // printf("Rank %d: Passed initial mesh/topography barrier.\n", rank_); fflush(stdout);

    double T = 0.0;
    std::size_t nt_step = 0;
    const int DEBUG_PRINT_RANK_LIMIT = 4;
    const bool FIRST_STEP_DEBUG_PRINTS = false;
    const int MAX_COMPUTE_STEPS_FOR_DEBUG = 1;

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
    } else {
        update_bcs(h0_, hu0_, hv0_);
        exchange_halos_for_field(h0_);
        exchange_halos_for_field(hu0_);
        exchange_halos_for_field(hv0_);
    }

    current_output_idx_ = 0;

    // printf("Rank %d: Before initial h0 output block.\n", rank_); fflush(stdout);
    if (output_n_param > 0) {
        const std::string h_h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_h_" + std::to_string(current_output_idx_) + ".h5";
        // printf("Rank %d: About to write initial h0 to %s.\n", rank_, h_h5_filename.c_str()); fflush(stdout);
        write_field_to_hdf5_parallel(h_h5_filename, "/h", h0_);
        // printf("Rank %d: After initial h0 write, before second barrier.\n", rank_); fflush(stdout);

        if (rank_ == 0 && writer_ptr_) {
            writer_ptr_->add_h_timestep(0.0);
        }
        if (rank_ == 0) {
            current_output_idx_++;
        }
    }
    MPI_Bcast(&current_output_idx_, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_);
    // printf("Rank %d: Passed initial h0 barrier. Entering main loop. Current output index: %zu\n", rank_, current_output_idx_); fflush(stdout);


    while (T < Tend) {
        if (nt_step > 0) {
            update_bcs(h0_, hu0_, hv0_);
            exchange_halos_for_field(h0_);
            exchange_halos_for_field(hu0_);
            exchange_halos_for_field(hv0_);
        }

        const double dt = this->compute_time_step(h0_, hu0_, hv0_, T, Tend);
        if (T + dt <= T + 1e-9 && (Tend - T) > 1e-9) { if(rank_==0) printf("\nDT too small.\n"); break;}
        if (dt <= 0.0) { if(rank_==0 && (Tend-T)>1e-9) printf("\nDT zero/negative.\n"); break;}
        const double T1 = T + dt;

        if (rank_ == 0 && (full_log || nt_step % 1 == 0 || (T1 >= Tend && T < Tend))) {
            printf("Step %zu: T = %.6f hr (dt = %.3e s), Progress: %.2f%%\r",
                   nt_step, T1, dt * 3600.0, std::min(100.0, T1 / Tend * 100.0));
            if (T1 >= Tend || full_log) printf("\n");
            fflush(stdout);
        }

        this->solve_step(dt, h0_, hu0_, hv0_, h1_, hu1_, hv1_);

        if (output_n_param > 0 && (((nt_step + 1) % output_n_param == 0) || (T1 >= Tend && (nt_step +1)%output_n_param !=0 && T < Tend ) )) {
            MPI_Bcast(&current_output_idx_, 1, MPI_UNSIGNED_LONG_LONG, 0, cart_comm_);

            const std::string h_h5_filename = filename_prefix_ + "/" + filename_prefix_ + "_h_" + std::to_string(current_output_idx_) + ".h5";
            // printf("Rank %d: About to write h1_ for timestep %zu to %s (HDF5 index %zu).\n", rank_, nt_step + 1, h_h5_filename.c_str(), current_output_idx_); fflush(stdout);
            write_field_to_hdf5_parallel(h_h5_filename, "/h", h1_);
            // printf("Rank %d: After h1_ write, before barrier for timestep %zu.\n", rank_, nt_step + 1); fflush(stdout);
            MPI_Barrier(cart_comm_);
            // printf("Rank %d: Passed barrier for timestep %zu.\n", rank_, nt_step + 1); fflush(stdout);

            if (rank_ == 0 && writer_ptr_) {
                writer_ptr_->add_h_timestep(T1);
                current_output_idx_++;
            }
        }

        std::swap(h0_, h1_); std::swap(hu0_, hu1_); std::swap(hv0_, hv1_);
        T = T1;
        nt_step++;
        if (T >= Tend) break;
    }

    if (rank_ == 0) {
        if (T < Tend && !(full_log || (nt_step > 0 && (nt_step-1) % 10 == 0))) { printf("\n"); }
        else if (T>= Tend && !(full_log || (nt_step > 0 && (nt_step-1)%10==0) ) ){
             printf("Step %zu: T = %.6f hr (dt = --- s), Progress: 100.00%%\n", nt_step > 0 ? nt_step-1:0, T);
        }
        std::cout << "Simulation finished after " << nt_step << " steps. Final time T = " << T << " hours." << std::endl;
    }
    return nt_step; // Return the total number of iterations
}


double SWESolver::compute_time_step(const std::vector<double> &h_local,
                                     const std::vector<double> &hu_local,
                                     const std::vector<double> &hv_local,
                                     const double T,
                                     const double Tend) const
{
  double local_max_nu_sqr = 1e-12;

  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi = i_local + halo_width_;
      const std::size_t pj = j_local + halo_width_;

      const double h_val = at(h_local, pi, pj);
      if (h_val < 1e-6) continue;

      const double hu_val = at(hu_local, pi, pj);
      const double hv_val = at(hv_local, pi, pj);

      const double u_val = hu_val / h_val;
      const double v_val = hv_val / h_val;

      const double wave_speed_h = sqrt(g * h_val);
      const double nu_u = std::fabs(u_val) + wave_speed_h;
      const double nu_v = std::fabs(v_val) + wave_speed_h;
      local_max_nu_sqr = std::max(local_max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    }
  }

  double global_max_nu_sqr = 0.0;
  MPI_Allreduce(&local_max_nu_sqr, &global_max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, cart_comm_);

  if (global_max_nu_sqr < 1e-9) {
      global_max_nu_sqr = 1e-9;
  }

  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;
  double dt = std::min(dx_global, dy_global) / (sqrt(2.0 * global_max_nu_sqr));

  if (T + dt > Tend) {
    dt = Tend - T;
  }
  if (dt <= 0 && Tend > T) {
      if (rank_ == 0) std::cerr << "Warning: dt calculation resulted in <=0 value (" << dt << ") before Tend." << std::endl;
      dt = (Tend - T) > 1e-7 ? 1e-7 : (Tend-T)*0.5;
      if (dt <=0 ) return 0;
  }
  return dt;
}

void SWESolver::compute_kernel(const std::size_t pi,
                               const std::size_t pj,
                               const double dt,
                               const std::vector<double> &h0_local,
                               const std::vector<double> &hu0_local,
                               const std::vector<double> &hv0_local,
                               std::vector<double> &h_local,
                               std::vector<double> &hu_local,
                               std::vector<double> &hv_local) const {
  const double dx_global = size_x_ / global_nx_;
  const double dy_global = size_y_ / global_ny_;
  const double C1x = 0.5 * dt / dx_global;
  const double C1y = 0.5 * dt / dy_global;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g;

  double hij = 0.25 * (at(h0_local, pi, pj - 1) + at(h0_local, pi, pj + 1) + at(h0_local, pi - 1, pj) + at(h0_local, pi + 1, pj))
               - C1x * (at(hu0_local, pi + 1, pj) - at(hu0_local, pi - 1, pj))
               - C1y * (at(hv0_local, pi, pj + 1) - at(hv0_local, pi, pj - 1));

  if (hij < 0.0) {
      hij = 1.0e-5;
  }
  at(h_local, pi, pj) = hij;

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
                           std::vector<double> &h0_local,
                           std::vector<double> &hu0_local,
                           std::vector<double> &hv0_local,
                           std::vector<double> &h_local,
                           std::vector<double> &hu_local,
                           std::vector<double> &hv_local)
{
  for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      this->compute_kernel(i_local + halo_width_, j_local + halo_width_, dt,
                           h0_local, hu0_local, hv0_local,
                           h_local, hu_local, hv_local);
    }
  }
}

void SWESolver::update_bcs(std::vector<double> &h_target, std::vector<double> &hu_target, std::vector<double> &hv_target)
{
  const double coef = this->reflective_ ? -1.0 : 1.0;

  if (neighbors_[0] == MPI_PROC_NULL) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi_owned = i_local + halo_width_;
      const std::size_t pj_halo_top = halo_width_ - 1;
      const std::size_t pj_owned_adjacent = halo_width_;

      at(h_target, pi_owned, pj_halo_top) = at(h_target, pi_owned, pj_owned_adjacent);
      at(hu_target, pi_owned, pj_halo_top) = at(hu_target, pi_owned, pj_owned_adjacent);
      at(hv_target, pi_owned, pj_halo_top) = coef * at(hv_target, pi_owned, pj_owned_adjacent);
    }
  }

  if (neighbors_[1] == MPI_PROC_NULL) {
    for (std::size_t i_local = 0; i_local < nx_; ++i_local) {
      const std::size_t pi_owned = i_local + halo_width_;
      const std::size_t pj_halo_bottom = ny_ + halo_width_;
      const std::size_t pj_owned_adjacent = ny_ + halo_width_ - 1;

      at(h_target, pi_owned, pj_halo_bottom) = at(h_target, pi_owned, pj_owned_adjacent);
      at(hu_target, pi_owned, pj_halo_bottom) = at(hu_target, pi_owned, pj_owned_adjacent);
      at(hv_target, pi_owned, pj_halo_bottom) = coef * at(hv_target, pi_owned, pj_owned_adjacent);
    }
  }

  if (neighbors_[2] == MPI_PROC_NULL) {
    for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
      const std::size_t pj_owned = j_local + halo_width_;
      const std::size_t pi_halo_left = halo_width_ - 1;
      const std::size_t pi_owned_adjacent = halo_width_;

      at(h_target, pi_halo_left, pj_owned) = at(h_target, pi_owned_adjacent, pj_owned);
      at(hu_target, pi_halo_left, pj_owned) = coef * at(hu_target, pi_owned_adjacent, pj_owned);
      at(hv_target, pi_halo_left, pj_owned) = at(hv_target, pi_owned_adjacent, pj_owned);
    }
  }

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

    constexpr int TAG_SEND_U_RECV_D = 10;
    constexpr int TAG_SEND_D_RECV_U = 11;
    constexpr int TAG_SEND_L_RECV_R = 12;
    constexpr int TAG_SEND_R_RECV_L = 13;

    if (neighbors_[0] != MPI_PROC_NULL) {
        MPI_Irecv(&at(data_field, halo_width_, 0), nx_, MPI_DOUBLE, neighbors_[0], TAG_SEND_D_RECV_U, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[1] != MPI_PROC_NULL) {
        MPI_Irecv(&at(data_field, halo_width_, ny_ + halo_width_), nx_, MPI_DOUBLE, neighbors_[1], TAG_SEND_U_RECV_D, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[2] != MPI_PROC_NULL) {
        MPI_Irecv(recv_buffer_from_left.data(), ny_, MPI_DOUBLE, neighbors_[2], TAG_SEND_R_RECV_L, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[3] != MPI_PROC_NULL) {
        MPI_Irecv(recv_buffer_from_right.data(), ny_, MPI_DOUBLE, neighbors_[3], TAG_SEND_L_RECV_R, cart_comm_, &reqs[req_count++]);
    }

    if (neighbors_[0] != MPI_PROC_NULL) {
        MPI_Isend(&at(data_field, halo_width_, halo_width_),
                  nx_, MPI_DOUBLE, neighbors_[0], TAG_SEND_U_RECV_D, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[1] != MPI_PROC_NULL) {
        MPI_Isend(&at(data_field, halo_width_, ny_ - 1 + halo_width_),
                  nx_, MPI_DOUBLE, neighbors_[1], TAG_SEND_D_RECV_U, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[2] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            send_buffer_to_left[j_local] = at(data_field, halo_width_, j_local + halo_width_);
        }
        MPI_Isend(send_buffer_to_left.data(), ny_, MPI_DOUBLE, neighbors_[2], TAG_SEND_L_RECV_R, cart_comm_, &reqs[req_count++]);
    }
    if (neighbors_[3] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            send_buffer_to_right[j_local] = at(data_field, nx_ - 1 + halo_width_, j_local + halo_width_);
        }
        MPI_Isend(send_buffer_to_right.data(), ny_, MPI_DOUBLE, neighbors_[3], TAG_SEND_R_RECV_L, cart_comm_, &reqs[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, stats);
    }

    if (neighbors_[2] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            at(data_field, 0, j_local + halo_width_) = recv_buffer_from_left[j_local];
        }
    }
    if (neighbors_[3] != MPI_PROC_NULL) {
        for (std::size_t j_local = 0; j_local < ny_; ++j_local) {
            at(data_field, nx_ + halo_width_, j_local + halo_width_) = recv_buffer_from_right[j_local];
        }
    }
}

void SWESolver::print_debug_perimeter_and_halos(const std::vector<double>& data_field, const std::string& label) const {
    if (nx_padded_ == 0 || ny_padded_ == 0) {
        printf("[Rank %d (%d,%d)] %s: Padded grid is empty (nx_padded_=%zu, ny_padded_=%zu).\n",
               rank_, coords_[0], coords_[1], label.c_str(), nx_padded_, ny_padded_);
        fflush(stdout);
        return;
    }

    printf("[Rank %d (%d,%d)] %s (Printing top-left corner):\n", rank_, coords_[0], coords_[1], label.c_str());

    const std::size_t corner_print_owned_size = 5;

    std::size_t print_extent_i = std::min(nx_padded_, halo_width_ + corner_print_owned_size);
    std::size_t print_extent_j = std::min(ny_padded_, halo_width_ + corner_print_owned_size);

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

    for (std::size_t pj = 0; pj < print_extent_j; ++pj) {
        printf("  %5zu |", pj);
        for (std::size_t pi = 0; pi < print_extent_i; ++pi) {
            printf("%8.1f ", at(data_field, pi, pj));
        }
        printf("\n");
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