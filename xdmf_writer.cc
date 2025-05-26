// xdmf_writer.cc
#include "xdmf_writer.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip> // For std::fixed, std::setprecision

#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mkdir
#include <cerrno>      // For errno
#include <cstring>   // for strerror()

XDMFWriter::XDMFWriter(const std::string& filename_prefix,
                       const std::size_t global_nx,
                       const std::size_t global_ny,
                       const double size_x,
                       const double size_y) :
  filename_prefix_(filename_prefix), global_nx_(global_nx), global_ny_(global_ny), size_x_(size_x), size_y_(size_y)
{
  if (mkdir(filename_prefix_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
    if (errno != EEXIST) {
      std::cerr << "Error creating directory " << filename_prefix_ << ": " << strerror(errno) << std::endl;
    }
  }
}

void
XDMFWriter::add_h_timestep(const double t)
{
  time_steps_.push_back(t);
  this->write_root_xdmf();
}

void
XDMFWriter::write_root_xdmf() const
{
  std::ofstream xdmf_file(filename_prefix_ + "/" + filename_prefix_ + ".xdmf");

  const std::size_t n_vertices = (global_nx_ + 1) * (global_ny_ + 1);
  const std::size_t n_cells = global_nx_ * global_ny_;

  if (xdmf_file.is_open())
  {
    xdmf_file << "<?xml version=\"1.0\" ?>\n";
    xdmf_file << "<Xdmf Version=\"3.0\" xmlns:xi=\"https://www.w3.org/2001/XInclude\">\n";

    xdmf_file << "  <Domain>\n";
    // Define the mesh using explicit Topology and Geometry from _mesh.h5
    xdmf_file << "    <Grid Name=\"mesh_explicit\" GridType=\"Uniform\">\n"; // Give it a unique name
    xdmf_file << "        <Topology TopologyType=\"Quadrilateral\" NumberOfElements=\"" << n_cells << "\">\n";
    xdmf_file << "          <DataItem DataType=\"Int\" Format=\"HDF\" Dimensions=\"" << n_cells << " 4\">"
              << filename_prefix_ << "_mesh.h5:/cells</DataItem>\n";
    xdmf_file << "        </Topology>\n";
    xdmf_file << "        <Geometry GeometryType=\"XY\">\n";
    xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << n_vertices
              << " 2\">" << filename_prefix_ << "_mesh.h5:/vertices</DataItem>\n";
    xdmf_file << "        </Geometry>\n";
    xdmf_file << "    </Grid>\n";


    if (!time_steps_.empty())
    {
      xdmf_file << "    <Grid Name=\"h_collection\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
      for (std::size_t i = 0; i < time_steps_.size(); ++i)
      {
        xdmf_file << "      <Grid Name=\"h_timestep_" << i << "\" GridType=\"Uniform\">\n";
        // Include the topology and geometry from the explicitly defined mesh
        xdmf_file
          << "        <xi:include xpointer =\"xpointer(/Xdmf/Domain/Grid[@Name='mesh_explicit']/*[self::Topology " // <--- Point to 'mesh_explicit'
             "or self::Geometry])\" />\n";
        xdmf_file << "        <Time Value=\"" << time_steps_[i] << "\"/>\n";

        xdmf_file << "        <Attribute Name=\"h\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << global_ny_ << " " << global_nx_
                  << "\">" << filename_prefix_ << "_h_" << i << ".h5:/h</DataItem>\n";
        xdmf_file << "        </Attribute>\n";

        xdmf_file << "        <Attribute Name=\"topography\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << global_ny_ << " " << global_nx_
                  << "\">" << filename_prefix_ << "_topography.h5:/topography</DataItem>\n";
        xdmf_file << "        </Attribute>\n";

        xdmf_file << "      </Grid>\n";
      }
      xdmf_file << "    </Grid>\n";
    }

    xdmf_file << "  </Domain>\n";
    xdmf_file << "</Xdmf>\n";

    xdmf_file.close();
  }
  else
  {
    std::cerr << "Error creating XDMF file: " << filename_prefix_ + "/" + filename_prefix_ + ".xdmf" << std::endl;
  }
}


void
XDMFWriter::create_vertices_parallel(int rank, int num_procs, std::vector<double>& local_block_vertices) const
{
  const std::size_t n_global_vertices_total = (global_nx_ + 1) * (global_ny_ + 1);

  const std::size_t base_n_local_v = n_global_vertices_total / num_procs;
  const std::size_t remainder_n_local_v = n_global_vertices_total % num_procs;

  const std::size_t num_vertices_this_rank_writes = base_n_local_v + (rank < remainder_n_local_v ? 1 : 0);
  const std::size_t global_start_vertex_idx = rank * base_n_local_v + std::min(rank, (int)remainder_n_local_v);

  local_block_vertices.clear();
  if (num_vertices_this_rank_writes == 0) return;
  local_block_vertices.reserve(num_vertices_this_rank_writes * 2);

  const double dx = size_x_ / global_nx_;
  const double dy = size_y_ / global_ny_;

  for (std::size_t k = 0; k < num_vertices_this_rank_writes; ++k)
  {
    const std::size_t current_global_vertex_idx = global_start_vertex_idx + k;
    const std::size_t global_idx_i = current_global_vertex_idx % (global_nx_ + 1); // Vertex column index (0 to global_nx_)
    const std::size_t global_idx_j = current_global_vertex_idx / (global_nx_ + 1); // Vertex row index (0 to global_ny_)

    local_block_vertices.push_back(static_cast<double>(global_idx_i) * dx);
    local_block_vertices.push_back(static_cast<double>(global_idx_j) * dy);
  }
}

void
XDMFWriter::create_cells_parallel(int rank, int num_procs, std::vector<int>& local_block_cells) const
{
  const std::size_t n_global_cells_total = global_nx_ * global_ny_;

  const std::size_t base_n_local_c = n_global_cells_total / num_procs;
  const std::size_t remainder_n_local_c = n_global_cells_total % num_procs;

  const std::size_t num_cells_this_rank_writes = base_n_local_c + (rank < remainder_n_local_c ? 1 : 0);
  const std::size_t global_start_cell_idx = rank * base_n_local_c + std::min(rank, (int)remainder_n_local_c);
  
  local_block_cells.clear();
  if (num_cells_this_rank_writes == 0) return;
  local_block_cells.reserve(num_cells_this_rank_writes * 4);

  for (std::size_t k = 0; k < num_cells_this_rank_writes; ++k)
  {
    const std::size_t current_global_cell_idx = global_start_cell_idx + k;
    const std::size_t global_cell_i = current_global_cell_idx % global_nx_; // Cell column index (0 to global_nx_ - 1)
    const std::size_t global_cell_j = current_global_cell_idx / global_nx_; // Cell row index (0 to global_ny_ - 1)

    // Vertex indices for this cell (global_cell_i, global_cell_j)
    // (bottom-left, bottom-right, top-right, top-left)
    local_block_cells.push_back(global_cell_j * (global_nx_ + 1) + global_cell_i);
    local_block_cells.push_back(global_cell_j * (global_nx_ + 1) + global_cell_i + 1);
    local_block_cells.push_back((global_cell_j + 1) * (global_nx_ + 1) + global_cell_i + 1);
    local_block_cells.push_back((global_cell_j + 1) * (global_nx_ + 1) + global_cell_i);
  }
}

void
XDMFWriter::write_mesh_hdf5_parallel(MPI_Comm comm, int rank, int num_procs) const
{
  std::string mesh_h5_filepath = filename_prefix_ + "/" + filename_prefix_ + "_mesh.h5";
  if (rank == 0) {
      printf("Attempting to write parallel mesh HDF5 file: %s\n", mesh_h5_filepath.c_str());
      fflush(stdout);
  }

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

  hid_t file_id = H5Fcreate(mesh_h5_filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id); // Close fapl_id after it's used by H5Fcreate

  if (file_id < 0) {
    std::cerr << "Rank " << rank << ": Error creating HDF5 mesh file: " << mesh_h5_filepath << std::endl;
    return;
  }

  // === Write Vertices ===
  std::vector<double> local_vertices_data;
  create_vertices_parallel(rank, num_procs, local_vertices_data);

  const hsize_t n_global_vertices_total = (global_nx_ + 1) * (global_ny_ + 1);
  hsize_t global_dims_v[2] = {n_global_vertices_total, 2};
  
  hsize_t num_vertices_this_rank_writes = local_vertices_data.size() / 2; // Each vertex has 2 components
  hsize_t local_count_v[2] = {num_vertices_this_rank_writes, 2};

  const std::size_t base_n_local_v = n_global_vertices_total / num_procs;
  const std::size_t remainder_n_local_v = n_global_vertices_total % num_procs;
  const std::size_t global_start_vertex_idx_for_hdf5 = rank * base_n_local_v + std::min(rank, (int)remainder_n_local_v);
  hsize_t offset_v[2] = {global_start_vertex_idx_for_hdf5, 0};
  
  hid_t filespace_v_id = H5Screate_simple(2, global_dims_v, NULL);
  hid_t memspace_v_id = H5Screate_simple(2, local_count_v, NULL);
  
  hid_t dataset_v_id = H5Dcreate2(file_id, "/vertices", H5T_NATIVE_DOUBLE, filespace_v_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_v_id < 0) {
    std::cerr << "Rank " << rank << ": Error creating /vertices dataset" << std::endl;
    H5Sclose(memspace_v_id);
    H5Sclose(filespace_v_id);
    H5Fclose(file_id);
    return;
  }

  H5Sselect_hyperslab(filespace_v_id, H5S_SELECT_SET, offset_v, NULL, local_count_v, NULL);

  hid_t xfer_plist_v_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer_plist_v_id, H5FD_MPIO_COLLECTIVE);

  if (num_vertices_this_rank_writes > 0) { // Important check for empty local data
    herr_t status_v = H5Dwrite(dataset_v_id, H5T_NATIVE_DOUBLE, memspace_v_id, filespace_v_id,
                               xfer_plist_v_id, local_vertices_data.data());
    if (status_v < 0) {
      std::cerr << "Rank " << rank << ": Error writing /vertices data" << std::endl;
    }
  } else {
    // Rank has no vertices to write, still participate in collective call if necessary
    // H5Dwrite with zero elements might still be needed by some HDF5 versions for true collective behavior
    // Or, ensure that ranks with no data do not call H5Dcreate/H5Dwrite if that's problematic.
    // For hyperslab, if count is 0, it should be fine.
    // The current create_vertices_parallel ensures local_vertices_data is empty if count is 0.
  }
  
  H5Pclose(xfer_plist_v_id);
  H5Dclose(dataset_v_id);
  H5Sclose(memspace_v_id);
  H5Sclose(filespace_v_id);

  // === Write Cells ===
  std::vector<int> local_cells_data;
  create_cells_parallel(rank, num_procs, local_cells_data);

  const hsize_t n_global_cells_total = global_nx_ * global_ny_;
  hsize_t global_dims_c[2] = {n_global_cells_total, 4};

  hsize_t num_cells_this_rank_writes = local_cells_data.size() / 4; // Each cell has 4 components
  hsize_t local_count_c[2] = {num_cells_this_rank_writes, 4};

  const std::size_t base_n_local_c = n_global_cells_total / num_procs;
  const std::size_t remainder_n_local_c = n_global_cells_total % num_procs;
  const std::size_t global_start_cell_idx_for_hdf5 = rank * base_n_local_c + std::min(rank, (int)remainder_n_local_c);
  hsize_t offset_c[2] = {global_start_cell_idx_for_hdf5, 0};

  hid_t filespace_c_id = H5Screate_simple(2, global_dims_c, NULL);
  hid_t memspace_c_id = H5Screate_simple(2, local_count_c, NULL);

  hid_t dataset_c_id = H5Dcreate2(file_id, "/cells", H5T_NATIVE_INT, filespace_c_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_c_id < 0) {
    std::cerr << "Rank " << rank << ": Error creating /cells dataset" << std::endl;
    H5Sclose(memspace_c_id);
    H5Sclose(filespace_c_id);
    H5Fclose(file_id);
    return;
  }
  
  H5Sselect_hyperslab(filespace_c_id, H5S_SELECT_SET, offset_c, NULL, local_count_c, NULL);

  hid_t xfer_plist_c_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer_plist_c_id, H5FD_MPIO_COLLECTIVE);

  if (num_cells_this_rank_writes > 0) { // Important check for empty local data
     herr_t status_c = H5Dwrite(dataset_c_id, H5T_NATIVE_INT, memspace_c_id, filespace_c_id,
                                xfer_plist_c_id, local_cells_data.data());
     if (status_c < 0) {
       std::cerr << "Rank " << rank << ": Error writing /cells data" << std::endl;
     }
  }

  H5Pclose(xfer_plist_c_id);
  H5Dclose(dataset_c_id);
  H5Sclose(memspace_c_id);
  H5Sclose(filespace_c_id);

  // Close the file
  H5Fclose(file_id);

  if (rank == 0) {
    std::cout << "Parallel mesh HDF5 file created: " << mesh_h5_filepath << std::endl;
  }
}