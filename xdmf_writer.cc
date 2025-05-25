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
XDMFWriter::create_vertices(std::vector<double>& vertices) const
{
  vertices.clear();
  vertices.reserve((global_nx_ + 1) * (global_ny_ + 1) * 2);
  const double dx = static_cast<double>(size_x_) / global_nx_;
  const double dy = static_cast<double>(size_y_) / global_ny_;
  for (std::size_t j = 0; j <= global_ny_; ++j)
  {
    for (std::size_t i = 0; i <= global_nx_; ++i)
    {
      vertices.push_back(i * dx);
      vertices.push_back(j * dy);
    }
  }
}

void
XDMFWriter::create_cells(std::vector<int>& cells) const
{
  cells.clear();
  cells.reserve(global_nx_ * global_ny_ * 4);
  for (std::size_t j = 0; j < global_ny_; ++j)
  {
    for (std::size_t i = 0; i < global_nx_; ++i)
    {
      cells.push_back(j * (global_nx_ + 1) + i);
      cells.push_back(j * (global_nx_ + 1) + i + 1);
      cells.push_back((j + 1) * (global_nx_ + 1) + i + 1);
      cells.push_back((j + 1) * (global_nx_ + 1) + i);
    }
  }
}

void
XDMFWriter::write_mesh_hdf5() const
{
  // Create the HDF5 file
  hid_t file_id = H5Fcreate((filename_prefix_ + "_mesh.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error creating HDF5 file: " << filename_prefix_ + "_mesh.h5" << std::endl;
    return;
  }

  std::vector<double> vertices;
  this->create_vertices(vertices);

  std::vector<int> cells;
  this->create_cells(cells);

  // Write vertices
  hsize_t dims[2];
  dims[0] = vertices.size() / 2;
  dims[1] = 2;
  hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
  hid_t dataset_id =
    H5Dcreate2(file_id, "/vertices", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error creating vertices dataset" << std::endl;
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return;
  }

  H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vertices.data());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Write cells
  dims[0] = cells.size() / 4;
  dims[1] = 4;
  dataspace_id = H5Screate_simple(2, dims, NULL);
  dataset_id = H5Dcreate2(file_id, "/cells", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error creating cells dataset" << std::endl;
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return;
  }

  H5Dwrite(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cells.data());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Close the file
  H5Fclose(file_id);

  // std::cout << "Mesh HDF5 file created: " << filename + "_mesh.h5" << std::endl;
}
