
#include "xdmf_writer.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cassert>

XDMFWriter::XDMFWriter(const std::string& filename_prefix,
                       const std::size_t nx,
                       const std::size_t ny,
                       const std::size_t size_x,
                       const std::size_t size_y,
                       const std::vector<double>& topography) :
  filename_prefix_(filename_prefix), nx_(nx), ny_(ny), size_x_(size_x), size_y_(size_y)
{
  this->write_mesh_hdf5();
  this->write_topography_hdf5(topography);
  this->write_root_xdmf();
}

void
XDMFWriter::add_h(const std::vector<double>& h, const double t)
{
  assert(h.size() == nx_ * ny_);
  time_steps_.push_back(t);

  this->write_root_xdmf();

  const std::string filename = filename_prefix_ + "_h_" + std::to_string(time_steps_.size() - 1) + ".h5";

  write_array_to_hdf5(filename, "h", h);
}

void
XDMFWriter::write_root_xdmf() const
{
  std::ofstream xdmf_file(filename_prefix_ + ".xdmf");

  const std::size_t n_vertices = (nx_ + 1) * (ny_ + 1);
  const std::size_t n_cells = nx_ * ny_;

  if (xdmf_file.is_open())
  {
    xdmf_file << "<?xml version=\"1.0\" ?>\n";
    xdmf_file << "<Xdmf Version=\"3.0\" xmlns:xi=\"https://www.w3.org/2001/XInclude\">\n";

    xdmf_file << "  <Domain>\n";
    xdmf_file << "    <Grid Name=\"mesh\" GridType=\"Uniform\">\n";
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
      xdmf_file << "    <Grid Name=\"h\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
      for (std::size_t i = 0; i < time_steps_.size(); ++i)
      {
        xdmf_file << "      <Grid Name=\"h\" GridType=\"Uniform\">\n";
        xdmf_file
          << "        <xi:include xpointer =\"xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology "
             "or self::Geometry])\" />\n";
        xdmf_file << "        <Time Value=\"" << time_steps_[i] << "\"/>\n";

        xdmf_file << "        <Attribute Name=\"h\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << n_cells
                  << "\">" << filename_prefix_ << "_h_" << i << ".h5:/h</DataItem>\n";
        xdmf_file << "        </Attribute>\n";

        xdmf_file << "        <Attribute Name=\"topography\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << n_cells
                  << "\">" << filename_prefix_ << "_topography.h5:/topography</DataItem>\n";
        xdmf_file << "        </Attribute>\n";

        xdmf_file << "      </Grid>\n";
      }
      xdmf_file << "    </Grid>\n";
    }

    xdmf_file << "  </Domain>\n";
    xdmf_file << "</Xdmf>\n";

    xdmf_file.close();
    // std::cout << "XDMF file created: " << filename_prefix_ + ".xdmf" << std::endl;
  }
  else
  {
    std::cerr << "Error creating XDMF file: " << filename_prefix_ + ".xdmf" << std::endl;
  }
}

void
XDMFWriter::create_vertices(std::vector<double>& vertices) const
{
  vertices.clear();
  vertices.reserve((nx_ + 1) * (ny_ + 1) * 2);
  const double dx = static_cast<double>(size_x_) / nx_;
  const double dy = static_cast<double>(size_y_) / ny_;
  for (std::size_t j = 0; j <= ny_; ++j)
  {
    for (std::size_t i = 0; i <= nx_; ++i)
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
  cells.reserve(nx_ * ny_ * 4);
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      cells.push_back(j * (nx_ + 1) + i);
      cells.push_back(j * (nx_ + 1) + i + 1);
      cells.push_back((j + 1) * (nx_ + 1) + i + 1);
      cells.push_back((j + 1) * (nx_ + 1) + i);
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

void
XDMFWriter::write_topography_hdf5(const std::vector<double>& topography) const
{
  const std::string filename = filename_prefix_ + "_topography.h5";
  write_array_to_hdf5(filename, "topography", topography);
}

void
XDMFWriter::write_array_to_hdf5(const std::string& filename,
                                const std::string& dataset_name,
                                const std::vector<double>& data)
{
  // Create the HDF5 file
  hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error creating HDF5 file: " << filename << std::endl;
    return;
  }
  // Write data.
  hsize_t dims[1];
  dims[0] = data.size();
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  const std::string dataset_name_ = "/" + dataset_name;
  hid_t dataset_id =
    H5Dcreate2(file_id, dataset_name_.c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error creating dataset" << std::endl;
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return;
  }
  H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  // Close the file
  H5Fclose(file_id);
}