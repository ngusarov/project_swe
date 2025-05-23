// xdmf_writer.cc
#include "xdmf_writer.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
// Removed HDF5 includes
#include <cassert>

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
  // Create directory (this part remains the same, as XDMF file itself goes here)
  // S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH gives rwxrwxr-x permissions
  if (mkdir(filename_prefix_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
    if (errno != EEXIST) { // EEXIST means directory already exists, which is fine.
      std::cerr << "Error creating directory " << filename_prefix_ << ": " << strerror(errno) << std::endl;
      // You might want to throw an exception or exit if directory creation is critical and fails for other reasons.
    }
  }
  // No HDF5 writing here anymore. This is now handled by SWESolver.
  // The root XDMF file will be written for the first time by SWESolver after
  // initial mesh and topography HDF5 files are created.
}

void
XDMFWriter::add_h_timestep(const double t)
{
  time_steps_.push_back(t);
  // The root XDMF file is rewritten to include the new time step
  this->write_root_xdmf();
}

void
XDMFWriter::write_root_xdmf() const
{
  // XDMF file itself is inside the directory
  std::ofstream xdmf_file(filename_prefix_ + "/" + filename_prefix_ + ".xdmf");

  // For XDMF, we need the global number of vertices and cells
  // The mesh is (global_nx) x (global_ny) cells, meaning (global_nx+1) x (global_ny+1) vertices.
  const std::size_t n_vertices = (global_nx_ + 1) * (global_ny_ + 1);
  const std::size_t n_cells = global_nx_ * global_ny_;

  if (xdmf_file.is_open())
  {
    xdmf_file << "<?xml version=\"1.0\" ?>\n";
    xdmf_file << "<Xdmf Version=\"3.0\" xmlns:xi=\"https://www.w3.org/2001/XInclude\">\n";

    xdmf_file << "  <Domain>\n";
    xdmf_file << "    <Grid Name=\"mesh\" GridType=\"Uniform\">\n";
    xdmf_file << "        <Topology TopologyType=\"Quadrilateral\" NumberOfElements=\"" << n_cells << "\">\n";
    // Path inside XDMF: HDF5 file is in the same directory as the XDMF file
    // Dimensions for cells are global_nx * global_ny, and 4 vertices per cell
    xdmf_file << "          <DataItem DataType=\"Int\" Format=\"HDF\" Dimensions=\"" << n_cells << " 4\">"
              << filename_prefix_ << "_mesh.h5:/cells</DataItem>\n"; // Filename relative to XDMF
    xdmf_file << "        </Topology>\n";
    xdmf_file << "        <Geometry GeometryType=\"XY\">\n";
    // Dimensions for vertices are (global_ny+1) * (global_nx+1) and 2 coordinates per vertex
    // Note: HDF5 stores row-major, so (num_rows, num_cols). For vertices, it's (total_vertices, 2).
    xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << n_vertices
              << " 2\">" << filename_prefix_ << "_mesh.h5:/vertices</DataItem>\n"; // Filename relative to XDMF
    xdmf_file << "        </Geometry>\n";
    xdmf_file << "    </Grid>\n";

    if (!time_steps_.empty())
    {
      xdmf_file << "    <Grid Name=\"h_collection\" GridType=\"Collection\" CollectionType=\"Temporal\">\n"; // Unique name for collection
      for (std::size_t i = 0; i < time_steps_.size(); ++i)
      {
        xdmf_file << "      <Grid Name=\"h_timestep_" << i << "\" GridType=\"Uniform\">\n"; // Unique name for each timestep grid
        xdmf_file
          << "        <xi:include xpointer =\"xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology "
             "or self::Geometry])\" />\n";
        xdmf_file << "        <Time Value=\"" << time_steps_[i] << "\"/>\n";

        xdmf_file << "        <Attribute Name=\"h\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        // Filename relative to XDMF. Dimensions for h are global_nx * global_ny (number of cells)
        xdmf_file << "          <DataItem DataType=\"Float\" Precision=\"8\" Format=\"HDF\" Dimensions=\"" << n_cells
                  << "\">" << filename_prefix_ << "_h_" << i << ".h5:/h</DataItem>\n";
        xdmf_file << "        </Attribute>\n";

        xdmf_file << "        <Attribute Name=\"topography\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        // Filename relative to XDMF. Dimensions for topography are global_nx * global_ny (number of cells)
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
  }
  else
  {
    std::cerr << "Error creating XDMF file: " << filename_prefix_ + "/" + filename_prefix_ + ".xdmf" << std::endl;
  }
}
