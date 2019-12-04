#include "output.h"

#include <string> // for std::string
#include <fstream> // for std::fstream
#include <iostream>
#include <sstream>
#include <time.h> // for time_t, localtime, strftime

inline
const std::string current_date()
{
  /* get current time */
  time_t     now = time(NULL);

  /* Format and print the time, "ddd yyyy-mm-dd hh:mm:ss zzz" */
  struct tm  *ts;
  ts = localtime(&now);

  char       buf[80];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %Z", ts);

  return std::string(buf);

} // current_date

/**
 * \brief Save using XSM file format.
 * A one line header used in xsmurf.
 */
void save_bin(real_t* data, const char* prefix, int time, int sizeX, int sizeY)
{
  FILE *fp;
  char number[16];
  char filename[256] = "";
  
  // make file name
  strcat(filename,prefix);
  sprintf(number,"%03d.xsm",time);
  strcat(filename,number);
  
  // open, write data, close
  fp = fopen(filename,"w");
  fprintf(fp,"Binary 1 %dx%d %d(4 byte reals)\n",sizeX,sizeY,sizeX*sizeY);
  fwrite(data,sizeof(real_t), sizeX*sizeY, fp);
  fclose(fp);
  
} // save_bin

/**
 * \brief Save using PGM file format.
 */
void save_pgm(real_t* data, const char* prefix, int time, int sizeX, int sizeY)
{
  FILE *fp;
  char number[16];
  char filename[256] = "";
  
  // make file name
  strcat(filename,prefix);
  sprintf(number,"%03d.pgm",time);
  strcat(filename,number);
  
  // open, write data, close
  fp = fopen(filename,"w");
  fprintf(fp,"P5\n%d %d\n255\n",sizeX,sizeY);
  for (int index=0; index<sizeX*sizeY; ++index) {
    unsigned char tmp = (unsigned char) (data[index]*255.0f);
    fwrite(&tmp,sizeof(unsigned char), 1, fp);
  }
  fclose(fp);

} // save_pgm

/**
 * \brief Save results using MGL library and png file format
 */
void save_mgl(real_t* data, const char* prefix, int time, int sizeX, int sizeY)
{

#ifdef USE_MGL

  // build filename
  char number[16];
  char filename[256] = "";

  strcat(filename,prefix);
  sprintf(number,"%03d.png",time);
  strcat(filename,number);
  
  // create mglData object
  mglGraph *gr = new mglGraph;
  mglData a(sizeX,sizeY);
  a.Set(data,sizeX,sizeY);
  
  // do the plot
  gr->Rotate(40,60);
  gr->Light(true);
  gr->Box();
  gr->Surf(a);
  
  // save to file
  gr->WritePNG(filename, 0, true); 
  delete gr;

#else

  (void) data;
  (void) prefix;
  (void) time;
  (void) sizeX;
  (void) sizeY;

  printf("Warning : library MathGL is not available or disabled !\n");

#endif // USE_MGL

} // save_mgl

/**
 * \brief Save results using VTK file format (more precisely VTI for
 * image data).
 * \see http://www.vtk.org/Wiki/VTK_XML_Formats
 */
#define VTK_ASCII
void save_vtk(real_t* data, const char* prefix, int time)
{
  FILE *fp;
  char number[16];
  char filename[256] = "";
  
  // make file name
  strcat(filename,prefix);
  sprintf(number,"%03d.vti",time);
  strcat(filename,number);

  // open file
  fp = fopen(filename,"w");

  // if writing raw binary data (file does not respect XML standard)
#ifdef VTK_ASCII
  fprintf(fp,"<?xml version=\"1.0\"?>\n");
#endif // VTK_ASCII

  // write xml data header
  if (isBigEndian())
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n");
  else
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

  fprintf(fp, "  <ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",0,NX-1,0,NY-1,0,NZ-1);
  fprintf(fp, "  <Piece Extent=\"%d %d %d %d %d %d\">\n",0,NX-1,0,NY-1,0,NZ-1);
  fprintf(fp, "    <PointData>\n");
 

#ifdef VTK_ASCII

  // write ascii data
#ifdef USE_DOUBLE
  fprintf(fp, "      <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\" >", "scalar_data");
#else
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\" >", "scalar_data");
#endif // USE_DOUBLE

  for(unsigned int k = 0; k < NZ; k++) {
    for(unsigned int j = 0; j < NY; j++) {
      for(unsigned int i = 0; i < NX; i++) {
	unsigned int index = i+NX*(j+NY*k);
	fprintf(fp, "%g ", data[index]);
      }
      fprintf(fp, "\n");
    }
  }
  fprintf(fp, "      </DataArray>\n");

  // write trailer
  fprintf(fp, "    </PointData>\n");
  fprintf(fp, "    <CellData>\n");
  fprintf(fp, "    </CellData>\n");
  fprintf(fp, "  </Piece>\n");
  fprintf(fp, "  </ImageData>\n");
  fprintf(fp, "</VTKFile>\n");
	    
#else

  // write binary data using appended (but no base 64 encoding)
#ifdef USE_DOUBLE
  fprintf(fp, "      <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"0\" />\n","scalar_data"); 
#else
  fprintf(fp, "      <DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\" />\n","scalar_data"); 
#endif // USE_DOUBLE

  fprintf(fp, "    </PointData>\n");
  fprintf(fp, "    <CellData>\n");
  fprintf(fp, "    </CellData>\n");
  fprintf(fp, "  </Piece>\n");
  fprintf(fp, "  </ImageData>\n");
  
  fprintf(fp, "  <AppendedData encoding=\"raw\">\n");
  
  // write the leading undescore
  fprintf(fp, "_");
  
  // then write heavy data (column major format)
  unsigned int nbOfWords = NX*NY*NZ*sizeof(real_t);
  fwrite( (void *) &nbOfWords, sizeof(unsigned int), 1, fp);
  
  fwrite((void *)data, sizeof(real_t), NX*NY*NZ, fp);
  
  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
	
#endif // VTK_ASCII

  // close vtk file
  fclose(fp);

} // save_vtk

/**
 * \brief Save using HDF5 file format.
 *
 */
void save_hdf5(real_t* data, 
	       const char* prefix, 
	       int time, 
	       int compressionLevel)
{
#if USE_HDF5
  
  herr_t   status;
 
  FILE *fp;
  char number[16];
  char filename[256] = "";
  
  // make file name
  strcat(filename,prefix);
  sprintf(number,"%03d.h5",time);
  strcat(filename,number);

  // Create a new file using default properties.
  hid_t file_id = H5Fcreate(filename, 
			    H5F_ACC_TRUNC |  H5F_ACC_DEBUG, 
			    H5P_DEFAULT, 
			    H5P_DEFAULT);

  // create dataspace for memory and file
  hsize_t  dims_memory[3];
  hsize_t  dims_file[3];
  hid_t    dataspace_memory, dataspace_file;  
  if (NZ==1) { // 2D
    dims_memory[0] = NY; 
    dims_memory[1] = NX;
    dims_file[0] = NY;
    dims_file[1] = NX;
    dataspace_memory = H5Screate_simple(2, dims_memory, NULL);
    dataspace_file   = H5Screate_simple(2, dims_file  , NULL);
  } else { //3D
    dims_memory[0] = NZ; 
    dims_memory[1] = NY;
    dims_memory[2] = NX;
    dims_file[0] = NZ;
    dims_file[1] = NY;
    dims_file[2] = NX;
    dataspace_memory = H5Screate_simple(3, dims_memory, NULL);
    dataspace_file   = H5Screate_simple(3, dims_file  , NULL);
  }

  // create dataType
  hid_t dataType;
  if (sizeof(real_t) == sizeof(float))
    dataType = H5T_NATIVE_FLOAT;
  else
    dataType = H5T_NATIVE_DOUBLE;

  // Create the datasets
  if (NZ==1) {
    hsize_t  start[2] = {0, 0};
    hsize_t stride[2] = {1, 1};
    hsize_t  count[2] = {NY, NX};
    hsize_t  block[2] = {1, 1}; // row-major instead of column-major here
    status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
  } else {
    hsize_t  start[3] = {0, 0, 0};
    hsize_t stride[3] = {1, 1, 1};
    hsize_t  count[3] = {NZ, NY, NX};
    hsize_t  block[3] = {1, 1, 1}; // row-major instead of column-major here
    status = H5Sselect_hyperslab(dataspace_memory, H5S_SELECT_SET, start, stride, count, block);
  }

  /*
   * property list for compression
   */
  // get compression level (0=no compression; 9 is highest level of compression)
  if (compressionLevel < 0 or compressionLevel > 9) {
    fprintf(stderr, "Invalid value for compression level; must be an integer between 0 and 9 !!!\n");
    fprintf(stderr, "compression level is then set to default value 0; i.e. no compression !!\n");
    compressionLevel = 0;
  }

  hid_t propList_create_id = H5Pcreate(H5P_DATASET_CREATE);
  
  if (NZ==1) {
    const hsize_t chunk_size2D[2] = {NY, NX};
    H5Pset_chunk (propList_create_id, 2, chunk_size2D);
  } else {
    const hsize_t chunk_size3D[3] = {NZ, NY, NX};
    H5Pset_chunk (propList_create_id, 3, chunk_size3D);
  }
  
  H5Pset_shuffle (propList_create_id);
  H5Pset_deflate (propList_create_id, compressionLevel);
  
  /*
   * write heavy data to HDF5 file
   */
  
  // write temperature
   hid_t dataset_id = H5Dcreate2(file_id, "/temperature", dataType, dataspace_file, 
   				H5P_DEFAULT, propList_create_id, H5P_DEFAULT);
   status = H5Dwrite(dataset_id, dataType, dataspace_memory, dataspace_file, H5P_DEFAULT, data);

  /*
   * write time step number
   */
  hid_t ds_id;
  hid_t attr_id;
  {
    ds_id   = H5Screate(H5S_SCALAR);
    attr_id = H5Acreate2(file_id, "time step", H5T_NATIVE_INT, 
			 ds_id,
			 H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_INT, &time);
    status = H5Sclose(ds_id);
    status = H5Aclose(attr_id);
  }

  /*
   * write creation date
   */
  std::string dataString = current_date();
  const char *dataChar = dataString.c_str();
  hsize_t   dimsAttr[1] = {1};
  hid_t type = H5Tcopy (H5T_C_S1);
  status = H5Tset_size (type, H5T_VARIABLE);
  hid_t root_id = H5Gopen2(file_id, "/", H5P_DEFAULT);
  hid_t dataspace_id = H5Screate_simple(1, dimsAttr, NULL);
  attr_id = H5Acreate2(root_id, "creation date", type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attr_id, type, &dataChar);
  status = H5Aclose(attr_id);
  status = H5Gclose(root_id);
  status = H5Tclose(type);
  status = H5Sclose(dataspace_id);
  
  // close/release resources.
  H5Pclose(propList_create_id);
  //H5Pclose(propList_xfer_id);
  H5Sclose(dataspace_memory);
  H5Sclose(dataspace_file);
  //H5Dclose(dataset_id);
  H5Fflush(file_id, H5F_SCOPE_LOCAL);  H5Fclose(file_id);
  
#else

  (void) data;
  (void) prefix;
  (void) time;
  (void) compressionLevel;
  printf("Warning : library HDF5 is not available or disabled !\n");

#endif // USE_HDF5
} // save_hdf5

/**
 * \brief Write XDMF wrapper file allowing to load data in Paraview/Visit.
 *
 */
void write_xdmf_wrapper(const char* prefix, int totalNumberOfSteps, int deltaT) 
{

  // get data type as a string for Xdmf
  std::string dataTypeName;
  if (sizeof(real_t) == sizeof(float))
    dataTypeName = "Float";
  else
    dataTypeName = "Double";
  
  /*
   * 1. open XDMF and write header lines
   */
  std::string outputDir("./");
  std::string outputPrefix(prefix);
  std::string xdmfFilename = outputDir + outputPrefix + ".xmf";
  std::fstream xdmfFile;
  xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);
  
  xdmfFile << "<?xml version=\"1.0\" ?>"                       << std::endl;
  xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>"         << std::endl;
  xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
  xdmfFile << "  <Domain>"                                     << std::endl;
  xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;
  
  // for each time step write a <grid> </grid> item
  int startStep=0;
  int stopStep =totalNumberOfSteps;
  int deltaStep=deltaT;
  
  for (int nStep=startStep, iStep=0; nStep<stopStep; nStep+=deltaStep, iStep++) {
    
    std::ostringstream outNum;
    outNum.width(7);
    outNum.fill('0');
    outNum << iStep;
    
    // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
    char number[16];
    char hdf5Filename[256] = "";
    strcat(hdf5Filename,prefix);
    sprintf(number,"_%03d.h5",iStep);
    strcat(hdf5Filename,number);
    
    xdmfFile << "    <Grid Name=\"" << prefix << "\" GridType=\"Uniform\">" << std::endl;
    xdmfFile << "    <Time Value=\"" << iStep << "\" />"                      << std::endl;
      
    // topology CoRectMesh
    if (NZ==1) 
      xdmfFile << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"" << NY << " " << NX << "\"/>" << std::endl;
    else
      xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << NZ << " " << NY << " " << NX << "\"/>" << std::endl;
      
    // geometry
    if (NZ==1) {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDY\">"        << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"2\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1"                                 << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    } else {
      xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">"      << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Origin\""                     << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       0 0 0"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    <DataStructure"                         << std::endl;
      xdmfFile << "       Name=\"Spacing\""                    << std::endl;
      xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
      xdmfFile << "       Dimensions=\"3\""                    << std::endl;
      xdmfFile << "       Format=\"XML\">"                     << std::endl;
      xdmfFile << "       1 1 1"                               << std::endl;
      xdmfFile << "    </DataStructure>"                       << std::endl;
      xdmfFile << "    </Geometry>"                            << std::endl;
    }
      
    // temperature
    xdmfFile << "      <Attribute Center=\"Node\" Name=\"temperature\">" << std::endl;
    xdmfFile << "        <DataStructure"                             << std::endl;
    xdmfFile << "           DataType=\"" << dataTypeName <<  "\""    << std::endl;
    if (NZ==1)
      xdmfFile << "           Dimensions=\"" << NY << " " << NX << "\"" << std::endl;
    else
      xdmfFile << "           Dimensions=\"" << NZ << " " << NY << " " << NX << "\"" << std::endl;
    xdmfFile << "           Format=\"HDF\">"                         << std::endl;
    xdmfFile << "           "<<hdf5Filename<<":/temperature"             << std::endl;
    xdmfFile << "        </DataStructure>"                           << std::endl;
    xdmfFile << "      </Attribute>"                                 << std::endl;
      
    // finalize grid file for the current time step
    xdmfFile << "   </Grid>" << std::endl;
      
  } // end for loop over time step
    
    // finalize Xdmf wrapper file
  xdmfFile << "   </Grid>" << std::endl;
  xdmfFile << " </Domain>" << std::endl;
  xdmfFile << "</Xdmf>"    << std::endl;

} // write_xdmf_wrapper

/**
 * \brief Save using XSM file format.
 * A one line header used in xsmurf.
 */
void save_bin_3d(real_t* data, const char* prefix, int time, 
		 int sizeX, int sizeY, int sizeZ)
{
  FILE *fp;
  char number[16];
  char filename[256] = "";
  
  // make file name
  strcat(filename,prefix);
  sprintf(number,"%03d.xsm",time);
  strcat(filename,number);
  
  // open, write data, close
  fp = fopen(filename,"w");
  fprintf(fp,"Binary 1 %dx%dx%d %d(4 byte reals)\n",sizeX,sizeY,sizeZ,sizeX*sizeY*sizeZ);
  fwrite(data,sizeof(real_t), sizeX*sizeY*sizeZ, fp);
  fclose(fp);
  
} // save_bin_3d

/**
 * \brief Save results using MGL library and png file format
 */
void save_mgl_3d(real_t* data, const char* prefix, int time, 
		 int sizeX, int sizeY, int sizeZ)
{

#ifdef USE_MGL

  // build filename
  char number[16];
  char filename[256] = "";

  strcat(filename,prefix);
  sprintf(number,"%03d.png",time);
  strcat(filename,number);
  
  // create mglData object
  mglGraph *gr = new mglGraph;
  mglData a(sizeX,sizeY,sizeZ);
  a.Set(data,sizeX,sizeY,sizeZ);
  
  // do the plot
  gr->Rotate(40,60);
  gr->Light(true);
  gr->Box();
  
  gr->Alpha(true);
  gr->Surf3(a);
  /*gr->DensX(a.Sum("x"),"",-1);
  gr->DensY(a.Sum("y"),"",1);
  gr->DensZ(a.Sum("z"),"",-1);*/
  
  // save to file
  gr->WritePNG(filename, 0, true); 
  delete gr;

#else

  (void) data;
  (void) prefix;
  (void) time;
  (void) sizeX;
  (void) sizeY;
  (void) sizeZ;

  printf("Warning : library MathGL is not available or disabled !\n");

#endif // USE_MGL

} // save_mgl_3d
