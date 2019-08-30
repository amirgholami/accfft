#include <pnetcdf.h>
#include <cstdlib>
#include <string>
#include <iostream>

#define PNETCDF_HANDLE_ERROR {              \
    if (err != NC_NOERR)              \
      printf("PNetCDF Error at line %d (%s)\n", __LINE__,     \
       ncmpi_strerror(err));            \
}

enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};

template <typename T> inline MPI_Datatype real_mpitype();
template<> inline
MPI_Datatype real_mpitype<float>() { return MPI_FLOAT; }
template<> inline
MPI_Datatype real_mpitype<double>() { return MPI_DOUBLE; }

template <typename T> inline nc_type real_nctype();
template<> inline nc_type real_nctype<float>() { return NC_FLOAT; }
template<> inline nc_type real_nctype<double>() { return NC_DOUBLE; }

/**
 * Read a parallel-netcdf file.
 *
 * We assume here that localData is a scalar.
 *
 * Pnetcdf uses row-major format (same as FFTW).
 *
 * \param[in]  filename  : PnetCDF filename
 * \param[in]  topol : offset to where to start reading data
 * \param[in]  counts    : number of elements read (3D sub-domain inside global)
 * \param[in]  c_comm    : MPI Communicator
 * \param[in]  gsizes    : global sizes
 * \param[out] localData : actual data buffer (size : nx*ny*nz*sizeof(T))
 *
 * localData must have been allocated prior to calling this routine.
 */
template <typename T>
void read_pnetcdf(const std::string &filename,
    const MPI_Offset starts[3],
    const MPI_Offset counts[3],
    const MPI_Comm c_comm,
    const int gsizes[3],
    T *localData) {

  int myRank;
  MPI_Comm_rank(c_comm, &myRank);

  // netcdf file id
  int ncFileId;
  int err;

  // file opening mode
  int ncOpenMode = NC_NOWRITE;

  int nbVar=1;
  int varIds[nbVar];
  MPI_Info mpi_info_used;

  /*
   * Open NetCDF file
   */
  err = ncmpi_open(c_comm, filename.c_str(),
      ncOpenMode,
      MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_open() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
  }

  /*
   * Query NetCDF mode
   */
  int NC_mode;
  err = ncmpi_inq_version(ncFileId, &NC_mode);
  // if (myRank==0) {
  //  if (NC_mode == NC_64BIT_DATA)
  //  std::cout << "Pnetcdf Input mode : NC_64BIT_DATA (CDF-5)\n";
  //  else if (NC_mode == NC_64BIT_OFFSET)
  //  std::cout << "Pnetcdf Input mode : NC_64BIT_OFFSET (CDF-2)\n";
  //  else
  //  std::cout << "Pnetcdf Input mode : unknown " << NC_mode << std::endl;
  // }

  /*
   * Query information about variable named "data"
   */
  {
    int ndims, nvars, ngatts, unlimited;
    err = ncmpi_inq(ncFileId, &ndims, &nvars, &ngatts, &unlimited);
    PNETCDF_HANDLE_ERROR;

    err = ncmpi_inq_varid(ncFileId, "data", &varIds[0]);
    PNETCDF_HANDLE_ERROR;
  }

  /*
   * Define expected data types (no conversion done here)
   */
  MPI_Datatype mpiDataType = real_mpitype<T>();
  // MPI_FLOAT / MPI_DOUBLE;

  /*
   * Get all the MPI_IO hints used (just in case, we want to print it after
   * reading data...
   */
  err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
  PNETCDF_HANDLE_ERROR;

  /*
   * Read heavy data (take care of row-major / column major format !)
   */
  int nItems = counts[IX]*counts[IY]*counts[IZ];
  {

    err = ncmpi_get_vara_all(ncFileId,
        varIds[0],
        starts,
        counts,
        localData,
        nItems,
        mpiDataType);
    PNETCDF_HANDLE_ERROR;
  } // end reading heavy data

  /*
   * close the file
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;

} // read_pnetcdf

/**
 * Write a parallel-netcdf file.
 *
 * We assume here that localData is a scalar.
 *
 * Pnetcdf uses row-major format (same as FFTW).
 *
 * \param[in]  filename  : PnetCDF filename
 * \param[in]  starts    : offset to where to start reading data
 * \param[in]  counts    : number of elements read (3D sub-domain inside global)
 * \param[in]  c_comm    : MPI Communicator
 * \param[in]  gsizes    : global sizes
 * \param[in]  localData : actual data buffer (size : nx*ny*nz*sizeof(T))
 *
 */
template <typename T>
void write_pnetcdf(const std::string &filename,
    const MPI_Offset starts[3],
    const MPI_Offset counts[3],
    const MPI_Comm c_comm,
    const int gsizes[3],
    const T *localData) {
  int myRank;
  MPI_Comm_rank(c_comm, &myRank);

  // netcdf file id
  int ncFileId;
  int err;

  // file creation mode
  int ncCreationMode = NC_CLOBBER;

  // CDF-5 is almost mandatory for very large files (>= 2x10^9 cells)
  // not useful here
  bool useCDF5 = false;
  if (useCDF5)
    ncCreationMode = NC_CLOBBER|NC_64BIT_DATA;
  else // use CDF-2 file format
    ncCreationMode = NC_CLOBBER|NC_64BIT_OFFSET;

  // verbose log ?
  //bool pnetcdf_verbose = false;

  int nbVar=1;
  int dimIds[3], varIds[nbVar];
  //MPI_Offset write_size, sum_write_size;
  MPI_Info mpi_info_used;
  //char str[512];

  // time measurement variables
  //double write_timing, max_write_timing, write_bw;

  /*
   * Create NetCDF file
   */
  err = ncmpi_create(c_comm, filename.c_str(),
      ncCreationMode,
      MPI_INFO_NULL, &ncFileId);
  if (err != NC_NOERR) {
    printf("Error: ncmpi_create() file %s (%s)\n",filename.c_str(),ncmpi_strerror(err));
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(1);
  }

  /*
   * Define global dimensions
   */
  err = ncmpi_def_dim(ncFileId, "x", gsizes[0], &dimIds[0]);
  PNETCDF_HANDLE_ERROR;

  err = ncmpi_def_dim(ncFileId, "y", gsizes[1], &dimIds[1]);
  PNETCDF_HANDLE_ERROR;

  err = ncmpi_def_dim(ncFileId, "z", gsizes[2], &dimIds[2]);
  PNETCDF_HANDLE_ERROR;

  /*
   * Define variables to write (give a name)
   */
  nc_type ncDataType = real_nctype<T>(); // NC_FLOAT or NC_DOUBLE
  MPI_Datatype mpiDataType = real_mpitype<T>();

  err = ncmpi_def_var(ncFileId, "data", ncDataType, 3, dimIds, &varIds[0]);
  PNETCDF_HANDLE_ERROR;

  /*
   * global attributes
   */
  // did we use CDF-2 or CDF-5
  {
    int useCDF5_int = useCDF5 ? 1 : 0;
    err = ncmpi_put_att_int(ncFileId, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &useCDF5_int);
    PNETCDF_HANDLE_ERROR;
  }

  /*
   * exit the define mode
   */
  err = ncmpi_enddef(ncFileId);
  PNETCDF_HANDLE_ERROR;

  /*
   * Get all the MPI_IO hints used
   */
  err = ncmpi_get_file_info(ncFileId, &mpi_info_used);
  PNETCDF_HANDLE_ERROR;

  // copy data to write in intermediate buffer
  int nItems = counts[IX]*counts[IY]*counts[IZ];

  {

    // debug
    // printf("Pnetcdf [rank=%d] starts=%lld %lld %lld, counts =%lld %lld %lld, gsizes=%d %d %d\n",
    //     myRank,
    //     starts[0],starts[1],starts[2],
    //     counts[0],counts[1],counts[2],
    //     gsizes[0],gsizes[1],gsizes[2]);

    /*
     * make sure PNetCDF doesn't complain when starts is outside of global domain
     * bound. When nItems is null, off course we don't write anything, but starts
     * offset have to be inside global domain.
     * So there is no harm, setting starts to origin.
     */
    MPI_Offset start2[3] = {starts[0], starts[1], starts[2]};
    if (nItems == 0) {
      start2[0]=0;
      start2[1]=0;
      start2[2]=0;
    }

    err = ncmpi_put_vara_all(ncFileId,
        varIds[0],
        start2,
        counts,
        localData,
        nItems,
        mpiDataType);
    PNETCDF_HANDLE_ERROR;
  }

  /*
   * close the file
   */
  err = ncmpi_close(ncFileId);
  PNETCDF_HANDLE_ERROR;

} // write_pnetcdf

/* template instantiations */
template void read_pnetcdf(const std::string &, const MPI_Offset starts[3],
  const MPI_Offset counts[3], const MPI_Comm, const int gsizes[3], float *);
template void write_pnetcdf(const std::string &, const MPI_Offset starts[3],
  const MPI_Offset counts[3], const MPI_Comm, const int gsizes[3], const float *);
template void read_pnetcdf(const std::string &, const MPI_Offset starts[3],
  const MPI_Offset counts[3], const MPI_Comm, const int gsizes[3], double *);
template void write_pnetcdf(const std::string &, const MPI_Offset starts[3],
  const MPI_Offset counts[3], const MPI_Comm, const int gsizes[3], const double *);

