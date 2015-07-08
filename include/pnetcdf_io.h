#ifndef _PNETCDF_IO_H_
#define _PNETCDF_IO_H_

void read_pnetcdf(const std::string &filename,
		  MPI_Offset         starts[3],
		  MPI_Offset         counts[3],
		  int                gsizes[3],
		  double            *localData);

void write_pnetcdf(const std::string &filename,
		   MPI_Offset         starts[3],
		   MPI_Offset         counts[3],
		   int                gsizes[3],
		   double            *localData);

#endif // _PNETCDF_IO_H_
