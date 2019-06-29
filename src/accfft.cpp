/**
 * @file
 * CPU functions of AccFFT
 */
/*
 *  Copyright (c) 2014-2015, Amir Gholami, George Biros
 *  All rights reserved.
 *  This file is part of AccFFT library.
 *
 *  AccFFT is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  AccFFT is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with AccFFT.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "accfft.h"
#include "dtypes.h"
#define VERBOSE 0
#define PCOUT if(procid==0) std::cout
#ifdef ACCFFT_MKL
#include "mkl.h"
#endif

fftwf_plan plan_many_dft_r2c(int rank, const int *n, int howmany,
                  float *in, const int *inembed,
                  int istride, int idist,
                  Complexf *out, const int *onembed,
                  int ostride, int odist,
                  unsigned flags) {
    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, flags);
}
fftw_plan plan_many_dft_r2c(int rank, const int *n, int howmany,
                  double *in, const int *inembed,
                  int istride, int idist,
                  Complex *out, const int *onembed,
                  int ostride, int odist,
                  unsigned flags) {
    return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, flags);
}

fftwf_plan plan_many_dft_c2r(int rank, const int *n, int howmany,
                  Complexf *in, const int *inembed,
                  int istride, int idist,
                  float *out, const int *onembed,
                  int ostride, int odist,
                  unsigned flags) {
    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, flags);
}
fftw_plan plan_many_dft_c2r(int rank, const int *n, int howmany,
                  Complex *in, const int *inembed,
                  int istride, int idist,
                  double *out, const int *onembed,
                  int ostride, int odist,
                  unsigned flags) {
    return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, flags);
}

fftwf_plan plan_many_dft(int rank, const int *n, int howmany,
                  Complexf *in, const int *inembed,
                  int istride, int idist,
                  Complexf *out, const int *onembed,
                  int ostride, int odist,
                  int sign, unsigned flags) {
    return fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, sign, flags);
}
fftw_plan plan_many_dft(int rank, const int *n, int howmany,
                  Complex *in, const int *inembed,
                  int istride, int idist,
                  Complex *out, const int *onembed,
                  int ostride, int odist,
                  int sign, unsigned flags) {
    return fftw_plan_many_dft(rank, n, howmany, in, inembed, istride,
                    idist, out, onembed, ostride, odist, sign, flags);
}

fftwf_plan plan_guru_dft(int rank, const fftwf_iodim *n, int dims,
		  const fftwf_iodim *m, Complexf *in, Complexf *out,
		  int sign, unsigned flags) {
    return fftwf_plan_guru_dft(rank, n, dims, m, in, out, sign, flags);
}
fftw_plan plan_guru_dft(int rank, const fftw_iodim *n, int dims,
		  const fftw_iodim *m, Complex *in, Complex *out,
		  int sign, unsigned flags) {
    return fftw_plan_guru_dft(rank, n, dims, m, in, out, sign, flags);
}

/**
 * Creates a 3D R2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */

CMETHOD1(AccFFT)(
       const int *n, real *data, cplx *data_out,
       MPI_Comm comm, unsigned flags) {
  r2c_plan_baked = false;
  c2c_plan_baked = false;
  T_plan_1 = NULL;
  T_plan_1i = NULL;
  T_plan_2 = NULL;
  T_plan_2i = NULL;
  T_plan_x = NULL;
  T_plan_xi = NULL;
  T_plan_y = NULL;
  T_plan_yi = NULL;
  Mem_mgr = NULL;
  fplan_y = NULL;
  fplan_x = NULL;
  iplan_y = NULL;
  iplan_x = NULL;

  // TODO: manage comm ourselves?
  //accfft_create_comm(comm, c_dims, &c_comm);
  c_comm = comm;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Cart_get(c_comm, 2, np, periods, coord);
  MPI_Comm_split(c_comm, coord[0], coord[1], &row_comm);
  MPI_Comm_split(c_comm, coord[1], coord[0], &col_comm);

  N[0] = n[0];
  N[1] = n[1];
  N[2] = n[2];

  data_r = data;
  data_out_c = data_out;
  if (data_out == (cplx *)data) {
    inplace = true;
  } else {
    inplace = false;
  }

  unsigned fftw_flags;
  if (flags == ACCFFT_ESTIMATE)
    fftw_flags = FFTW_ESTIMATE;
  else
    fftw_flags = FFTW_MEASURE;

  if(np[1] == 1)
    oneD = true;
  else
    oneD = false;

  int alloc_local;
  int alloc_max = 0;
  int n_tuples_i = inplace == true ? (n[2] / 2 + 1) * 2 : n[2];
  int n_tuples_o = (n[2] / 2 + 1) * 2;

  //int isize[3],osize[3],istart[3],ostart[3];
  alloc_max = accfft_local_size_dft_r2c<real>(n, isize, istart,
      osize, ostart, c_comm);

  dfft_get_local_size<real>(n[0], n[1], n_tuples_o / 2, osize_0, ostart_0, c_comm);
  dfft_get_local_size<real>(n[0], n_tuples_o / 2, n[1], osize_1, ostart_1, c_comm);
  dfft_get_local_size<real>(n[1], n_tuples_o / 2, n[0], osize_2, ostart_2, c_comm);

  std::swap(osize_1[1], osize_1[2]);
  std::swap(ostart_1[1], ostart_1[2]);

  std::swap(ostart_2[1], ostart_2[2]);
  std::swap(ostart_2[0], ostart_2[1]);
  std::swap(osize_2[1], osize_2[2]);
  std::swap(osize_2[0], osize_2[1]);

  // osize_y is the configuration after y transpose
  dfft_get_local_size<real>(n[0], n[2], n[1], osize_y, ostart_y, c_comm);
  std::swap(osize_y[1], osize_y[2]);
  std::swap(ostart_y[1], ostart_y[2]);
  osize_yi[0] = osize_y[0];
  osize_yi[1] = osize_y[1] / 2 + 1;
  osize_yi[2] = osize_y[2];

  std::swap(osize_y[1], osize_y[2]); // N0/P0 x N2/P1 x N1
  std::swap(osize_yi[1], osize_yi[2]);
  // for x only fft. The strategy is to divide (N1/P1 x N2) by P0 completely.
  // So we treat the last size to be just 1.
  dfft_get_local_size<real>(isize[1] * n[2], n[2], n[0], osize_x, ostart_x, c_comm);
  osize_x[1] = osize_x[0];
  osize_x[0] = osize_x[2];
  osize_x[2] = 1;
  std::swap(osize_x[0], osize_x[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1

  dfft_get_local_size<real>(isize[1] * n[2], n[2], n[0] / 2 + 1, osize_xi, ostart_x, c_comm);
  osize_xi[1] = osize_xi[0];
  osize_xi[0] = osize_xi[2];
  osize_xi[2] = 1;
  std::swap(osize_xi[0], osize_xi[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1
  ostart_x[0] = 0;
  ostart_x[1] = -1<<8; // starts have no meaning in this approach
  ostart_x[2] = -1<<8;

  for (int i = 0; i < 3; i++) {
    osize_1i[i] = osize_1[i];
    osize_2i[i] = osize_2[i];
    ostart_1i[i] = ostart_1[i];
    ostart_2i[i] = ostart_2[i];
  }

  // FFT Plans
  {
    real *dummy_data = (real *)accfft_alloc(alloc_max);
    fplan_0 = plan_many_dft_r2c(1, &n[2],
        osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
        data, NULL,         //real *in, const int *inembed,
        1, n_tuples_i,      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n_tuples_o / 2,        // int ostride, int odist,
        fftw_flags);
    if (fplan_0 == NULL)
      std::cout << "!!! fplan_0 not created in r2c plan!!!" << std::endl;

    iplan_0 = plan_many_dft_c2r(1, &n[2],
        osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
        data_out, NULL, //real *in, const int *inembed,
        1, n_tuples_o / 2,      //int istride, int idist,
        data, NULL, //fftw_cplx *out, const int *onembed,
        1, n_tuples_i,        // int ostride, int odist,
        fftw_flags);
    if (iplan_0 == NULL)
      std::cout << "!!! iplan_0 not created in r2c plan!!!" << std::endl;

    // ---- fplan 1

    fftw_iodim dims, howmany_dims[2];
    dims.n = osize_1[1];
    dims.is = osize_1[2];
    dims.os = osize_1[2];

    howmany_dims[0].n = osize_1[2];
    howmany_dims[0].is = 1;
    howmany_dims[0].os = 1;

    howmany_dims[1].n = osize_1[0];
    howmany_dims[1].is = osize_1[1] * osize_1[2];
    howmany_dims[1].os = osize_1[1] * osize_1[2];

#ifdef ACCFFT_MKL
    fplan_1 = plan_many_dft(1, &n[1], osize_1[2] * osize_1[0], //int rank, const int *n, int howmany
    data_out, NULL,        //real *in, const int *inembed,
        1, n[1],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n[1],        // int ostride, int odist,
        FFTW_FORWARD, fftw_flags);
    if (fplan_1 == NULL)
      std::cout << "!!! fplan1 not created in r2c plan !!!" << std::endl;

    iplan_1 = plan_many_dft(1, &n[1], osize_1[2] * osize_1[0], //int rank, const int *n, int howmany
    data_out, NULL,        //real *in, const int *inembed,
        1, n[1],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n[1],        // int ostride, int odist,
        FFTW_BACKWARD, fftw_flags);
    if (iplan_1 == NULL)
      std::cout << "!!! iplan1 not created in r2c plan !!!" << std::endl;

#else
    fplan_1 = plan_guru_dft(1, &dims, 2, howmany_dims,
        data_out, data_out, -1,
        fftw_flags);
    if (fplan_1 == NULL)
      std::cout << "!!! fplan1f not created in r2c plan!!!" << std::endl;
    iplan_1 = plan_guru_dft(1, &dims, 2, howmany_dims,
        data_out, data_out, 1,
        fftw_flags);
    if (iplan_1 == NULL)
      std::cout << "!!! iplan1f not created in r2c plan !!!" << std::endl;
#endif

    // fplan_y ----
    fplan_y = plan_many_dft_r2c(1, &n[1],
        osize_y[0]*osize_y[1], //int rank, const int *n, int howmany
        dummy_data, NULL,         //real *in, const int *inembed,
        1, n[1],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1,n[1]/2+1,        // int ostride, int odist,
        fftw_flags);
    if (fplan_y == NULL)
      std::cout << "!!! fplan_y not created in r2c plan!!!" << std::endl;
    iplan_y = plan_many_dft_c2r(1, &n[1],
        osize_y[0]*osize_y[1], //int rank, const int *n, int howmany
        data_out, NULL, //real *in, const int *inembed,
        1, n[1]/2+1,      //int istride, int idist,
        (real *)data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n[1],        // int ostride, int odist,
        fftw_flags);
    if (iplan_y == NULL)
      std::cout << "!!! iplan_y not created in r2c plan!!!" << std::endl;
    // ----

#ifdef ACCFFT_MKL
    fplan_2 = plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
        data_out, NULL,        //real *in, const int *inembed,
        1,n[0],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1,n[0],        // int ostride, int odist,
        FFTW_FORWARD, fftw_flags);
    if (fplan_2 == NULL)
      std::cout << "!!! fplan2 not created in r2c plan !!!" << std::endl;
    iplan_2 = plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
        data_out, NULL,        //double *in, const int *inembed,
        1,n[0],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1,n[0],        // int ostride, int odist,
        FFTW_BACKWARD, fftw_flags);
    if (iplan_2 == NULL)
      std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;
#else
    fplan_2 = plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
    data_out, NULL,        //real *in, const int *inembed,
        osize_2[2] * osize_2[1], 1,      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
        FFTW_FORWARD, fftw_flags);
    if (fplan_2 == NULL)
      std::cout << "!!! fplan2 not created in r2c plan !!!" << std::endl;

    iplan_2 = plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
    data_out, NULL,        //real *in, const int *inembed,
        osize_2[2] * osize_2[1], 1,      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
        FFTW_BACKWARD, fftw_flags);
    if (iplan_2 == NULL)
      std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;
#endif

    // fplan_x
    // PCOUT << "osize_x[0] = " << osize_x[0] << " osize_x[1] = " << osize_x[1] << " osize_x[2] = " << osize_x[2] << std::endl;
    fplan_x = plan_many_dft_r2c(1, &n[0],
        osize_x[0], //int rank, const int *n, int howmany
        dummy_data, NULL,         //real *in, const int *inembed,
        1, n[0],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1,n[0]/2+1,        // int ostride, int odist,
        fftw_flags);
    if (fplan_x == NULL)
      std::cout << "!!! fplan_x not created in r2c plan!!!" << std::endl;
    iplan_x = plan_many_dft_c2r(1, &n[0],
        osize_x[0], //int rank, const int *n, int howmany
        data_out, NULL, //real *in, const int *inembed,
        1, n[0]/2+1,      //int istride, int idist,
        (real *)data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n[0],        // int ostride, int odist,
        fftw_flags);
    if (iplan_x == NULL)
      std::cout << "!!! iplan_x not created in r2c plan!!!" << std::endl;

    //fplan_x = plan_many_dft_r2c(1, &n[0],
    //    osize_x[0] * osize_x[2], //int rank, const int *n, int howmany
    //    dummy_data, NULL,         //real *in, const int *inembed,
    //    osize_x[0] * osize_x[2], 1,      //int istride, int idist,
    //    data_out, NULL, //fftw_cplx *out, const int *onembed,
    //    osize_xi[0] * osize_xi[2], 1,        // int ostride, int odist,
    //    fftw_flags);
    //if (fplan_x == NULL)
    //  std::cout << "!!! fplan_xf not created in r2c plan!!!" << std::endl;

    //iplan_x = plan_many_dft_c2r(1, &n[0],
    //    osize_xi[0] * osize_xi[2], //int rank, const int *n, int howmany
    //    data_out, NULL, //real *in, const int *inembed,
    //    osize_xi[0] * osize_xi[2], 1,      //int istride, int idist,
    //    data_out, NULL, //fftw_cplx *out, const int *onembed,
    //    osize_x[0] * osize_x[2], 1,        // int ostride, int odist,
    //    fftw_flags);
    //if (iplan_x == NULL)
    //  std::cout << "!!! iplan_xf not created in r2c plan!!!" << std::endl;
    accfft_free(dummy_data);
  }

  // 1D Decomposition
  if (oneD) {
    Mem_mgr = new Mem_Mgr<real>(n[0], n[1], n_tuples_o, c_comm, 1, alloc_max);
    T_plan_2 = new T_Plan<real>(n[0], n[1], n_tuples_o,
        Mem_mgr, c_comm);
    T_plan_2i = new T_Plan<real>(n[1], n[0], n_tuples_o,
        Mem_mgr, c_comm);
    T_plan_1 = NULL;
    T_plan_1i = NULL;

    T_plan_2->alloc_local = alloc_max;
    T_plan_2i->alloc_local = alloc_max;

    if (flags == ACCFFT_MEASURE) {
      T_plan_2->which_fast_method(T_plan_2, (real *)data_out, 2);
    } else {
      T_plan_2->method = 2;
      T_plan_2->kway = 2;
    }
    T_plan_2i->method = -T_plan_2->method;
    T_plan_2i->kway = T_plan_2->kway;
    T_plan_2i->kway_async = T_plan_2->kway_async;
    data = data;

    data = data;

  } // end 1D Decomp r2c

  // 2D Decomposition
  if (!oneD) {
    Mem_mgr = new Mem_Mgr<real>(n[1], n_tuples_o / 2, 2,
        row_comm, osize_0[0], alloc_max);
    T_plan_1 = new T_Plan<real>(n[1], n_tuples_o / 2, 2,
        Mem_mgr, row_comm, osize_0[0]);
    T_plan_2 = new T_Plan<real>(n[0], n[1], osize_2[2] * 2,
        Mem_mgr, col_comm);
    T_plan_2i = new T_Plan<real>(n[1], n[0], osize_2i[2] * 2,
        Mem_mgr, col_comm);
    T_plan_1i = new T_Plan<real>(n_tuples_o / 2, n[1], 2,
        Mem_mgr, row_comm, osize_1i[0]);


    T_plan_1->alloc_local = alloc_max;
    T_plan_2->alloc_local = alloc_max;
    T_plan_2i->alloc_local = alloc_max;
    T_plan_1i->alloc_local = alloc_max;


    if (flags == ACCFFT_MEASURE) {
      if (coord[0] == 0) {
        T_plan_1->which_fast_method(T_plan_1,
            (real *)data_out, 2, osize_0[0], coord[0]);
      }
    } else {
      if (coord[0] == 0) {
        T_plan_1->method = 2;
        T_plan_1->kway = 2;
      }
    }

    MPI_Bcast(&T_plan_1->method, 1, MPI_INT, 0, c_comm);
    MPI_Bcast(&T_plan_1->kway, 1, MPI_INT, 0, c_comm);
    MPI_Bcast(&T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);

    T_plan_2->method = T_plan_1->method;
    T_plan_2i->method = -T_plan_1->method;
    T_plan_1i->method = -T_plan_1->method;

    T_plan_2->kway = T_plan_1->kway;
    T_plan_2i->kway = T_plan_1->kway;
    T_plan_1i->kway = T_plan_1->kway;

    T_plan_2->kway_async = T_plan_1->kway_async;
    T_plan_2i->kway_async = T_plan_1->kway_async;
    T_plan_1i->kway_async = T_plan_1->kway_async;

    data = data;
  } // end 2D r2c

  // T_plan_x has to be created for both oneD and !oneD
  // Note that the T method is set via T_plan_2 for oneD case
  // to transpose N0/P0 x N1/P1 x N2 -> N0 x (N1/P1 x N2)/P0
  T_plan_x = new T_Plan<real>(n[0], isize[1] * isize[2], 1,
      Mem_mgr, col_comm, 1);
  // to transpose N0 x (N1/P1 x N2)/P0 -> N0/P0 x N1/P1 x N2
  T_plan_xi = new T_Plan<real>(isize[1] * isize[2], n[0], 1,
      Mem_mgr, col_comm, 1);
  T_plan_x->alloc_local = alloc_max;
  T_plan_xi->alloc_local = alloc_max;
  T_plan_x->method = T_plan_2->method;
  T_plan_xi->method = T_plan_2->method; // notice that we do not use minus for T_planxi
  T_plan_x->kway = T_plan_2->kway;
  T_plan_xi->kway = T_plan_2->kway;
  T_plan_x->kway_async = T_plan_2->kway_async;
  T_plan_xi->kway_async = T_plan_2->kway_async;

  // to transpose N0/P0 x N1/P1 x N2 -> N0/P0 x N1 x N2/P1
  T_plan_y = new T_Plan<real>(n[1], n[2], 1,
      Mem_mgr, row_comm, isize[0]);
  // to transpose N0/P0 x N1 x N2/P1 -> N0/P0 x N1/P1 x N2
  T_plan_yi = new T_Plan<real>(n[2], n[1], 1,
      Mem_mgr, row_comm, isize[0]);
  T_plan_y->alloc_local = alloc_max;
  T_plan_yi->alloc_local = alloc_max;
  T_plan_y->method = T_plan_2->method;
  T_plan_yi->method = T_plan_2->method; // should not be set to minus
  T_plan_y->kway_async = T_plan_2->kway_async;
  T_plan_yi->kway_async = T_plan_2->kway_async;
  T_plan_y->kway = T_plan_2->kway;
  T_plan_yi->kway = T_plan_2->kway;


  r2c_plan_baked = true;
} // end accfft_plan_dft_3d_r2c

/**
 * Creates a 3D C2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */

CMETHOD1(AccFFT)(
       const int *n, cplx *data, cplx *data_out,
       MPI_Comm ic_comm, unsigned flags) {
  r2c_plan_baked = false;
  c2c_plan_baked = false;
  T_plan_1 = NULL;
  T_plan_1i = NULL;
  T_plan_2 = NULL;
  T_plan_2i = NULL;
  T_plan_x = NULL;
  T_plan_xi = NULL;
  T_plan_y = NULL;
  T_plan_yi = NULL;
  Mem_mgr = NULL;
  fplan_y = NULL;
  fplan_x = NULL;
  iplan_y = NULL;
  iplan_x = NULL;

  c_comm = ic_comm;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Cart_get(c_comm, 2, np, periods, coord);
  MPI_Comm_split(c_comm, coord[0], coord[1], &row_comm);
  MPI_Comm_split(c_comm, coord[1], coord[0], &col_comm);
  N[0] = n[0];
  N[1] = n[1];
  N[2] = n[2];

  data_c = data;
  data_out_c = data_out;
  if (data_out == data) {
    inplace = true;
  } else {
    inplace = false;
  }

  unsigned fftw_flags;
  if (flags == ACCFFT_ESTIMATE)
    fftw_flags = FFTW_ESTIMATE;
  else
    fftw_flags = FFTW_MEASURE;

  if (np[1] == 1)
    oneD = true;
  else
    oneD = false;

  // FFT Plans
  int alloc_local;
  int alloc_max = 0, n_tuples = (n[2] / 2 + 1) * 2;

  //int isize[3],osize[3],istart[3],ostart[3];
  alloc_max = accfft_local_size_dft_c2c<real>(n, isize, istart,
      osize, ostart, c_comm);

  dfft_get_local_size<real>(n[0], n[1], n[2], osize_0, ostart_0, c_comm);
  dfft_get_local_size<real>(n[0], n[2], n[1], osize_1, ostart_1, c_comm);
  dfft_get_local_size<real>(n[1], n[2], n[0], osize_2, ostart_2, c_comm);

  std::swap(osize_1[1], osize_1[2]);
  std::swap(ostart_1[1], ostart_1[2]);

  std::swap(ostart_2[1], ostart_2[2]);
  std::swap(ostart_2[0], ostart_2[1]);
  std::swap(osize_2[1], osize_2[2]);
  std::swap(osize_2[0], osize_2[1]);

  for (int i = 0; i < 3; i++) {
    osize_1i[i] = osize_1[i];
    osize_2i[i] = osize_2[i];
    ostart_1i[i] = ostart_1[i];
    ostart_2i[i] = ostart_2[i];
  }

  {
    // fplan_0
    fplan_0 = plan_many_dft(1, &n[2], osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
    data, NULL,         //real *in, const int *inembed,
        1, n[2],      //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        1, n[2],        // int ostride, int odist,
        FFTW_FORWARD, fftw_flags);
    if (fplan_0 == NULL)
      std::cout << "!!! fplan0 not created in c2c plan!!!" << std::endl;
    iplan_0 = plan_many_dft(1, &n[2], osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
    data_out, NULL,         //real *in, const int *inembed,
        1, n[2],      //int istride, int idist,
        data, NULL, //fftw_cplx *out, const int *onembed,
        1, n[2],        // int ostride, int odist,
        FFTW_BACKWARD, fftw_flags);
    if (iplan_0 == NULL)
      std::cout << "!!! iplan0 not created in c2c plan!!!" << std::endl;

    // fplan_1
    fftw_iodim dims, howmany_dims[2];
    dims.n = osize_1[1];
    dims.is = osize_1[2];
    dims.os = osize_1[2];

    howmany_dims[0].n = osize_1[2];
    howmany_dims[0].is = 1;
    howmany_dims[0].os = 1;

    howmany_dims[1].n = osize_1[0];
    howmany_dims[1].is = osize_1[1] * osize_1[2];
    howmany_dims[1].os = osize_1[1] * osize_1[2];

    fplan_1 = plan_guru_dft(1, &dims, 2, howmany_dims, data_out,
        data_out, -1, fftw_flags);
    if (fplan_1 == NULL)
      std::cout << "!!! fplan1 not created in c2c plan !!!" << std::endl;
    iplan_1 = plan_guru_dft(1, &dims, 2, howmany_dims,
        (cplx *)data_out, (cplx *)data_out, 1,
        fftw_flags);
    if (iplan_1 == NULL)
      std::cout << "!!! iplan1 not created in c2c plan !!!" << std::endl;

    // fplan_2
    fplan_2 = plan_many_dft(1, &n[0], osize_2[1] * osize_2[2], //int rank, const int *n, int howmany
    data_out, NULL,         //real *in, const int *inembed,
        osize_2[1] * osize_2[2], 1,     //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        osize_2[1] * osize_2[2], 1,       // int ostride, int odist,
        FFTW_FORWARD, fftw_flags);
    if (fplan_2 == NULL)
      std::cout << "!!! fplan2 not created in c2c plan!!!" << std::endl;
    iplan_2 = plan_many_dft(1, &n[0], osize_2[1] * osize_2[2], //int rank, const int *n, int howmany
    data_out, NULL,         //flaot *in, const int *inembed,
        osize_2[1] * osize_2[2], 1,     //int istride, int idist,
        data_out, NULL, //fftw_cplx *out, const int *onembed,
        osize_2[1] * osize_2[2], 1,       // int ostride, int odist,
        FFTW_BACKWARD, fftw_flags);
    if (iplan_2 == NULL)
      std::cout << "!!! fplan2 not created in c2c plan!!!" << std::endl;

  }

  // 1D Decomposition
  if (oneD) {
    int NX = n[0], NY = n[1], NZ = n[2];
    Mem_mgr = new Mem_Mgr<real>(NX, NY, (NZ) * 2, c_comm, 1, alloc_max);
    T_plan_2 = new T_Plan<real>(NX, NY, (NZ) * 2, Mem_mgr,
        c_comm);
    T_plan_2i = new T_Plan<real>(NY, NX, NZ * 2, Mem_mgr,
        c_comm);

    T_plan_1 = NULL;
    T_plan_1i = NULL;

    alloc_max = alloc_max;
    T_plan_2->alloc_local = alloc_max;
    T_plan_2i->alloc_local = alloc_max;

    if (flags == ACCFFT_MEASURE) {
      T_plan_2->which_fast_method(T_plan_2, (real *)data_out, 2);
    } else {
      T_plan_2->method = 2;
      T_plan_2->kway = 2;
    }
    T_plan_2i->method = -T_plan_2->method;
    T_plan_2i->kway = T_plan_2->kway;
    T_plan_2i->kway_async = T_plan_2->kway_async;

  } // end 1D decomp c2c

  // 2D Decomposition
  if (!oneD) {
    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers

    Mem_mgr = new Mem_Mgr<real>(n[1], n[2], 2, row_comm,
        osize_0[0], alloc_max);
    T_plan_1 = new T_Plan<real>(n[1], n[2], 2, Mem_mgr,
        row_comm, osize_0[0]);
    T_plan_2 = new T_Plan<real>(n[0], n[1], 2 * osize_2[2],
        Mem_mgr, col_comm);
    T_plan_2i = new T_Plan<real>(n[1], n[0], 2 * osize_2i[2],
        Mem_mgr, col_comm);
    T_plan_1i = new T_Plan<real>(n[2], n[1], 2, Mem_mgr,
        row_comm, osize_1i[0]);

    T_plan_1->alloc_local = alloc_max;
    T_plan_2->alloc_local = alloc_max;
    T_plan_2i->alloc_local = alloc_max;
    T_plan_1i->alloc_local = alloc_max;

    if (flags == ACCFFT_MEASURE) {
      if (coord[0] == 0) {
        T_plan_1->which_fast_method(T_plan_1,
            (real *)data_out, 2, osize_0[0], coord[0]);
      }
    } else {
      if (coord[0] == 0) {
        T_plan_1->method = 2;
        T_plan_1->kway = 2;
      }
    }

    MPI_Bcast(&T_plan_1->method, 1, MPI_INT, 0, c_comm);
    MPI_Bcast(&T_plan_1->kway, 1, MPI_INT, 0, c_comm);
    MPI_Bcast(&T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);

    T_plan_2->method = T_plan_1->method;
    T_plan_2i->method = -T_plan_1->method;
    T_plan_1i->method = -T_plan_1->method;

    T_plan_2->kway = T_plan_1->kway;
    T_plan_2i->kway = T_plan_1->kway;
    T_plan_1i->kway = T_plan_1->kway;

    T_plan_2->kway_async = T_plan_1->kway_async;
    T_plan_2i->kway_async = T_plan_1->kway_async;
    T_plan_1i->kway_async = T_plan_1->kway_async;

  } // end 2D Decomp c2c

  c2c_plan_baked = true;
} // end accfft_plan_dft_3d_c2c

/**
 * Execute R2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in spatial domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
CMETHOD(void, execute_r2c)(
        real *data, cplx *data_out, double *timer, std::bitset<3> XYZ) {
  if (!r2c_plan_baked) {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
    return;
  }

  this->execute(-1, data, (real *)data_out, timer, XYZ);
  return;
}

/**
 * Execute C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transform, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
CMETHOD(void, execute_c2r)(cplx *data,
        real *data_out, double *timer, std::bitset<3> XYZ) {
  if (!r2c_plan_baked) {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
    return;
  }

  this->execute(1, (real *)data, data_out, timer, XYZ);
  return;
}

/**
 * Helper function so that execute_plan can be templated.
 * @param fftw plan
 * @param input data
 * @param output data
 */
static void execute_fftw(fftwf_plan p, Complexf *data, Complexf *out) {
    fftwf_execute_dft(p, data, out);
}
static void execute_fftw(fftw_plan p, Complex *data, Complex *out) {
    fftw_execute_dft(p, data, out);
}
static void execute_fftw_r2c(fftwf_plan p, float *data, Complexf *out) {
    fftwf_execute_dft_r2c(p, data, out);
}
static void execute_fftw_r2c(fftw_plan p, double *data, Complex *out) {
    fftw_execute_dft_r2c(p, data, out);
}
static void execute_fftw_c2r(fftwf_plan p, Complexf *data, float *out) {
    fftwf_execute_dft_c2r(p, data, out);
}
static void execute_fftw_c2r(fftw_plan p, Complex *data, double *out) {
    fftw_execute_dft_c2r(p, data, out);
}

/**
 * Execute C2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
/*#define C2C_ARGS(real, cplx, fftw_tplan, accfft_plan) \
        accfft_plan *plan, int direction, cplx *data, cplx *data_out, double *timer, std::bitset<3> XYZ
#define C2C_RET(real, cplx, fftw_tplan, accfft_plan) void
INST_TPL(accfft_execute_c2c, C2C_RET, C2C_ARGS, plan, direction, data, data_out, timer, XYZ) */

CMETHOD(void, execute_c2c)(int direction,
        cplx *data, cplx *data_out, double *timer, std::bitset<3> XYZ) {
  if (!c2c_plan_baked) {
    if (procid == 0)
      std::cout
          << "Error. c2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
    return;
  }

  if (data == NULL)
    data = data_c;
  if (data_out == NULL)
    data_out = data_out_c;
  double fft_time = 0;
  double timings[5] = { 0 };

  if (direction == -1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    fft_time -= MPI_Wtime();
    if (XYZ[2])
      execute_fftw(fplan_0, data, data_out);
    else
      data_out = data;
    fft_time += MPI_Wtime();
    // TODO: replace remaining code with execute(110 & XYZ)

    if (!oneD) {
      T_plan_1->execute(T_plan_1, (real *)data_out, timings,
          2, osize_0[0], coord[0]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
    if (XYZ[1])
      execute_fftw(fplan_1, (cplx *)data_out, (cplx *)data_out);
    fft_time += MPI_Wtime();

    if (!oneD) {
      T_plan_2->execute(T_plan_2, (real *)data_out, timings,
          2, 1, coord[1]);
    } else {
      T_plan_2->execute(T_plan_2, (real *)data_out, timings,
          2);
    }
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
    if (XYZ[0])
      execute_fftw(fplan_2, (cplx *)data_out, (cplx *)data_out);
    fft_time += MPI_Wtime();
  } else if (direction == 1) {
    fft_time -= MPI_Wtime();
    if (XYZ[0])
      execute_fftw(iplan_2, (cplx *)data, (cplx *)data);
    fft_time += MPI_Wtime();

    if (!oneD) {
      T_plan_2i->execute(T_plan_2i, (real *)data, timings, 1,
          1, coord[1]);
    } else {
      T_plan_2i->execute(T_plan_2i, (real *)data, timings,
          1);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
    if (XYZ[1])
      execute_fftw(iplan_1, (cplx *)data, (cplx *)data);
    fft_time += MPI_Wtime();

    if (!oneD) {
      T_plan_1i->execute(T_plan_1i, (real *)data, timings, 1,
          osize_1i[0], coord[0]);
    }
    //MPI_Barrier(c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    // IFFT in Z direction
    fft_time -= MPI_Wtime();
    if (XYZ[2])
      execute_fftw(iplan_0, data, data_out);
    else
      data_out = data;
    fft_time += MPI_Wtime();

  }

  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //MPI_Barrier(c_comm);

  return;
}

/**
 * Helper function so that accfft_destroy_plan can be templated.
 * @param fftw plan to be destroyed
 */
static void destroy_fftw(fftwf_plan p) {
    fftwf_destroy_plan(p);
}
static void destroy_fftw(fftw_plan p) {
    fftw_destroy_plan(p);
}

/**
 * Destroy AccFFT CPU plan.
 * @param plan Input plan to be destroyed.
 */
CMETHOD1(~AccFFT)() {
    if (T_plan_1 != NULL)
      delete (T_plan_1);
    if (T_plan_1i != NULL)
      delete (T_plan_1i);
    if (T_plan_2 != NULL)
      delete (T_plan_2);
    if (T_plan_2i != NULL)
      delete (T_plan_2i);
    if (Mem_mgr != NULL)
      delete (Mem_mgr);
    if (fplan_0 != NULL)
      destroy_fftw(fplan_0);
    if (fplan_1 != NULL)
      destroy_fftw(fplan_1);
    if (fplan_2 != NULL)
      destroy_fftw(fplan_2);
    if (iplan_0 != NULL)
      destroy_fftw(iplan_0);
    if (iplan_1 != NULL)
      destroy_fftw(iplan_1);
    if (iplan_2 != NULL)
      destroy_fftw(iplan_2);
    if (fplan_x != NULL)
      destroy_fftw(fplan_x);
    if (fplan_y != NULL)
      destroy_fftw(fplan_y);
    if (iplan_x != NULL)
      destroy_fftw(iplan_x);
    if (iplan_y != NULL)
      destroy_fftw(iplan_y);
    if (T_plan_x != NULL)
      delete (T_plan_x);
    if (T_plan_xi != NULL)
      delete (T_plan_xi);
    if (T_plan_y != NULL)
      delete (T_plan_y);
    if (T_plan_yi != NULL)
      delete (T_plan_yi);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

CMETHOD(void, execute_z)(int direction,
        real *data, real *data_out, double *timer) {

  if (data == NULL)
    data = data_r;
  if (data_out == NULL)
    data_out = (real *)data_out_c;
  double fft_time = 0;
  double timings[5] = { 0 };

  if (direction == -1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    fft_time -= MPI_Wtime();
      execute_fftw_r2c(fplan_0, (real *)data, (cplx *)data_out);
    fft_time += MPI_Wtime();

  } else if (direction == 1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // IFFT in Z direction
    fft_time -= MPI_Wtime();
      execute_fftw_c2r(iplan_0, (cplx *)data, (real *)data_out);
    fft_time += MPI_Wtime();
    //MPI_Barrier(c_comm);
  }

  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //MPI_Barrier(c_comm);

  return;
}

CMETHOD(void, execute_r2c_z)(
        real *data, cplx *data_out, double *timer) {
  if (r2c_plan_baked) {
    this->execute_z(-1, data, (real *)data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}

CMETHOD(void, execute_c2r_z)(
        cplx *data, real *data_out, double *timer) {
  if (r2c_plan_baked) {
    this->execute_z(1, (real *)data, data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}

// templates for execution only in y direction
CMETHOD(void, execute_y)(int direction,
        real *data, real *data_out, double *timer) {

  if (data == NULL)
    data = data_r;
  if (data_out == NULL)
    data_out = (real *)data_out_c;
  double fft_time = 0;
  double timings[5] = { 0 };

  // 1D Decomposition
  real *cwork = Mem_mgr->buffer_3;
  int64_t N_local = isize[0] * isize[1] * isize[2];

  if (direction == -1) {
    // timings[0] += -MPI_Wtime();
    // memcpy(cwork, data, N_local * sizeof(real));
    // timings[0] += +MPI_Wtime();
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // Perform N0/P0 transpose
    // if (!oneD) {
    //   T_plan_y->execute(T_plan_y, cwork, timings, 2,
    //       osize_0[0], coord[0]);
    // }
    T_plan_y->execute(T_plan_y, data, timings, 0,
        osize_y[0], coord[0], cwork);
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
      execute_fftw_r2c(fplan_y, (real *)cwork, (cplx *)data_out);
    fft_time += MPI_Wtime();

  } else if (direction == 1) {
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
      execute_fftw_c2r(iplan_y, (cplx *)data, (real *)cwork);
    fft_time += MPI_Wtime();

    // if (!oneD) {
    //   T_plan_yi->execute(T_plan_yi, cwork, timings, 1,
    //       osize_yi[0], coord[0]);
    // }

    T_plan_yi->execute(T_plan_yi, cwork, timings, 0,
        osize_yi[0], coord[0], data_out);

    // timings[0] += -MPI_Wtime();
    // memcpy(data_out, cwork, N_local * sizeof(real));
    // timings[0] += +MPI_Wtime();
    //MPI_Barrier(c_comm);
  }

  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //MPI_Barrier(c_comm);

  return;
}

CMETHOD(void, execute_r2c_y)(
        real *data, cplx *data_out, double *timer) {
  if (r2c_plan_baked) { 
    this->execute_y(-1, data, (real *)data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}

CMETHOD(void, execute_c2r_y)(
        cplx *data, real *data_out, double *timer) {
  if (r2c_plan_baked) {
    this->execute_y(1, (real *)data, data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}


CMETHOD(void, execute_x)(int direction,
        real *data, real *data_out, double *timer) {

  if (data == NULL)
    data = data_r;
  if (data_out == NULL)
    data_out = (real *)data_out_c;
  double fft_time = 0;
  double timings[5] = { 0 };

  // 1D Decomposition
  int64_t N_local = isize[0] * isize[1] * isize[2];

  real *cwork = Mem_mgr->buffer_3;
  if (direction == -1) {
    // timings[0] += -MPI_Wtime();
    // memcpy(cwork, data, N_local * sizeof(real));
    // timings[0] += +MPI_Wtime();
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // T_plan_x->execute(T_plan_x, cwork, timings);
    T_plan_x->execute(T_plan_x, data, timings, 0, 1, 0, cwork);
    /**************************************************************/
    /****************  (N1/P1 x N2)/P0 x N0 x 1 *******************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
    execute_fftw_r2c(fplan_x, (real *)cwork, (cplx *)data_out);
    fft_time += MPI_Wtime();
    // memcpy(data_out, cwork, alloc_max);

  } else if (direction == 1) {
    /**************************************************************/
    /****************  (N1/P1 x N2)/P0 x N0 x 1 *******************/
    /**************************************************************/
    fft_time -= MPI_Wtime();
    execute_fftw_c2r(iplan_x, (cplx *)data, (real *)data);
    fft_time += MPI_Wtime();

    // T_plan_xi->execute(T_plan_xi, data, timings);
    T_plan_xi->execute(T_plan_xi, data, timings, 0, 1, 0, data_out);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    // timings[0] += -MPI_Wtime();
    // memcpy(data_out, data, N_local * sizeof(real));
    // timings[0] += +MPI_Wtime();
  }
  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //MPI_Barrier(c_comm);

  return;
}

CMETHOD(void, execute_r2c_x)(
        real *data, cplx *data_out, double *timer) {
  if (r2c_plan_baked) { 
    this->execute_x(-1, data, (real *)data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}

CMETHOD(void, execute_c2r_x)(
        cplx *data, real *data_out, double *timer) {
  if (r2c_plan_baked) {
    this->execute_x(1, (real *)data, data_out, timer);
  } else {
    if (procid == 0)
      std::cout
          << "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
          << std::endl;
  }
}

CMETHOD(void, execute)(int direction,
        real *data, real *data_out, double *timer, std::bitset<3> XYZ) {
  if (data == NULL)
    data = data_r;
  if (data_out == NULL)
    data_out = (real *)data_out_c;
  double fft_time = 0;
  double timings[5] = { 0 };

  // 1D Decomposition
  if (direction == -1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    fft_time -= MPI_Wtime();
    if (XYZ[2])
      execute_fftw_r2c(fplan_0, (real *)data, (cplx *)data_out);
    else
      data_out = data;
    fft_time += MPI_Wtime();

    // Perform N0/P0 transpose
    if (!oneD) {
      T_plan_1->execute(T_plan_1, data_out, timings, 2,
          osize_0[0], coord[0]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
#ifdef ACCFFT_MKL
    if (XYZ[1]){
      MKL_cplx alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      for(int i = 0; i < osize_1[0]; ++i)
        mkl_cimatcopy ('r', 't',  //const char ordering, const char trans,
                       osize_1[1], osize_1[2], //size_t rows, size_t cols,
                       alpha, (MKL_cplx*)&data_out[2*i*osize_1[1]*osize_1[2]], //const MKL_cplx alpha, MKL_cplx *AB,
                       osize_1[2],osize_1[1]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

      fft_time -= MPI_Wtime();
      execute_fftw(fplan_1, (cplx *)data_out, (cplx *)data_out);
      fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      for(int i = 0; i < osize_1[0]; ++i)
        mkl_cimatcopy ('r', 't',  //const char ordering, const char trans,
                       osize_1[2], osize_1[1],//size_t rows, size_t cols,
                       alpha, (MKL_cplx*)&data_out[2*i*(osize_1[1])*osize_1[2]], //const MKL_cplx alpha, MKL_cplx *AB,
                       osize_1[1],osize_1[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();
    }
#else
    fft_time -= MPI_Wtime();
    if (XYZ[1])
      execute_fftw(fplan_1, (cplx *)data_out, (cplx *)data_out);
    fft_time += MPI_Wtime();
#endif

    if (oneD) {
      T_plan_2->execute(T_plan_2, data_out, timings, 2);
    } else {
      T_plan_2->execute(T_plan_2, data_out, timings, 2, 1,
          coord[1]);
    }
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
#ifdef ACCFFT_MKL
    if (XYZ[0]){
      MKL_cplx alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      mkl_cimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[0], osize_2[1] * osize_2[2], //size_t rows, size_t cols,
                     alpha, (MKL_cplx*)data_out, //const MKL_cplx alpha, MKL_cplx *AB,
                     osize_2[1] * osize_2[2], osize_2[0]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

      fft_time -= MPI_Wtime();
      execute_fftw(fplan_2, (cplx *)data_out, (cplx *)data_out);
      fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      mkl_cimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[1] * osize_2[2], osize_2[0],//size_t rows, size_t cols,
                     alpha, (MKL_cplx*)data_out, //const MKL_cplx alpha, MKL_cplx *AB,
                     osize_2[0], osize_2[1] * osize_2[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

    }
#else
    fft_time -= MPI_Wtime();
    if (XYZ[0])
      execute_fftw(fplan_2, (cplx *)data_out, (cplx *)data_out);
    fft_time += MPI_Wtime();
#endif
  } else if (direction == 1) {
#ifdef ACCFFT_MKL
    if (XYZ[0]) {
      MKL_cplx alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      mkl_cimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[0], osize_2[1] * osize_2[2], //size_t rows, size_t cols,
                     alpha, (MKL_cplx*)data, //const MKL_cplx alpha, MKL_cplx *AB,
                     osize_2[1] * osize_2[2], osize_2[0]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

      fft_time -= MPI_Wtime();
      execute_fftw(iplan_2, (cplx *)data, (cplx *)data);
      fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      mkl_cimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[1] * osize_2[2], osize_2[0],//size_t rows, size_t cols,
                     alpha, (MKL_cplx*)data, //const MKL_cplx alpha, MKL_cplx *AB,
                     osize_2[0], osize_2[1] * osize_2[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();
    }
#else
    fft_time -= MPI_Wtime();
    if (XYZ[0])
      execute_fftw(iplan_2, (cplx *)data, (cplx *)data);
    fft_time += MPI_Wtime();
#endif

    if (oneD) {
      T_plan_2i->execute(T_plan_2i, data, timings, 1);
    } else {
      T_plan_2i->execute(T_plan_2i, data, timings, 1, 1,
          coord[1]);
    }

    //MPI_Barrier(c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
#ifdef ACCFFT_MKL
    if (XYZ[1]){
      MKL_cplx alpha;
      alpha.real =1.0;
      alpha.imag = 0;
      for(int i = 0; i < osize_1[0]; ++i){
        mkl_cimatcopy ('r','t',  //const char ordering, const char trans,
                       osize_1i[1], osize_1i[2],//size_t rows, size_t cols,
                       alpha, (MKL_cplx*)&data[2*i*osize_1i[1]*osize_1i[2]], //const MKL_cplx alpha, MKL_cplx *AB,
                       osize_1i[2], osize_1i[1]); //size_t lda, size_t ldb
      }

      fft_time -= MPI_Wtime();
      execute_fftw(iplan_1, (cplx *)data, (cplx *)data);
      fft_time += MPI_Wtime();

      for(int i = 0; i < osize_1[0]; ++i)
        mkl_cimatcopy ('r','t',  //const char ordering, const char trans,
                       osize_1i[2], osize_1i[1],//size_t rows, size_t cols,
                       alpha, (MKL_cplx*)&data[2*i*(osize_1i[1])*osize_1i[2]], //const MKL_cplx alpha, MKL_cplx *AB,
                       osize_1i[1],osize_1i[2]); //size_t lda, size_t ldb
    }
#else
    fft_time -= MPI_Wtime();
    if (XYZ[1])
      execute_fftw(iplan_1, (cplx *)data, (cplx *)data);
    fft_time += MPI_Wtime();
#endif

    if (!oneD) {
      T_plan_1i->execute(T_plan_1i, data, timings, 1, osize_1i[0], coord[0]);
    }
    //MPI_Barrier(c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // IFFT in Z direction
    fft_time -= MPI_Wtime();
    if (XYZ[2])
      execute_fftw_c2r(iplan_0, (cplx *)data, (real *)data_out);
    else
      data_out = data;
    fft_time += MPI_Wtime();
    //MPI_Barrier(c_comm);

  }

  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //MPI_Barrier(c_comm);

  return;
}
