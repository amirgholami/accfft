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

#ifndef ACCFFT_H
#define ACCFFT_H
#include <string.h>
#include <string>
#include <bitset>
#include "transpose.h"
#include "accfft_common.h"

/**
 * template parameters:
 *    real = T = double / float
 *    cplx = Tc = Complex / Complexf
 *    fftw_ptype = P = fftw_plan / fftwf_plan
 */
template <typename real> class get_fftw_plan_type;

template<> class get_fftw_plan_type<float> {
  public:
    typedef fftwf_plan result;
};
template<> class get_fftw_plan_type<double> {
  public:
    typedef fftw_plan result;
};

template <typename real>
class AccFFT {
  typedef typename get_fftw_plan_type<real>::result fftw_ptype;
  typedef real cplx[2];
  int alloc_max;
  T_Plan<real> *T_plan_1;
  T_Plan<real> *T_plan_2;
  T_Plan<real> *T_plan_2i;
  T_Plan<real> *T_plan_1i;
  T_Plan<real> *T_plan_y;
  T_Plan<real> *T_plan_yi;
  T_Plan<real> *T_plan_x;
  T_Plan<real> *T_plan_xi;

  fftw_ptype fplan_0, iplan_0, fplan_1, iplan_1, fplan_2, iplan_2;
  fftw_ptype fplan_y, fplan_x, iplan_y, iplan_x;

  int coord[2], np[2], periods[2];
  MPI_Comm row_comm, col_comm;

  real* data_r;
  cplx* data_c;
  cplx* data_out_c;
  int procid;
  bool oneD;
  bool inplace;

  public:

    /* FIXME: these are required by operators.cpp for now */
    int N[3];
    Mem_Mgr<real> *Mem_mgr;
    MPI_Comm c_comm;

    int osize_0[3], ostart_0[3];
    int osize_1[3], ostart_1[3];
    int osize_2[3], ostart_2[3];
    int osize_1i[3], ostart_1i[3];
    int osize_2i[3], ostart_2i[3];
    int osize_y[3], osize_yi[3], ostart_y[3];
    int osize_x[3], osize_xi[3], ostart_x[3];

    int isize[3], istart[3];
    int osize[3], ostart[3];

    bool r2c_plan_baked;
    bool c2c_plan_baked;
    /* FIXME: operators should be implemented with better defined scope */

      AccFFT(const int *n, real *data, cplx *data_out, MPI_Comm c_comm, unsigned flags = ACCFFT_MEASURE);
      AccFFT(const int *n, cplx *data, cplx *data_out, MPI_Comm c_comm, unsigned flags = ACCFFT_MEASURE);
      ~AccFFT();

    void execute_r2c(real *data = NULL, cplx *data_out = NULL,
            double *timer = NULL, std::bitset<3> xyz = 111);
    void execute_c2r(cplx *data = NULL, real *data_out = NULL,
            double *timer = NULL, std::bitset<3> xyz = 111);
    void execute_c2c(int direction, cplx *data = NULL, cplx *data_out = NULL,
            double *timer = NULL, std::bitset<3> xyz = 111);
    void execute(int direction, real *data = NULL, real *data_out = NULL,
            double *timer = NULL, std::bitset<3> xyz = 111);
    void execute_z(int direction, real *data, real *out, double *timer);
    void execute_y(int direction, real *data, real *out, double *timer);
    void execute_x(int direction, real *data, real *out, double *timer);

    void execute_r2c_z(real *data, cplx *out, double *timer = NULL);
    void execute_c2r_z(cplx *data, real *out, double *timer = NULL);
    void execute_r2c_y(real *data, cplx *out, double *timer = NULL);
    void execute_c2r_y(cplx *data, real *out, double *timer = NULL);
    void execute_r2c_x(real *data, cplx *out, double *timer = NULL);
    void execute_c2r_x(cplx *data, real *out, double *timer = NULL);
};

typedef class AccFFT<float> AccFFTs;
typedef class AccFFT<double> AccFFTd;

template class AccFFT<float>;
template class AccFFT<double>;

int accfft_init(int nthreads);

void accfft_cleanup();

#endif
