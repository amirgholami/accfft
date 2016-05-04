/*
 * File: step4.cpp
 * Project: AccFFT
 * Created by Amir Gholami on 12/23/2014
 * Contact: contact@accfft.org
 * Copyright (c) 2014-2015
 *
 * A very simple test for solving Laplacian(\phi) = rho using FFT in 3D.
 *
 * Laplacian operator can be considered as a low-pass filter.
 * Here we implement 2 types of filters :
 *
 * method 0 : see Numerical recipes in C, section 19.4
 * method 1 : just divide right hand side by -(kx^2+ky^2+kz^2) in Fourier
 *
 * Test case 0:  rho(x,y,z) = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(2*pi*z/Lz)
 * Test case 1:  rho(x,y,z) = (4*alpha*alpha*(x^2+y^2+z^2)-6*alpha)*exp(-alpha*(x^2+y^2+z^2))
 * Test case 2:  rho(x,y,z) = ( r=sqrt(x^2+y^2+z^2) < R ) ? 1 : 0
 *
 * Example of use:
 * ./step4 -x 64 -y 64 -z 64 -m 1 -t 2
 *
 * \author Pierre Kestener
 * \date June 1st, 2015
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h> // for M_PI
#include <unistd.h> // for getopt

#include <accfft.h>
#include <accfft_utils.h>

#include <string>

#define SQR(x) ((x)*(x))

enum TESTCASE {
  TESTCASE_SINE=0,
  TESTCASE_GAUSSIAN=1,
  TESTCASE_UNIFORM_BALL=2
};

struct PoissonParams {

  // global domain sizes
  int nx;
  int ny;
  int nz;

  // there 3 testcases
  int testcase;

  // method number: 2 variants (see head of this file)
  int methodNb;

  // only valid for the uniform ball testcase (coordinate of center of the ball)
  double xC;
  double yC;
  double zC;
  double radius;

  // only valid for the gaussian testcase
  double alpha;

  // define constructor for default values
  PoissonParams() :
    nx(128),
    ny(128),
    nz(128),
    testcase(TESTCASE_SINE),
    methodNb(0),
    xC(0.5),
    yC(0.5),
    zC(0.5),
    radius(0.1),
    alpha(30.0)
  {
    
  } // PoissonParams::PoissonParams

}; // struct PoissonParams


// =======================================================
// =======================================================
/*
 * RHS of testcase sine: eigenfunctions of Laplacian
 */
double testcase_sine_rhs(double x, double y, double z) {

  return sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z);

} // testcase_sine_rhs

// =======================================================
// =======================================================
/*
 * Solution of testcase sine: eigenfunctions of Laplacian
 */
double testcase_sine_sol(double x, double y, double z) {

  return -sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z) / ( 3*(4*M_PI*M_PI) );

} // testcase_sine_sol

// =======================================================
// =======================================================
/*
 * RHS of testcase gaussian
 */
double testcase_gaussian_rhs(double x, double y, double z, double alpha) {

  return (4*alpha*alpha*(x*x+y*y+z*z)-6*alpha)*exp(-alpha*(x*x+y*y+z*z));

} // testcase_gaussian_rhs

// =======================================================
// =======================================================
/*
 * Solution of testcase gaussian
 */
double testcase_gaussian_sol(double x, double y, double z, double alpha) {

  return exp(-alpha*(x*x+y*y+z*z));

} // testcase_gaussian_sol

// =======================================================
// =======================================================
/*
 * RHS of testcase uniform ball
 */
double testcase_uniform_ball_rhs(double x,  double y,  double z,
				 double xC, double yC, double zC,
				 double R) {
  
  double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) + (z-zC)*(z-zC) );

  double res = r < R ? 1.0 : 0.0;
  return res;

} // testcase_uniform_ball_rhs

// =======================================================
// =======================================================
/*
 * Solution of testcase uniform ball
 */
double testcase_uniform_ball_sol(double x,  double y,  double z,
				 double xC, double yC, double zC,
				 double R) {
  
  double r = sqrt( (x-xC)*(x-xC) + (y-yC)*(y-yC) + (z-zC)*(z-zC) );

  double res = r < R ? r*r/6.0 : -R*R*R/(3*r)+R*R/2;
  return res;

} // testcase_uniform_ball_sol


// =======================================================
// =======================================================
/*
 * Initialize the rhs of Poisson equation, and exact known solution.
 *
 * \param[out] rho Poisson rhs array
 * \param[out] sol known exact solution to corresponding Poisson problem
 */
template<const TESTCASE testcase_id>
void initialize(double *rho, double *sol, int *n, MPI_Comm c_comm, PoissonParams &params)
{
  double pi=M_PI;
  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  /*
   * testcase gaussian parameters
   */
  double alpha=1.0;
  if (testcase_id == TESTCASE_GAUSSIAN)
    alpha = params.alpha;

  /*
   * testcase uniform ball parameters
   */
  // uniform ball function center
  double xC = params.xC;
  double yC = params.yC;
  double zC = params.zC;
	  
  // uniform ball radius
  double R = params.radius;

  {
    double X,Y,Z;
    double x0    = 0.5;
    double y0    = 0.5;
    double z0    = 0.5;
    long int ptr;

    for (int i=0; i<isize[0]; i++) {
      for (int j=0; j<isize[1]; j++) {
        for (int k=0; k<isize[2]; k++) {

          X = 1.0*(i+istart[0])/n[0];
          Y = 1.0*(j+istart[1])/n[1];
          Z = 1.0*(k+istart[2])/n[2];

          ptr = i*isize[1]*isize[2] + j*isize[2] + k;

	  if (testcase_id == TESTCASE_SINE) {

	    rho[ptr] = testcase_sine_rhs(X,Y,Z);
	    sol[ptr] = testcase_sine_sol(X,Y,Z);

	  } else if (testcase_id == TESTCASE_GAUSSIAN) {

	    rho[ptr] = testcase_gaussian_rhs(X-x0,Y-x0,Z-x0,alpha);
	    sol[ptr] = testcase_gaussian_sol(X-x0,Y-x0,Z-x0,alpha);

	  } else if (testcase_id == TESTCASE_UNIFORM_BALL) {

	    rho[ptr] = testcase_uniform_ball_rhs(X,Y,Z,xC,yC,zC,R);
	    sol[ptr] = testcase_uniform_ball_sol(X,Y,Z,xC,yC,zC,R);

	  }

        } // end for k
      } // end for j
    } // end for i


    /*
     * rescale exact solution to ease comparison with numerical solution which has a zero average value
     */
    if (testcase_id == TESTCASE_UNIFORM_BALL) {
      
      // make exact solution allways positive to ease comparison with numerical one

      // compute local min
      double minVal = sol[0];
      for (int i=0; i<isize[0]; i++){
	for (int j=0; j<isize[1]; j++){
	  for (int k=0; k<isize[2]; k++){

	    ptr = i*isize[1]*n_tuples + j*n_tuples + k;
	    if (sol[ptr] < minVal)
	      minVal = sol[ptr];

	  } // end for k
	} // end for j
      } // end for i

      // compute global min
      double minValG;
      MPI_Allreduce(&minVal, &minValG, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

      for (int i=0; i<isize[0]; i++){
	for (int j=0; j<isize[1]; j++){
	  for (int k=0; k<isize[2]; k++){

	    ptr = i*isize[1]*n_tuples + j*n_tuples + k;
	    sol[ptr] -= minValG;

	  } // end for k
	} // end for j
      } // end for i

    } // end TESTCASE_UNIFORM_BALL

  }

  return;

} // end initialize

// =======================================================
// =======================================================
/*
 * Poisson fourier filter.
 * Divide fourier coefficients by -(kx^2+ky^2+kz^2).
 */
void poisson_fourier_filter(Complex *data_hat,
			    int N[3],
			    int isize[3],
			    int istart[3],
			    int methodNb) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  double NX = N[0];
  double NY = N[1];
  double NZ = N[2];

  double Lx = 1.0;
  double Ly = 1.0;
  double Lz = 1.0;

  double dx = Lx/NX;
  double dy = Ly/NY;
  double dz = Lz/NZ;

  for (int i=0; i < isize[0]; i++) {
    for (int j=0; j < isize[1]; j++) {
      for (int k=0; k < isize[2]; k++) {

	double kx = istart[0]+i;
	double ky = istart[1]+j;
	double kz = istart[2]+k;

	double kkx = (double) kx;
	double kky = (double) ky;
	double kkz = (double) kz;

	if (kx>NX/2)
	  kkx -= NX;
	if (ky>NY/2)
	  kky -= NY;
	if (kz>NZ/2)
	  kkz -= NZ;

	int index = i*isize[1]*isize[2]+j*isize[2]+k;

	double scaleFactor = 0.0;

	if (methodNb==0) {

	  /*
	   * method 0 (See Eq. 19.4.5 of Numerical recipes 2nd Ed.)
	   */

	  scaleFactor=2*(
			 (cos(1.0*2*M_PI*kx/NX) - 1)/(dx*dx) +
			 (cos(1.0*2*M_PI*ky/NY) - 1)/(dy*dy) +
			 (cos(1.0*2*M_PI*kz/NZ) - 1)/(dz*dz) );

	} else if (methodNb==1) {

	  /*
	   * method 1 (just from Continuous Fourier transform of
	   * Poisson equation)
	   */
	  //scaleFactor=-4*M_PI*M_PI*(kkx*kkx + kky*kky + kkz*kkz)/;
	  scaleFactor=-(4*M_PI*M_PI/Lx/Lx*kkx*kkx + 4*M_PI*M_PI/Ly/Ly*kky*kky + 4*M_PI*M_PI/Ly/Ly*kkz*kkz);

	}

  scaleFactor*=NX*NY*NZ; // FFT scaling factor

	if (kx!=0 or ky!=0 or kz!=0) {
	  data_hat[index][0] /= scaleFactor;
	  data_hat[index][1] /= scaleFactor;
	} else { // enforce mean value is zero since you cannot recover the zero frequency
	  data_hat[index][0] = 0.0;
	  data_hat[index][1] = 0.0;
	}

      }
    }
  }


} // poisson_fourier_filter

// =======================================================
// =======================================================
/*
 * Rescale Numerical Solution.
 * The zeroth frequency data (i.e. the mean of the solution)
 * cannot be recovered in the Poisson problem without using additional
 * data, such as boundary condition. Without such information
 * any u=u+constant would satisfy the problem.
 *
 * Here we use specific information for each test case
 * to "calibrate" the numerical solution with the
 * correct constant.
 */
void rescale_numerical_solution(double *phi,
    int *isize,
    PoissonParams &params) {

  const int testcase_id = params.testcase;
  int n_tuples=params.nz;

  if (testcase_id == TESTCASE_GAUSSIAN) {

    // compute local max
    double maxVal = phi[0];
    for (int index=0;
        index < isize[0]*isize[1]*isize[2];
        index++) {

      if (phi[index] > maxVal)
        maxVal = phi[index];

    } // end for index

    // compute global max
    double maxValG;
    MPI_Allreduce(&maxVal, &maxValG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (int index=0;
        index < isize[0]*isize[1]*isize[2];
        index++) {

      phi[index] += 1-maxValG;

    } // end for index

  } // end TESTCASE_GAUSSIAN

  if (testcase_id == TESTCASE_UNIFORM_BALL) {

    // make exact solution allways positive to ease comparison with numerical one

    // compute local min
    double minVal = phi[0];
    for (int index=0;
        index < isize[0]*isize[1]*isize[2];
        index++) {

      if (phi[index] < minVal)
        minVal = phi[index];

    }

    // compute global min
    double minValG;
    MPI_Allreduce(&minVal, &minValG, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    for (int index=0;
        index < isize[0]*isize[1]*isize[2];
        index++) {

      phi[index] -= minValG;

    }

  } // end TESTCASE_UNIFORM_BALL

} // rescale_numerical_solution

// =======================================================
// =======================================================
void  compute_L2_error(double *phi,
    double *phi_exact,
    int *isize,
    PoissonParams &params) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* global domain size */
  int n[3];
  n[0] = params.nx;
  n[1] = params.ny;
  n[2] = params.nz;
  int n_tuples=n[2];
  int N=n[0]*n[1]*n[2];

  // compute L2 difference between FFT-based solution (phi) and
  // expected analytical solution
  double L2_diff = 0.0;
  double L2_phi  = 0.0;

  // global values
  double L2_diff_G = 0.0;
  double L2_phi_G = 0.0;

  long int ptr;
  for (int index=0;
      index < isize[0]*isize[1]*isize[2];
      index++) {

    L2_phi += phi_exact[index]*phi_exact[index];

    L2_diff += (phi[index]-phi_exact[index])*(phi[index]-phi_exact[index]);

  }

  // global L2
  MPI_Reduce(&L2_phi, &L2_phi_G, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&L2_diff, &L2_diff_G, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (procid==0) {
    std::cout << "#################################################################" << std::endl;
    std::cout << "L2 relative error between phi and exact solution : "
      <<  L2_diff_G / L2_phi_G
      //<< " ( = " <<  L2_diff_G << "/" << L2_phi_G << ")"
      << std::endl;
    std::cout << "#################################################################" << std::endl;
  }

} // compute_L2_error

// =======================================================
// =======================================================
/*
 * FFT-based poisson solver.
 *
 * \param[in] params parameters parsed from the command line arguments
 * \param[in] nthreads number of threads
 */
void poisson_solve(PoissonParams &params, int nthreads) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // which testcase
  const int testCaseNb = params.testcase;

  // which method ? variant of FFT-based Poisson solver : 0 or 1
  const int methodNb   = params.methodNb;
  if (procid==0)
    printf("---> Using Fourier filter method %d\n",methodNb);

  /* global domain size */
  int n[3];
  n[0] = params.nx;
  n[1] = params.ny;
  n[2] = params.nz;

  /* Create Cartesian Communicator */
  int c_dims[2] = {0};
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);
  //printf("[mpi rank %d] c_dims = %d %d\n", procid, c_dims[0], c_dims[1]);


  // Governing Equation: Laplace(\phi)=rho
  double *rho, *exact_solution;
  Complex *phi_hat;
  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;
  int alloc_max=0;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  printf("[mpi rank %d] isize  %3d %3d %3d osize  %3d %3d %3d\n", procid,
      isize[0],isize[1],isize[2],
      osize[0],osize[1],osize[2]
      );

  printf("[mpi rank %d] istart %3d %3d %3d ostart %3d %3d %3d\n", procid,
      istart[0],istart[1],istart[2],
      ostart[0],ostart[1],ostart[2]
      );

  rho=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  phi_hat=(Complex*)accfft_alloc(alloc_max);

  exact_solution=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));

  accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan * plan = accfft_plan_dft_3d_r2c(n,
      rho, (double*)phi_hat,
      c_comm, ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /*  Initialize rho (force) */
  switch(testCaseNb) {
    case TESTCASE_SINE:
      initialize<TESTCASE_SINE>(rho, exact_solution, n, c_comm, params);
      break;
    case TESTCASE_GAUSSIAN:
      initialize<TESTCASE_GAUSSIAN>(rho, exact_solution, n, c_comm, params);
      break;
    case TESTCASE_UNIFORM_BALL:
      initialize<TESTCASE_UNIFORM_BALL>(rho, exact_solution, n, c_comm, params);
      break;
  }
  MPI_Barrier(c_comm);

  // optional : save rho (rhs)
#ifdef USE_PNETCDF
  {
    std::string filename = "rho.nc";
    MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
    MPI_Offset isize_mpi[3]  = { isize[0],  isize[1],  isize[2] };
    write_pnetcdf(filename,
        istart_mpi,
        isize_mpi,
        c_comm,
        n,
        rho);
  }
#else
  {
    if (procid==0)
      std::cout << "[WARNING] You have to enable PNETCDF to be enable to dump data into files\n";
  }
#endif // USE_PNETCDF

  /*
   * Perform forward FFT
   */
  f_time-=MPI_Wtime();
  accfft_execute_r2c(plan,rho,phi_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);

  /*
   * here perform fourier filter associated to poisson ...
   */
  poisson_fourier_filter(phi_hat, n, osize, ostart, methodNb);

  /*
   * Perform backward FFT
   */
  double * phi=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  i_time-=MPI_Wtime();
  accfft_execute_c2r(plan,phi_hat,phi);
  i_time+=MPI_Wtime();

  /* rescale numerical solution before computing L2 */
  rescale_numerical_solution(phi, isize, params);

  /* L2 error between phi and phi_exact */
  compute_L2_error(phi, exact_solution, isize, params);

  /* optional : save phi (solution to poisson) and exact solution */

#ifdef USE_PNETCDF
  {
    std::string filename = "phi.nc";
    MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
    MPI_Offset isize_mpi[3]  = { isize[0],  isize[1],  isize[2] };
    write_pnetcdf(filename,
        istart_mpi,
        isize_mpi,
        c_comm,
        n,
        phi);
  }
  {
    std::string filename = "phi_exact.nc";
    MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
    MPI_Offset isize_mpi[3]  = { isize[0],  isize[1],  isize[2] };
    write_pnetcdf(filename,
        istart_mpi,
        isize_mpi,
        c_comm,
        n,
        exact_solution);
  }
#else
  {
    if (procid==0)
      std::cout << "[WARNING] You have to enable PNETCDF to be enable to dump data into files\n";
  }
#endif // USE_PNETCDF


  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  accfft_free(rho);
  accfft_free(exact_solution);
  accfft_free(phi_hat);
  accfft_free(phi);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);

  return;

} // end poisson_solve

// =======================================================
// =======================================================
/*
 * Read poisson parameters from command line argument.
 *
 * \param[in] argc
 * \param[in] argv
 * \param[out] params a reference to a PoissonParams structure.
 *
 * options:
 * - x,y,z are for global domain sizes.
 * - t         for testcase
 * - m         for method
 * - a,b,c,r   for uniform ball parameter (center location + radius)
 * - l         for alpha
 */
void getPoissonParams(const int argc, char *argv[],
    PoissonParams &params) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //opterr = 0;
  char *value = NULL;
  int c;
  int tmp;

  /*
   *
   */
  while ((c = getopt (argc, argv, "x:y:z:m:t:a:b:c:r:l:")) != -1)
    switch (c)
    {
      case 'x':
        value = optarg;
        params.nx = atoi(value);
        break;
      case 'y':
        value = optarg;
        params.ny = atoi(value);
        break;
      case 'z':
        value = optarg;
        params.nz = atoi(value);
        break;
      case 'm':
        value = optarg;
        tmp = atoi(value);
        if (tmp < 0 || tmp > 1) {
          // wrong value, defaulting to 1
          tmp = 1;
          if (procid==0) std::cout << "wrong value for option -m (method); defaulting to 0\n";
        }
        params.methodNb = tmp;
        break;
      case 't':
        value = optarg;
        tmp = atoi(value);
        if (tmp < 0 || tmp > 2) {
          // wrong value, defaulting to 0
          tmp = 0;
          if (procid==0) std::cout << "wrong value for option -t (testcase); defaulting to 0\n";
        }
        params.testcase = tmp;
        break;
      case 'a':
        value = optarg;
        params.xC = atof(value);
        break;
      case 'b':
        value = optarg;
        params.yC = atof(value);
        break;
      case 'c':
        value = optarg;
        params.zC = atof(value);
        break;
      case 'r':
        value = optarg;
        params.radius = atof(value);
        break;
      case 'l':
        value = optarg;
        params.alpha = atof(value);
        break;
      case '?':
        if (procid==0) std::cerr << "#### All options require an argument. ####\n";
      default:
        ;
    }

} // getPoissonParams

/******************************************************/
/******************************************************/
/******************************************************/
int main(int argc, char *argv[])
{

  MPI_Init (&argc, &argv);
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* parse command line arguments and fill params structure */
  PoissonParams params = PoissonParams();
  getPoissonParams(argc, argv, params);

  // test case number
  const int testCaseNb = params.testcase;
  if (testCaseNb < 0 || testCaseNb > 2) {
    if (procid == 0) {
      std::cerr << "---> Wrong test case. Must be integer < 2 !!!\n";
    }
  } else {
    if (procid == 0) {
      std::cout << "---> Using test case number : " << testCaseNb << std::endl;
    }
  }


  int nthreads=1;
  poisson_solve(params, nthreads);

  MPI_Finalize();
  return 0;
} // end main
