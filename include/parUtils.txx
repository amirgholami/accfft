/*
 *  Copyright (c) 2014-2015, George Biros
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

/*
 * The following people have contributed to this file:
 * @author Rahul S. Sampath, rahul.sampath@gmail.com
 * @author Hari Sundar, hsundar@gmail.com
 * @author Shravan Veerapaneni, shravan@seas.upenn.edu
 * @author Santi Swaroop Adavani, santis@gmail.com
 * @author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * @author Amir Gholami, gholami@accfft.org
*/

#include <mpi.h>
#include <omp.h>
#include "dtypes.h"
#include "ompUtils.h"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace par {

  template <class T, bool ASYNC>
  int Mpi_Alltoallv_dense(T* sbuff_, int* s_cnt_, int* sdisp_,
                          T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c, int kway){
    //double tt=omp_get_wtime();
    //MPI_Barrier(c); if(!pid) std::cout<<"Init: "<<omp_get_wtime()-tt<<'\n'; tt=omp_get_wtime(); ///////////////////////////////////
    int np, pid;
    MPI_Comm_size(c, &np);
    MPI_Comm_rank(c, &pid);
    if(np==1) memcpy(rbuff_,sbuff_,s_cnt_[0]*sizeof(T));

    char* sptr=(char*)sbuff_;
    char* rptr=(char*)rbuff_;

    static std::vector<char> sbuff;
    static std::vector<char> rbuff;

    static std::vector<int> s_cnt; s_cnt.resize(np+1,0);
    static std::vector<int> s_dsp; s_dsp.resize(np+1,0);
    static std::vector<int> r_cnt; r_cnt.resize(np+1,0);
    static std::vector<int> r_dsp; r_dsp.resize(np+1,0);
    static std::vector<int> sorig; sorig.resize(np  ,0);
    static std::vector<int> rorig; rorig.resize(np  ,0);

    #pragma omp parallel for
    for(size_t i=0;i<np;i++){
      s_cnt[i]=s_cnt_[i]*sizeof(T);
      sorig[i]=pid;
    }
    omp_par::scan(&s_cnt[0],&s_dsp[0],np+1);

    size_t range[2]={0,np};
    for(size_t np_new=np; np_new>1; np_new/=kway){
      double tt=omp_get_wtime();
      if(kway>np_new) kway=np_new;
      assert((np_new/kway)*kway==np_new); // Number of processes must be: p=c*kway^n, where c<kway
      static std::vector<int> new_range; new_range.resize(kway+1);
      for(int i=0;i<=kway;i++) new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
      int p_class=((pid-range[0])*kway)/(range[1]-range[0]);
      int new_pid=pid-new_range[p_class];

      for(int i=0;i<kway;i++){ //Exchange send sizes.
        MPI_Status status;
        size_t si=(p_class+new_pid+(np_new+kway)+i)%kway;
        size_t ri=(p_class-new_pid+(np_new+kway)-i)%kway;
        int spartner=new_range[si]+new_pid;
        int rpartner=new_range[ri]+new_pid;
        MPI_Sendrecv(&s_cnt[si*np/kway], np/kway, MPI_INT, spartner, 0,
                     &r_cnt[ri*np/kway], np/kway, MPI_INT, rpartner, 0, c, &status);
        MPI_Sendrecv(&sorig[si*np/kway], np/kway, MPI_INT, spartner, 0,
                     &rorig[ri*np/kway], np/kway, MPI_INT, rpartner, 0, c, &status);
      }
      omp_par::scan(&r_cnt[0],&r_dsp[0],np+1);
      rbuff.resize(r_dsp[np]); rptr=&rbuff[0];

      { //Exchange data.
        MPI_Status status0;
        static std::vector<MPI_Request> reqst; reqst .resize(kway*2);
        static std::vector<MPI_Status> status; status.resize(kway*2);
        for(int i=0;i<kway;i++){ //Exchange data.
          size_t si=(p_class+new_pid+(np_new+kway)+i)%kway;
          size_t ri=(p_class-new_pid+(np_new+kway)-i)%kway;
          int spartner=new_range[si]+new_pid;
          int rpartner=new_range[ri]+new_pid;
          size_t sa=(si+0)*np/kway;
          size_t sb=(si+1)*np/kway;
          size_t ra=(ri+0)*np/kway;
          size_t rb=(ri+1)*np/kway;
          if(!ASYNC) MPI_Sendrecv(&sptr[s_dsp[sa]], s_dsp[sb]-s_dsp[sa], MPI_BYTE, spartner, 0,
                                  &rptr[r_dsp[ra]], r_dsp[rb]-r_dsp[ra], MPI_BYTE, rpartner, 0, c, &status0);

          if(ASYNC) MPI_Issend(&sptr[s_dsp[sa]], s_dsp[sb]-s_dsp[sa], MPI_BYTE, spartner, 0, c, &reqst[i*2+1]);
          if(ASYNC) MPI_Irecv (&rptr[r_dsp[ra]], r_dsp[rb]-r_dsp[ra], MPI_BYTE, rpartner, 0, c, &reqst[i*2+0]);
        }
        if(ASYNC) MPI_Waitall(kway*2, &reqst[0], &status[0]);
      }

      { //Rearrange data.
        if(np_new<=kway){
          #pragma omp parallel for
          for(int i=0;i<np;i++){
            memcpy(rbuff_+rdisp_[rorig[i]], rptr+r_dsp[i], r_cnt[i]);
          }
        }else{
          sbuff.resize(rbuff.size()); sptr=&sbuff[0];
          size_t blk_size=(np/np_new);
          #pragma omp parallel for
          for(int i=0;i<np_new;i++){
            size_t k=i%kway;
            size_t j=i/kway;
            for(size_t l=0;l<blk_size;l++){
              s_cnt[l+(k+j*kway)*blk_size]=r_cnt[l+(j+k*(np_new/kway))*blk_size];
              sorig[l+(k+j*kway)*blk_size]=rorig[l+(j+k*(np_new/kway))*blk_size];
            }
          }
          omp_par::scan(&s_cnt[0],&s_dsp[0],np+1);

          #pragma omp parallel for
          for(int i=0;i<np_new;i++){
            size_t k=i%kway;
            size_t j=i/kway;

            size_t sdsp0=s_dsp[(0+k+j*kway)*blk_size];
            size_t sdsp1=s_dsp[(1+k+j*kway)*blk_size];
            size_t rdsp0=r_dsp[(0+j+k*(np_new/kway))*blk_size];
            size_t rdsp1=r_dsp[(1+j+k*(np_new/kway))*blk_size];
            memcpy(sptr+sdsp0, rptr+rdsp0, sdsp1-sdsp0);
            assert(sdsp1-sdsp0==rdsp1-rdsp0);
          }
        }
      }

      range[0]=new_range[p_class+0];
      range[1]=new_range[p_class+1];
    }

    return 0;
  }

#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
  template <class T, bool ASYNC>
  int Mpi_Alltoallv_dense_gpu(T* sbuff_, int* s_cnt_, int* sdisp_,
                              T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c, int kway){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //double tt=omp_get_wtime();
    //MPI_Barrier(c); if(!pid) std::cout<<"Init: "<<omp_get_wtime()-tt<<'\n'; tt=omp_get_wtime(); ///////////////////////////////////
    int np, pid;
    MPI_Comm_size(c, &np);
    MPI_Comm_rank(c, &pid);
    if(np==1){ // cudaMemcpy
      cudaError_t error=cudaMemcpy(rbuff_,sbuff_,s_cnt_[0]*sizeof(T),cudaMemcpyDeviceToDevice);
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }

    static std::vector<char> sbuff;
    static std::vector<char> rbuff;
    char* sptr=(sbuff.size()?&sbuff[0]:NULL);
    char* rptr=(rbuff.size()?&rbuff[0]:NULL);

    static std::vector<int> s_cnt; s_cnt.resize(np+1,0);
    static std::vector<int> s_dsp; s_dsp.resize(np+1,0);
    static std::vector<int> r_cnt; r_cnt.resize(np+1,0);
    static std::vector<int> r_dsp; r_dsp.resize(np+1,0);
    static std::vector<int> sorig; sorig.resize(np  ,0);
    static std::vector<int> rorig; rorig.resize(np  ,0);

    #pragma omp parallel for
    for(size_t i=0;i<np;i++){
      s_cnt[i]=s_cnt_[i]*sizeof(T);
      sorig[i]=pid;
    }
    omp_par::scan(&s_cnt[0],&s_dsp[0],np+1);
    if(sbuff.size()<s_dsp[np]){ // Resize sbuff
      if(sbuff.size()){ // cudaHostUnregister
        cudaError_t error=cudaHostUnregister(&sbuff[0]);
        if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
        assert(error == cudaSuccess);
      }
      sbuff.resize(s_dsp[np]); sptr=&sbuff[0];
      { // cudaHostRegister
        cudaError_t error=cudaHostRegister(&sbuff[0], sbuff.size(), cudaHostRegisterPortable);
        if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
        assert(error == cudaSuccess);
      }
    }

    size_t range[2]={0,np};
    for(size_t np_new=np; np_new>1; np_new/=kway){
      double tt=omp_get_wtime();
      if(kway>np_new) kway=np_new;
      assert((np_new/kway)*kway==np_new); // Number of processes must be: p=c*kway^n, where c<kway
      static std::vector<int> new_range; new_range.resize(kway+1);
      for(int i=0;i<=kway;i++) new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
      int p_class=((pid-range[0])*kway)/(range[1]-range[0]);
      int new_pid=pid-new_range[p_class];

      for(int i=0;i<kway;i++){ //Exchange send sizes.
        MPI_Status status;
        size_t si=(p_class+new_pid+(np_new+kway)+i)%kway;
        size_t ri=(p_class-new_pid+(np_new+kway)-i)%kway;
        int spartner=new_range[si]+new_pid;
        int rpartner=new_range[ri]+new_pid;
        MPI_Sendrecv(&s_cnt[si*np/kway], np/kway, MPI_INT, spartner, 0,
                     &r_cnt[ri*np/kway], np/kway, MPI_INT, rpartner, 0, c, &status);
        MPI_Sendrecv(&sorig[si*np/kway], np/kway, MPI_INT, spartner, 0,
                     &rorig[ri*np/kway], np/kway, MPI_INT, rpartner, 0, c, &status);
      }
      omp_par::scan(&r_cnt[0],&r_dsp[0],np+1);
      if(rbuff.size()<r_dsp[np]){ // Resize rbuff
        if(rbuff.size()){ // cudaHostUnregister
          cudaError_t error=cudaHostUnregister(&rbuff[0]);
          if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
          assert(error == cudaSuccess);
        }
        rbuff.resize(r_dsp[np]); rptr=&rbuff[0];
        { // cudaHostRegister
          cudaError_t error=cudaHostRegister(&rbuff[0], rbuff.size(), cudaHostRegisterPortable);
          if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
          assert(error == cudaSuccess);
        }
      }

      { //Exchange data.
        //MPI_Status status0;
        static std::vector<MPI_Request> reqst; reqst .resize(kway*2);
        static std::vector<MPI_Status> status; status.resize(kway*2);
        for(int i=0;i<kway;i++){ //Post receives.
          size_t ri=(p_class-new_pid+(np_new+kway)-i)%kway;
          int rpartner=new_range[ri]+new_pid;
          size_t ra=(ri+0)*np/kway;
          size_t rb=(ri+1)*np/kway;
          MPI_Irecv (&rptr[r_dsp[ra]], r_dsp[rb]-r_dsp[ra], MPI_BYTE, rpartner, 0, c, &reqst[i]);
        }

        if(np_new==np){ // cudaMemcpy DeviceToHost first block
          size_t si=(p_class+new_pid+(np_new+kway)+0)%kway;
          size_t sa=(si+0)*np/kway;
          size_t sb=(si+1)*np/kway;
          { // cudaMemcpy DeviceToHost
            cudaError_t error=cudaMemcpyAsync(sptr+s_dsp[sa], ((char*)sbuff_)+s_dsp[sa], s_dsp[sb]-s_dsp[sa], cudaMemcpyDeviceToHost, stream);
            if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
            assert(error == cudaSuccess);
          }
        }

        for(int i=0;i<kway;i++){ //Exchange data.
          size_t si=(p_class+new_pid+(np_new+kway)+i)%kway;
          size_t ri=(p_class-new_pid+(np_new+kway)-i)%kway;
          int spartner=new_range[si]+new_pid;
          int rpartner=new_range[ri]+new_pid;
          size_t sa=(si+0)*np/kway;
          size_t sb=(si+1)*np/kway;

          { // cudaStreamSynchronize
            cudaError_t error=cudaStreamSynchronize(stream);
            if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
            assert(error == cudaSuccess);
          }

          if(np_new==np && i<kway-1){ // cudaMemcpy DeviceToHost next block
            size_t si=(p_class+new_pid+(np_new+kway)+(i+1))%kway;
            size_t sa=(si+0)*np/kway;
            size_t sb=(si+1)*np/kway;
            { // cudaMemcpy DeviceToHost
              cudaError_t error=cudaMemcpyAsync(sptr+s_dsp[sa], ((char*)sbuff_)+s_dsp[sa], s_dsp[sb]-s_dsp[sa], cudaMemcpyDeviceToHost,stream);
              if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
              assert(error == cudaSuccess);
            }
          }

          //MPI_Sendrecv(&sptr[s_dsp[sa]], s_dsp[sb]-s_dsp[sa], MPI_BYTE, spartner, 0,
          //             &rptr[r_dsp[ra]], r_dsp[rb]-r_dsp[ra], MPI_BYTE, rpartner, 0, c, &status0);
          if(!ASYNC && i>0) MPI_Wait(&reqst[(i-1)+kway], &status[(i-1)+kway]); // Wait for previous send
          MPI_Issend(&sptr[s_dsp[sa]], s_dsp[sb]-s_dsp[sa], MPI_BYTE, spartner, 0, c, &reqst[i+kway]);

          if(np_new==kway && i>0){ // cudaMemcpy HostToDevice previous block
            MPI_Waitall(1, &reqst[i-1], &status[i-1]); // Wait for previous receive
            size_t ri=(p_class-new_pid+(np_new+kway)-(i-1))%kway;
            size_t ra=(ri+0)*np/kway;
            size_t rb=(ri+1)*np/kway;
            for(int i=ra;i<rb;i++){
              cudaError_t error=cudaMemcpyAsync(rbuff_+rdisp_[rorig[i]], rptr+r_dsp[i], r_cnt[i],cudaMemcpyHostToDevice);
              if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
              assert(error == cudaSuccess);
            }
          }
        }

        if(np_new==kway){ // cudaMemcpy HostToDevice last block
          MPI_Wait(&reqst[kway-1], &status[kway-1]); // Wait for last receive
          size_t ri=(p_class-new_pid+(np_new+kway)-(kway-1))%kway;
          size_t ra=(ri+0)*np/kway;
          size_t rb=(ri+1)*np/kway;
          for(int i=ra;i<rb;i++){
            cudaError_t error=cudaMemcpyAsync(rbuff_+rdisp_[rorig[i]], rptr+r_dsp[i], r_cnt[i],cudaMemcpyHostToDevice);
            if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
            assert(error == cudaSuccess);
          }
          cudaDeviceSynchronize();
          cudaStreamDestroy(stream);
        }else{ // Wait for all receives
          MPI_Waitall(kway, &reqst[0], &status[0]);
        }

        // Wait for all sends
        if(ASYNC) MPI_Waitall(kway, &reqst[kway], &status[kway]);
        else MPI_Wait(&reqst[2*kway-1], &status[2*kway-1]);
      }

      { //Rearrange data.
        if(np_new>kway){
          if(sbuff.size()<rbuff.size()){ // Resize sbuff
            if(sbuff.size()){ // cudaHostUnregister
              cudaError_t error=cudaHostUnregister(&sbuff[0]);
              if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
              assert(error == cudaSuccess);
            }
            sbuff.resize(rbuff.size()); sptr=&sbuff[0];
            { // cudaHostRegister
              cudaError_t error=cudaHostRegister(&sbuff[0], sbuff.size(), cudaHostRegisterPortable);
              if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
              assert(error == cudaSuccess);
            }
          }
          size_t blk_size=(np/np_new);
          #pragma omp parallel for
          for(int i=0;i<np_new;i++){
            size_t k=i%kway;
            size_t j=i/kway;
            for(size_t l=0;l<blk_size;l++){
              s_cnt[l+(k+j*kway)*blk_size]=r_cnt[l+(j+k*(np_new/kway))*blk_size];
              sorig[l+(k+j*kway)*blk_size]=rorig[l+(j+k*(np_new/kway))*blk_size];
            }
          }
          omp_par::scan(&s_cnt[0],&s_dsp[0],np+1);

          #pragma omp parallel for
          for(int i=0;i<np_new;i++){
            size_t k=i%kway;
            size_t j=i/kway;

            size_t sdsp0=s_dsp[(0+k+j*kway)*blk_size];
            size_t sdsp1=s_dsp[(1+k+j*kway)*blk_size];
            size_t rdsp0=r_dsp[(0+j+k*(np_new/kway))*blk_size];
            size_t rdsp1=r_dsp[(1+j+k*(np_new/kway))*blk_size];
            memcpy(sptr+sdsp0, rptr+rdsp0, sdsp1-sdsp0);
            assert(sdsp1-sdsp0==rdsp1-rdsp0);
          }
        }
      }

      range[0]=new_range[p_class+0];
      range[1]=new_range[p_class+1];
    }

    return 0;
  }

  inline void testAlltoallv(size_t N=10000000, MPI_Comm c=MPI_COMM_WORLD){
    int np, pid;
    MPI_Comm_size(c, &np);
    MPI_Comm_rank(c, &pid);
    srand48(pid);
    N=N/np;

    double tt=0;
    std::vector<int> scnt(np,0);
    std::vector<int> sdsp(np,0);
    for(size_t i=0;i<np;i++){
      scnt[i]=N*drand48();
    }
    omp_par::scan(&scnt[0],&sdsp[0],np);

    std::vector<char> send_buff;
    { // Create send data
      send_buff.resize(scnt[np-1]+sdsp[np-1]);
      for(size_t i=0;i<send_buff.size();i++){
        send_buff[i]=drand48()*256;
      }
    }

    std::vector<int> rcnt(np,0);
    std::vector<int> rdsp(np,0);
    MPI_Alltoall(&scnt[0], 1, MPI_INT,
                 &rcnt[0], 1, MPI_INT, c);
    omp_par::scan(&rcnt[0],&rdsp[0],np);

    std::vector<char> recv_buff0(rcnt[np-1]+rdsp[np-1]);
    std::vector<char> recv_buff1(rcnt[np-1]+rdsp[np-1]);


    char* send_buff_d=&send_buff [0];
    char* recv_buff_d=&recv_buff0[0];
    { // cudaMalloc send_buff_d
      cudaError_t error=cudaMalloc((void**)&send_buff_d,send_buff .size()*sizeof(send_buff [0]));
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }
    { // cudaMalloc recv_buff_d
      cudaError_t error=cudaMalloc((void**)&recv_buff_d,recv_buff0.size()*sizeof(recv_buff0[0]));
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }
    { // cudaMemcpy send_buff_d
      cudaError_t error=cudaMemcpy(send_buff_d, &send_buff[0], send_buff .size()*sizeof(send_buff [0]), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }

    Mpi_Alltoallv_dense_gpu<char,false>(&send_buff_d[0], &scnt[0], &sdsp[0],
                                        &recv_buff_d[0], &rcnt[0], &rdsp[0], c,2);
    MPI_Barrier(c);
    tt=-omp_get_wtime();
    Mpi_Alltoallv_dense_gpu<char,false>(&send_buff_d[0], &scnt[0], &sdsp[0],
                                        &recv_buff_d[0], &rcnt[0], &rdsp[0], c,2);
    MPI_Barrier(c);
    tt+=omp_get_wtime();
    if(!pid) std::cout<<N<<"    "<<tt;

    { // cudaMemcpy recv_buff_d
      cudaError_t error=cudaMemcpy(&recv_buff0[0], recv_buff_d, recv_buff0.size()*sizeof(recv_buff0[0]), cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }
    { // cudaFree send_buff_d
      cudaError_t error=cudaFree(send_buff_d);
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }
    { // cudaFree recv_buff_d
      cudaError_t error=cudaFree(recv_buff_d);
      if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
      assert(error == cudaSuccess);
    }



    MPI_Alltoallv      (&send_buff [0], &scnt[0], &sdsp[0],MPI_BYTE,
                        &recv_buff1[0], &rcnt[0], &rdsp[0],MPI_BYTE, c);
    MPI_Barrier(c);
    tt=-omp_get_wtime();
    MPI_Alltoallv      (&send_buff [0], &scnt[0], &sdsp[0],MPI_BYTE,
                        &recv_buff1[0], &rcnt[0], &rdsp[0],MPI_BYTE, c);
    MPI_Barrier(c);
    tt+=omp_get_wtime();
    if(!pid) std::cout<<"    "<<tt<<'\n';




    for(size_t i=0;i<recv_buff0.size();i++){
//      std::cout<<(int)recv_buff0[i]<<' ';
      assert(recv_buff0[i]==recv_buff1[i]);
    }
//    std::cout<<'\n';
  }

#endif
}

