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

#include <cstdlib>
#include <omp.h>
#include <iterator>
#include <vector>

template <class T, class I>
T omp_par::reduce(T* A, I cnt){
  T sum=0;
  #pragma omp parallel for reduction(+:sum)
  for(I i = 0; i < cnt; i++)
    sum+=A[i];
  return sum;
}

template <class T, class I>
void omp_par::scan(T* A, T* B,I cnt){
  int p=omp_get_max_threads();
  if(cnt<100*p){
    for(I i=1;i<cnt;i++)
      B[i]=B[i-1]+A[i-1];
    return;
  }

  I step_size=cnt/p;

  #pragma omp parallel for
  for(int i=0; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    if(i!=0)B[start]=0;
    for(I j=start+1; j<end; j++)
      B[j]=B[j-1]+A[j-1];
  }

  T* sum=new T[p];
  sum[0]=0;
  for(int i=1;i<p;i++)
    sum[i]=sum[i-1]+B[i*step_size-1]+A[i*step_size-1];

  #pragma omp parallel for
  for(int i=1; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    T sum_=sum[i];
    for(I j=start; j<end; j++)
      B[j]+=sum_;
  }
  delete[] sum;
}


