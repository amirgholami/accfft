Timing AccFFT                        {#timer}
============

AccFFT *execute* functions get a double timer of size 5, 
where the timing for different parts of the algorithm is written to:

1. timer[2]: Comm time
2. timer[4]: Local FFT execution time.
3. timer[0],timer[1],timer[3]: Scratch data

None of the entries in timer denote the total time.
To get the total execution time, you should wrap the execution
function call with MPI_Wtime(). For example:

double exec_time=0;
exec_time-=MPI_Wtime();
accfft_execute(...,timer);
exec_time+=MPI_Wtime();

It is recommended that you perform 1-2 warmup runs by calling the
corresponding *execute* function, before profiling
the code. 
