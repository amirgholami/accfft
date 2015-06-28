Timing AccFFT                        {#timer}
============

AccFFT *execute* functions get a double timer of size 5, 
where the timing for different parts of the algorithm is written to:

1. timer[0]: Total global transpose time.
2. timer[4]: Local FFT execution time.

It is recommended that you perform 1-2 warmup runs by calling the
corresponding *execute* function, before profiling
the code. 
