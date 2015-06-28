AccFFT Flags                      {#flags}
============

Flags are used to tune the library during the setup time
to use the best configuration.
Currently two flags are supported:

1. ACCFFT_ESTIMATE: Minimal/no tuning.
2. ACCFFT_MEASURE: Tunes for local FFT execution algorithm, as well as global transposes to
reduce communication time.
