#include "fft.h"
#include <cilk/cilk.h>

#define MAX_LINEAR 128


void transform(carray& x, direction dir);
void cilk_transform(carray& x, direction dir);
void combine(carray& x, size_t N, direction dir, carray& E, carray& O);
void cilk_combine(carray& x, size_t N, direction dir, carray& E, carray& O);

void cilk_fft(carray& x)
{
  cilk_transform(x, FORWARD);
}

void fft(carray& x)
{
  transform(x, FORWARD);
}

void ifft(carray& x)
{
  transform(x, REVERSE);
  for(int i=0; i<x.size(); i++)
    x[i] /= x.size();
}

// slightly modified from https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
void transform(carray& x, direction dir)
{
    const size_t N = x.size();

    // recursion base case
    if (N <= 1)
      return;

    // even side recursive call
    carray E = x[std::slice(0, N/2, 2)];
    transform(E, dir);

    // odd side recursive call
    carray O = x[std::slice(1, N/2, 2)];
    transform(O, dir);

    // combine
    combine(x, N, dir, E, O);
    //double v = (2*(dir==REVERSE)-1) * 2 * PI / N;
    //for (size_t k = 0; k < N/2; ++k)
    //{
    //    cdouble t = std::polar(1.0, v * k) * O[k];
    //    x[k] = E[k] + t;
    //    x[k+N/2] = E[k] - t;
    //}
}

const size_t cilk_max_recombine = MAX_LINEAR;

void combine(carray& x, size_t N, direction dir, carray& E, carray& O)
{
    // combine
    double v = (2*(dir==REVERSE)-1) * 2 * PI / N;
    for (size_t k = 0; k < N/2; ++k)
    {
        cdouble t = std::polar(1.0, v * k) * O[k];
        x[k] = E[k] + t;
        x[k+N/2] = E[k] - t;
    }
} 
void cilk_combine(carray& x, size_t N, direction dir, carray& E, carray& O)
{
    if (N <= cilk_max_recombine) {
      // combine
      combine(x, N, dir, E, O);
    } else {
          carray small = x[std::slice(0, N/2, 1)];
          carray large = x[std::slice(N/2, N/2, 1)];
          cilk_spawn cilk_combine(small, N/2, dir, E, O);
          cilk_spawn cilk_combine(large, N/2, dir, E, O);
          cilk_sync;
    }
} 

void cilk_transform(carray& x, direction dir)
{
    const size_t N = x.size();

    // recursion base case
    if (N <= 1)
      return;

    // even side recursive call
    carray E = x[std::slice(0, N/2, 2)];
    cilk_spawn transform(E, dir);

    // odd side recursive call
    carray O = x[std::slice(1, N/2, 2)];
    cilk_spawn transform(O, dir);

    cilk_sync;

    // combine
    combine(x, N, dir, E, O);
}
