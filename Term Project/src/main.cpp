#include <iostream>
#include <complex>
#include <cmath>
#include "dft.h"
#include "fft.h"
#include "cilktime.h"

using std::complex;
using namespace std::literals::complex_literals;

//#define DFT
#define FFT
//#define DEBUG

bool isPow2(int n) {
  return n && (!(n & (n-1)));
}

int main( int argc, char *argv[] )
{
  // read cmdline args to get number of elements to transform /*{{{*/
  int n=0;
  if ( argc != 2 ) { // argc should be 2 for correct execution
    std::cout << "ERROR: usage: " << argv[0] << " needs number of elements to transform\n";
    return -1;
  }
  else {
    n = std::stoi(argv[1]);
    if (!isPow2(n)) {
      std::cout << "ERROR: Argument " << n << " should be power of 2!\n";
      return -1;
    }
  }
/*}}}*/
#ifdef DFT /*{{{*/
  // dft test
  carray x(n);
  for(int i=0; i<n; i++)
    x[i] = i+1;

  dft(x);
  for(int i=0; i<n; i++)
    std::cout << i << ": " << x[i] << std::endl;
  std::cout << std::endl;

  idft(x);
  for(int i=0; i<n; i++)
    std::cout << i << ": " << x[i] << std::endl;
  std::cout << std::endl;
#endif
/*}}}*/
#ifdef FFT /*{{{*/
  // fft test
  // cilk FFT (cooley-tukey) /*{{{*/
  carray cilk_y(n);
  for(int i=0; i<n; i++)
    cilk_y[i] = i+1;
 
  unsigned long long start_c = cilk_getticks();   
  cilk_fft(cilk_y);
  unsigned long long stop_c  = cilk_getticks();
  unsigned long long ticks_c = stop_c - start_c;
  std::cout << "cilk fft: " << cilk_ticks_to_seconds(ticks_c)*1000 << std::endl;
/*}}}*/
  // sequential FFT (cooley-tukey) /*{{{*/
  carray y(n);
  for(int i=0; i<n; i++)
    y[i] = i+1;
  
  unsigned long long start_s = cilk_getticks();   
  fft(y);
  unsigned long long stop_s  = cilk_getticks();
  unsigned long long ticks_s = stop_s - start_s;
  std::cout << "non_cilk fft: " << cilk_ticks_to_seconds(ticks_s)*1000 << std::endl;
/*}}}*/

#ifdef DEBUG
  for(int i=0; i<n; i++)
    std::cout << i << ": " << y[i] << std::endl;
  std::cout << std::endl;
#endif

  ifft(y);
  ifft(cilk_y);
#ifdef DEBUG
  for(int i=0; i<n; i++) {
    std::cout << "noncilk - " << i << ": " << y[i] << std::endl;
    std::cout << "cilk    - " << i << ": " << cilk_y[i] << std::endl;
  }
  std::cout << std::endl;
#endif // DEBUG
#endif // FFT
/*}}}*/
  return 0;
}
