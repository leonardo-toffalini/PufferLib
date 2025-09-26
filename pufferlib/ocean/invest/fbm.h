#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Calculate next power of 2
uint32_t next_power_of_2(uint32_t n) {
  if (n == 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

// In-place FFT (Cooley-Tukey)
void fft(double complex *x, int N, bool inverse) {
  if (N <= 1)
    return;

  // Bit-reversal permutation
  for (int i = 1, j = 0; i < N; i++) {
    int bit = N >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      double complex temp = x[i];
      x[i] = x[j];
      x[j] = temp;
    }
  }

  // FFT stages
  for (int len = 2; len <= N; len <<= 1) {
    double angle = 2 * M_PI / len * (inverse ? 1 : -1);
    double complex wlen = cos(angle) + sin(angle) * I;
    for (int i = 0; i < N; i += len) {
      double complex w = 1.0;
      for (int j = 0; j < len / 2; j++) {
        double complex u = x[i + j];
        double complex v = x[i + j + len / 2] * w;
        x[i + j] = u + v;
        x[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }

  // Scaling for inverse FFT
  if (inverse) {
    for (int i = 0; i < N; i++) {
      x[i] /= N;
    }
  }
}

// Box-Muller transform for normal distribution
void box_muller(double *out1, double *out2) {
  double u, v, s;
  do {
    u = rand() / (double)RAND_MAX * 2.0 - 1.0;
    v = rand() / (double)RAND_MAX * 2.0 - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);
  s = sqrt(-2.0 * log(s) / s);
  *out1 = u * s;
  *out2 = v * s;
}

// Covariance function for fractional Gaussian noise
double r_k(double H, int k) {
  if (k == 0)
    return 1.0;
  return 0.5 * (pow(k + 1, 2 * H) - 2 * pow(k, 2 * H) + pow(abs(k - 1), 2 * H));
}

// Fractional Brownian motion generator
double *simulate_fBm(double H, int n, double T) {
  // Adjust to power-of-two size
  if ((n & (n - 1)) != 0 || n < 2) {
    n = next_power_of_2(n);
  }

  // Allocate memory
  double *c = malloc(2 * n * sizeof(double));
  double *lam = malloc(2 * n * sizeof(double));
  double complex *Z = malloc(2 * n * sizeof(double complex));
  double complex *Y = malloc(2 * n * sizeof(double complex));
  double *fGn = malloc(n * sizeof(double));
  double *fBm = malloc((n + 1) * sizeof(double));

  // Construct covariance vector
  for (int k = 0; k < n; k++) {
    c[k] = r_k(H, k);
  }
  c[n] = r_k(H, n);
  for (int k = 1; k < n; k++) {
    c[2 * n - k] = c[k];
  }

  // FFT of covariance vector
  double complex *c_fft = malloc(2 * n * sizeof(double complex));
  for (int i = 0; i < 2 * n; i++) {
    c_fft[i] = c[i];
  }
  fft(c_fft, 2 * n, false);
  free(c);

  // Compute eigenvalues (real part)
  for (int i = 0; i < 2 * n; i++) {
    lam[i] = creal(c_fft[i]);
    if (lam[i] < 0)
      lam[i] = 0;
  }
  free(c_fft);

  // Generate complex Gaussian vector
  double n1, n2;
  Z[0] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  Z[n] = (double)rand() / RAND_MAX * 2.0 - 1.0;

  for (int k = 1; k < n; k++) {
    box_muller(&n1, &n2);
    Z[k] = (n1 + n2 * I) * M_SQRT1_2;
    Z[2 * n - k] = conj(Z[k]);
  }

  // Spectral synthesis
  for (int i = 0; i < 2 * n; i++) {
    Y[i] = sqrt(lam[i]) * Z[i];
  }
  free(lam);
  free(Z);

  // Inverse FFT
  fft(Y, 2 * n, true);

  // Extract fractional Gaussian noise
  for (int i = 0; i < n; i++) {
    fGn[i] = creal(Y[i]);
  }
  free(Y);

  // Scale for interval [0,T]
  double scale = sqrt(T / n) * pow(T / n, H - 0.5);
  for (int i = 0; i < n; i++) {
    fGn[i] *= scale;
  }

  // Construct fBm path (cumulative sum)
  fBm[0] = 0.0;
  for (int i = 0; i < n; i++) {
    fBm[i + 1] = fBm[i] + fGn[i];
  }
  free(fGn);

  return fBm;
}
