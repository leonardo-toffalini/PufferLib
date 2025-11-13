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
  double s = 1.0 / sqrt((double)N);
  if (inverse) {
    for (int i = 0; i < N; i++) {
      x[i] *= s;
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
  // Preserve requested number of steps; use an internal embedding size that may grow
  int n_req = n < 2 ? 2 : n;
  int n_embed = next_power_of_2(n_req);

  double *fBm = malloc((n_req + 1) * sizeof(double));
  if (!fBm)
    return NULL;

  double *fGn = NULL;

  while (true) {
    int m = 2 * n_embed; // FFT size for circulant embedding

    // Allocate temporaries for this embedding size
    double *c = malloc(m * sizeof(double));
    double *lam = malloc(m * sizeof(double));
    double complex *c_fft = malloc(m * sizeof(double complex));
    double complex *Z = malloc(m * sizeof(double complex));
    double complex *Y = malloc(m * sizeof(double complex));
    double *fGn_tmp = malloc(n_embed * sizeof(double));

    if (!c || !lam || !c_fft || !Z || !Y || !fGn_tmp) {
      free(c);
      free(lam);
      free(c_fft);
      free(Z);
      free(Y);
      free(fGn_tmp);
      free(fBm);
      return NULL;
    }

    // Construct covariance vector for fractional Gaussian noise
    for (int k = 0; k < n_embed; k++) {
      c[k] = r_k(H, k);
    }
    c[n_embed] = r_k(H, n_embed);
    for (int k = 1; k < n_embed; k++) {
      c[m - k] = c[k];
    }

    // FFT of covariance vector
    for (int i = 0; i < m; i++) {
      c_fft[i] = c[i];
    }
    fft(c_fft, m, false);
    free(c);

    // Compute eigenvalues and check positivity
    double min_lam = INFINITY;
    for (int i = 0; i < m; i++) {
      lam[i] = creal(c_fft[i]);
      if (lam[i] < min_lam)
        min_lam = lam[i];
    }
    free(c_fft);

    if (min_lam < 0) {
      // Increase embedding size until all eigenvalues are non-negative
      free(lam);
      free(Z);
      free(Y);
      free(fGn_tmp);
      n_embed <<= 1;
      continue;
    }

    // Generate complex Gaussian vector with Hermitian symmetry
    double g0, g1;
    box_muller(&g0, &g1);
    Z[0] = g0;           // real N(0,1)
    Z[n_embed] = g1;     // real N(0,1)

    for (int k = 1; k < n_embed; k++) {
      double n1, n2;
      box_muller(&n1, &n2); // N(0,1), N(0,1)
      Z[k] = (n1 + n2 * I) * M_SQRT1_2; // E|Z[k]|^2 = 1
      Z[m - k] = conj(Z[k]);
    }

    // Spectral synthesis (no extra normalization; inverse FFT already divides by m)
    for (int i = 0; i < m; i++) {
      double eig_sqrt = sqrt(lam[i] > 0 ? lam[i] : 0);
      Y[i] = eig_sqrt * Z[i];
    }
    free(lam);

    // Inverse FFT
    fft(Y, m, true);

    // Extract fractional Gaussian noise (first n_embed points)
    for (int i = 0; i < n_embed; i++) {
      fGn_tmp[i] = creal(Y[i]);
    }
    free(Y);
    free(Z);

    // Keep the result
    fGn = fGn_tmp;
    break;
  }

  // Scale for interval [0, T] using self-similarity: scale entire path by (T/n_req)^H
  double dt = T / (double)n_req;
  double inc_scale = pow(dt, H);

  // Construct fBm path (cumulative sum) from the first n_req increments
  fBm[0] = 0.0;
  for (int i = 0; i < n_req; i++) {
    double inc = fGn[i] * inc_scale;
    fBm[i + 1] = fBm[i] + inc;
  }
  free(fGn);

  return fBm;
}
