#include <torch/extension.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

namespace torch {
namespace audio {

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

#define KALDI_COMPLEXFFT_BLOCKSIZE 8192

typedef int MatrixIndexT;

//! ComplexMul implements, inline, the complex multiplication b *= a.
template <typename Real>
inline void ComplexMul(const Real &a_re, const Real &a_im, Real *b_re,
                       Real *b_im) {
  Real tmp_re = (*b_re * a_re) - (*b_im * a_im);
  *b_im = *b_re * a_im + *b_im * a_re;
  *b_re = tmp_re;
}

template <typename Real>
inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                              const Real &b_re, const Real &b_im, Real *c_re,
                              Real *c_im) {
  *c_re += b_re * a_re - b_im * a_im;
  *c_im += b_re * a_im + b_im * a_re;
}

template <typename Real>
inline void ComplexImExp(Real x, Real *a_re, Real *a_im) {
  *a_re = std::cos(x);
  *a_im = std::sin(x);
}

template <typename Real>
void ComplexFftRecursive(Real *data, int nffts, int N, const int *factor_begin,
                         const int *factor_end, bool forward,
                         vector<Real> *tmp_vec) {
  if (factor_begin == factor_end) {
    assert(N == 1);
    return;
  }

  {  // an optimization: compute in smaller blocks.
    // this block of code could be removed and it would still work.
    MatrixIndexT size_perblock = N * 2 * sizeof(Real);
    if (nffts > 1 && size_perblock * nffts >
                         KALDI_COMPLEXFFT_BLOCKSIZE) {  // can break it up...
      // Break up into multiple blocks.  This is an optimization.  We make
      // no progress on the FFT when we do this.
      int block_skip =
          KALDI_COMPLEXFFT_BLOCKSIZE / size_perblock;  // n blocks per call
      if (block_skip == 0) block_skip = 1;
      if (block_skip < nffts) {
        int blocks_left = nffts;
        while (blocks_left > 0) {
          int skip_now = std::min(blocks_left, block_skip);
          ComplexFftRecursive(data, skip_now, N, factor_begin, factor_end,
                              forward, tmp_vec);
          blocks_left -= skip_now;
          data += skip_now * N * 2;
        }
        return;
      }  // else do the actual algorithm.
    }    // else do the actual algorithm.
  }

  int P = *factor_begin;
  assert(P > 1);
  int Q = N / P;

  if (P > 1 &&
      Q > 1) {  // Do the rearrangement.   C.f. eq. (8) below.  Transform
    // (a) to (b).
    Real *data_thisblock = data;
    if (tmp_vec->size() < (MatrixIndexT)N) tmp_vec->resize(N);
    Real *data_tmp = &(tmp_vec->at(0));
    for (int thisfft = 0; thisfft < nffts; thisfft++, data_thisblock += N * 2) {
      for (int offset = 0; offset < 2; offset++) {  // 0 == real, 1 == im.
        for (int p = 0; p < P; p++) {
          for (int q = 0; q < Q; q++) {
            int aidx = q * P + p, bidx = p * Q + q;
            data_tmp[bidx] = data_thisblock[2 * aidx + offset];
          }
        }
        for (int n = 0; n < P * Q; n++)
          data_thisblock[2 * n + offset] = data_tmp[n];
      }
    }
  }

  {  // Recurse.
    ComplexFftRecursive(data, nffts * P, Q, factor_begin + 1, factor_end,
                        forward, tmp_vec);
  }

  int exp_sign = (forward ? -1 : 1);
  Real rootN_re, rootN_im;  // Nth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / N), &rootN_re, &rootN_im);

  Real rootP_re, rootP_im;  // Pth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / P), &rootP_re, &rootP_im);

  {  // Do the multiplication
    // could avoid a bunch of complex multiplies by moving the loop over
    // data_thisblock inside.
    if (tmp_vec->size() < (MatrixIndexT)(P * 2)) tmp_vec->resize(P * 2);
    Real *temp_a = &(tmp_vec->at(0));

    Real *data_thisblock = data, *data_end = data + (N * 2 * nffts);
    for (; data_thisblock != data_end;
         data_thisblock += N * 2) {   // for each separate fft.
      Real qd_re = 1.0, qd_im = 0.0;  // 1^(q'/N)
      for (int qd = 0; qd < Q; qd++) {
        Real pdQ_qd_re = qd_re,
             pdQ_qd_im =
                 qd_im;  // 1^((p'Q+q') / N) == 1^((p'/P) + (q'/N))
                         // Initialize to q'/N, corresponding to p' == 0.
        for (int pd = 0; pd < P; pd++) {  // pd == p'
          {  // This is the p = 0 case of the loop below [an optimization].
            temp_a[pd * 2] = data_thisblock[qd * 2];
            temp_a[pd * 2 + 1] = data_thisblock[qd * 2 + 1];
          }
          {  // This is the p = 1 case of the loop below [an optimization]
            // **** MOST OF THE TIME (>60% I think) gets spent here. ***
            ComplexAddProduct(pdQ_qd_re, pdQ_qd_im,
                              data_thisblock[(qd + Q) * 2],
                              data_thisblock[(qd + Q) * 2 + 1],
                              &(temp_a[pd * 2]), &(temp_a[pd * 2 + 1]));
          }
          if (P > 2) {
            Real p_pdQ_qd_re = pdQ_qd_re,
                 p_pdQ_qd_im = pdQ_qd_im;  // 1^(p(p'Q+q')/N)
            for (int p = 2; p < P; p++) {
              ComplexMul(pdQ_qd_re, pdQ_qd_im, &p_pdQ_qd_re,
                         &p_pdQ_qd_im);  // p_pdQ_qd *= pdQ_qd.
              int data_idx = p * Q + qd;
              ComplexAddProduct(p_pdQ_qd_re, p_pdQ_qd_im,
                                data_thisblock[data_idx * 2],
                                data_thisblock[data_idx * 2 + 1],
                                &(temp_a[pd * 2]), &(temp_a[pd * 2 + 1]));
            }
          }
          if (pd != P - 1)
            ComplexMul(rootP_re, rootP_im, &pdQ_qd_re,
                       &pdQ_qd_im);  // pdQ_qd *= (rootP == 1^{1/P})
          // (using 1/P == Q/N)
        }
        for (int pd = 0; pd < P; pd++) {
          data_thisblock[(pd * Q + qd) * 2] = temp_a[pd * 2];
          data_thisblock[(pd * Q + qd) * 2 + 1] = temp_a[pd * 2 + 1];
        }
        ComplexMul(rootN_re, rootN_im, &qd_re, &qd_im);  // qd *= rootN.
      }
    }
  }
}

template <class I>
void Factorize(I m, std::vector<I> *factors) {
  // Splits a number into its prime factors, in sorted order from
  // least to greatest,  with duplication.  A very inefficient
  // algorithm, which is mainly intended for use in the
  // mixed-radix FFT computation (where we assume most factors
  // are small).
  assert(factors != NULL);
  assert(m >= 1);  // Doesn't work for zero or negative numbers.
  factors->clear();
  I small_factors[10] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

  // First try small factors.
  for (I i = 0; i < 10; i++) {
    if (m == 1) return;  // We're done.
    while (m % small_factors[i] == 0) {
      m /= small_factors[i];
      factors->push_back(small_factors[i]);
    }
  }
  // Next try all odd numbers starting from 31.
  for (I j = 31;; j += 2) {
    if (m == 1) return;
    while (m % j == 0) {
      m /= j;
      factors->push_back(j);
    }
  }
}

template <typename Real>
void ComplexFft(vector<Real> *v, bool forward, vector<Real> *tmp_in = NULL) {
  assert(v != NULL);

  if (v->size() <= 1) return;
  assert(v->size() % 2 == 0);  // complex input.
  int N = v->size() / 2;
  std::vector<int> factors;
  Factorize(N, &factors);
  int *factor_beg = NULL;
  if (factors.size() > 0) factor_beg = &(factors[0]);
  vector<Real> tmp;  // allocated in ComplexFftRecursive.
  ComplexFftRecursive(&(v->at(0)), 1, N, factor_beg,
                      factor_beg + factors.size(), forward,
                      (tmp_in ? tmp_in : &tmp));
}

// See the long comment below for the math behind this.
template <typename Real>
void RealFft(vector<Real> *v, bool forward) {
  assert(v != NULL);
  MatrixIndexT N = v->size(), N2 = N / 2;
  assert(N % 2 == 0);
  if (N == 0) return;

  if (forward) ComplexFft(v, true);

  Real *data = &(v->at(0));
  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI / N * forward_sign), &rootN_re,
               &rootN_im);
  Real kN_re = -forward_sign,
       kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2 * k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5 * (data[2 * k] + data[N - 2 * k]);
    Ck_im = 0.5 * (data[2 * k + 1] - data[N - 2 * k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5 * (data[2 * k + 1] + data[N - 2 * k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im = -0.5 * (data[2 * k] - data[N - 2 * k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2 * k] = Ck_re;  // A_k <-- C_k
    data[2 * k + 1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2 * k]),
                      &(data[2 * k + 1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2 * kdash] = Ck_re;  // A_k' <-- C_k'
      data[2 * kdash + 1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 *
      // (1^(k/N))^* so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2 * kdash]),
                        &(data[2 * kdash + 1]));
    }
  }

  {  // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1], n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }

  if (!forward) {
    ComplexFft(v, false);
    // v->Scale(2.0);  // This is so we get a factor of N increase, rather than
    // N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in
    // sizeension N/2. It's for consistency with our normal FFT convensions.
  }
}

at::Tensor fft(at::Tensor input) {
  cout << "fft" << endl;
  auto input_a = input.accessor<float, 2>();
  torch::Tensor out =
      torch::rand({input_a.size(0), input_a.size(1) / 2 + 1, 2},
                  torch::TensorOptions().dtype(torch::kFloat32));
  auto out_a = out.accessor<float, 3>();

  for (int i = 0; i < input_a.size(0); i++) {
    std::vector<float> v;
    for (int j = 0; j < input_a.size(1); j++) {
      v.push_back(input_a[i][j]);
    }
    RealFft(&v, true);
    for (int j = 1; 2 * j + 1 < input_a.size(1); j++) {
      out_a[i][j][0] = v[2 * j];
      out_a[i][j][1] = v[2 * j + 1];
    }
    out_a[i][0][0] = v[0];
    out_a[i][0][1] = 0;
    out_a[i][input_a.size(1) / 2][0] = v[1];
    out_a[i][input_a.size(1) / 2][1] = 0;

    printf("[%.10f, %.10f],\n", v[0], 0);
    for (int j = 1; 2 * j + 1 < input_a.size(1); j++) {
      printf("[%.10f, %.10f],\n", v[2 * j], v[2 * j + 1]);
    }
    printf("[%.10f, %.10f],\n", v[1], 0);
  }

  return out;
}

}  // namespace audio
}  // namespace torch

PYBIND11_MODULE(_fft, m) { m.def("fft", &torch::audio::fft, "Description"); }