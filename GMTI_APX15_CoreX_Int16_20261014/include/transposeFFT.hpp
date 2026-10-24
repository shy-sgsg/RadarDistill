// transpose_fft_cols.hpp
#pragma once
#include <vector>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <fftw3.h>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>         // OpenMP 支持
#endif

class ColFFTTranspose
{
public:
    ColFFTTranspose() = default;
    ~ColFFTTranspose() { destroy(); }

    // 初始化（建议全流程只建一次）
    bool init(int H, int W, bool use_threads = false, int nthreads = 1)
    {
        H_ = H;
        W_ = W;
        if (H_ <= 0 || W_ <= 0)
            return false;

        // 线程版（可选）
        if (use_threads)
        {
#ifdef FFTW3_THREADS
            static bool th_inited = false;
            if (!th_inited)
            {
                fftw_init_threads();
                th_inited = true;
            }
            fftw_plan_with_nthreads(std::max(1, nthreads));
#else
            (void)nthreads;
#endif
        }

        // 批量行向 FFT：howmany=W，每行长度=H
        int rank = 1;
        int n[1] = {H_};
        int howmany = W_;
        int istride = 1, ostride = 1;
        int idist = H_, odist = H_;
        int *inembed = n;
        int *onembed = n;

        // 用哑元缓冲建计划（执行时可用任何同布局地址）
        const size_t total = (size_t)H_ * (size_t)W_;
        fftw_complex *dummy = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * total);
        if (!dummy)
            return false;

        plan_row_fwd_ = fftw_plan_many_dft(rank, n, howmany,
                                           dummy, inembed, istride, idist,
                                           dummy, onembed, ostride, odist,
                                           FFTW_FORWARD, FFTW_MEASURE);
        fftw_free(dummy);
        if (!plan_row_fwd_)
            return false;

        // 工作区：W x H（用 fftw_malloc 对齐，便于 SIMD/预取）
        at_bytes_ = sizeof(std::complex<double>) * total;
        at_ = (std::complex<double> *)fftw_malloc(at_bytes_);
        if (!at_)
            return false;

        return true;
    }

    void destroy()
    {
        if (plan_row_fwd_)
        {
            fftw_destroy_plan(plan_row_fwd_);
            plan_row_fwd_ = nullptr;
        }
        if (at_)
        {
            fftw_free(at_);
            at_ = nullptr;
        }
        at_bytes_ = 0;
        H_ = W_ = 0;
    }

    // 执行：对 in(HxW) 做“列向 FFT + 预乘(-1)^r（等效 fftshift）”，输出到 out(HxW)
    bool execute(const std::complex<double> *in, std::complex<double> *out)
    {
        if (!plan_row_fwd_ || H_ <= 0 || W_ <= 0 || !in || !out)
            return false;

        // 1) 预乘 (-1)^r 并转置到 at_(WxH)
        fuse_preflip_and_transpose_(in, at_, H_, W_);

        // 2) 在 at_(WxH) 上批量行向 FFT（howmany=W，长度=H），就地
        fftw_execute_dft(plan_row_fwd_,
                         reinterpret_cast<fftw_complex *>(at_),
                         reinterpret_cast<fftw_complex *>(at_));

        // 3) 把 (WxH) 再转置回 (HxW)
        transpose_blocked_(at_, out, W_, H_); // 输入 WxH，输出 HxW

        return true;
    }

private:
    int H_ = 0, W_ = 0;
    fftw_plan plan_row_fwd_ = nullptr;

    std::complex<double> *at_ = nullptr; // 工作区 (W x H)
    size_t at_bytes_ = 0;

    // 把 in(HxW) 预乘 (-1)^r 后转置到 out(WxH)
    static inline void fuse_preflip_and_transpose_(const std::complex<double> *in,
                                                   std::complex<double> *out,
                                                   int H, int W)
    {
        // 并行每一行
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < H; ++r)
        {
            const double s = (r & 1) ? -1.0 : 1.0; // (-1)^r
            const std::complex<double> *row_in = in + (size_t)r * W;
            std::complex<double> *col_out = out + r; // 写到 (c, r) => out[c*H + r]
            for (int c = 0; c < W; ++c)
            {
                col_out[(size_t)c * H] = row_in[c] * s;
            }
        }
    }

    // 分块转置：in(rows x cols) → out(cols x rows)，行优先
    static inline void transpose_blocked_(const std::complex<double> *in,
                                          std::complex<double> *out,
                                          int rows, int cols, int TILE = 32)
    {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
        for (int r0 = 0; r0 < rows; r0 += TILE)
        {
            for (int c0 = 0; c0 < cols; c0 += TILE)
            {
                const int r1 = std::min(r0 + TILE, rows);
                const int c1 = std::min(c0 + TILE, cols);
                for (int r = r0; r < r1; ++r)
                {
                    const size_t in_base = (size_t)r * cols + c0;
                    for (int c = c0; c < c1; ++c)
                    {
                        out[(size_t)c * rows + r] = in[in_base + (size_t)(c - c0)];
                    }
                }
            }
        }
    }
};