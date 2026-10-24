#include "../include/transposeFFT.hpp"

#include <complex>
#include <cstddef>
#include <memory>

struct D0FftContext
{
    explicit D0FftContext(int nfft, int width) : nfft(nfft), width(width) {}

    int nfft;
    int width;
    ColFFTTranspose fft;
};

extern "C" void *radardistill_d0_fft_create(int nfft, int width)
{
    if (nfft <= 0 || width <= 0)
        return nullptr;
    std::unique_ptr<D0FftContext> context(new D0FftContext(nfft, width));
    if (!context->fft.init(nfft, width, false, 1))
        return nullptr;
    return context.release();
}

extern "C" void radardistill_d0_fft_destroy(void *opaque)
{
    delete static_cast<D0FftContext *>(opaque);
}

extern "C" int radardistill_d0_fft_execute_many(
    void *opaque,
    int batch_size,
    const std::complex<double> *input,
    std::complex<double> *output)
{
    D0FftContext *context = static_cast<D0FftContext *>(opaque);
    if (!context || batch_size <= 0 || !input || !output)
        return 0;

    const std::size_t sample_size =
        static_cast<std::size_t>(context->nfft) * static_cast<std::size_t>(context->width);
    for (int sample = 0; sample < batch_size; ++sample)
    {
        const std::size_t offset = static_cast<std::size_t>(sample) * sample_size;
        if (!context->fft.execute(input + offset, output + offset))
            return 0;
    }
    return 1;
}
