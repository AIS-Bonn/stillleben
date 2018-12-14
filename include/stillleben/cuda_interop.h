// CUDA integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_CUDA_INTEROP_H
#define STILLLEBEN_CUDA_INTEROP_H

#include <Magnum/GL/RectangleTexture.h>

#include <cstdlib>
#include <memory>

namespace sl
{

class CUDAMapper
{
public:
    explicit CUDAMapper(Magnum::GL::RectangleTexture& tex, std::size_t bytesPerPixel);
    ~CUDAMapper();

    /**
     * @brief Copy data into CUDA buffer
     *
     * Copies the data in the texture into the CUDA address @p dest.
     * Note that it is the user's responsibility to ensure that
     * bytesPerPixel * tex.width() * tex.height() bytes are available in the
     * buffer.
     **/
    void readInto(void* dest) const;
private:
    class Private;

    std::unique_ptr<Private> m_d;
};

}

#endif
