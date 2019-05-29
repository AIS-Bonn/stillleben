// CUDA integration
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_CUDA_INTEROP_H
#define STILLLEBEN_CUDA_INTEROP_H

#include <Magnum/GL/RectangleTexture.h>

#include <cstdlib>
#include <memory>

namespace sl
{

class CUDAMapper;

class CUDAMap
{
public:
    class Private;

    explicit CUDAMap(CUDAMapper& mapper, Magnum::GL::RectangleTexture& texture, std::size_t bytesPerPixel);
    ~CUDAMap();

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
    CUDAMapper& m_parent;
    std::unique_ptr<Private> m_d;
};

class CUDAMapper
{
public:
    CUDAMapper();
    ~CUDAMapper();

    void mapAll();
    void unmapAll();
private:
    class Private;
    friend class CUDAMap;

    void registerMap(CUDAMap::Private& map);
    void unregisterMap(CUDAMap::Private& map);

    std::unique_ptr<Private> m_d;
};

}

#endif
