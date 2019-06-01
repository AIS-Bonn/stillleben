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

class CUDATexture : public Magnum::GL::RectangleTexture
{
public:
    class Private;

    explicit CUDATexture(CUDAMapper& mapper);
    explicit CUDATexture(CUDAMapper& mapper, Magnum::NoCreateT);
    CUDATexture(const CUDATexture&) = delete;
    CUDATexture(CUDATexture&& other);

    ~CUDATexture();

    CUDATexture& operator=(const CUDATexture&) = delete;
    CUDATexture& operator=(CUDATexture&& other);

    void setStorage(Magnum::GL::TextureFormat internalFormat, const Magnum::Vector2i& size) = delete;
    void setStorage(Magnum::GL::TextureFormat internalFormat, std::size_t pixelSize, const Magnum::Vector2i& size);

    /**
     * @brief Copy data into CUDA buffer
     *
     * Copies the data in the texture into the CUDA address @p dest.
     * Note that it is the user's responsibility to ensure that
     * bytesPerPixel * tex.width() * tex.height() bytes are available in the
     * buffer.
     **/
    void readIntoCUDA(void* dest);
private:
    CUDAMapper& m_parent;
    std::unique_ptr<Private> m_d;
};

class CUDAMapper
{
public:
    explicit CUDAMapper(bool active);
    ~CUDAMapper();

    void mapAll();
    void unmapAll();

    constexpr bool active() const
    { return m_active; }
private:
    class Private;
    friend class CUDATexture;

    void registerMap(CUDATexture::Private& map);
    void unregisterMap(CUDATexture::Private& map);

    bool m_active;
    std::unique_ptr<Private> m_d;
};

}

#endif
