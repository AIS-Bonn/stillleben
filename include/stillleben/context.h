// stillleben context
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_CONTEXT_H
#define STILLLEBEN_CONTEXT_H

#include <memory>
#include <string>

#include <Magnum/GL/Texture.h>

#include <Corrade/Containers/Pointer.h>

namespace Magnum
{
    namespace DebugTools
    {
        class ResourceManager;
    }
    namespace Trade
    {
        class AbstractImporter;
        class AbstractImageConverter;
        class MaterialData;
    }
    namespace GL
    {
        class RectangleTexture;
    }
}

namespace physx
{
    class PxPhysics;
    class PxCooking;
    class PxMaterial;
}

namespace sl
{

class Mesh;

class Context
{
public:
    using Ptr = std::shared_ptr<Context>;

    Context(const Context& other) = delete;
    Context(const Context&& other) = delete;

    static Ptr Create(const std::string& installPrefix = {});
    static Ptr CreateCUDA(unsigned int device = 0, const std::string& installPrefix = {});

    bool makeCurrent();

    std::string importerPluginPath() const;
    std::string imageConverterPluginPath() const;
    std::string sceneConverterPluginPath() const;

    Magnum::GL::RectangleTexture loadTexture(const std::string& path);
    Magnum::GL::Texture2D loadTexture2D(const std::string& path);

    physx::PxPhysics& physxPhysics();
    physx::PxCooking& physxCooking();
    physx::PxMaterial& physxDefaultMaterial();

    const Magnum::Trade::MaterialData& defaultMaterial() const;

    Magnum::DebugTools::ResourceManager& debugResourceManager();


    int visualID() const;
    void* eglConfig() const;

    void* x11Display() const;
private:
    class Private;

    Context(const std::string& installPrefix = {});

    std::unique_ptr<Private> m_d;
};

}

#endif
