// Obtain tangent attribute from GLTF
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <stillleben/mesh_tools/tangents.h>

#include <Corrade/Containers/ArrayView.h>

#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/MeshData.h>

#include <MagnumExternal/TinyGltf/tiny_gltf.h>

using namespace Corrade::Containers;
using namespace Magnum;

namespace
{

// The following is taken from TinyGltfImporter.cpp

std::size_t elementSize(const tinygltf::Accessor& accessor)
{
    /* GetTypeSizeInBytes() is totally bogus and misleading name, it should
       have been called GetTypeComponentCount but who am I to judge. */
    return tinygltf::GetComponentSizeInBytes(accessor.componentType)*tinygltf::GetNumComponentsInType(accessor.type);
}

Containers::ArrayView<const char> bufferView(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const std::size_t bufferElementSize = elementSize(accessor);
    CORRADE_INTERNAL_ASSERT(std::size_t(accessor.bufferView) < model.bufferViews.size());
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    CORRADE_INTERNAL_ASSERT(std::size_t(bufferView.buffer) < model.buffers.size());
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    CORRADE_INTERNAL_ASSERT(bufferView.byteStride == 0 || bufferView.byteStride == bufferElementSize);
    return {reinterpret_cast<const char*>(buffer.data.data()) + bufferView.byteOffset + accessor.byteOffset, accessor.count*bufferElementSize};
}

template<class T> Containers::ArrayView<const T> bufferView(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    CORRADE_INTERNAL_ASSERT(elementSize(accessor) == sizeof(T));
    return Containers::arrayCast<const T>(bufferView(model, accessor));
}

}

namespace sl
{

Optional<Containers::Array<Vector3>> extractTangents(const Trade::AbstractImporter& importer, const Trade::MeshData& meshData)
{
    auto model = reinterpret_cast<const tinygltf::Model*>(importer.importerState());
    auto mesh = reinterpret_cast<const tinygltf::Mesh*>(meshData.importerState());

    if(mesh->primitives.size() != 1)
    {
        Warning{} << "Mesh has multiple primitives. extractTangents() does not support that and will not return tangents.";
        return {};
    }

    const auto& primitive = mesh->primitives[0];

    auto tangentIt = primitive.attributes.find("TANGENT");
    if(tangentIt == primitive.attributes.end())
        return {};

    const auto& accessor = model->accessors[tangentIt->second];

    if(accessor.type != TINYGLTF_TYPE_VEC4) {
        Warning{} << "extractTangents(): expected type of TANGENT is VEC3, got" << accessor.type;
        return {};
    }

    Containers::Array<Vector3> tangents{accessor.count};
    const auto buffer = bufferView<Vector4>(*model, accessor);
    for(std::size_t i = 0; i < buffer.size(); ++i)
    {
        const Vector4& vec = buffer[i];
        if(std::abs(vec.w() - 1.0f) > 1e-9)
        {
            Warning{} << "extractTangents(): We only support w=1.0 at the moment, will not return tangents";
            return {};
        }

        tangents[i] = vec.xyz();
    }

    return tangents;
}

}
