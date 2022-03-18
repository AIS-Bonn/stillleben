// Vertex shader

// Mesh attributes
layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec4 position;

layout(location = TEXTURECOORDINATES_ATTRIBUTE_LOCATION)
in highp vec2 textureCoords;

layout(location = COLOR_ATTRIBUTE_LOCATION)
in mediump vec4 vertexColors;

layout(location = NORMAL_ATTRIBUTE_LOCATION)
in mediump vec3 normal;

layout(location = TANGENT_ATTRIBUTE_LOCATION)
in mediump vec4 tangent;

// for differentiable renderer
layout(location = VERTEX_INDEX_ATTRIBUTE_LOCATION)
in uint vertexIndex;


// Uniform parameters
layout(location = UNIFORM_MESH_TO_OBJECT)
uniform highp mat4 meshToObject = mat4(1.0);

layout(location = UNIFORM_OBJECT_TO_WORLD)
uniform highp mat4 objectToWorld = mat4(1.0);

layout(location = UNIFORM_PROJECTION)
uniform highp mat4 projectionMatrix = mat4(1.0);

layout(location = UNIFORM_WORLD_TO_CAM)
uniform mediump mat4 worldToCam = mat4(1.0);

layout(location = UNIFORM_NORMAL_TO_WORLD)
uniform mediump mat3 normalToWorld = mat3(1.0);

layout(location = UNIFORM_NORMAL_TO_CAM)
uniform mediump mat3 normalToCam = mat3(1.0);

// Sticker simulator
layout(location = UNIFORM_STICKER_PROJECTION)
uniform mat4 stickerProjection = mat4(1.0);

layout(location = UNIFORM_STICKER_RANGE)
uniform vec4 stickerRange = vec4(0.0);


// Outputs
out DataBridge primitiveData;
flat out uint gsVertexIndex;
out highp vec4 vsPosition;


void main()
{
    // Mesh points to object coordinates
    highp vec4 objectCoordinates4 = meshToObject * position;
    primitiveData.objectCoordinates = objectCoordinates4 / objectCoordinates4.w;

    // Object coordinates to world coordinates
    highp vec4 worldCoordinates4 = objectToWorld * objectCoordinates4;
    primitiveData.worldCoordinates = (worldCoordinates4 / worldCoordinates4.w).xyz;

    // World coordinates to camera coordinates
    highp vec4 camCoordinates4 = worldToCam * worldCoordinates4;
    primitiveData.camCoordinates = camCoordinates4.xyz / camCoordinates4.w;

    // Output depth in fourth channel of the coordinate output
    primitiveData.objectCoordinates.w = primitiveData.camCoordinates.z;

    // Transformed normal, tangent, bitangent vector
    primitiveData.normalInCam = normalize(normalToCam * normal);
    primitiveData.normalInWorld = normalize(normalToWorld * normal);
    primitiveData.tangentInWorld = normalize(normalToWorld * tangent.xyz);
    primitiveData.bitangentInWorld = normalize(cross(primitiveData.normalInWorld, primitiveData.tangentInWorld)) * tangent.w;

    // Transform the position
    vsPosition = projectionMatrix*camCoordinates4;

    // Texture coordinates
    primitiveData.interpolatedTextureCoords = textureCoords;

    primitiveData.interpolatedVertexColors = vertexColors;

    gsVertexIndex = vertexIndex;

    /* Project into sticker frame */
    highp vec4 stickerPos = stickerProjection * objectCoordinates4;
    stickerPos = stickerPos / stickerPos.w;

    primitiveData.stickerCoordinates = (stickerPos.xy - stickerRange.xy) / stickerRange.zw;
}
