// Most of the code taken from Magnum engine

#ifndef EXPLICIT_UNIFORM_LOCATION
#error This shader needs GL_ARB_explicit_uniform_location
#endif
#ifndef EXPLICIT_ATTRIB_LOCATION
#error This shader needs GL_ARB_explicit_attrib_location
#endif

layout(location = 0)
uniform highp mat4 meshToObject = mat4(1.0);

layout(location = 1)
uniform highp mat4 objectToCam = mat4(1.0);

layout(location = 2)
uniform highp mat4 projectionMatrix = mat4(1.0);

layout(location = 3)
uniform mediump mat3 normalMatrix = mat3(1.0);

layout(location = 4)
uniform highp vec3 lightPosition; /* defaults to zero */


// Sticker simulator
layout(location = 16)
uniform mat4 stickerProjection = mat4(1.0);

layout(location = 17)
uniform vec4 stickerRange = vec4(0.0);


layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec4 position;

layout(location = NORMAL_ATTRIBUTE_LOCATION)
in mediump vec3 normal;

#ifdef TEXTURED
layout(location = TEXTURECOORDINATES_ATTRIBUTE_LOCATION)
in mediump vec2 textureCoords;
#endif

#ifdef VERTEX_COLORS
layout(location = VERTEXCOLORS_ATTRIBUTE_LOCATION)
in mediump vec4 vertexColors;
#endif

#ifdef TEXTURED
out mediump vec2 interpolatedTextureCoords;
#endif

#ifdef VERTEX_COLORS
out mediump vec4 interpolatedVertexColors;
#endif

out mediump vec3 transformedNormal;
out highp vec3 lightDirection;
out highp vec3 cameraDirection;

centroid out highp vec4 objectCoordinates;
centroid out highp vec3 camCoordinates;

out mediump vec2 stickerCoordinates;

void main()
{
    /* Mesh points to object coordinates */
    highp vec4 objectCoordinates4 = meshToObject * position;
    objectCoordinates = objectCoordinates4 / objectCoordinates4.w;

    /* Object coordinates to camera coordinates */
    highp vec4 camCoordinates4 = objectToCam * objectCoordinates4;
    camCoordinates = camCoordinates4.xyz / camCoordinates4.w;

    /* Output depth in fourth channel of the coordinate output */
    objectCoordinates.w = camCoordinates.z;

    /* Transformed normal vector */
    transformedNormal = normalMatrix*normal;

    /* Direction to the light */
    lightDirection = normalize(lightPosition - camCoordinates);

    /* Direction to the camera */
    cameraDirection = -camCoordinates;

    /* Transform the position */
    gl_Position = projectionMatrix*camCoordinates4;

    #ifdef TEXTURED
    /* Texture coordinates, if needed */
    interpolatedTextureCoords = textureCoords;
    #endif

    #ifdef VERTEX_COLORS
    interpolatedVertexColors = vertexColors;
    #endif

    /* Project into sticker frame */
    highp vec4 stickerPos = stickerProjection * objectCoordinates4;
    stickerPos = stickerPos / stickerPos.w;

    stickerCoordinates = (stickerPos.xy - stickerRange.xy) / stickerRange.zw;
}
