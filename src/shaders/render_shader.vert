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

// for differentiable renderer 
layout(location = VERTEXINDEX_ATTRIBUTE_LOCATION)
in uint vertexIndex;


out DataBridge primitiveData;
flat out uint gsVertexIndex;;
out highp vec4 vsPosition;



void main()
{
    /* Mesh points to object coordinates */
    highp vec4 objectCoordinates4 = meshToObject * position;
    primitiveData.objectCoordinates = objectCoordinates4 / objectCoordinates4.w;

    /* Object coordinates to camera coordinates */
    highp vec4 camCoordinates4 = objectToCam * objectCoordinates4;
    primitiveData.camCoordinates = camCoordinates4.xyz / camCoordinates4.w;

    /* Output depth in fourth channel of the coordinate output */
    primitiveData.objectCoordinates.w = primitiveData.camCoordinates.z;

    /* Transformed normal vector */
    primitiveData.transformedNormal = normalMatrix*normal;

    /* Direction to the light */
    primitiveData.lightDirection = normalize(lightPosition - primitiveData.camCoordinates);

    /* Direction to the camera */
    primitiveData.cameraDirection = -primitiveData.camCoordinates;

    /* Transform the position */
    vsPosition = projectionMatrix*camCoordinates4;

    #ifdef TEXTURED
    /* Texture coordinates, if needed */
    primitiveData.interpolatedTextureCoords = textureCoords;
    #endif

    #ifdef VERTEX_COLORS
    primitiveData.interpolatedVertexColors = vertexColors;
    #endif

    gsVertexIndex = vertexIndex;
    /* Project into sticker frame */
    highp vec4 stickerPos = stickerProjection * objectCoordinates4;
    stickerPos = stickerPos / stickerPos.w;

    primitiveData.stickerCoordinates = (stickerPos.xy - stickerRange.xy) / stickerRange.zw;
}
