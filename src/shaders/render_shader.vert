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

layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec4 position;

layout(location = NORMAL_ATTRIBUTE_LOCATION)
in mediump vec3 normal;

#ifdef TEXTURED
layout(location = TEXTURECOORDINATES_ATTRIBUTE_LOCATION)
in mediump vec2 textureCoords;
#endif

#ifdef TEXTURED
out mediump vec2 interpolatedTextureCoords;
#endif

out mediump vec3 transformedNormal;
out highp vec3 lightDirection;
out highp vec3 cameraDirection;

out highp vec3 objectCoordinates;

void main()
{
    /* Mesh points to object coordinates */
    highp vec4 objectCoordinates4 = meshToObject * position;
    objectCoordinates = objectCoordinates4.xyz / objectCoordinates4.w;

    /* Object coordinates to camera coordinates */
    highp vec4 camCoordinates4 = objectToCam * objectCoordinates4;
    highp vec3 camCoordinates = camCoordinates4.xyz / camCoordinates4.w;

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
}
