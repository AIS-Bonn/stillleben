// Most of the code taken from Magnum engine

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 0)
#endif
uniform highp mat4 transformationMatrix = mat4(1.0);

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 1)
#endif
uniform highp mat4 projectionMatrix = mat4(1.0);

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 2)
#endif
uniform mediump mat3 normalMatrix = mat3(1.0);

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 3)
#endif
uniform highp vec3 lightPosition; /* defaults to zero */

#ifdef EXPLICIT_ATTRIB_LOCATION
layout(location = POSITION_ATTRIBUTE_LOCATION)
#endif
in highp vec4 position;

#ifdef EXPLICIT_ATTRIB_LOCATION
layout(location = NORMAL_ATTRIBUTE_LOCATION)
#endif
in mediump vec3 normal;

#ifdef TEXTURED
#ifdef EXPLICIT_ATTRIB_LOCATION
layout(location = TEXTURECOORDINATES_ATTRIBUTE_LOCATION)
#endif
in mediump vec2 textureCoords;
#endif

layout(location = 4)
in highp vec3 offsetCoords;

#ifdef EXPLICIT_ATTRIB_LOCATION
layout(location = 10)
#endif
uniform highp mat4 worldTransformMatrix = mat4(1.0);

#ifdef EXPLICIT_ATTRIB_LOCATION
layout(location = 11)
#endif
uniform highp mat4 pretransformMatrix = mat4(1.0);

#ifdef TEXTURED
out mediump vec2 interpolatedTextureCoords;
#endif

out mediump vec3 transformedNormal;
out highp vec3 lightDirection;
out highp vec3 cameraDirection;

out highp vec3 worldPosition;

void main()
{
    /* Transformed vertex position */
    highp vec4 worldPosition4 = worldTransformMatrix*position;
    worldPosition = worldPosition4.xyz/worldPosition4.w;

    highp vec4 deformed = vec4(worldPosition + offsetCoords, 1.0);
    deformed = pretransformMatrix * deformed;

    highp vec4 transformedPosition4 = transformationMatrix*deformed;
    highp vec3 transformedPosition = transformedPosition4.xyz / transformedPosition4.w;

    /* Transformed normal vector */
    transformedNormal = normalMatrix*normal;

    /* Direction to the light */
    lightDirection = normalize(lightPosition - transformedPosition);

    /* Direction to the camera */
    cameraDirection = -transformedPosition;

    /* Transform the position */
    gl_Position = projectionMatrix*transformedPosition4;

    #ifdef TEXTURED
    /* Texture coordinates, if needed */
    interpolatedTextureCoords = textureCoords;
    #endif
}
