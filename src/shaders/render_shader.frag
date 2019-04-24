// Most of the code taken from Magnum engine

#ifndef RUNTIME_CONST
#define const
#endif



#ifdef AMBIENT_TEXTURE
layout(binding = 0)
uniform lowp sampler2D ambientTexture;
#endif

layout(location = 5)
uniform lowp vec4 ambientColor
    #ifndef AMBIENT_TEXTURE
    = vec4(0.0)
    #else
    = vec4(1.0)
    #endif
    ;

#ifdef DIFFUSE_TEXTURE
layout(binding = 1)
uniform lowp sampler2D diffuseTexture;
#endif

layout(location = 6)
uniform lowp vec4 diffuseColor = vec4(1.0);

#ifdef SPECULAR_TEXTURE
layout(binding = 2)
uniform lowp sampler2D specularTexture;
#endif

layout(location = 7)
uniform lowp vec4 specularColor = vec4(1.0);

layout(location = 8)
uniform lowp vec4 lightColor = vec4(1.0);

layout(location = 9)
uniform mediump float shininess = 80.0;

#ifdef ALPHA_MASK
layout(location = 10)
uniform lowp float alphaMask = 0.5;
#endif

// Segmentation information
layout(location = 11)
uniform uint classIndex = 0u;

layout(location = 12)
uniform uint instanceIndex = 0u;

in mediump vec3 transformedNormal;
in highp vec3 lightDirection;
in highp vec3 cameraDirection;

#if defined(AMBIENT_TEXTURE) || defined(DIFFUSE_TEXTURE) || defined(SPECULAR_TEXTURE)
in mediump vec2 interpolatedTextureCoords;
#endif

#ifdef VERTEX_COLORS
in mediump vec4 interpolatedVertexColors;
#endif

centroid in highp vec3 objectCoordinates;

layout(location = 0) out lowp vec4 color;
layout(location = 1) out highp vec3 objectCoordinatesOut;
layout(location = 2) out uint classIndexOut;
layout(location = 3) out uint instanceIndexOut;
layout(location = 4) out highp vec4 normalOut;

void main()
{
    objectCoordinatesOut = objectCoordinates;
    classIndexOut = classIndex;
    instanceIndexOut = instanceIndex;

    lowp const vec4 finalAmbientColor =
        #ifdef AMBIENT_TEXTURE
        texture(ambientTexture, interpolatedTextureCoords)*
        #endif
        ambientColor;
    lowp const vec4 finalDiffuseColor =
        #ifdef DIFFUSE_TEXTURE
        texture(diffuseTexture, interpolatedTextureCoords)*
        #endif
        #ifdef VERTEX_COLORS
        interpolatedVertexColors;
        #else
        diffuseColor;
        #endif
    lowp const vec4 finalSpecularColor =
        #ifdef SPECULAR_TEXTURE
        texture(specularTexture, interpolatedTextureCoords)*
        #endif
        specularColor;

    /* Ambient color */
    color = finalAmbientColor;

    mediump vec3 normalizedTransformedNormal = -normalize(transformedNormal);
    if(!gl_FrontFacing)
        normalizedTransformedNormal = -normalizedTransformedNormal;

    /* Output the normal and dot product with camera ray */
    normalOut.xyz = normalizedTransformedNormal;
    normalOut.w = dot(normalizedTransformedNormal, cameraDirection);

    highp vec3 normalizedLightDirection = normalize(lightDirection);

    /* Add diffuse color */
    lowp float intensity = max(0.0, dot(normalizedTransformedNormal, normalizedLightDirection));
    color += finalDiffuseColor*lightColor*intensity;

    /* Add specular color, if needed */
    if(intensity > 0.001) {
        highp vec3 reflection = reflect(-normalizedLightDirection, normalizedTransformedNormal);
        mediump float specularity = pow(max(0.0, dot(normalize(cameraDirection), reflection)), shininess);
        color += finalSpecularColor*specularity;
    }

    color.a = 1.0;

    #ifdef ALPHA_MASK
    if(color.a < alphaMask) discard;
    #endif
}
