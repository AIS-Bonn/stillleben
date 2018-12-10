// Most of the code taken from Magnum engine

#ifndef RUNTIME_CONST
#define const
#endif

#ifdef AMBIENT_TEXTURE
#ifdef EXPLICIT_TEXTURE_LAYER
layout(binding = 0)
#endif
uniform lowp sampler2D ambientTexture;
#endif

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 4)
#endif
uniform lowp vec4 ambientColor
    #ifndef AMBIENT_TEXTURE
    = vec4(0.0)
    #else
    = vec4(1.0)
    #endif
    ;

#ifdef DIFFUSE_TEXTURE
#ifdef EXPLICIT_TEXTURE_LAYER
layout(binding = 1)
#endif
uniform lowp sampler2D diffuseTexture;
#endif

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 5)
#endif
uniform lowp vec4 diffuseColor = vec4(1.0);

#ifdef SPECULAR_TEXTURE
#ifdef EXPLICIT_TEXTURE_LAYER
layout(binding = 2)
#endif
uniform lowp sampler2D specularTexture;
#endif

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 6)
#endif
uniform lowp vec4 specularColor = vec4(1.0);

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 7)
#endif
uniform lowp vec4 lightColor = vec4(1.0);

#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 8)
#endif
uniform mediump float shininess = 80.0;

#ifdef ALPHA_MASK
#ifdef EXPLICIT_UNIFORM_LOCATION
layout(location = 9)
#endif
uniform lowp float alphaMask = 0.5;
#endif

in mediump vec3 transformedNormal;
in highp vec3 lightDirection;
in highp vec3 cameraDirection;

#if defined(AMBIENT_TEXTURE) || defined(DIFFUSE_TEXTURE) || defined(SPECULAR_TEXTURE)
in mediump vec2 interpolatedTextureCoords;
#endif

in highp vec3 worldPosition;

out lowp vec4 color;
out highp vec3 worldPositionOut;

void main()
{
    worldPositionOut = worldPosition;

    lowp const vec4 finalAmbientColor =
        #ifdef AMBIENT_TEXTURE
        texture(ambientTexture, interpolatedTextureCoords)*
        #endif
        ambientColor;
    lowp const vec4 finalDiffuseColor =
        #ifdef DIFFUSE_TEXTURE
        texture(diffuseTexture, interpolatedTextureCoords)*
        #endif
        diffuseColor;
    lowp const vec4 finalSpecularColor =
        #ifdef SPECULAR_TEXTURE
        texture(specularTexture, interpolatedTextureCoords)*
        #endif
        specularColor;

    /* Ambient color */
    color = finalAmbientColor;

    mediump vec3 normalizedTransformedNormal = normalize(transformedNormal);
    if(!gl_FrontFacing)
        normalizedTransformedNormal = -normalizedTransformedNormal;

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

    #ifdef ALPHA_MASK
    if(color.a < alphaMask) discard;
    #endif
}
