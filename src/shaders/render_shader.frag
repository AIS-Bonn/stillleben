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

// Do we use a light map?
layout(location = 13)
uniform bool useLightMap = false;

layout(binding = 3)
uniform lowp sampler2D lightMapDiffuse;
layout(binding = 4)
uniform lowp sampler2D lightMapSpecular;

in mediump vec3 transformedNormal;
in highp vec3 lightDirection;
in highp vec3 cameraDirection;

#if defined(AMBIENT_TEXTURE) || defined(DIFFUSE_TEXTURE) || defined(SPECULAR_TEXTURE)
in mediump vec2 interpolatedTextureCoords;
#endif

#ifdef VERTEX_COLORS
in mediump vec4 interpolatedVertexColors;
#endif

centroid in highp vec4 objectCoordinates;

layout(location = 0) out lowp vec4 color;
layout(location = 1) out highp vec4 objectCoordinatesOut;
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
        #elif defined(DIFFUSE_TEXTURE)
        texture(diffuseTexture, interpolatedTextureCoords)*
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

    mediump vec3 normalizedTransformedNormal = normalize(transformedNormal);
    /* Output the normal and dot product with camera ray */
    normalOut.xyz = normalizedTransformedNormal;
    normalOut.w = dot(normalizedTransformedNormal, cameraDirection);

    #ifdef FLAT
    color = finalDiffuseColor;
    #else
    // Start with ambient color
    color = finalAmbientColor;

    if(useLightMap)
    {
        if(!gl_FrontFacing)
            normalizedTransformedNormal = -normalizedTransformedNormal;

        mediump vec3 reflected = normalize(reflect(-normalize(cameraDirection), normalizedTransformedNormal));

        // Convert to spherical coordinates
        mediump vec2 longlat_diffuse = vec2(atan(normalizedTransformedNormal.y, normalizedTransformedNormal.x), acos(normalizedTransformedNormal.z));
        mediump vec2 longlat_specular = vec2(atan(reflected.y, reflected.x), acos(reflected.z));

        // normalize
        longlat_diffuse = longlat_diffuse / vec2(2.0*M_PI, M_PI);
        longlat_specular = longlat_specular / vec2(2.0*M_PI, M_PI);

        // Lookup!
        lowp vec4 diffuse = 50.0 * texture2D(lightMapDiffuse, longlat_diffuse);
        lowp vec4 specular = texture2D(lightMapSpecular, longlat_specular);

        color += diffuse*finalDiffuseColor + specular*finalSpecularColor;
    }
    else
    {
        // NOTE: from here on, normalizedTransformedNormal points *into* the mesh
        if(gl_FrontFacing)
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
    }
    #endif

    color.a = 1.0;

    #ifdef ALPHA_MASK
    if(color.a < alphaMask) discard;
    #endif
}
