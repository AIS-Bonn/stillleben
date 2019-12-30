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

// Material parameters
layout(location = 14)
uniform float metallic = 0.04;

layout(location = 15)
uniform float roughness = 0.5;


layout(binding = 3)
uniform highp samplerCube lightMapIrradiance;

layout(binding = 4)
uniform highp samplerCube lightMapPrefilter;

layout(binding = 5)
uniform highp sampler2D lightMapBRDFLUT;

// Sticker simulator
layout(binding = 6)
uniform highp sampler2DRect stickerTexture;

layout(binding = 7)
uniform highp sampler2DRect depthTexture;


in DataBridge fragmentData;

flat in centroid uvec3 g_vertexIndices;
in centroid vec3 g_barycentricCoeffs;


layout(location = 0) out lowp vec4 color;
layout(location = 1) out highp vec4 objectCoordinatesOut;
layout(location = 2) out uint classIndexOut;
layout(location = 3) out uint instanceIndexOut;
layout(location = 4) out highp vec4 normalOut;
layout(location = 5) out uvec3 vertexIndices;
layout(location = 6) out vec3 barycentricCoeffs;
layout(location = 7) out highp vec3 camCoordinatesOut;

vec2 dirToSpherical(vec3 dir)
{
    // We are in the camera coordinate system, so Y is down, X is right, Z is
    // into the image.

    return vec2(
        atan(-dir.x, dir.z) + M_PI,
        acos(dir.y)
    );
}

vec3 toGamma(vec3 v, float gamma)
{
    return pow(v, vec3(1.0 / gamma));
}

vec4 toGamma(vec4 v, float gamma)
{
    return vec4(toGamma(v.rgb, gamma), v.a);
}

vec3 toLinear(vec3 v, float gamma)
{
    return pow(v, vec3(gamma));
}

vec4 toLinear(vec4 v, float gamma)
{
    return vec4(toLinear(v.rgb, gamma), v.a);
}

const vec3 plastic_F0 = vec3(0.04);

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 cubemapCoord(vec3 camSpaceCoord)
{
    // camera coordinates are in y-down frame.
    // FIXME: We should use world coordinates here!
    return vec3(camSpaceCoord.z, -camSpaceCoord.x, -camSpaceCoord.y);
}

void main()
{
    // Depth peeling operation
    // discard fragments that closer to the camera than the current depth.
    if(fragmentData.objectCoordinates.w - 0.00001 <= texelFetch(depthTexture, ivec2(gl_FragCoord.xy)).w)
        discard;

    objectCoordinatesOut = fragmentData.objectCoordinates;
    camCoordinatesOut = fragmentData.camCoordinates;
    classIndexOut = classIndex;
    instanceIndexOut = instanceIndex;

    lowp vec4 finalAmbientColor =
        #ifdef AMBIENT_TEXTURE
        texture(ambientTexture, fragmentData.interpolatedTextureCoords)*
        #elif defined(DIFFUSE_TEXTURE)
        texture(diffuseTexture, fragmentData.interpolatedTextureCoords)*
        #endif
        ambientColor;
    lowp vec4 finalDiffuseColor =
        #ifdef DIFFUSE_TEXTURE
        texture(diffuseTexture, fragmentData.interpolatedTextureCoords)*
        #endif
        #ifdef VERTEX_COLORS
        fragmentData.interpolatedVertexColors;
        #else
        diffuseColor;
        #endif
    lowp vec4 finalSpecularColor =
        #ifdef SPECULAR_TEXTURE
        texture(specularTexture, fragmentData.interpolatedTextureCoords)*
        #endif
        specularColor;

    /* Are we inside the simulated sticker? */
    if(fragmentData.stickerCoordinates.x >= 0 &&
       fragmentData.stickerCoordinates.y >= 0 &&
       fragmentData.stickerCoordinates.x < 1 &&
       fragmentData.stickerCoordinates.y < 1)
    {
        lowp vec4 stickerColor = texture(stickerTexture, fragmentData.stickerCoordinates * textureSize(stickerTexture));
        finalDiffuseColor = mix(finalDiffuseColor, stickerColor, stickerColor.a);
    }

    mediump vec3 N = normalize(fragmentData.transformedNormal);
    /* Output the normal and dot product with camera ray */
    normalOut.xyz = N;
    normalOut.w = dot(N, fragmentData.cameraDirection);

    #ifdef FLAT
    color = finalDiffuseColor;
    #else
    // Start with ambient color
    color = finalAmbientColor;

    if(!gl_FrontFacing)
        N = -N;

    if(useLightMap)
    {
        float gamma = 1.8;

        vec3 V = normalize(fragmentData.cameraDirection); // cameraDirection: camera - object
        vec3 R = reflect(-V, N);

        color = toLinear(color, gamma);

        // HACK: the diffuse textures we load are not designed for PBR
        // and usually result in very dark results.
        vec3 albedo = 5.0 * toLinear(finalDiffuseColor.rgb, gamma);

        // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
        // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
        vec3 F0 = vec3(0.04);
        F0 = mix(F0, albedo, metallic);

        // ambient lighting (we now use IBL as the ambient term)
        vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

        vec3 kS = F;
        vec3 kD = 1.0 - kS;
        kD *= 1.0 - metallic;

        vec3 irradiance = texture(lightMapIrradiance, cubemapCoord(N)).rgb;
        vec3 diffuse      = irradiance * albedo;

        // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
        const float MAX_REFLECTION_LOD = 4.0;
        vec3 prefilteredColor = textureLod(lightMapPrefilter, cubemapCoord(R), roughness * MAX_REFLECTION_LOD).rgb;
        vec2 brdf  = texture(lightMapBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
        vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

        vec3 ambient = (kD * diffuse + specular); // FIXME: * ao

        color += vec4(ambient, 1.0);

        // HDR tonemapping (Reinhard operator)
        color = color / (color + vec4(1.0));

        color = toGamma(color, gamma);
    }
    else
    {
        highp vec3 normalizedLightDirection = normalize(fragmentData.lightDirection);

        /* Add diffuse color */
        lowp float intensity = max(0.0, dot(N, normalizedLightDirection));
        color += finalDiffuseColor*lightColor*intensity;

        /* Add specular color, if needed */
        if(intensity > 0.001) {
            highp vec3 reflection = reflect(-normalizedLightDirection, N);
            mediump float specularity = pow(max(0.0, dot(normalize(fragmentData.cameraDirection), reflection)), shininess);
            color += finalSpecularColor*specularity;
        }
    }
    #endif

    color.a = 1.0;

    #ifdef ALPHA_MASK
    if(color.a < alphaMask) discard;
    #endif


    vertexIndices = g_vertexIndices;
    barycentricCoeffs = g_barycentricCoeffs;
}
