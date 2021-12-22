// Fragment shader

// Texture samplers
//@{
layout(binding = BASE_COLOR_TEXTURE)
uniform lowp sampler2D baseColorTexture;

layout(binding = NORMAL_TEXTURE)
uniform lowp sampler2D normalTexture;

layout(binding = METALLIC_ROUGHNESS_TEXTURE)
uniform lowp sampler2D metallicRoughnessTexture;

layout(binding = EMISSIVE_TEXTURE)
uniform lowp sampler2D emissiveTexture;

layout(binding = OCCLUSION_TEXTURE)
uniform lowp sampler2D occlusionTexture;

layout(binding = LIGHTMAP_IRRADIANCE_TEXTURE)
uniform highp samplerCube lightMapIrradiance;

layout(binding = LIGHTMAP_PREFILTER_TEXTURE)
uniform highp samplerCube lightMapPrefilter;

layout(binding = LIGHTMAP_BRDF_LUT_TEXTURE)
uniform highp sampler2D lightMapBRDFLUT;

layout(binding = STICKER_TEXTURE)
uniform highp sampler2DRect stickerTexture;

layout(binding = DEPTH_TEXTURE)
uniform highp sampler2DRect depthTexture;
//@}

// Uniform parameters
//@{
layout(location = UNIFORM_MATERIAL)
uniform vec4 materialParameters[3];

#define baseColorFactor materialParameters[0]
#define emissiveFactor materialParameters[1]
#define alphaCutoff materialParameters[2].x
#define metallicFactor materialParameters[2].y
#define roughnessFactor materialParameters[2].z

layout(location = UNIFORM_AVAILABLE_TEXTURES)
uniform uint availableTextures = 0u;

// Segmentation information
layout(location = UNIFORM_CLASS_INDEX)
uniform uint classIndex = 0u;

layout(location = UNIFORM_INSTANCE_INDEX)
uniform uint instanceIndex = 0u;

layout(location = UNIFORM_CAM_POSITION)
uniform vec3 camPosition;
//@}

// Inputs from vertex or geometry shader
in DataBridge fragmentData;

flat in centroid uvec3 g_vertexIndices;
in centroid vec3 g_barycentricCoeffs;

// Outputs!
//@{
layout(location = COLOR_OUTPUT_ATTRIBUTE_LOCATION)
out lowp vec4 color;

layout(location = OBJECT_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION)
out highp vec4 objectCoordinatesOut;

layout(location = CLASS_INDEX_OUTPUT_ATTRIBUTE_LOCATION)
out uint classIndexOut;

layout(location = INSTANCE_INDEX_OUTPUT_ATTRIBUTE_LOCATION)
out uint instanceIndexOut;

layout(location = NORMAL_OUTPUT_ATTRIBUTE_LOCATION)
out highp vec4 normalOut;

layout(location = VERTEX_INDEX_OUTPUT_ATTRIBUTE_LOCATION)
out uvec3 vertexIndices;

layout(location = BARYCENTRIC_COEFFS_OUTPUT_ATTRIBUTE_LOCATION)
out vec3 barycentricCoeffs;

layout(location = CAM_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION)
out highp vec4 camCoordinatesOut;
//@}


#define DIELECTRIC_SPECULAR 0.04
#define MIN_ROUGHNESS 0.045

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

vec3 Tonemap_ACES(vec3 x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + vec3(b))) / (x * (c * x + vec3(d)) + vec3(e));
}

vec3 diffuseColor(vec3 baseColor, float metallic)
{
    return baseColor * (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metallic);
}

bool haveTexture(uint tex_code)
{
    return (availableTextures & (1 << tex_code)) != 0;
}

float clampDot(vec3 v1, vec3 v2)
{
    return clamp(dot(v1, v2), 0.0, 1.0);
}

void main()
{
    // Depth peeling operation
    // discard fragments that closer to the camera than the current depth.
    if(fragmentData.objectCoordinates.w - 0.00001 <= texelFetch(depthTexture, ivec2(gl_FragCoord.xy)).w)
    {
        discard;
        return;
    }

    float gamma = 2.2;

    vec4 baseColor = baseColorFactor;

    if(haveTexture(BASE_COLOR_TEXTURE))
        baseColor *= toLinear(texture2D(baseColorTexture, fragmentData.interpolatedTextureCoords), gamma);

    if(baseColor.w < alphaCutoff)
    {
        discard;
        return;
    }

    // Are we inside the simulated sticker?
    if(fragmentData.stickerCoordinates.x >= 0 &&
       fragmentData.stickerCoordinates.y >= 0 &&
       fragmentData.stickerCoordinates.x < 1 &&
       fragmentData.stickerCoordinates.y < 1)
    {
        lowp vec4 stickerColor = texture(stickerTexture, fragmentData.stickerCoordinates * textureSize(stickerTexture));
        baseColor = mix(baseColor, stickerColor, stickerColor.a);
    }

    // Compute normal
    vec3 normal;

    if(haveTexture(NORMAL_TEXTURE))
    {
        normal = texture2D(normalTexture, fragmentData.interpolatedTextureCoords).xyz * 2.0 - 1.0;
        normal = normalize(normal.x * fragmentData.tangentInWorld + normal.y * fragmentData.bitangentInWorld + normal.z * fragmentData.normalInWorld);
    }
    else
        normal = fragmentData.normalInWorld;

    if(!gl_FrontFacing)
        normal = -normal;

    highp vec3 cameraDirection = normalize(camPosition - fragmentData.worldCoordinates);
    vec3 lightDir = reflect(-cameraDirection, normal);
    vec3 H = normalize(lightDir + cameraDirection);
    float VoH = clampDot(cameraDirection, H);
    float NoV = clamp(dot(normal, cameraDirection), 1e-5, 1.0);


    // Retrieve material parameters
    float roughness = roughnessFactor;
    float metallic = metallicFactor;

    if(haveTexture(METALLIC_ROUGHNESS_TEXTURE))
    {
        vec2 roughnessMetal = texture2D(metallicRoughnessTexture, fragmentData.interpolatedTextureCoords).yz;
        roughness *= roughnessMetal.x;
        metallic *= roughnessMetal.y;
    }

    roughness = max(roughness, MIN_ROUGHNESS);

    float occlusion = 1.0f;
    if(haveTexture(OCCLUSION_TEXTURE))
        occlusion = texture2D(occlusionTexture, fragmentData.interpolatedTextureCoords).x;

    vec3 emissive = vec3(emissiveFactor);
    if(haveTexture(EMISSIVE_TEXTURE))
        emissive *= toLinear(texture2D(emissiveTexture, fragmentData.interpolatedTextureCoords), gamma).xyz;

    // Load env texture data
    vec2 f_ab = texture2D(lightMapBRDFLUT, vec2(NoV, roughness)).xy;
    float lodLevel = roughness * 4.0;
    vec3 radiance = textureLod(lightMapPrefilter, lightDir, lodLevel).xyz;
    vec3 irradiance = textureLod(lightMapIrradiance, normal, 0).xyz;

    // From GLTF spec
    vec3 c_diff = diffuseColor(baseColor.rgb, metallic);
    vec3 F0 = mix(vec3(DIELECTRIC_SPECULAR), baseColor.xyz, metallic);

    // Roughness dependent fresnel, from Fdez-Aguera
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    vec3 k_S = F0 + Fr * pow(1.0 - NoV, 5.0);

    vec3 FssEss = k_S * f_ab.x + f_ab.y;

    // Multiple scattering, from Fdez-Aguera
    float Ems = (1.0 - (f_ab.x + f_ab.y));
    vec3 F_avg = F0 + (1.0 - F0) / 21.0;
    vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    vec3 k_D = c_diff * (1.0 - FssEss - FmsEms);
    vec3 selfColor = FssEss * radiance + (FmsEms + k_D) * irradiance;

    // RGB output (linear space/HDR!)
    color = vec4(selfColor * occlusion + emissive, baseColor.w);

    // Simple semantic information outputs
    objectCoordinatesOut = fragmentData.objectCoordinates;
    camCoordinatesOut = vec4(fragmentData.camCoordinates, 1.0);
    classIndexOut = classIndex;
    instanceIndexOut = instanceIndex;

    // Output the normal and dot product with camera ray
    normalOut.xyz = normalize(fragmentData.normalInCam);
    normalOut.w = dot(normal, cameraDirection);

    // Outputs for the differentiable renderer
    vertexIndices = g_vertexIndices;
    barycentricCoeffs = g_barycentricCoeffs;
}
