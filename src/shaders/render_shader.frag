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

layout(binding = SHADOW_MAP_TEXTURE)
uniform highp sampler2DArrayShadow shadowMap;
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

// Lighting
layout(location = UNIFORM_LIGHT_MAP_AVAILABLE)
uniform uint lightMapAvailable = 0u;

layout(location = UNIFORM_LIGHT_DIRECTIONS)
uniform vec3 lightDirections[NUM_LIGHTS];

layout(location = UNIFORM_LIGHT_COLORS)
uniform vec3 lightColors[NUM_LIGHTS];

layout(location = UNIFORM_SHADOW_MATRICES)
uniform mat4 shadowMatrices[NUM_LIGHTS];

layout(location = UNIFORM_AMBIENT_LIGHT)
uniform vec3 ambientLight = vec3(0.0);

// Segmentation information
layout(location = UNIFORM_CLASS_INDEX)
uniform uint classIndex = 0u;

layout(location = UNIFORM_INSTANCE_INDEX)
uniform uint instanceIndex = 0u;

layout(location = UNIFORM_CAM_POSITION)
uniform vec3 camPosition;

layout(location = UNIFORM_WORLD_TO_CAM)
uniform mediump mat4 worldToCam = mat4(1.0);
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
    return (availableTextures & (1u << tex_code)) != 0u;
}

float clampDot(vec3 v1, vec3 v2)
{
    return clamp(dot(v1, v2), 0.0, 1.0);
}


////////////////////////////////////////////////////////////////////////////////

//DFG equations from https://learnopengl.com/PBR/Lighting
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = M_PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

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

////////////////////////////////////////////////////////////////////////////////

void main()
{
    // Depth peeling operation
    // discard fragments which are closer to the camera than the current depth.
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
        lowp vec4 stickerColor = toLinear(texture(stickerTexture, fragmentData.stickerCoordinates * textureSize(stickerTexture)), gamma);
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

    color = vec4(0.0, 0.0, 0.0, baseColor.w);

    // From GLTF spec
    vec3 c_diff = diffuseColor(baseColor.rgb, metallic);
    vec3 F0 = mix(vec3(DIELECTRIC_SPECULAR), baseColor.xyz, metallic);

    // Roughness dependent fresnel, from Fdez-Aguera
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    vec3 k_S = F0 + Fr * pow(1.0 - NoV, 5.0);

    ivec3 shadowMapSize = textureSize(shadowMap, 0);
    vec2 shadowMapScale = 1.0 / shadowMapSize.xy;

    for(int i = 0; i < NUM_LIGHTS; ++i)
    {
        vec3 lightColor = lightColors[i];
        vec3 lightDirection = lightDirections[i];
        if(lightColor == vec3(0.0) || lightDirection == vec3(0.0))
            continue;

        // Shadow lookup
        vec4 projCoords = shadowMatrices[i] * vec4(fragmentData.worldCoordinates, 1.0);
        projCoords = projCoords / projCoords.w;

        // NDC to texture
        projCoords = 0.5 * projCoords + 0.5;

        float inverseShadow = 0.0;

        for(float y = -1.5; y <= 1.5; y += 1.0)
        {
            for(float x = -1.5; x <= 1.5; x += 1.0)
            {
                vec2 offset = vec2(x,y) * shadowMapScale;
                inverseShadow += texture(shadowMap, vec4(projCoords.xy + offset, i, projCoords.z-0.00003)).r;
            }
        }
        inverseShadow /= 16;

        // calculate per-light radiance
        vec3 L = normalize(-lightDirection);
        vec3 H = normalize(cameraDirection + L);
        float attenuation = 1.0;
        vec3 radiance = lightColor * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(normal, H, roughness);
        float G   = GeometrySmith(normal, cameraDirection, L, roughness);
        vec3 F    = k_S;

        vec3 nominator    = NDF * G * F;
        float denominator = 4 * NoV * max(dot(normal, L), 0.0);
        vec3 specular = nominator / max(denominator, 0.001); // prevent divide by zero for NdotV=0.0 or NdotL=0.0

        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;

        // scale light by NdotL
        float NdotL = max(dot(normal, L), 0.0);

        // add to outgoing radiance Lo
        color.rgb += inverseShadow * (kD * baseColor.rgb / M_PI + specular) * radiance * NdotL;
    }

    // Ambient light
    color.rgb += ambientLight * baseColor.rgb;

    if(lightMapAvailable != 0)
    {
        // Load env texture data
        vec2 f_ab = texture2D(lightMapBRDFLUT, vec2(NoV, roughness)).xy;
        float lodLevel = roughness * 4.0;
        vec3 radiance = textureLod(lightMapPrefilter, lightDir, lodLevel).xyz;
        vec3 irradiance = textureLod(lightMapIrradiance, normal, 0).xyz;

        vec3 FssEss = k_S * f_ab.x + f_ab.y;

        // Multiple scattering, from Fdez-Aguera
        float Ems = (1.0 - (f_ab.x + f_ab.y));
        vec3 F_avg = F0 + (1.0 - F0) / 21.0;
        vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
        vec3 k_D = c_diff * (1.0 - FssEss - FmsEms);
        vec3 selfColor = FssEss * radiance + (FmsEms + k_D) * irradiance;

        // RGB output (linear space/HDR!)
        color.rgb += selfColor * occlusion;
    }

    // Emissive
    color.rgb += emissive;

    // Simple semantic information outputs
    objectCoordinatesOut = fragmentData.objectCoordinates;
    camCoordinatesOut = vec4(fragmentData.camCoordinates, 1.0);
    classIndexOut = classIndex;
    instanceIndexOut = instanceIndex;

    // Output the normal and dot product with camera ray
    normalOut.xyz = normalize(mat3(worldToCam) * normal);
    normalOut.w = dot(normal, cameraDirection);

    // Outputs for the differentiable renderer
    vertexIndices = g_vertexIndices;
    barycentricCoeffs = g_barycentricCoeffs;
}
