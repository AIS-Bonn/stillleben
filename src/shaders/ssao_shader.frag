
layout(binding = 0) uniform highp sampler2DRect positions;
layout(binding = 1) uniform highp sampler2DRect normals;
layout(binding = 2) uniform highp sampler2D noiseSampler;

layout(location = 0)
uniform vec3 samples[64];

layout(location = 65)
uniform mat4 projection;

// parameters (you'd probably want to use them as uniforms to more easily tweak the effect)
const int kernelSize = 64;
const float radius = 0.1;
const float bias = 0.0025;

// Outputs
layout(location = 0) out highp float ao;

void main()
{
    ivec2 texSize = textureSize(positions);
    vec2 texCoord = gl_FragCoord.xy;
//     texCoord.y = texSize.y - texCoord.y;

    // get input for SSAO algorithm
    vec3 fragPos = texture(positions, texCoord).xyz;
    vec3 normal = normalize(texture(normals, texCoord).rgb);
    vec3 randomVec = normalize(texture(noiseSampler, texCoord / 4.0).xyz);
    // create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    // iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * samples[i]; // from tangent to view-space
        samplePos = fragPos + samplePos * radius;

        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0

        // get sample depth
        float sampleDepth = texture(positions, offset.xy * texSize).z; // get depth value of kernel sample

        // range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth <= samplePos.z - bias ? 1.0 : 0.0) * rangeCheck;
    }

    ao = 1.0 - (occlusion / kernelSize);
}
