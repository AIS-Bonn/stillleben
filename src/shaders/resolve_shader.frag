
layout(binding = 0) uniform highp sampler2DMS rgb;

layout(binding = 1) uniform highp sampler2DMS objectCoordinates;

layout(binding = 2) uniform highp usampler2DMS classIndex;

layout(binding = 3) uniform highp usampler2DMS instanceIndex;

// Inputs
in highp vec2 textureCoords;

// Outputs
layout(location = 0) out highp vec4 fragmentColor;

layout(location = 1) out highp vec3 objectCoordinatesOut;
layout(location = 2) out highp uint classIndexOut;
layout(location = 3) out highp uint instanceIndexOut;
layout(location = 4) out highp uint validMaskOut;

highp vec4 multisampleAverage(sampler2DMS sampler, ivec2 coord)
{
    vec4 color = vec4(0.0);

    for (int i = 0; i < MSAA_SAMPLES; i++)
        color += texelFetch(sampler, coord, i);

    color /= float(MSAA_SAMPLES);

    return color;
}

highp vec3 multisampleVec3First(sampler2DMS sampler, ivec2 coord)
{
    return texelFetch(sampler, coord, 0);
}

void main()
{
    ivec2 texSize = textureSize(rgb);
    ivec2 texCoord = ivec2(textureCoords * texSize);

    fragmentColor = multisampleAverage(rgb, texCoord);
//     fragmentColor = vec4(textureCoords.xy, 0.0, 1.0);

    objectCoordinatesOut = texelFetch(objectCoordinates, texCoord, 0).rgb;
    classIndexOut = texelFetch(classIndex, texCoord, 0).r;

    instanceIndexOut = texelFetch(instanceIndex, texCoord, 0).r;
//     validMaskOut = 255u;
//     for(int i = 1; i < MSAA_SAMPLES; ++i)
//     {
//         if(texelFetch(instanceIndex, texCoord, i).r != instanceIndexOut)
            validMaskOut = 0u;
//     }
}
