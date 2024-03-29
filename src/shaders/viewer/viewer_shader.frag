
layout(binding = 0) uniform highp sampler2DRect rgb;
layout(binding = 1) uniform highp sampler2DRect objectCoordinates;
layout(binding = 2) uniform highp sampler2DRect normal;
layout(binding = 3) uniform highp usampler2DRect instanceIndex;
layout(binding = 4) uniform highp usampler2DRect classIndex;

layout(location = 0) uniform vec3 bbox[MAX_INSTANCE+1];
layout(location = MAX_INSTANCE+1) uniform vec4 instanceColors[MAX_INSTANCE+1];
layout(location = MAX_INSTANCE+1+MAX_INSTANCE+1) uniform vec4 classColors[MAX_CLASS+1];

// Inputs
in highp vec2 textureCoords;

// Outputs
layout(location = 0) out highp vec4 rgbOut;
layout(location = 1) out highp vec4 normalOut;
layout(location = 2) out highp vec4 instanceIndexOut;
layout(location = 3) out highp vec4 classIndexOut;
layout(location = 4) out highp vec4 objectCoordinatesOut;

void main()
{
    ivec2 texSize = textureSize(normal);
    ivec2 texCoord = ivec2(textureCoords * texSize);

    vec4 rgbVec = texelFetch(rgb, texCoord);
    vec4 objectCoordinatesVec = texelFetch(objectCoordinates, texCoord);
    uint instanceIndexVal = texelFetch(instanceIndex, texCoord).x;
    uint classIndexVal = texelFetch(classIndex, texCoord).x;
    vec4 normalVec = texelFetch(normal, texCoord);

    vec3 objBBox = bbox[instanceIndexVal];

    rgbOut = vec4(rgbVec.rgb, 1.0);

    // no object -> bg color
    if(objectCoordinatesVec.w > 2000.0)
    {
        objectCoordinatesOut = instanceColors[0];
        normalOut = instanceColors[0];
    }
    else
    {
        objectCoordinatesOut.rgb = clamp(objectCoordinatesVec.xyz / objBBox + 0.5, 0, 1);
        objectCoordinatesOut.a = 1.0;

        normalOut.rgb = normalVec.xyz / 2.0 + 0.5;
        normalOut.a = 1.0;
    }

    instanceIndexOut = instanceColors[instanceIndexVal];
    classIndexOut = classColors[classIndexVal];
}
