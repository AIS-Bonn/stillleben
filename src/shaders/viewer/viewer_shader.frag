
layout(binding = 0) uniform highp sampler2DRect rgb;
layout(binding = 1) uniform highp sampler2DRect objectCoordinates;
layout(binding = 2) uniform highp sampler2DRect normal;
layout(binding = 3) uniform highp usampler2DRect instanceIndex;
layout(binding = 4) uniform highp usampler2DRect classIndex;

layout(location = 0) uniform vec3 bbox[MAX_CLASS+1];

// Inputs
in highp vec2 textureCoords;

// Outputs
layout(location = 0) out highp vec4 rgbOut;
layout(location = 1) out highp vec4 normalOut;
layout(location = 2) out highp vec4 segmentationOut;
layout(location = 3) out highp vec4 objectCoordinatesOut;

void main()
{
    ivec2 texSize = textureSize(normal);
    ivec2 texCoord = ivec2(textureCoords * texSize);

    vec4 rgbVec = texture(rgb, texCoord);
    vec4 objectCoordinatesVec = texture(objectCoordinates, texCoord);
    uint instanceIndexVal = texture(instanceIndex, texCoord).x;
    uint classIndexVal = texture(classIndex, texCoord).x;
    vec4 normalVec = texture(normal, texCoord);

    vec3 objBBox = bbox[classIndexVal];

    rgbOut = rgbVec;

    objectCoordinatesOut.rgb = clamp(objectCoordinatesVec.rgb / objBBox + 0.5, 0, 1);
    objectCoordinatesOut.a = 1.0;

    segmentationOut.rgb = vec3(clamp(float(instanceIndexVal) / 10.0, 0, 1));
    segmentationOut.a = 1.0;

    normalOut.rgb = normalVec.xyz / 2.0 + 0.5;
    normalOut.a = 1.0;
}
