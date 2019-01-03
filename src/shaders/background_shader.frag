
layout(binding = 0) uniform highp sampler2DRect rgb;

// Inputs
in highp vec2 textureCoords;

// Outputs
layout(location = 0) out highp vec4 fragmentColor;

void main()
{
    ivec2 texSize = textureSize(rgb);
    ivec2 texCoord = ivec2(textureCoords * texSize);

    fragmentColor = texture(rgb, texCoord);
}
