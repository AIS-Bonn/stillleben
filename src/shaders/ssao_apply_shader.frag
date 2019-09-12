
layout(binding = 0) uniform highp sampler2DRect rgbSampler;
layout(binding = 1) uniform highp sampler2DRect aoSampler;
layout(binding = 2) uniform highp sampler2DRect coordinateSampler;

// Outputs
layout(location = 0) out highp vec4 outputColor;

vec3 toLinear(vec3 v, float gamma)
{
    return pow(v, vec3(gamma));
}

vec4 toLinear(vec4 v, float gamma)
{
    return vec4(toLinear(v.rgb, gamma), v.a);
}

vec3 toGamma(vec3 v, float gamma)
{
    return pow(v, vec3(1.0 / gamma));
}

vec4 toGamma(vec4 v, float gamma)
{
    return vec4(toGamma(v.rgb, gamma), v.a);
}

const float KERNEL_RADIUS = 3.0;
const float g_Sharpness = 300.0;

float bilateralBlur(vec2 uv, float r, float center_c, float center_d, inout float w_total)
{
    float c = texture(aoSampler, uv).r;
    float d = texture(coordinateSampler, uv).z;

    const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    return c*w;
}

float smoothAO(vec2 texCoord)
{
    float center_c = texture(aoSampler, texCoord).r;
    float center_d = texture(coordinateSampler, texCoord).z;

    float result = 0.0;
    float w_total = 0.0;
    for (int x = -2; x < 2; ++x)
    {
        for (int y = -2; y < 2; ++y)
        {
            vec2 offset = vec2(float(x), float(y));
            result += bilateralBlur(texCoord + offset, sqrt(x*x+y*y), center_c, center_d, w_total);
        }
    }
    return result / w_total;
}

void main()
{
    ivec2 texSize = textureSize(rgbSampler);
    vec2 texCoord = gl_FragCoord.xy;

    vec4 rgb = texture(rgbSampler, texCoord);

    float ao = smoothAO(texCoord);

//     rgb = toLinear(rgb, 1.8);

    rgb.rgb *= vec3(ao);

//     rgb = toGamma(rgb, 1.8);

    outputColor = rgb;
}
