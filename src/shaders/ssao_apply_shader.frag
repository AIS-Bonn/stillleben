
layout(binding = 0) uniform highp sampler2D rgbSampler;
layout(binding = 1) uniform highp sampler2D aoSampler;
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

float bilateralBlur(ivec2 uv, float r, float center_c, float center_d, inout float w_total)
{
    float c = texelFetch(aoSampler, uv, 0).r;
    float d = texture(coordinateSampler, uv).z;

    const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    return c*w;
}

float smoothAO(ivec2 texCoord)
{
    float center_c = texelFetch(aoSampler, texCoord, 0).r;
    float center_d = texture(coordinateSampler, texCoord).z;

    float result = 0.0;
    float w_total = 0.0;
    for (int x = -2; x < 2; ++x)
    {
        for (int y = -2; y < 2; ++y)
        {
            ivec2 offset = ivec2(x, y);
            result += bilateralBlur(texCoord + offset, sqrt(x*x+y*y), center_c, center_d, w_total);
        }
    }
    return result / w_total;
}

void main()
{
    ivec2 texCoord = ivec2(gl_FragCoord.xy);

    vec4 rgb = texelFetch(rgbSampler, texCoord, 0);

    float ao = smoothAO(texCoord);

    rgb.rgb *= vec3(ao);

    outputColor = rgb;
}
