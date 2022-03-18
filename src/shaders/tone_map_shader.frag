
// Uniforms
layout(location = UNIFORM_MANUAL_EXPOSURE)
uniform highp float manualExposure = -1.0;

// Texture samplers
layout(binding = COLOR_TEXTURE)
uniform highp sampler2D rgbSampler;

layout(binding = OBJECT_LUMINANCE_TEXTURE)
uniform highp sampler2D luminanceSampler;

// Outputs
layout(location = 0)
out highp vec4 outputColor;

// Taken from RTR vol 4 pg. 278
#define RGB_TO_LUM vec3(0.2125, 0.7154, 0.0721)

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

vec3 convertRGB2XYZ(vec3 _rgb)
{
    // Reference:
    // RGB/XYZ Matrices
    // http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    vec3 xyz;
    xyz.x = dot(vec3(0.4124564, 0.3575761, 0.1804375), _rgb);
    xyz.y = dot(vec3(0.2126729, 0.7151522, 0.0721750), _rgb);
    xyz.z = dot(vec3(0.0193339, 0.1191920, 0.9503041), _rgb);
    return xyz;
}

vec3 convertXYZ2RGB(vec3 _xyz)
{
    vec3 rgb;
    rgb.x = dot(vec3( 3.2404542, -1.5371385, -0.4985314), _xyz);
    rgb.y = dot(vec3(-0.9692660,  1.8760108,  0.0415560), _xyz);
    rgb.z = dot(vec3( 0.0556434, -0.2040259,  1.0572252), _xyz);
    return rgb;
}

vec3 convertXYZ2Yxy(vec3 _xyz)
{
    // Reference:
    // http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
    float inv = 1.0/dot(_xyz, vec3(1.0, 1.0, 1.0) );
    return vec3(_xyz.y, _xyz.x*inv, _xyz.y*inv);
}

vec3 convertYxy2XYZ(vec3 _Yxy)
{
    // Reference:
    // http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
    vec3 xyz;
    xyz.x = _Yxy.x*_Yxy.y/_Yxy.z;
    xyz.y = _Yxy.x;
    xyz.z = _Yxy.x*(1.0 - _Yxy.y - _Yxy.z)/_Yxy.z;
    return xyz;
}

vec3 convertRGB2Yxy(vec3 _rgb)
{
    return convertXYZ2Yxy(convertRGB2XYZ(_rgb) );
}

vec3 convertYxy2RGB(vec3 _Yxy)
{
    return convertXYZ2RGB(convertYxy2XYZ(_Yxy) );
}

// By Krzysztof Narkowicz
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main()
{
    vec2 texCoord = gl_FragCoord.xy;

    vec4 color = texelFetch(rgbSampler, ivec2(texCoord), 0);

    vec3 Yxy = convertRGB2Yxy(color.rgb);

    if(manualExposure >= 0)
        Yxy.x *= manualExposure;
    else
    {
        vec4 avgVec = texelFetch(luminanceSampler, ivec2(0,0), textureQueryLevels(luminanceSampler)-1);

        // The average is taken over all scene pixels, but only those on objects
        // contribute to alpha. So divide by alpha to get the average over all
        // object pixels.
        // FIXME: I think the average luminance calculation is flawed. Fix it and
        //  get rid of the 0.1 twiddle factor.
        float lum = 0.1 * dot(RGB_TO_LUM, avgVec.rgb / avgVec.a);
        Yxy.x /= (9.6 * lum + 0.0001);
    }

    color.rgb = convertYxy2RGB(Yxy);

    color.rgb = ACESFilm(color.rgb);

    outputColor = toGamma(color, 2.2);
    outputColor = color;
}
