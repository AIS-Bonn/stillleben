/* Source: Joey De Vries' http://learnopengl.com
 * (licensed under CC BY-NC 4.0) */

layout(location = 0)
out vec4 FragColor;

layout(location = 0)
in vec3 WorldPos;

layout(binding = 0)
uniform sampler2D equirectangularMap;

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.y, v.x), asin(v.z));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{
    vec2 uv = SampleSphericalMap(normalize(WorldPos));
    vec3 color = texture(equirectangularMap, uv).rgb;

    FragColor = vec4(color, 1.0);
}
