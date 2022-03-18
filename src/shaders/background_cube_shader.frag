
layout(location = 0)
in vec3 worldPosition;

layout(binding = 0)
uniform samplerCube cubeMap;

layout(location = 0)
out vec4 color;

void main()
{
    vec3 envColor = texture(cubeMap, worldPosition).rgb;

    color = vec4(envColor, 0.0);
}
