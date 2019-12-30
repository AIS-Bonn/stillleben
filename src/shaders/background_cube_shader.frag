
layout(location = 0)
in vec3 worldPosition;

layout(binding = 0)
uniform samplerCube cubeMap;

layout(location = 0)
out vec4 color;

void main()
{
    vec3 envColor = texture(cubeMap, worldPosition).rgb;

    // HDR tonemap and gamma correct
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2));

    color = vec4(envColor, 1.0);
}
