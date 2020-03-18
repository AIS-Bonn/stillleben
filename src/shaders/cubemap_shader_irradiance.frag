/* Source: Joey De Vries' http://learnopengl.com
 * (licensed under CC BY-NC 4.0) */

layout(location = 0)
out vec4 FragColor;

layout(location = 0)
in vec3 WorldPos;

layout(binding = 0)
uniform samplerCube environmentMap;

const float PI = 3.14159265359;

void main()
{
    vec3 N = normalize(WorldPos);

    vec3 irradiance = vec3(0.0);

    // tangent space calculation from origin point
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(up, N);
    up            = cross(N, right);

    float sampleDelta = 0.020;
    float nrSamples = 0.0f;
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N;

            irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(nrSamples));

    FragColor = vec4(irradiance, 1.0);
}
