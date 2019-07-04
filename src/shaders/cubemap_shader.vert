
layout(location = 0) in vec3 aPos;

layout(location = 0) out vec3 WorldPos;

layout(location = 0)
uniform mat4 projection;

layout(location = 1)
uniform mat4 view;

void main()
{
    WorldPos = aPos;
    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}
