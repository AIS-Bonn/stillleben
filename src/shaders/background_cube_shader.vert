
layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec3 position;

layout(location = 0)
uniform mat4 view;

layout(location = 1)
uniform mat4 projection;

layout(location = 0)
out vec3 worldPosition;

void main()
{
    worldPosition = position;

    mat4 rotView = mat4(mat3(view));
    vec4 clipPos = projection * rotView * vec4(position, 1.0);

    gl_Position = clipPos.xyww;
}
