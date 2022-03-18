// Shader for creating shadow maps
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec3 position;

layout(location = UNIFORM_TRANSFORMATION)
uniform highp mat4 transformation;

void main()
{
    gl_Position = transformation * vec4(position, 1.0);
}
