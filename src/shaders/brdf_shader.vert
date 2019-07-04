/* Source: Joey De Vries' http://learnopengl.com
 * (licensed under CC BY-NC 4.0) */

layout (location = POSITION_ATTRIBUTE_LOCATION)
in vec3 aPos;

layout (location = TEXTURECOORDINATES_ATTRIBUTE_LOCATION)
in vec2 aTexCoords;

layout(location = 0)
out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 1.0);
}
