// Simple as possible for 2D blitting

layout(location = POSITION_ATTRIBUTE_LOCATION)
in highp vec2 position;

out highp vec2 textureCoords;

void main() {
    // We draw directly in normalized device coordinates (-1 to 1)
    gl_Position.xywz = vec4(position, 1.0, 0.0);

    textureCoords = position / 2.0 + vec2(0.5, 0.5);
}
