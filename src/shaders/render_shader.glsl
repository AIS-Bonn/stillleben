struct DataBridge
{
    mediump vec2 interpolatedTextureCoords;

    mediump vec4 interpolatedVertexColors;

    mediump vec3 normalInCam;
    mediump vec3 normalInWorld;
    mediump vec3 tangentInWorld;
    mediump vec3 bitangentInWorld;

    highp vec4 objectCoordinates;
    highp vec3 worldCoordinates;
    highp vec3 camCoordinates;
    mediump vec2 stickerCoordinates;
};
