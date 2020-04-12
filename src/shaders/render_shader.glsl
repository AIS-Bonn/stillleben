struct DataBridge
{
    #ifdef TEXTURED
    mediump vec2 interpolatedTextureCoords;
    #endif

    #ifdef VERTEX_COLORS
    mediump vec4 interpolatedVertexColors;
    #endif

    mediump vec3 normalInCam;
    mediump vec3 normalInWorld;

    highp vec4 objectCoordinates;
    highp vec3 worldCoordinates;
    highp vec3 camCoordinates;
    mediump vec2 stickerCoordinates;
};
