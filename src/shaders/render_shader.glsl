struct DataBridge
{
    #ifdef TEXTURED
    mediump vec2 interpolatedTextureCoords;
    #endif

    #ifdef VERTEX_COLORS
    mediump vec4 interpolatedVertexColors;
    #endif

    mediump vec3 transformedNormal;
    highp vec3 lightDirection;
    highp vec3 cameraDirection;

    centroid highp vec4 objectCoordinates;
    centroid highp vec3 camCoordinates;
    mediump vec2 stickerCoordinates;
};
