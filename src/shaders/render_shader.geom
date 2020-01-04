layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;


in DataBridge primitiveData[];
flat in uint gsVertexIndex[];
in highp vec4 vsPosition[];

out DataBridge fragmentData;
flat out centroid uvec3 g_vertexIndices;
out centroid vec3 g_barycentricCoeffs;

void main()
{
    // vertexIndices is same for all vertices to avoid interpolation
    g_vertexIndices = uvec3(gsVertexIndex[0], gsVertexIndex[1], gsVertexIndex[2]);

    // simply pass vsFragmentOut & vsPosition through
    fragmentData = primitiveData[0];
    gl_Position = vsPosition[0];
    g_barycentricCoeffs = vec3(1.0, 0.0, 0.0);
    EmitVertex();

    fragmentData = primitiveData[1];
    gl_Position = vsPosition[1];
    g_barycentricCoeffs = vec3(0.0, 1.0, 0.0);
    EmitVertex();

    fragmentData = primitiveData[2];
    gl_Position = vsPosition[2];
    g_barycentricCoeffs = vec3(0.0, 0.0, 1.0);
    EmitVertex();

    EndPrimitive();
}
