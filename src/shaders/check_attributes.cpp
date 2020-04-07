// Enforce GLSL <-> C++ consistency of attribute locations
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "common.glsl"
#include "render_shader.h"

namespace sl
{

// Inputs
static_assert(RenderShader::Position::Location == POSITION_ATTRIBUTE_LOCATION);
static_assert(RenderShader::TextureCoordinates::Location == TEXTURECOORDINATES_ATTRIBUTE_LOCATION);
static_assert(RenderShader::VertexColors::Location == COLOR_ATTRIBUTE_LOCATION);
static_assert(RenderShader::Tangent::Location == TANGENT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::VertexIndex::Location == VERTEX_INDEX_ATTRIBUTE_LOCATION);
static_assert(RenderShader::Normal::Location == NORMAL_ATTRIBUTE_LOCATION);

// Outputs
static_assert(RenderShader::ColorOutput == COLOR_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::ObjectCoordinatesOutput == OBJECT_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::ClassIndexOutput == CLASS_INDEX_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::InstanceIndexOutput == INSTANCE_INDEX_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::NormalOutput == NORMAL_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::VertexIndexOutput == VERTEX_INDEX_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::BarycentricCoeffsOutput == BARYCENTRIC_COEFFS_OUTPUT_ATTRIBUTE_LOCATION);
static_assert(RenderShader::CamCoordinatesOutput == CAM_COORDINATES_OUTPUT_ATTRIBUTE_LOCATION);

}
