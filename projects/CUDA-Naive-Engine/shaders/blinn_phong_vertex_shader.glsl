#version 330 core
#pragma optionNV(unroll all)

#define LIGHT_COUNT 5

#define SPOT_LIGHT 1
#define POSITIONAL_LIGHT 2
#define DIRECTIONAL_LIGHT 3

#define SPOTLIGHT_OUTER_ANGLE 0.97
 
/* Input Attributes (vertex properties shared between shaders) */
in vec4 VertexPosition;

in vec4 VertexNormal;

in vec4 VertexAmbient;
in vec4 VertexDiffuse;
in vec4 VertexSpecular;
in float VertexShininess;

/* Uniforms */
uniform mat4 ModelMatrix;

layout(std140) uniform SharedMatrices {

	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

struct LightSource {

	vec4 Position;
	vec4 Direction;

	vec4 Color;

	float CutOff;

	float AmbientIntensity;
	float DiffuseIntensity;
	float SpecularIntensity;

	float ConstantAttenuation;
	float LinearAttenuation;
	float ExponentialAttenuation;

	int LightType;
};

layout(std140) uniform SharedLightSources {

	LightSource LightSources[LIGHT_COUNT];
};

/* Output Attributes (Same as the Blinn-Phong Fragment Shader) */
out vec4 FragmentPosition;

out vec3 FragmentNormal;

out vec4 FragmentAmbient;
out vec4 FragmentDiffuse;
out vec4 FragmentSpecular;
out float FragmentShininess;

out vec4 FragmentRayOrigin;
out vec4 FragmentRayNormal;

void main() {

	/* Normal transformation matrices */
	mat3 NormalMatrix = inverse(transpose(mat3(ViewMatrix * ModelMatrix)));
	mat3 LightMatrix = inverse(transpose(mat3(ViewMatrix)));

	/* Vertex Position and Normal to View Space */
    FragmentPosition = ViewMatrix * ModelMatrix * VertexPosition;
    FragmentNormal = NormalMatrix * vec3(VertexNormal);

	/* Vertex Material */
	FragmentAmbient = VertexAmbient;
	FragmentDiffuse = VertexDiffuse;
	FragmentSpecular = VertexSpecular;
	FragmentShininess = VertexShininess;

	/* Ray Tracing */
	FragmentRayOrigin = ModelMatrix * VertexPosition;
	FragmentRayNormal = vec4(inverse(transpose(mat3(ModelMatrix))) * vec3(VertexNormal), 1.0f);

	/* Vertex Position to Clip Space */
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * VertexPosition;
}