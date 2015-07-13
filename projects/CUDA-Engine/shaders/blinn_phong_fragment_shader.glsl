#version 330 core
#pragma optionNV(unroll all)

#define LIGHT_COUNT 5

#define SPOT_LIGHT 1
#define POSITIONAL_LIGHT 2
#define DIRECTIONAL_LIGHT 3

#define SPOTLIGHT_OUTER_ANGLE 0.97

/* Input Attributes (Passed from the Blinn-Phong Vertex Shader) */
in vec4 FragmentPosition;

in vec3 FragmentNormal;

in vec4 FragmentAmbient;
in vec4 FragmentDiffuse;
in vec4 FragmentSpecular;
in float FragmentShininess;

in vec4 FragmentRayOrigin;
in vec4 FragmentRayNormal;

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

/* Output Attributes (Fragment Color) */
layout(location=0) out vec4 DiffuseColor;
layout(location=1) out vec4 SpecularColor;
layout(location=2) out vec4 RayOrigin;
layout(location=3) out vec4 RayNormal;

void main() {

	/* Fragment Diffuse and Specular Colors */
	DiffuseColor = FragmentDiffuse;
	SpecularColor = vec4(FragmentSpecular.rgb, FragmentShininess);

	/* Fragment Position and Normal */
	RayOrigin = FragmentRayOrigin;
	RayNormal = normalize(FragmentRayNormal);
}