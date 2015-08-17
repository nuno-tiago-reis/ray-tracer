#version 330 core
#pragma optionNV(unroll all)

#define LIGHT_COUNT 5

#define SPOT_LIGHT 1
#define POSITIONAL_LIGHT 2
#define DIRECTIONAL_LIGHT 3

#define SPOTLIGHT_OUTER_ANGLE 0.97

/* Input Attributes (Same as the other Vertex Shaders) */
in vec4 VertexPosition;

in vec4 VertexNormal;
in vec4 VertexTangent;

in vec2 VertexTextureUV;

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

uniform sampler2D DiffuseTexture;
uniform sampler2D NormalTexture;

/* Output Attributes (Same as the Bump-Map Fragment Shader) */
out vec4 FragmentPosition;

out vec3 FragmentNormal;
out vec3 FragmentTangent;
out vec3 FragmentBitangent;

out vec2 FragmentTextureUV;

out vec4 FragmentAmbient;
out vec4 FragmentDiffuse;
out vec4 FragmentSpecular;
out float FragmentShininess;

out vec3 LightDirection[LIGHT_COUNT];
out vec3 HalfwayVector[LIGHT_COUNT];

/* Converts a Vector to Tangent Space */
vec3 convertToTangentSpace(vec3 Vector, vec3 Tangent, vec3 Bitangent, vec3 Normal) {

	vec3 Result;

	Result.x = dot(Vector, Tangent);
	Result.y = dot(Vector, Bitangent);
	Result.z = dot(Vector, Normal);

	return normalize(Result);
}

void main() {

	/* Normal transformation matrices */
	mat3 NormalMatrix = inverse(transpose(mat3(ViewMatrix * ModelMatrix)));
	mat3 LightMatrix = inverse(transpose(mat3(ViewMatrix)));

	/* Vertex Position, Normal, Tangent and Bitangent to View Space */
    FragmentPosition = ViewMatrix * ModelMatrix * VertexPosition;

    FragmentNormal = normalize(NormalMatrix * vec3(VertexNormal));
	FragmentTangent = normalize(NormalMatrix * vec3(VertexTangent));
	FragmentBitangent = cross(FragmentNormal,FragmentTangent) * VertexTangent.w;

	/* Vertex Texture Coordinate */
	FragmentTextureUV = VertexTextureUV;

	/* Vertex Material */
	FragmentAmbient = VertexAmbient;
	FragmentDiffuse = VertexDiffuse;
	FragmentSpecular = VertexSpecular;
	FragmentShininess = VertexShininess;

	/* Light computing */
	for(int i=0; i<LIGHT_COUNT; i++) {

		switch(LightSources[i].LightType) { 

			case POSITIONAL_LIGHT:	LightDirection[i] = convertToTangentSpace(
										vec3((ViewMatrix * LightSources[i].Position) - FragmentPosition), 
										FragmentTangent, FragmentBitangent, FragmentNormal);

									HalfwayVector[i] = convertToTangentSpace(
										vec3(-FragmentPosition) + vec3((ViewMatrix * LightSources[i].Position) - FragmentPosition), 
										FragmentTangent, FragmentBitangent, FragmentNormal);
									break;

			case DIRECTIONAL_LIGHT:	LightDirection[i] = convertToTangentSpace(
										LightMatrix * vec3(LightSources[i].Direction), 
										FragmentTangent, FragmentBitangent, FragmentNormal);

									HalfwayVector[i] = LightDirection[i];
									break;
		
			case SPOT_LIGHT:		LightDirection[i] = convertToTangentSpace(
										LightMatrix * vec3(LightSources[i].Direction), 
										FragmentTangent, FragmentBitangent, FragmentNormal);
									
									HalfwayVector[i] = convertToTangentSpace(
										vec3(-FragmentPosition) + vec3((ViewMatrix * LightSources[i].Position) - FragmentPosition), 
										FragmentTangent, FragmentBitangent, FragmentNormal);
									break;
		}
	}

	/* Vertex Position to Clip Space */
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * VertexPosition;
}