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

uniform sampler3D Noise;
uniform float NoiseScale;

/* Output Attributes (Same as the Real-Wood Fragment Shader) */
out vec4 FragmentPosition;

out vec3 FragmentNormal;

out vec4 FragmentWoodColor;
out vec4 FragmentModelPosition;

out vec4 FragmentAmbient;
out vec4 FragmentDiffuse;
out vec4 FragmentSpecular;
out float FragmentShininess;

out vec3 LightDirection[LIGHT_COUNT];
out vec3 HalfwayVector[LIGHT_COUNT];

void main() {

	/* Normal transformation matrices */
	mat3 NormalMatrix = inverse(transpose(mat3(ViewMatrix * ModelMatrix)));
	mat3 LightMatrix = inverse(transpose(mat3(ViewMatrix)));

	/* Vertex Position and Normal to View Space */
    FragmentPosition = ViewMatrix * ModelMatrix * VertexPosition;
    FragmentNormal = NormalMatrix * vec3(VertexNormal);

	/******************************************* WOOD MAPPING ********************************************************/
	/* Wood Color Calculations * /
	vec4 DarkShade=vec4(0.4, 0.2, 0.07, 1.0);
	vec4 LightShade=vec4(0.6, 0.3, 0.1, 1.0);
	float RepetitionRate=20;
	float RepetitionScale=3.14;
	float RepetitionSharpness=4352.0;

	mat4 RotationMatrix;
	RotationMatrix[0]=vec4(0.75,-0.21651,0.625,0);
	RotationMatrix[1]=vec4(0.43301,0.875,-0.21651,0);
	RotationMatrix[2]=vec4(-0.5,0.43301,0.75,0);
	RotationMatrix[3]=vec4(0,0,0,1);
	
	vec3 RotatedPosition =  (RotationMatrix * ModelMatrix * VertexPosition * NoiseScale).xyz;
	float RotatedNoise = 2.0 * texture(Noise, RotatedPosition).r - 1.0;
	float r = fract(RepetitionRate * ((ModelMatrix * VertexPosition).y * NoiseScale) + RepetitionScale * RotatedNoise);
	float invMax = pow(RepetitionSharpness, RepetitionSharpness / (RepetitionSharpness - 1.0))/(RepetitionSharpness - 1.0);
	float ring = invMax * (r - pow(r, RepetitionSharpness));
	float lerp = ring + RotatedNoise;
	
	FragmentWoodColor = mix(DarkShade, LightShade, lerp);
	/ *****************************************************************************************************************/
	FragmentModelPosition = ModelMatrix * VertexPosition;

	/* Vertex Material */
	FragmentAmbient = VertexAmbient;
	FragmentDiffuse = VertexDiffuse;
	FragmentSpecular = VertexSpecular;
	FragmentShininess = VertexShininess;

	/* Light computing */
	for(int i=0; i<LIGHT_COUNT; i++) {

		switch(LightSources[i].LightType) { 

			case POSITIONAL_LIGHT:	LightDirection[i] = vec3((ViewMatrix * LightSources[i].Position) - FragmentPosition);
									HalfwayVector[i] = vec3(-FragmentPosition) + LightDirection[i];
									break;
			
			case DIRECTIONAL_LIGHT:	LightDirection[i] = LightMatrix * vec3(LightSources[i].Direction);
									HalfwayVector[i] = LightDirection[i];
									break;

			case SPOT_LIGHT:		LightDirection[i] = LightMatrix * vec3(LightSources[i].Direction);
									HalfwayVector[i] = vec3(-FragmentPosition) + LightDirection[i];
									break;
		}
	}

	/* Vertex Position to Clip Space */
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * VertexPosition;
}