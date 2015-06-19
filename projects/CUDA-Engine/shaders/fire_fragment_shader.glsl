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

in vec3 LightDirection[LIGHT_COUNT];
in vec3 HalfwayVector[LIGHT_COUNT];

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
out vec4 FragmentColor;

vec4 positionalLight(int i, vec3 Normal) {
	
	/* Light LightDistance / Direction */
	float LightDistance = length(ViewMatrix * LightSources[i].Position - FragmentPosition);

	/* Light Intensity */
	float LightIntensity = 1.0f / (LightSources[i].ConstantAttenuation + LightSources[i].LinearAttenuation * LightDistance + LightSources[i].ExponentialAttenuation * LightDistance * LightDistance);

	/* Ambient Component */
	vec4 AmbientColor = FragmentAmbient * LightSources[i].Color * LightSources[i].AmbientIntensity;
	vec4 DiffuseColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);                                            
	vec4 SpecularColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);

	/* Diffuse Component */
	float DiffuseFactor = max(dot(Normal, normalize(LightDirection[i])), 0.0f);

	if (DiffuseFactor > 0.0f) {

		DiffuseColor = FragmentDiffuse * LightSources[i].Color * LightSources[i].DiffuseIntensity * DiffuseFactor;

		/* Specular Component */
		float SpecularAngle = max(dot(Normal, normalize(HalfwayVector[i])), 0.0f);
			                          
		float SpecularFactor = pow(SpecularAngle, FragmentShininess);                
		if(SpecularFactor > 0.0f)
			SpecularColor = FragmentSpecular * LightSources[i].Color * LightSources[i].SpecularIntensity * SpecularFactor;
	}

	/* Final Calculation */
	return AmbientColor + (DiffuseColor + SpecularColor) * LightIntensity;
}

vec4 directionalLight(int i, vec3 Normal) {

	/* Ambient Component */
	vec4 AmbientColor = FragmentAmbient * LightSources[i].Color * LightSources[i].AmbientIntensity;
	vec4 DiffuseColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);                                            
	vec4 SpecularColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);

	/* Diffuse Component */
	float DiffuseFactor = max(dot(Normal, -normalize(LightDirection[i])), 0.0f);

	if (DiffuseFactor > 0.0f) {

		DiffuseColor = FragmentDiffuse * LightSources[i].Color * LightSources[i].DiffuseIntensity * DiffuseFactor;

		/* Specular Component */
		float SpecularAngle = max(dot(Normal, normalize(HalfwayVector[i])), 0.0f);
			                          
		float SpecularFactor = pow(SpecularAngle, FragmentShininess);                               
		if(SpecularFactor > 0.0f)
			SpecularColor = FragmentSpecular * LightSources[i].Color * LightSources[i].SpecularIntensity * SpecularFactor;
	}

	/* Final Calculation */
	return AmbientColor + DiffuseColor + SpecularColor;
}

vec4 spotLight(int i, vec3 Normal) {

	/* Light LightDistance / Direction */
	vec3 LightToVertex = vec3(ViewMatrix * LightSources[i].Position - FragmentPosition);
	float LightDistance = length(LightToVertex);  
	LightToVertex = normalize(LightToVertex);

	/* Light Intensity */
	float LightIntensity = 1.0f / (LightSources[i].ConstantAttenuation + LightSources[i].LinearAttenuation * LightDistance + LightSources[i].ExponentialAttenuation * LightDistance * LightDistance);

	/* Ambient Component */
	vec4 AmbientColor = FragmentAmbient * LightSources[i].Color * LightSources[i].AmbientIntensity;
	vec4 DiffuseColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);                                            
	vec4 SpecularColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	
	/* Diffuse Component */
	float DiffuseFactor = max(dot(Normal, LightToVertex), 0.0f);

	if (DiffuseFactor > 0.0f) {

		/* SpotLight Circular effect fading around the edges */
		float CosineDifference = SPOTLIGHT_OUTER_ANGLE - LightSources[i].CutOff ;

		float SpotEffect = clamp((dot(normalize(LightDirection[i]), -LightToVertex) - LightSources[i].CutOff) / CosineDifference, 0.0f, 1.0f);

		DiffuseColor = FragmentDiffuse * LightSources[i].Color * LightSources[i].DiffuseIntensity * DiffuseFactor * SpotEffect;

		/* Specular Component */			                
		float SpecularAngle = max(dot(Normal, normalize(HalfwayVector[i])), 0.0f);
			                          
		float SpecularFactor = pow(SpecularAngle, FragmentShininess);                               
		if(SpecularFactor > 0.0f)
			SpecularColor = FragmentSpecular *  LightSources[i].Color * LightSources[i].SpecularIntensity * SpecularFactor * SpotEffect;
	}

	/* Final Calculation */
	return AmbientColor + (DiffuseColor + SpecularColor) * LightIntensity;
}

void main() {

	/* Fragment Normal */
	vec3 Normal = normalize(FragmentNormal);

	/* Fragment Color */
	FragmentColor = vec4(0.0f);

	FragmentColor = vec4(1.0f,0.0f,0.0f,0.75f);
	
	/* Light computing */
	/*for(int i=0; i<LIGHT_COUNT; i++) {

		switch(LightSources[i].LightType) { 

			case POSITIONAL_LIGHT:	FragmentColor += positionalLight(i, Normal);
									break;

			case DIRECTIONAL_LIGHT:	FragmentColor += directionalLight(i, Normal);
									break;

			case SPOT_LIGHT:		FragmentColor += spotLight(i, Normal);
									break;
		}
	}*/
}