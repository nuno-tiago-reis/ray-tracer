#version 330 core
#pragma optionNV(unroll all)

#define LIGHT_COUNT 5

#define SPOT_LIGHT 1
#define POSITIONAL_LIGHT 2
#define DIRECTIONAL_LIGHT 3

#define SPOTLIGHT_OUTER_ANGLE 0.97

/* Input Attributes (Passed from the Blinn-Phong Vertex Shader) */
smooth in vec4 FragmentPosition;

smooth in vec3 FragmentNormal;

in vec4 FragmentAmbient;
in vec4 FragmentDiffuse;
in vec4 FragmentSpecular;
in float FragmentShininess;

smooth in vec4 FragmentRayOrigin;
smooth in vec4 FragmentRayNormal;

smooth in vec3 LightDirection[LIGHT_COUNT];
smooth in vec3 HalfwayVector[LIGHT_COUNT];

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
layout(location=0) out vec4 FragmentColor;
layout(location=1) out vec4 RayOrigin;
layout(location=2) out vec4 RayReflection;
layout(location=3) out vec4 RayRefraction;

vec4 positionalLight(int i, vec3 Normal) {
	
	/* Light LightDistance / Direction */
	float LightDistance = length(ViewMatrix * LightSources[i].Position - FragmentPosition);

	/* Light Intensity */
	float LightIntensity = 1.0f / (1.0f + LightSources[i].ConstantAttenuation + LightSources[i].LinearAttenuation * LightDistance + LightSources[i].ExponentialAttenuation * LightDistance * LightDistance);

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
	
	/* Light computing */
	for(int i=0; i<LIGHT_COUNT; i++) {

		switch(LightSources[i].LightType) { 

			case POSITIONAL_LIGHT:	FragmentColor += positionalLight(i, Normal);
									break;

			case DIRECTIONAL_LIGHT:	FragmentColor += directionalLight(i, Normal);
									break;

			case SPOT_LIGHT:		FragmentColor += spotLight(i, Normal);
									break;
		}
	}

	/* Ray Origin */
	RayOrigin = FragmentRayOrigin;
	/* Ray Reflection */
	RayReflection = normalize(FragmentRayNormal);
	/* Ray Refraction */
	RayRefraction = vec4(refract(FragmentRayNormal.xyz, Normal, 1.0f / 0.5f), 1.0f);
}