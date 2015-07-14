#include "PositionalLight.h"

PositionalLight::PositionalLight(string name) : Light(name) {

	this->constantAttenuation = 0.0f;
	this->linearAttenuation = 0.0f;
	this->exponentialAttenuation = 0.0f;
}

PositionalLight::~PositionalLight() {
}

void PositionalLight::loadUniforms() {

	GLfloat lightSourceArray[LIGHT_SOURCE_STRUCT_SIZE];

	/* Position */
	for(int i=0; i<4; i++)
		lightSourceArray[i] = this->position[i];
	/* Direction */
	for(int i=0; i<4; i++)
		lightSourceArray[i+4] = 0.0f;
	/* Color */
	for(int i=0; i<4; i++)
		lightSourceArray[i+8] = this->color[i];

	/* Cut Off */
	lightSourceArray[12] = 0.0f;
	/* Intensity (Ambient, Diffuse, Specular) */
	lightSourceArray[13] = this->ambientIntensity;
	lightSourceArray[14] = this->diffuseIntensity;
	lightSourceArray[15] = this->specularIntensity;
	/* Attenuation (Constant, Linear, Exponential) */
	lightSourceArray[16] = this->constantAttenuation;
	lightSourceArray[17] = this->linearAttenuation;
	lightSourceArray[18] = this->exponentialAttenuation;

	GLint lightType = POSITIONAL_LIGHT;
	
	/* Uniform Locations */
	glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBufferIndex);

		glBufferSubData(GL_UNIFORM_BUFFER, (sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * this->identifier, 
											sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE, lightSourceArray);
		glBufferSubData(GL_UNIFORM_BUFFER, (sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * this->identifier + sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE, 
											sizeof(GLint), &lightType);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	Utility::checkOpenGLError("ERROR: PositionalLight \"" + this->name + "\" Uniform loading failed.");
}

GLfloat PositionalLight::getConstantAttenuation() {

	return this->constantAttenuation;
}

GLfloat PositionalLight::getLinearAttenuation() {

	return this->linearAttenuation;
}

GLfloat PositionalLight::getExponentinalAttenuation() {

	return this->exponentialAttenuation;
}

void PositionalLight::setConstantAttenuation(GLfloat constantAttenuation) {

	this->constantAttenuation = constantAttenuation;
}

void PositionalLight::setLinearAttenuation(GLfloat linearAttenuation) {

	this->linearAttenuation = linearAttenuation;
}

void PositionalLight::setExponentialAttenuation(GLfloat exponentialAttenuation) {

	this->exponentialAttenuation = exponentialAttenuation;
}

void PositionalLight::dump() {

	/* Light Dump */
	Light::dump();

	/* PositionalLight Attenuation */
	cout << "<PositionalLight Constant Attenuation> = " << this->constantAttenuation << endl;
	cout << "<PositionalLight Linear Attenuation> = " << this->linearAttenuation << endl;
	cout << "<PositionalLight Exponential Attenuation> = " << this->exponentialAttenuation << endl;
}
