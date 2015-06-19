#include "SpotLight.h"

SpotLight::SpotLight(string name) : Light(name) {

	this->cutOff = 45.0f;

	this->constantAttenuation = 0.0f;
	this->linearAttenuation = 0.0f;
	this->exponentialAttenuation = 0.0f;
}

SpotLight::~SpotLight() {
}

void SpotLight::loadUniforms() {

	GLfloat lightSourceArray[LIGHT_SOURCE_STRUCT_SIZE];

	/* Position */
	for(int i=0; i<4; i++)
		lightSourceArray[i] = this->position[i];
	/* Direction */
	for(int i=0; i<4; i++)
		lightSourceArray[i+4] = this->direction[i];
	/* Color */
	for(int i=0; i<4; i++)
		lightSourceArray[i+8] = this->color[i];

	/* Cut Off */
	lightSourceArray[12] = this->cutOff;
	/* Intensity (Ambient, Diffuse, Specular) */
	lightSourceArray[13] = this->ambientIntensity;
	lightSourceArray[14] = this->diffuseIntensity;
	lightSourceArray[15] = this->specularIntensity;
	/* Attenuation (Constant, Linear, Exponential) */
	lightSourceArray[16] = this->constantAttenuation;
	lightSourceArray[17] = this->linearAttenuation;
	lightSourceArray[18] = this->exponentialAttenuation;

	GLint lightType = SPOT_LIGHT;

	/* Uniform Locations */
	glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBufferIndex);

	glBufferSubData(GL_UNIFORM_BUFFER, (sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * this->identifier, 
										sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE, lightSourceArray);
	glBufferSubData(GL_UNIFORM_BUFFER, (sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * this->identifier + sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE, 
										sizeof(GLint), &lightType);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	Utility::checkOpenGLError("ERROR: SpotLight \"" + this->name + "\" Uniform loading failed.");
}

Vector SpotLight::getDirection() {

	return this->direction;
}

GLfloat SpotLight::getCutOff() {

	return this->cutOff;
}

GLfloat SpotLight::getConstantAttenuation() {

	return this->constantAttenuation;
}

GLfloat SpotLight::getLinearAttenuation() {

	return this->linearAttenuation;
}

GLfloat SpotLight::getExponentinalAttenuation() {

	return this->exponentialAttenuation;
}

void SpotLight::setDirection(Vector direction) {

	this->direction = direction;
}

void SpotLight::setCutOff(GLfloat cutOff) {

	this->cutOff = cutOff;
}

void SpotLight::setConstantAttenuation(GLfloat constantAttenuation) {

	this->constantAttenuation = constantAttenuation;
}

void SpotLight::setLinearAttenuation(GLfloat linearAttenuation) {

	this->linearAttenuation = linearAttenuation;
}

void SpotLight::setExponentialAttenuation(GLfloat exponentialAttenuation) {

	this->exponentialAttenuation = exponentialAttenuation;
}

void SpotLight::dump() {

	/* Light Dump */
	Light::dump();

	/* SpotLight Direction */
	cout << "<SpotLight Direction> = " << endl;
	this->direction.dump();

	/* SpotLight Cut Off */
	cout << "<SpotLight Cut Off> = " << this->cutOff << endl;

	/* SpotLight Attenuation */
	cout << "<SpotLight Constant Attenuation> = " << this->constantAttenuation << endl;
	cout << "<SpotLight Linear Attenuation> = " << this->linearAttenuation << endl;
	cout << "<SpotLight Exponential Attenuation> = " << this->exponentialAttenuation << endl;
}
