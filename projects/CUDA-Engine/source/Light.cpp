#include "Light.h"

Light::Light(string name) {

	this->name = name;

	this->identifier = 0;

	this->ambientIntensity = 0.0f;
	this->diffuseIntensity = 0.0f;
	this->specularIntensity = 0.0f;
}

Light::~Light() {
}

void Light::loadUniforms() {
}

string Light::getName() {
	
	return this->name;
}

GLint Light::getIdentifier() {

	return this->identifier;
}

Vector Light::getPosition() {

	return this->position;
}

Vector Light::getColor() {

	return this->color;
}

GLfloat Light::getAmbientIntensity() {

	return this->ambientIntensity;
}

GLfloat Light::getDiffuseIntensity() {

	return this->diffuseIntensity;
}

GLfloat Light::getSpecularIntensity() {

	return this->specularIntensity;
}

GLuint Light::getUniformBufferIndex() {

	return this->uniformBufferIndex;
}

void Light::setName(string name) {

	this->name = name;
}

void Light::setIdentifier(GLint identifier) {

	this->identifier = identifier;
}

void Light::setPosition(Vector position) {

	this->position = position;
}

void Light::setColor(Vector color) {

	this->color = color;
}

void Light::setAmbientIntensity(GLfloat ambientIntensity) {

	this->ambientIntensity = ambientIntensity;
}

void Light::setDiffuseIntensity(GLfloat diffuseIntensity) {

	this->diffuseIntensity = diffuseIntensity;
}

void Light::setSpecularIntensity(GLfloat specularIntensity) {

	this->specularIntensity = specularIntensity;
}

void Light::setUniformBufferIndex(GLuint uniformBufferIndex) {

	this->uniformBufferIndex = uniformBufferIndex;
}

void Light::dump() {

	cout << "<Light \"" << this->name << "\" Dump>" << endl;

	/* Light Position */
	cout << "<Light Position> = " << endl;
	this->position.dump();

	/* Light Color */
	cout << "<Light Color> = " << endl;
	this->color.dump();

	/* Light Intensity */
	cout << "<Light Ambient Intensity> = " << this->ambientIntensity << endl;
	cout << "<Light Diffuse Intensity> = " << this->diffuseIntensity << endl;
}
