#include "Light.h"

Light::Light(string name) {

	this->name = name;

	this->cutOff = 45.0f;

	this->diffuseIntensity = 0.0f;
	this->specularIntensity = 0.0f;

	this->constantAttenuation = 0.0f;
	this->linearAttenuation = 0.0f;
	this->exponentialAttenuation = 0.0f;
}

Light::~Light() {
}

string Light::getName() {
	
	return this->name;
}

Vector Light::getPosition() {

	return this->position;
}

Vector Light::getDirection() {

	return this->direction;
}

Vector Light::getColor() {

	return this->color;
}

float Light::getCutOff() {

	return this->cutOff;
}

float Light::getDiffuseIntensity() {

	return this->diffuseIntensity;
}

float Light::getSpecularIntensity() {

	return this->specularIntensity;
}

float Light::getConstantAttenuation() {

	return this->constantAttenuation;
}

float Light::getLinearAttenuation() {

	return this->linearAttenuation;
}

float Light::getExponentinalAttenuation() {

	return this->exponentialAttenuation;
}

void Light::setName(string name) {

	this->name = name;
}

void Light::setPosition(Vector position) {

	this->position = position;
}

void Light::setDirection(Vector direction) {

	this->direction = direction;
}

void Light::setCutOff(float cutOff) {

	this->cutOff = cutOff;
}

void Light::setColor(Vector color) {

	this->color = color;
}

void Light::setDiffuseIntensity(float diffuseIntensity) {

	this->diffuseIntensity = diffuseIntensity;
}

void Light::setSpecularIntensity(float specularIntensity) {

	this->specularIntensity = specularIntensity;
}

void Light::setConstantAttenuation(float constantAttenuation) {

	this->constantAttenuation = constantAttenuation;
}

void Light::setLinearAttenuation(float linearAttenuation) {

	this->linearAttenuation = linearAttenuation;
}

void Light::setExponentialAttenuation(float exponentialAttenuation) {

	this->exponentialAttenuation = exponentialAttenuation;
}

void Light::dump() {

	cout << "<Light \"" << this->name << "\" Dump>" << endl;

	/* Light Position */
	cout << "<Light Position> = "; this->position.dump();
	/* SpotLight Direction */
	cout << "<SpotLight Direction> = ";	this->direction.dump();

	/* SpotLight Cut Off */
	cout << "<SpotLight Cut Off> = " << this->cutOff << endl;

	/* Light Color */
	cout << "<Light Color> = ";	this->color.dump();

	/* Light Intensity */
	cout << "<Light Diffuse Intensity> = " << this->diffuseIntensity << endl;
	cout << "<Light Specular Intensity> = " << this->specularIntensity << endl;

	/* Light Attenuation */
	cout << "<PositionalLight Constant Attenuation> = " << this->constantAttenuation << endl;
	cout << "<PositionalLight Linear Attenuation> = " << this->linearAttenuation << endl;
	cout << "<PositionalLight Exponential Attenuation> = " << this->exponentialAttenuation << endl;
}
