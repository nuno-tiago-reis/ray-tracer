#include "Object.h"

Object::Object(int identifier) {

	this->identifier = identifier;

	diffuseIntensity = 0.0f;
	specularIntensity = 0.0f;
	
	shininess = 0.0f;

	transmittance = 0.0f;
	refractionIndex = 0.0f;

	boundingBox = NULL;
}

Object::~Object() {
}

GLint Object::getIdentifier() {

	return identifier;
}


Vector Object::getColor() {

	return color;
}

GLfloat Object::getDiffuseIntensity() {

	return diffuseIntensity;
}

GLfloat Object::getSpecularIntensity() {

	return specularIntensity;
}

GLfloat Object::getShininess() {

	return shininess;
}

GLfloat Object::getTransmittance() {

	return transmittance;
}

GLfloat Object::getRefractionIndex() {

	return refractionIndex;
}

BoundingBox* Object::getBoundingBox() {
	return boundingBox;
}


void Object::setIdentifier(GLint identifier) {

	this->identifier = identifier;
}

void Object::setColor(Vector color) {

	this->color = color;
}

void Object::setDiffuseIntensity(GLfloat diffuseIntensity) {

	this->diffuseIntensity = diffuseIntensity;
}

void Object::setSpecularIntensity(GLfloat specularIntensity) {

	this->specularIntensity = specularIntensity;
}

void Object::setShininess(GLfloat shininess) {

	this->shininess = shininess;
}

void Object::setTransmittance(GLfloat transmittance) {

	this->transmittance = transmittance;
}

void Object::setRefractionIndex(GLfloat refractionIndex) {

	this->refractionIndex = refractionIndex;
}

void Object::setBoundingBox(BoundingBox *boundingBox) {

	this->boundingBox = boundingBox;
}

void Object::dump() {

	cout << "Debugging Object [" << identifier << "]" << endl;

	cout << "\tColor = "; color.dump();

	cout << "\tDiffuseIntensity = " << diffuseIntensity << endl;
	cout << "\tSpecularIntensity = " << specularIntensity << endl;
	cout << "\tShininess = " << shininess << endl;
	cout << "\tTransmittance = " << transmittance << endl;
	cout << "\tRefractionIndex = " << refractionIndex << endl;
}