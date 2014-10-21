#include "Light.h"

Light::Light(GLint identifier) {

	this->identifier = identifier;
}

Light::~Light() {
}

GLint Light::getIdentifier() {

	return identifier;
}

Vector Light::getPosition() {

	return position;
}

Vector Light::getColor() {

	return color;
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

void Light::dump() {

	cout << "Debugging Light" << endl;
 
	cout << "Pos = "; position.dump();
	cout << "color = "; color.dump();
}