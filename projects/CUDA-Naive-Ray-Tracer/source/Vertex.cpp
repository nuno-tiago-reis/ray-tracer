#include "Vertex.h"

Vertex::Vertex(int id) {

	/* Initialize the Vertex Identifier */
	this->id = id;
}

Vertex::~Vertex() {
}

int Vertex::getID() {

	return this->id;
}

Vector Vertex::getPosition() {

	return this->position;
}

Vector Vertex::getNormal() {

	return this->normal;
}

Vector Vertex::getTangent() {

	return this->tangent;
}

Vector Vertex::getTextureCoordinates() {

	return this->textureCoordinates;
}

int Vertex::getMaterialID() {

	return this->materialID;
}

void Vertex::setID(int id) {

	this->id = id;
}

void Vertex::setPosition(Vector position) {

	this->position = position;
}

void Vertex::setNormal(Vector normal) {

	this->normal = normal;
}

void Vertex::setTangent(Vector tangent) {

	this->tangent = tangent;
}

void Vertex::setTextureCoordinates(Vector textureCoordinates) {

	this->textureCoordinates = textureCoordinates;
}

void Vertex::setMaterialID(int materialID) {

	this->materialID = materialID;
}

void Vertex::dump() {

	cout << "<Vertex \"" << this->id << "\" Dump>" << endl;

	/* Vertex Position */
	cout << "<Vertex Position> = "; this->position.dump();

	/* Vertex Normal and Tangent */
	cout << "<Vertex Normal> = "; this->normal.dump();
	cout << "<Vertex Tangent> = "; this->tangent.dump();

	/* Vertex Texture Coordinates */
	cout << "<Vertex Texture Coordinates> = "; this->textureCoordinates.dump();

	/* Vertex Material ID */
	cout << "<Vertex Material ID> = " << this->materialID << endl;
}