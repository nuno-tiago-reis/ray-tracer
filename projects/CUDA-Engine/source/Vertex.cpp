#include "Vertex.h"

Vertex::Vertex(int id) {

	// Initialize the Vertex Identifier
	this->id = id;
}

Vertex::~Vertex() {
}

VertexStructure Vertex::getVertexStructure() {

	VertexStructure vertexStructure;

	// Load the Vertices Position
	vertexStructure.position[0] = this->position[0];
	vertexStructure.position[1] = this->position[1];
	vertexStructure.position[2] = this->position[2];
	vertexStructure.position[3] = this->position[3];
	
	// Load the Vertices Normal
	vertexStructure.normal[0] = this->normal[0];
	vertexStructure.normal[1] = this->normal[1];
	vertexStructure.normal[2] = this->normal[2];
	vertexStructure.normal[3] = this->normal[3];
					
	// Load the Vertices Tangent				
	vertexStructure.tangent[0] = this->tangent[0];
	vertexStructure.tangent[1] = this->tangent[1];
	vertexStructure.tangent[2] = this->tangent[2];
	vertexStructure.tangent[3] = this->tangent[3];

	// Load the Vertices Texture Coordinates				
	vertexStructure.textureUV[0] = this->textureCoordinates[0];
	vertexStructure.textureUV[1] = this->textureCoordinates[1];
	vertexStructure.textureUV[2] = this->textureCoordinates[2];
	vertexStructure.textureUV[3] = this->textureCoordinates[3];	

	return vertexStructure;
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

void Vertex::dump() {

	cout << "<Vertex \"" << this->id << "\" Dump>" << endl;

	/* Vertex Position */
	cout << "<Vertex Position> = "; this->position.dump();

	/* Vertex Normal and Tangent */
	cout << "<Vertex Normal> = "; this->normal.dump();
	cout << "<Vertex Tangent> = "; this->tangent.dump();

	/* Vertex Texture Coordinates */
	cout << "<Vertex Texture Coordinates> = "; this->textureCoordinates.dump();
}