#include "Mesh.h"

Mesh::Mesh(string name, string meshFilename) {
	
	// Initialize the Mesh
	OBJ_Reader* objReader = OBJ_Reader::getInstance();

	objReader->loadMesh(meshFilename, this);
}

Mesh::~Mesh() {
}

string Mesh::getName() {

	return this->name;
}

BoundingSphere* Mesh::getBoundingSphere() {

	return this->boundingSphere;
}

void Mesh::setName(string name) {

	this->name = name;
}

void Mesh::setBoundingSphere(BoundingSphere* boundingSphere) {

	this->boundingSphere = boundingSphere;
}

int Mesh::getVertexCount() {

	return this->vertexMap.size();
}

void Mesh::addVertex(Vertex* vertex) {

	this->vertexMap[vertex->getID()] = vertex;
}

void Mesh::removeVertex(int vertexID) {

	this->vertexMap.erase(vertexID);
}

map<int, Vertex*> Mesh::getVertexMap() {

	return this->vertexMap;
}

void Mesh::dump() {

	cout << "<Mesh \"" << this->name << "\" Dump>" << endl;

	/* Buffer Object Vertex Attributes */
	cout << "<Mesh Vertex Count> = " << this->vertexMap.size() << " ;" << endl;

	cout << "<Mesh Vertex List> = "  << endl;
	for(map<int,Vertex*>::const_iterator vertexMapIterator = this->vertexMap.begin(); vertexMapIterator != this->vertexMap.end(); vertexMapIterator++) {

		Vertex* vertex = vertexMapIterator->second;

		cout << "\tVertex " << vertex->getID() << " Position: ";
		vertex->getPosition().dump();
	}
}