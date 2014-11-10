#include "Mesh.h"

Mesh::Mesh(string name, string meshFilename, string materialFilename) {

	OBJ_Reader* objReader = OBJ_Reader::getInstance();

	objReader->loadMesh(meshFilename, materialFilename, this);
}

Mesh::~Mesh() {

	/* Destroy Vertices */
	map<int,Vertex*>::const_iterator vertexIterator;
	for(vertexIterator = this->vertexMap.begin(); vertexIterator != this->vertexMap.end(); vertexIterator++)
		delete vertexIterator->second;

	/* Destroy Materials */
	map<int,Material*>::const_iterator materialIterator;
	for(materialIterator = this->materialMap.begin(); materialIterator != this->materialMap.end(); materialIterator++)
		delete materialIterator->second;
}

string Mesh::getName() {

	return this->name;
}

void Mesh::addVertex(Vertex* vertex) {

	this->vertexMap[vertex->getID()] = vertex;
}

void Mesh::removeVertex(int vertexID) {

	this->vertexMap.erase(vertexID);
}

Vertex* Mesh::getVertex(int vertexID) {

	if (vertexMap.find(vertexID) == vertexMap.end())
		return NULL;

	return vertexMap[vertexID];
}

map<int,Vertex*> Mesh::getVertexMap() {

	return this->vertexMap;
}

void Mesh::addMaterial(Material* material) {

	this->materialMap[material->getID()] = material;
}

void Mesh::removeMaterial(int materialID) {

	this->materialMap.erase(materialID);
}

Material* Mesh::getMaterial(int materialID) {

	if(this->materialMap.find(materialID) == this->materialMap.end())
		return NULL;

	return this->materialMap[materialID];
}

map<int,Material*> Mesh::getMaterialMap() {

	return this->materialMap;
}

void Mesh::dump() {

	cout << "<Mesh \"" << this->name << "\" Dump>" << endl;

	/* Vertex Map */
	map<int,Vertex*>::const_iterator vertexIterator;
	for(vertexIterator = this->vertexMap.begin(); vertexIterator != this->vertexMap.end(); vertexIterator++)
		vertexIterator->second->dump();

	/* Material Map */
	map<int,Material*>::const_iterator materialIterator;
	for(materialIterator = this->materialMap.begin(); materialIterator != this->materialMap.end(); materialIterator++)
		materialIterator->second->dump();
}