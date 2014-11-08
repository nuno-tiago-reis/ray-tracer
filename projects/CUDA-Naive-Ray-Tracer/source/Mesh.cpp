#include "Mesh.h"

Mesh::Mesh(string name, string meshFilename, string materialFilename) {

	OBJ_Reader* objReader = OBJ_Reader::getInstance();
	objReader->loadMesh(meshFilename, materialFilename, this);

	//createMesh();
}

Mesh::~Mesh() {

	delete[] this->vertices;
}

string Mesh::getName() {

	return this->name;
}

int Mesh::getVertexCount() {

	return this->vertexCount;
}

Vertex* Mesh::getVertices() {

	return this->vertices;
}

Vertex Mesh::getVertex(int vertexID) {

	return this->vertices[vertexID];
}

void Mesh::setName(string name) {

	this->name = name;
}

void Mesh::setVertexCount(int vertexCount) {

	this->vertexCount = vertexCount;
}

void Mesh::setVertices(Vertex* vertices, int vertexCount) {

	this->vertexCount = vertexCount;

	this->vertices = new Vertex[vertexCount];

	memcpy(this->vertices, vertices, sizeof(Vertex) * vertexCount);
}

void Mesh::dump() {

	cout << "<Mesh \"" << this->name << "\" Dump>" << endl;

	/* Buffer Object Vertex Attributes */
	cout << "<Mesh Vertex Count> = " << this->vertexCount << " ;" << endl;

	cout << "<Mesh Vertex List> = "  << endl;
	for(int i=0; i < this->vertexCount; i++) {

		cout << "\tVertex " << i << " Position: ";
		cout << "\t[" << this->vertices[i].position[0] << "][" << this->vertices[i].position[1] << "][" << this->vertices[i].position[2] << "][" << this->vertices[i].position[3] << "]" << endl;
	}
}