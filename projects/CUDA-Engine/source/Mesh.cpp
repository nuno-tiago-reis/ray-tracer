#include "Mesh.h"

Mesh::Mesh(string name, string meshFilename, string materialFilename) {

	OBJ_Reader* objReader = OBJ_Reader::getInstance();
	objReader->loadMesh(meshFilename, materialFilename, this);

	createMesh();
}

Mesh::~Mesh() {

	destroyMesh();

	delete[] this->vertices;
}

void Mesh::createMesh() {

	/* Generate the Array Object */
	glGenVertexArrays(1, &this->arrayObjectID);
	/* Bind the Array Object */
	glBindVertexArray(this->arrayObjectID);

		/* Generate the Buffer Object */
		glGenBuffers(1, &this->bufferObjectID);
		/* Bind the Buffer Object */
		glBindBuffer(GL_ARRAY_BUFFER, this->bufferObjectID);

			/* Allocate the memory for the Mesh */
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * this->vertexCount, this->vertices, GL_STATIC_DRAW);

			GLint offset = 0;

			/* Position - 4 Floats */
			glEnableVertexAttribArray(POSITION);
			glVertexAttribPointer(POSITION, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);

			offset += sizeof(this->vertices[0].position);

			/* Normal - 4 Normalized Floats */
			glEnableVertexAttribArray(NORMAL);
			glVertexAttribPointer(NORMAL, 4, GL_FLOAT, GL_TRUE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].normal);

			/* Tangent - 4 Normalized Floats */
			glEnableVertexAttribArray(TANGENT);
			glVertexAttribPointer(TANGENT, 4, GL_FLOAT, GL_TRUE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].tangent);

			/* Texture UV - 2 Floats */
			glEnableVertexAttribArray(TEXTURE_UV);
			glVertexAttribPointer(TEXTURE_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].textureUV);

			/* Material Ambient - 4 Floats */
			glEnableVertexAttribArray(AMBIENT);
			glVertexAttribPointer(AMBIENT, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].ambient);

			/* Material Diffuse - 4 Floats */
			glEnableVertexAttribArray(DIFFUSE);
			glVertexAttribPointer(DIFFUSE, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].diffuse);

			/* Material Specular - 4 Floats */
			glEnableVertexAttribArray(SPECULAR);
			glVertexAttribPointer(SPECULAR, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offset);

			offset += sizeof(this->vertices[0].specular);

			/* Material Shininess - 1 Float */
			glEnableVertexAttribArray(SHININESS);
			glVertexAttribPointer(SHININESS, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offset);

		/* Unbind the Buffer Object */
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	/* Unbind the Array Object */
	glBindVertexArray(0);

	/*glDisableVertexAttribArray(POSITION);
	
	glDisableVertexAttribArray(NORMAL);
	glDisableVertexAttribArray(TANGENT);

	glDisableVertexAttribArray(TEXTURE_UV);

	glDisableVertexAttribArray(AMBIENT);
	glDisableVertexAttribArray(DIFFUSE);
	glDisableVertexAttribArray(SPECULAR);
	glDisableVertexAttribArray(SHININESS);*/

	Utility::checkOpenGLError("ERROR: Buffer Object creation failed.");
}

void Mesh::destroyMesh() {

	/*glDisableVertexAttribArray(POSITION);
	
	glDisableVertexAttribArray(NORMAL);
	glDisableVertexAttribArray(TANGENT);

	glDisableVertexAttribArray(TEXTURE_UV);

	glDisableVertexAttribArray(AMBIENT);
	glDisableVertexAttribArray(DIFFUSE);
	glDisableVertexAttribArray(SPECULAR);
	glDisableVertexAttribArray(SHININESS);*/

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glDeleteBuffers(1, &this->bufferObjectID);
	glDeleteVertexArrays(1, &this->arrayObjectID);

	Utility::checkOpenGLError("ERROR: Buffer Object destruction failed.");
}

string Mesh::getName() {

	return this->name;
}

GLint Mesh::getVertexCount() {

	return this->vertexCount;
}

Vertex* Mesh::getVertices() {

	return this->vertices;
}

GLuint Mesh::getArrayObjectID() {

	return this->arrayObjectID;
}

GLuint Mesh::getBufferObjectID() {

	return this->bufferObjectID;
}

void Mesh::setName(string name) {

	this->name = name;
}

void Mesh::setVertexCount(GLint vertexCount) {

	this->vertexCount = vertexCount;
}

void Mesh::setVertices(Vertex* vertices, GLint vertexCount) {

	this->vertexCount = vertexCount;

	this->vertices = new Vertex[vertexCount];

	memcpy(this->vertices, vertices, sizeof(Vertex) * vertexCount);
}

void Mesh::setArrayObjectID(GLuint arrayObjectID) {

	this->arrayObjectID = arrayObjectID;
}

void Mesh::setBufferObjectID(GLuint bufferObjectID) {

	this->bufferObjectID = bufferObjectID;
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

	/* Buffer Object OpenGL IDs */
	cout << "<Mesh ArrayObject ID> = " << this->arrayObjectID << " ;" << endl;
	cout << "<Mesh BufferObject ID> = " << this->bufferObjectID << ";" << endl;
}