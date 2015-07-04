#include "Object.h"

Object::Object(string name) {

	/* Initialize the Objects Name */
	this->name.assign(name);
	this->parentName = "None";

	/* Initialize the Objects Mesh, Material and Transform */
	this->mesh = NULL;
	this->material = NULL;
	this->transform = NULL;
}

Object::~Object() {

	destroyMesh();

	if(this->mesh != NULL)
		delete this->mesh;

	if(this->material != NULL)
		delete this->material;

	if(this->transform != NULL)
		delete this->transform;
}

void Object::createMesh() {

	if(this->mesh == NULL) {

		cerr << "[" << this->name << "] Mesh not initialized." << endl;
		return;
	}

	if(this->material == NULL) {

		cerr << "[" << this->name << "] Material not initialized." << endl;
		return;
	}

	map<int,Vertex*> vertexMap = this->mesh->getVertexMap();

	// Create the Vertex List
	VertexStructure* vertexList = new VertexStructure[vertexMap.size()];

	// Populate the Vertex List
	for(map<int,Vertex*>::const_iterator vertexMapIterator = vertexMap.begin(); vertexMapIterator != vertexMap.end(); vertexMapIterator++) {
	
		Vertex* vertex = vertexMapIterator->second;

		// Load the Position
		Vector position = vertex->getPosition();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].position[i] = position[i];

		// Load the Normal
		Vector normal = vertex->getNormal();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].normal[i] = normal[i];

		// Load the Tangent
		Vector tangent = vertex->getTangent();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].tangent[i] = tangent[i];

		// Load the Texture Coordinates
		Vector textureCoordinates = vertex->getTextureCoordinates();
		for(int i=0; i<2; i++)
			vertexList[vertex->getID()].textureUV[i] = textureCoordinates[i];

		// Load the Ambient
		Vector ambient = material->getAmbient();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].ambient[i] = ambient[i];

		// Load the Diffuse
		Vector diffuse = material->getDiffuse();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].diffuse[i] = diffuse[i];

		// Load the Specular
		Vector specular = material->getSpecular();
		for(int i=0; i<4; i++)
			vertexList[vertex->getID()].specular[i] = specular[i];

		// Load the Specular Constant
		GLfloat specularConstant = material->getSpecularConstant();
		vertexList[vertex->getID()].specularConstant = specularConstant;
	}

	/* Generate the Array Object */
	glGenVertexArrays(1, &this->arrayObjectID);
	/* Bind the Array Object */
	glBindVertexArray(this->arrayObjectID);

		/* Generate the Buffer Object */
		glGenBuffers(1, &this->bufferObjectID);
		/* Bind the Buffer Object */
		glBindBuffer(GL_ARRAY_BUFFER, this->bufferObjectID);

			/* Allocate the memory for the Mesh */
			glBufferData(GL_ARRAY_BUFFER, sizeof(VertexStructure) * vertexMap.size(), vertexList, GL_STATIC_DRAW);

			GLint offset = 0;

			/* Position - 4 Floats */
			glEnableVertexAttribArray(POSITION);
			glVertexAttribPointer(POSITION, 4, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), 0);

			offset += sizeof(vertexList[0].position);

			/* Normal - 4 Normalized Floats */
			glEnableVertexAttribArray(NORMAL);
			glVertexAttribPointer(NORMAL, 4, GL_FLOAT, GL_TRUE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].normal);

			/* Tangent - 4 Normalized Floats */
			glEnableVertexAttribArray(TANGENT);
			glVertexAttribPointer(TANGENT, 4, GL_FLOAT, GL_TRUE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].tangent);

			/* Texture UV - 2 Floats */
			glEnableVertexAttribArray(TEXTURE_UV);
			glVertexAttribPointer(TEXTURE_UV, 2, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].textureUV);

			/* Material Ambient - 4 Floats */
			glEnableVertexAttribArray(AMBIENT);
			glVertexAttribPointer(AMBIENT, 4, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].ambient);

			/* Material Diffuse - 4 Floats */
			glEnableVertexAttribArray(DIFFUSE);
			glVertexAttribPointer(DIFFUSE, 4, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].diffuse);

			/* Material Specular - 4 Floats */
			glEnableVertexAttribArray(SPECULAR);
			glVertexAttribPointer(SPECULAR, 4, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), (GLvoid*)offset);

			offset += sizeof(vertexList[0].specular);

			/* Material Shininess - 1 Float */
			glEnableVertexAttribArray(SHININESS);
			glVertexAttribPointer(SHININESS, 1, GL_FLOAT, GL_FALSE, sizeof(VertexStructure), (GLvoid*)offset);

		/* Unbind the Buffer Object */
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	/* Unbind the Array Object */
	glBindVertexArray(0);

	Utility::checkOpenGLError("ERROR: Buffer Object creation failed.");
}

void Object::destroyMesh() {

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glDeleteBuffers(1, &this->bufferObjectID);
	glDeleteVertexArrays(1, &this->arrayObjectID);

	Utility::checkOpenGLError("ERROR: Buffer Object destruction failed.");
}

void Object::draw() {

	if(this->mesh == NULL) {

		cerr << "[" << this->name << "] Mesh not initialized." << endl;
		return;
	}

	if(this->material == NULL) {

		cerr << "[" << this->name << "] Material not initialized." << endl;
		return;
	}

	if(this->transform == NULL) {

		cerr << "[" << this->name << "] Transform not initialized." << endl;
		return;
	}

	/* Bind the Shader Program */
	glBindVertexArray(this->arrayObjectID);

		/* Bind the Shader Program */
		glUseProgram(this->material->getShaderProgram()->getProgramID());

			/* Get the Model Matrix */
			GLfloat modelMatrix[16];

			this->transform->getModelMatrix().getValue(modelMatrix);

			/* Update the Model Matrix Uniform */
			glUniformMatrix4fv(glGetUniformLocation(this->material->getShaderProgram()->getProgramID(), MODEL_MATRIX_UNIFORM), 1, GL_TRUE, modelMatrix);
			Utility::checkOpenGLError("ERROR: Uniform Location \"" MODEL_MATRIX_UNIFORM "\" error.");

			this->material->bind();

				/* Draw the Model */
				glDrawArrays(GL_TRIANGLES, 0, this->mesh->getVertexCount());

			this->material->unbind();

		/* Unbind the Shader Program */
		glUseProgram(0);

	/* Unbind the Array Object */
	glBindVertexArray(0);
			
	Utility::checkOpenGLError("ERROR: Object drawing failed.");
}

void Object::update() {

	this->transform->update();
}

void Object::update(GLfloat elapsedTime) {

	this->transform->update(elapsedTime);
}

GLfloat Object::isIntersecting(Vector origin, Vector direction) {

	/*Vertex* vertices = this->mesh->getVertices();

	Matrix modelMatrix = this->transform->getModelMatrix();

	for(int i=0; i < this->mesh->getVertexCount() / 3; i++) {

		Vector vertex0(vertices[i*3+0].position[VX],vertices[i*3+0].position[VY],vertices[i*3+0].position[VZ],1.0f);
		Vector vertex1(vertices[i*3+1].position[VX],vertices[i*3+1].position[VY],vertices[i*3+1].position[VZ],1.0f);
		Vector vertex2(vertices[i*3+2].position[VX],vertices[i*3+2].position[VY],vertices[i*3+2].position[VZ],1.0f);

		vertex0 = modelMatrix * vertex0;
		vertex1 = modelMatrix * vertex1;
		vertex2 = modelMatrix * vertex2;

		Vector edge1 = vertex1;
		edge1 -= vertex0;

		Vector edge2 = vertex2;		
		edge2 -= vertex0;

		Vector p = Vector::crossProduct(direction,edge2);

		GLfloat determinant = Vector::dotProduct(edge1,p);

		if(fabs(determinant) < Vector::threshold)
			continue;

		GLfloat invertedDeterminant = 1.0f / determinant;
 
		Vector t = origin;
		t -= vertex0;
 
		GLfloat u = Vector::dotProduct(t,p) * invertedDeterminant;

		if(u < 0.0f || u > 1.0f)
			continue;

		Vector q = Vector::crossProduct(t,edge1);
 
		GLfloat v = Vector::dotProduct(direction,q) * invertedDeterminant;

		if(v < 0.0f || u + v  > 1.0f)
			continue;
 
		GLfloat w = Vector::dotProduct(edge2,q) * invertedDeterminant;
 
		if(w > Vector::threshold)
			return w;
	}*/

	return 0.0f;
}

int Object::getID() {

	return this->id;
}

string Object::getName() {

	return this->name;
}

string Object::getParentName() {

	return this->parentName;
}

Mesh* Object::getMesh() {

	return this->mesh;
}

Material* Object::getMaterial() {

	return this->material;
}

Transform* Object::getTransform() {

	return this->transform;
}

GLuint Object::getArrayObjectID() {

	return this->arrayObjectID;
}

GLuint Object::getBufferObjectID() {

	return this->bufferObjectID;
}

void Object::setID(int id) {

	this->id = id;
}

void Object::setName(string name) {

	this->name = name;
}

void Object::setParentName(string parentName) {

	this->parentName = parentName;
}

void Object::setMesh(Mesh* mesh) {

	this->mesh = mesh;
}

void Object::setMaterial(Material* material) {

	this->material = material;
}

void Object::setTransform(Transform* transform) {

	this->transform = transform;
}

void Object::setArrayObjectID(GLuint arrayObjectID) {

	this->arrayObjectID = arrayObjectID;
}

void Object::setBufferObjectID(GLuint bufferObjectID) {

	this->bufferObjectID = bufferObjectID;
}

void Object::dump() {

	cout << "<Object \"" << this->name << "\" Dump>" << endl;

	/* Object Compnents: Mesh, Material and Transform */
	cout << "<Object Mesh> = ";
	this->mesh->dump();
	cout << "<Object Material> = ";
	this->material->dump();
	cout << "<Object Transform> = ";
	this->transform->dump();

	/* Buffer Object OpenGL IDs */
	cout << "<Object ArrayObject ID> = " << this->arrayObjectID << " ;" << endl;
	cout << "<Object BufferObject ID> = " << this->bufferObjectID << ";" << endl;
}