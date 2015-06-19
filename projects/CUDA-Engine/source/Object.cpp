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

	if(this->mesh != NULL)
		delete this->mesh;

	if(this->material != NULL)
		delete this->material;

	if(this->transform != NULL)
		delete this->transform;
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
	glBindVertexArray(this->mesh->getArrayObjectID());

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

	Vertex* vertices = this->mesh->getVertices();

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
	}

	return 0.0f;
}

string Object::getName() {

	return this->name;
}

string Object::getParentName() {

	return this->parentName;
}

Material* Object::getMaterial() {

	return this->material;
}

Transform* Object::getTransform() {

	return this->transform;
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

void Object::dump() {

	cout << "<Object \"" << this->name << "\" Dump>" << endl;

	/* Object Compnents: Mesh, Material and Transform */
	cout << "<Object Mesh> = ";
	this->mesh->dump();
	cout << "<Object Material> = ";
	this->material->dump();
	cout << "<Object Transform> = ";
	this->transform->dump();
}