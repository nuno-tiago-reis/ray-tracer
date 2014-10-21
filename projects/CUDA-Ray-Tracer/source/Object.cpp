#include "Object.h"

Object::Object(string name) {

	/* Initialize the Objects Name */
	this->name.assign(name);

	/* Initialize the Objects Mesh, Material and Transform */
	this->mesh = NULL;
	this->transform = NULL;
}

Object::~Object() {

	if(this->mesh != NULL)
		delete this->mesh;

	if(this->transform != NULL)
		delete this->transform;
}

void Object::update() {

	this->transform->update();
}

void Object::update(GLfloat elapsedTime) {

	this->transform->update(elapsedTime);
}

string Object::getName() {

	return this->name;
}

Mesh* Object::getMesh() {

	return this->mesh;
}

Transform* Object::getTransform() {

	return this->transform;
}

void Object::setName(string name) {

	this->name = name;
}

void Object::setMesh(Mesh* mesh) {

	this->mesh = mesh;
}

void Object::setTransform(Transform* transform) {

	this->transform = transform;
}

void Object::dump() {

	cout << "<Object \"" << this->name << "\" Dump>" << endl;

	/* Object Compnents: Mesh, Material and Transform */
	cout << "<Object Mesh> = ";
	this->mesh->dump();
	cout << "<Object Transform> = ";
	this->transform->dump();
}