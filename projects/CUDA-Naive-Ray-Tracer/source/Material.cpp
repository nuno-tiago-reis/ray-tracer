#include "Material.h"

Material::Material(int id, string name) {

	/* Initialize the Materials Identifiers */
	this->id = id;
	this->name = name;
}

Material::~Material() {
}

int Material::getID() {

	return this->id;
}

string Material::getName() {

	return this->name;
}

Vector Material::getAmbient() {

	return this->ambient;
}

Vector Material::getDiffuse() {

	return this->diffuse;
}

Vector Material::getSpecular() {

	return this->specular;
}

float Material::getSpecularConstant() {

	return this->specularConstant;
}

void Material::setID(int id) {

	this->id = id;
}

void Material::setName(string name) {

	this->name = name;
}

void Material::setAmbient(Vector ambient) {

	this->ambient = ambient;
}

void Material::setDiffuse(Vector diffuse) {

	this->diffuse = diffuse;
}

void Material::setSpecular(Vector specular) {

	this->specular = specular;
}

void Material::setSpecularConstant(float specularConstant) {

	this->specularConstant = specularConstant;
}

void Material::dump() {

	cout << "<Material \"" << this->id << "\" Dump>" << endl;

	/* Material Name */
	cout << "<Material Name> = " << this->name.c_str() << endl;

	/* Material Ambient, Diffuse, Specular Properties */
	cout << "<Material Ambient> = "; this->ambient.dump();
	cout << "<Material Diffuse> = "; this->diffuse.dump();
	cout << "<Material Specular> = "; this->specular.dump();

	/* Material Specular Constant */
	cout << "<Material Specular Constant> = " << this->specularConstant << endl;
}