#include "Material.h"

Material::Material(string name, string materialFilename, ShaderProgram* shaderProgram) {

	// Initialize the Material Name
	this->name = name;

	// Initialize the Material Shader Program
	this->shaderProgram = shaderProgram;

	// Initialize the Material
	OBJ_Reader* objReader = OBJ_Reader::getInstance();

	objReader->loadMaterial(materialFilename, this);
}

Material::~Material() {

	// Destroy the Materials Textures
	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++)
		delete textureIterator->second;
}

void Material::bind() {

	GLuint textureID = 0;

	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++) {
	
		Texture* texture = textureIterator->second;

		// Bind the Texture
		texture->bind(GL_TEXTURE0 + textureID);
		// Load the Texture Uniforms
		texture->loadUniforms(this->shaderProgram->getProgramID(), textureID);

		textureID++;
	}
}

void Material::unbind() {

	GLuint textureID = 0;

	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++) {
	
		Texture* texture = textureIterator->second;

		// Unbind the Texture
		texture->unbind(GL_TEXTURE0 + textureID);

		textureID++;
	}
}

MaterialStructure Material::getMaterialStructure() {

	MaterialStructure materialStructure;

	// Load the Materials Ambient Vector
	materialStructure.ambient[0] = this->ambient[0];
	materialStructure.ambient[1] = this->ambient[1];
	materialStructure.ambient[2] = this->ambient[2];
	materialStructure.ambient[3] = this->ambient[3];

	// Load the Materials Diffuse Vector
	materialStructure.diffuse[0] = this->diffuse[0];
	materialStructure.diffuse[1] = this->diffuse[1];
	materialStructure.diffuse[2] = this->diffuse[2];
	materialStructure.diffuse[3] = this->diffuse[3];
					
	// Load the Materials Specular Vector					
	materialStructure.specular[0] = this->specular[0];
	materialStructure.specular[1] = this->specular[1];
	materialStructure.specular[2] = this->specular[2];
	materialStructure.specular[3] = this->specular[3];

	// Load the Materials Specular Constant					
	materialStructure.specularConstant = this->specularConstant;

	return materialStructure;
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

ShaderProgram* Material::getShaderProgram() {

	return this->shaderProgram;
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

void Material::setShaderProgram(ShaderProgram* shaderProgram) {

	this->shaderProgram = shaderProgram;
}

void Material::addTexture(Texture* texture) {

	this->textureMap[texture->getName()] = texture;
}

void Material::removeTexture(string textureName) {

	this->textureMap.erase(textureName);
}

Texture* Material::getTexture(string textureName) {

	return this->textureMap[textureName];
}

void Material::dump() {

	cout << "<Material \"" << this->name << "\" Dump>" << endl;

	// Material Shader Program
	cout << "<Material Shader Program> = " << this->shaderProgram->getName() << endl;
	
	// Material Ambient Vector
	cout << "<Material Ambient Vector> = "; this->ambient.dump();
	// Material Diffuse Vector
	cout << "<Material Diffuse Vector> = "; this->diffuse.dump();
	// Material Diffuse Vector
	cout << "<Material Specular Vector> = "; this->specular.dump();

	// Material Texture Map
	cout << "<Material Texture Map> = " << endl;
	for(map<string,Texture*>::const_iterator textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++)
		textureIterator->second->dump();
}