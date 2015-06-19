#include "Material.h"

Material::Material(string name, ShaderProgram* shaderProgram) {

	/* Initialize the Material Name */
	this->name = name;

	/* Initialize the Material Shader Program */
	this->shaderProgram = shaderProgram;
}

Material::~Material() {

	/* Destroy the Materials Textures */
	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++)
		delete textureIterator->second;
}

void Material::bind() {

	/*this->textureMap[NORMAL_TEXTURE] = new Texture(GL_TEXTURE_2D, normalTextureFileName);
	_te-xtureMap[NORMAL_TEXTURE]->load();*/

	GLuint textureID = 0;

	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++) {
	
		Texture* texture = textureIterator->second;

		/* Bind the Texture */
		texture->bind(GL_TEXTURE0 + textureID);
		/* Load the Texture Uniforms */
		texture->loadUniforms(this->shaderProgram->getProgramID(), textureID);

		textureID++;
	}
}

void Material::unbind() {

	GLuint textureID = 0;

	map<string,Texture*>::const_iterator textureIterator;
	for(textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++) {
	
		Texture* texture = textureIterator->second;

		/* Unind the Texture */
		texture->unbind(GL_TEXTURE0 + textureID);

		textureID++;
	}
}

string Material::getName() {

	return this->name;
}

ShaderProgram* Material::getShaderProgram() {

	return this->shaderProgram;
}

void Material::setName(string name) {

	this->name = name;
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

	/* Material Shader Program */
	cout << "<Material Shader Program> = " << this->shaderProgram->getName() << endl;

	/* Material Texture Map */
	cout << "<Material Texture Map> = " << endl;
	for(map<string,Texture*>::const_iterator textureIterator = this->textureMap.begin(); textureIterator != this->textureMap.end(); textureIterator++)
		textureIterator->second->dump();
}