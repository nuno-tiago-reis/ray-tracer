#include "Texture.h"

Texture::Texture(string name) {

	/* Initialize the Textures Name */
	this->name = name;
	/* Initialize the Textures Handler */
	this->handler = UINT_MAX;
}

Texture::~Texture() {
}

void Texture::deleteTexture() {

	// Delete the Texture from OpenGL 
	glDeleteTextures(1, &handler);
	Utility::checkOpenGLError("glDeleteTextures()");
}

string Texture::getName() {

	return this->name;
}

unsigned int Texture::getHandler() {

	return this->handler;
}

void Texture::setName(string name) {

	this->name = name;
}

void Texture::setHandler(GLuint handler) {

	this->handler = handler;
}

void Texture::dump() {

	cout << "<Texture \"" << this->name << "\" Dump>" << endl;

	/* Texture Handler */
	cout << "<Texture Handler> = " << this->handler << endl;
}