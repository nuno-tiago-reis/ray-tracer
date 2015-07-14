#include "ScreenTexture.h"

ScreenTexture::ScreenTexture(string name, unsigned int width, unsigned int height) {
	
	/* Initialize the Textures Name */
	this->name = name;
	/* Initialize the Textures Handler */
	this->handler = UINT_MAX;

	/* Initialize the Textures Dimensions */
	this->width = width;
	this->height = height;
}

ScreenTexture::~ScreenTexture() {
	
	// Delete the Texture in case it already exists.
	this->deleteTexture();
}

void ScreenTexture::createTexture() {
	
	// Delete the Texture in case it already exists.
	this->deleteTexture();

	// Create the Texture to output the Ray-Tracing result.
	glGenTextures(1, &this->handler);
	glBindTexture(GL_TEXTURE_2D, this->handler);

		// Set the basic Texture parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// Define the basic Texture parameters
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		Utility::checkOpenGLError("glTexImage2D()");

	glBindTexture(GL_TEXTURE_2D, 0);
}

void ScreenTexture::deleteTexture() {

	// Delete the Texture from OpenGL 
	glDeleteTextures(1, &handler);
	Utility::checkOpenGLError("glDeleteTextures()");
}

string ScreenTexture::getName() {

	return this->name;
}

unsigned int ScreenTexture::getHandler() {

	return this->handler;
}

unsigned int ScreenTexture::getWidth() {

	return this->width;
}

unsigned int ScreenTexture::getHeight() {

	return this->height;
}

void ScreenTexture::setName(string name) {

	this->name = name;
}

void ScreenTexture::setHandler(unsigned int handler) {

	this->handler = handler;
}

void ScreenTexture::setWidth(unsigned int width) {

	this->width = width;
}

void ScreenTexture::setHeight(unsigned int height) {

	this->height = height;
}

void ScreenTexture::dump() {

	cout << "<ScreenTexture \"" << this->name << "\" Dump>" << endl;

	/* Texture Handler */
	cout << "<ScreenTexture Handler> = " << this->handler << endl;

	/* Texture Dimensions */
	cout << "<ScreenTexture Width> = " << this->width << endl;
	cout << "<ScreenTexture Format> = " << this->height << endl;
}