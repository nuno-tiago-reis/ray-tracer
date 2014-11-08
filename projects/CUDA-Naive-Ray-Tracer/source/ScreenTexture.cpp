#include "ScreenTexture.h"

ScreenTexture::ScreenTexture(string name, unsigned int width, unsigned int height) : Texture(name) {

	/* Initialize the Textures Dimensions */
	this->width = width;
	this->height = height;
}

ScreenTexture::~ScreenTexture() {
}

void ScreenTexture::createTexture() {

	// Delete the Texture in case it already exists.
	glDeleteTextures(1, &this->handler);

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

void ScreenTexture::replaceTexture() {

	// Copy the Output to the Texture 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->handler);

		glActiveTexture(GL_TEXTURE0);

		glBindTexture(GL_TEXTURE_2D, this->handler);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width, this->height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

unsigned int ScreenTexture::getWidth() {

	return this->width;
}

unsigned int ScreenTexture::getHeight() {

	return this->height;
}

void ScreenTexture::setWidth(unsigned int width) {

	this->width = width;
}

void ScreenTexture::setHeight(unsigned int height) {

	this->height = height;
}

void ScreenTexture::dump() {

	cout << "<ScreenTexture \"" << this->name << "\" Dump>" << endl;

	/* Texture Dimensions */
	cout << "<ScreenTexture Width> = " << this->width << endl;
	cout << "<ScreenTexture Format> = " << this->height << endl;
}