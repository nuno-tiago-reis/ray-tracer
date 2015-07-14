#include "Texture.h"

Texture::Texture(string name, GLuint format, string uniform, string filename) {

	/* Initialize the Textures Name */
	this->name = name;

	/* Initialize the Texture Format (eg. GL_TEXTURE_2D) */
	this->format = format;
	/* Initialize the Texture Uniform (specified by the Shader Program) */
	this->uniform = uniform;

	/* Initialize the Textures Filname */
	this->filename = filename;

	/* Initialize the Textures CUDA Array and Graphics Resource*/
	this->cudaArrayReference = NULL;
	this->cudaGraphicsResourceReference = NULL;
}

Texture::Texture(string name, GLuint format, string uniform) {

	/* Initialize the Textures Name */
	this->name = name;

	/* Texture Format (eg. GL_TEXTURE_2D) */
	this->format = format;
	/* Texture Uniform - Specified by the Shader Program */
	this->uniform = uniform;

	/* Initialize the Textures CUDA Array and Graphics Resource*/
	this->cudaArrayReference = NULL;
	this->cudaGraphicsResourceReference = NULL;
}

Texture::~Texture() {
}

void Texture::createTexture() {

	/* Load the Texture */
	this->handler = SOIL_load_OGL_texture(this->filename.c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y);

	/* Check for an error during the load process */
	if(this->handler == 0){

		cout << "[SOIL Error] Loading failed. (\"" << this->filename.c_str() << "\": " << SOIL_last_result() << std::endl;

		exit(1);
	}

	Utility::checkOpenGLError("ERROR: Texture \"" + this->name + "\" creation failed.");

	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&cudaGraphicsResourceReference, this->handler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}


void Texture::deleteTexture() {
	
	// Delete the Texture from OpenGL 
	glDeleteTextures(1, &this->handler);
	Utility::checkOpenGLError("ERROR: Texture \"" + this->name + "\" deletion failed.");

	// Free the used CUDA Array
	Utility::checkCUDAError("cudaArrayFree()",  cudaFreeArray(this->cudaArrayReference));
}

void Texture::loadUniforms(GLuint programID, GLuint textureID) {

	/* Load the Texture to the corresponding Uniform */
	glProgramUniform1i(programID, glGetUniformLocation(programID, this->uniform.c_str()), textureID);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->name + "\" loading failed.");
}

void Texture::bind(GLuint textureID) {

	glActiveTexture(textureID);

    glBindTexture(this->format, this->handler);

	glTexParameteri(this->format, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(this->format, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(this->format, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(this->format, GL_TEXTURE_WRAP_T, GL_REPEAT);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->name + "\" binding failed.");
}

void Texture::unbind(GLuint textureID) {

	glBindTexture(this->format, 0);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->name + "\" unbinding failed.");
}

void Texture::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->cudaGraphicsResourceReference, 0));
}

void Texture::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->cudaGraphicsResourceReference, 0));
}

cudaArray* Texture::getArrayPointer() {

	// Bind the CUDA Array to the Resource
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->cudaArrayReference, this->cudaGraphicsResourceReference, 0, 0));

	return cudaArrayReference;
}

string Texture::getName() {

	return this->name;
}

string Texture::getFilename() {

	return this->filename;
}

GLuint Texture::getHandler() {

	return this->handler;
}

GLenum Texture::getFormat() {

	return this->format;
}

string Texture::getUniform() {

	return this->uniform;
}

cudaArray* Texture::getCudaArrayReference() {

	return this->cudaArrayReference;
}

cudaGraphicsResource* Texture::getCudaGraphicsResourceReference() {

	return this->cudaGraphicsResourceReference;
}

void Texture::setName(string name) {

	this->name = name;
}

void Texture::setFilename(string filename) {

	this->filename = filename;
}

void Texture::setHandler(GLuint handler) {

	this->handler = handler;
}

void Texture::setFormat(GLenum format) {

	this->format = format;
}

void Texture::setUniform(string uniform) {

	this->uniform = uniform;
}

void Texture::setCudaArrayReference(cudaArray* cudaArrayReference) {

	this->cudaArrayReference = cudaArrayReference;
}

void Texture::setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference) {

	this->cudaGraphicsResourceReference = cudaGraphicsResourceReference;
}

void Texture::dump() {

	cout << "<Texture \"" << this->name << "\" Dump>" << endl;

	/* Texture Filename */
	cout << "<Texture Filename> = " << this->filename << endl;
	/* Texture Handler */
	cout << "<Texture Handler> = " << this->handler << endl;
	/* Texture Format */
	cout << "<Texture Format> = " << this->format << endl;
	/* Texture Uniform */
	cout << "<Texture Uniform> = " << this->uniform << endl;
}