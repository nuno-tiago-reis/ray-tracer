#include "ShadingTexture.h"

ShadingTexture::ShadingTexture(string name, string fileName) : Texture(name) {

	/* Initialize the Textures FileName */
	this->fileName = fileName;

	/* Initialize the Textures CUDA Array and Graphics Resource*/
	this->cudaArrayReference = NULL;
	this->cudaGraphicsResourceReference = NULL;
}

ShadingTexture::~ShadingTexture() {
}

void ShadingTexture::createTexture() {

	// Delete the Texture in case it already exists.
	glDeleteTextures(1, &this->handler);

	// Create the Texture to store the Chess Image.
	glGenTextures(1, &this->handler);

	// Load a Sample Texture
	this->handler = SOIL_load_OGL_texture(this->fileName.c_str(), SOIL_LOAD_RGBA, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y);

	// Check for an error during the loading process
	if(this->handler == 0) {

		cout << "[SOIL Error] Loading failed. (\"" << this->fileName << "\": " << SOIL_last_result() << std::endl;

		exit(1);
	}

	// Set the basic Texture parameters
	glBindTexture(GL_TEXTURE_2D, handler);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&cudaGraphicsResourceReference, this->handler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

void ShadingTexture::deleteTexture() {

	Texture::deleteTexture();

	// Free the used CUDA Array
	Utility::checkCUDAError("cudaArrayFree()",  cudaFreeArray(cudaArrayReference));
}

void ShadingTexture::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &cudaGraphicsResourceReference, 0));
}

void ShadingTexture::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &cudaGraphicsResourceReference, 0));
}

cudaArray* ShadingTexture::getArrayPointer() {

	// Bind the CUDA Array to the Resource
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&cudaArrayReference, cudaGraphicsResourceReference, 0, 0));

	return cudaArrayReference;
}

string ShadingTexture::getFileName() {

	return this->fileName;
}

cudaArray* ShadingTexture::getCudaArrayReference() {

	return this->cudaArrayReference;
}

cudaGraphicsResource* ShadingTexture::getCudaGraphicsResourceReference() {

	return this->cudaGraphicsResourceReference;
}

void ShadingTexture::setCudaArrayReference(cudaArray* cudaArrayReference) {

	this->cudaArrayReference = cudaArrayReference;
}

void ShadingTexture::setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference) {

	this->cudaGraphicsResourceReference = cudaGraphicsResourceReference;
}

void ShadingTexture::dump() {

	cout << "<ShadingTexture \"" << this->name << "\" Dump>" << endl;

	/* Texture Handler */
	cout << "<Texture FileName> = " << this->fileName << endl;
}