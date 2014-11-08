#include "BufferObject.h"

BufferObject::BufferObject(unsigned int width, unsigned int height) {

	/* Initialize the BufferObjects Handler */
	this->handler = UINT_MAX;

	/* Initialize the BufferObjects Dimensions */
	this->width = width;
	this->height = height;

	/* Initialize the BufferObjects CUDA Graphics Resource*/
	this->cudaGraphicsResourceReference = NULL;
}

BufferObject::~BufferObject() {
}

void BufferObject::createBufferObject() {

	unsigned int size = sizeof(GLubyte) * this->width * this->height * 4; //4 component RGBA

	// Delete the PixelBufferObject in case it already exists.
	glDeleteBuffers(1, &this->handler);
	Utility::checkOpenGLError("glDeleteBuffers()");

	// Create the PixelBufferObject to output the Ray-Tracing result.
	glGenBuffers(1, &this->handler);
	Utility::checkOpenGLError("glGenBuffers()");

	glBindBuffer(GL_ARRAY_BUFFER, this->handler);

		glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
		Utility::checkOpenGLError("glBufferData()");

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register the PixelBufferObject with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterBuffer()", cudaGraphicsGLRegisterBuffer(&this->cudaGraphicsResourceReference, this->handler, cudaGraphicsMapFlagsWriteDiscard));
}

void BufferObject::deleteBufferObject() {

	// Delete the BufferObject from OpenGL 
    glDeleteBuffers(1, &this->handler);
	Utility::checkOpenGLError("glDeleteBuffers()");
}

void BufferObject::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &cudaGraphicsResourceReference, 0));
}

void BufferObject::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &cudaGraphicsResourceReference, 0));
}

unsigned int* BufferObject::getDevicePointer() {

	unsigned int* devicePointer = NULL;

	Utility::checkCUDAError("cudaGraphicsResourceGetMappedPointer()", cudaGraphicsResourceGetMappedPointer((void**)&devicePointer, 0, cudaGraphicsResourceReference));

	return devicePointer;
}

unsigned int BufferObject::getHandler() {

	return this->handler;
}

unsigned int BufferObject::getWidth() {

	return this->width;
}

unsigned int BufferObject::getHeight() {

	return this->height;
}

cudaGraphicsResource* BufferObject::getCudaGraphicsResourceReference() {

	return this->cudaGraphicsResourceReference;
}

void BufferObject::setHandler(unsigned int handler) {

	this->handler = handler;
}

void BufferObject::setWidth(unsigned int width) {

	this->width = width;
}

void BufferObject::setHeight(unsigned int height) {

	this->height = height;
}

void BufferObject::setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference) {

	this->cudaGraphicsResourceReference = cudaGraphicsResourceReference;
}

void BufferObject::dump() {

	cout << "<BufferObject Dump>" << endl;

	/* BufferObjects Handler */
	cout << "<BufferObject Handler> = " << this->handler << endl;

	/* BufferObjects Dimensions */
	cout << "<BufferObject Width> = " << this->width << endl;
	cout << "<BufferObject Format> = " << this->height << endl;
}