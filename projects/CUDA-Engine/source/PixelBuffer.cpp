#include "PixelBuffer.h"

PixelBuffer::PixelBuffer(unsigned int width, unsigned int height) {

	/* Initialize the PixelBuffers Handler */
	this->handler = UINT_MAX;

	/* Initialize the PixelBuffers Dimensions */
	this->width = width;
	this->height = height;

	/* Initialize the PixelBuffers CUDA Graphics Resource*/
	this->cudaGraphicsResourceReference = NULL;
}

PixelBuffer::~PixelBuffer() {
	
	// Delete the PixelBuffer in case it already exists.
	this->deletePixelBuffer();
}

void PixelBuffer::createPixelBuffer() {
	
	// Delete the PixelBuffer in case it already exists.
	this->deletePixelBuffer();

	unsigned int size = sizeof(GLubyte) * this->width * this->height * 4; //4 component RGBA

	// Create the PixelPixelBuffer to output the Ray-Tracing result.
	glGenBuffers(1, &this->handler);
	Utility::checkOpenGLError("glGenBuffers()");

	glBindBuffer(GL_ARRAY_BUFFER, this->handler);

		glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
		Utility::checkOpenGLError("glBufferData()");

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register the PixelBuffer with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterBuffer()", cudaGraphicsGLRegisterBuffer(&this->cudaGraphicsResourceReference, this->handler, cudaGraphicsMapFlagsWriteDiscard));
}

void PixelBuffer::deletePixelBuffer() {

	// Delete the PixelBuffer from OpenGL 
    glDeleteBuffers(1, &this->handler);
	Utility::checkOpenGLError("glDeleteBuffers()");
}

void PixelBuffer::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &cudaGraphicsResourceReference, 0));
}

void PixelBuffer::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &cudaGraphicsResourceReference, 0));
}

unsigned int* PixelBuffer::getDevicePointer() {

	unsigned int* devicePointer = NULL;

	Utility::checkCUDAError("cudaGraphicsResourceGetMappedPointer()", cudaGraphicsResourceGetMappedPointer((void**)&devicePointer, 0, cudaGraphicsResourceReference));

	return devicePointer;
}

unsigned int PixelBuffer::getHandler() {

	return this->handler;
}

unsigned int PixelBuffer::getWidth() {

	return this->width;
}

unsigned int PixelBuffer::getHeight() {

	return this->height;
}

cudaGraphicsResource* PixelBuffer::getCudaGraphicsResourceReference() {

	return this->cudaGraphicsResourceReference;
}

void PixelBuffer::setHandler(unsigned int handler) {

	this->handler = handler;
}

void PixelBuffer::setWidth(unsigned int width) {

	this->width = width;
}

void PixelBuffer::setHeight(unsigned int height) {

	this->height = height;
}

void PixelBuffer::setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference) {

	this->cudaGraphicsResourceReference = cudaGraphicsResourceReference;
}

void PixelBuffer::dump() {

	cout << "<PixelBuffer Dump>" << endl;

	/* PixelBuffers Handler */
	cout << "<PixelBuffer Handler> = " << this->handler << endl;

	/* PixelBuffers Dimensions */
	cout << "<PixelBuffer Width> = " << this->width << endl;
	cout << "<PixelBuffer Format> = " << this->height << endl;
}