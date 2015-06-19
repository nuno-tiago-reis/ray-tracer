#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(GLint width, GLint height) {

	this->frameBufferHandler = UINT_MAX;
	this->depthBufferHandler = UINT_MAX;

	this->renderTextureHandler = UINT_MAX;

	this->width = width;
	this->height = height;

	/* Initialize the CUDA Graphics Resource*/
	this->cudaGraphicsResourceReference = NULL;
}

FrameBuffer::~FrameBuffer() {
	
	// Delete the FrameBuffer in case it already exists.
	this->deleteFrameBuffer();
}

void FrameBuffer::createFrameBuffer() {

	// Delete the FrameBuffer in case it already exists.
	this->deleteFrameBuffer();
	
	// Generate the Render Texture
	glGenTextures(1, &this->renderTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the RenderTexture.");

    glBindTexture(GL_TEXTURE_2D, this->renderTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Depth Buffer
	glGenRenderbuffers(1, &this->depthBufferHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the DepthBuffer.");

	glBindRenderbuffer(GL_RENDERBUFFER, this->depthBufferHandler);
 
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, this->width, this->height);
 
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	// Generate the FrameBuffer 
	glGenFramebuffers(1, &this->frameBufferHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the FrameBuffer.");

	glBindFramebuffer(GL_FRAMEBUFFER, this->frameBufferHandler);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->renderTextureHandler, 0);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->renderTextureHandler, 0);		
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->renderTextureHandler, 0);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->renderTextureHandler, 0);

		Utility::checkOpenGLError("ERROR: Failed to attached the Textures to the FrameBuffer.");

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->depthBufferHandler);
		
		Utility::checkOpenGLError("ERROR: Failed to attached the Depth Buffer to the FrameBuffer.");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->cudaGraphicsResourceReference, this->renderTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

void FrameBuffer::deleteFrameBuffer() {

	// Delete the FrameBuffer from OpenGL 
	glDeleteFramebuffers(1, &this->frameBufferHandler);
	// Delete the DepthBuffer from OpenGL 
	glDeleteRenderbuffers(1, &this->depthBufferHandler);
	// Delete the RenderTexture from OpenGL 
	glDeleteTextures(1, &this->renderTextureHandler);

	Utility::checkOpenGLError("deleteFrameBuffer()");
}

void FrameBuffer::reshape(GLint width, GLint height) {

	this->width = width;
	this->height = height;

	/* Reshape the Render Texture */
	glBindTexture(GL_TEXTURE_2D, this->renderTextureHandler);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
 
	/* Reshape the Depth Buffer */
	glBindRenderbuffer(GL_RENDERBUFFER, this->depthBufferHandler);

		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);

	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}


void FrameBuffer::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->cudaGraphicsResourceReference, 0));
}

void FrameBuffer::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->cudaGraphicsResourceReference, 0));
}

cudaArray* FrameBuffer::getArrayPointer() {

	// Bind the CUDA Array to the Resource
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->cudaArrayReference, this->cudaGraphicsResourceReference, 0, 0));

	return this->cudaArrayReference;
}

GLint FrameBuffer::getWidth() {

	return this->width;
}

GLint FrameBuffer::getHeight() {

	return this->height;
}

GLuint FrameBuffer::getFrameBufferHandler() {

	return this->frameBufferHandler;
}

GLuint FrameBuffer::getDepthBufferHandler() {

	return this->depthBufferHandler;
}

GLuint FrameBuffer::getRenderTextureHandler() {

	return this->renderTextureHandler;
}

cudaArray* FrameBuffer::getCudaArray() {

	return this->cudaArrayReference;
}

cudaGraphicsResource* FrameBuffer::getCudaGraphicsResource() {

	return this->cudaGraphicsResourceReference;
}

void FrameBuffer::setWidth(GLint width) {

	this->width = width;
}

void FrameBuffer::setHeight(GLint height) {

	this->height = height;
}

void FrameBuffer::setFrameBufferHandler(GLuint frameBufferHandler) {

	this->frameBufferHandler = frameBufferHandler;
}

void FrameBuffer::setDepthBufferHandler(GLuint depthBufferHandler) {

	this->depthBufferHandler = depthBufferHandler;
}

void FrameBuffer::setRenderTextureHandler(GLuint renderTextureHandler) {

	this->renderTextureHandler = renderTextureHandler;
}

void FrameBuffer::setCudaArray(cudaArray* cudaArrayReference) {

	this->cudaArrayReference = cudaArrayReference;
}

void FrameBuffer::setCudaGraphicsResource(cudaGraphicsResource* cudaGraphicsResourceReference) {

	this->cudaGraphicsResourceReference = cudaGraphicsResourceReference;
}