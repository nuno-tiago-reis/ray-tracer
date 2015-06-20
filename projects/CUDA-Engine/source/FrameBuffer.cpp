#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(GLint width, GLint height) {

	this->width = width;
	this->height = height;

	this->frameBufferHandler = UINT_MAX;
	this->depthBufferHandler = UINT_MAX;

	this->renderTextureHandler = UINT_MAX;

	this->rayOriginTextureHandler = UINT_MAX;
	this->rayReflectionTextureHandler = UINT_MAX;
	this->rayRefractionTextureHandler = UINT_MAX;

	// Initialize the CUDA Array references
	this->renderCudaArray = NULL;

	this->rayOriginCudaArray = NULL;
	this->rayReflectionCudaArray = NULL;
	this->rayRefractionCudaArray = NULL;
	
	// Initialize the CUDA Graphics Resource references
	this->renderCudaGraphicsResource = NULL;
		
	this->rayOriginCudaGraphicsResource = NULL;
	this->rayReflectionCudaGraphicsResource = NULL;
	this->rayRefractionCudaGraphicsResource = NULL;
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
	Utility::checkOpenGLError("ERROR: Failed to create the Render Texture.");

    glBindTexture(GL_TEXTURE_2D, this->renderTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Ray Origin Texture
	glGenTextures(1, &this->rayOriginTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Ray Origin Texture.");

    glBindTexture(GL_TEXTURE_2D, this->rayOriginTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Ray Reflection Texture
	glGenTextures(1, &this->rayReflectionTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Ray Reflection Texture.");

    glBindTexture(GL_TEXTURE_2D, this->rayReflectionTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Ray Refraction Texture
	glGenTextures(1, &this->rayRefractionTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Ray Refraction Texture.");

    glBindTexture(GL_TEXTURE_2D, this->rayRefractionTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

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
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->rayOriginTextureHandler, 0);		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, this->rayReflectionTextureHandler, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, this->rayRefractionTextureHandler, 0);

		Utility::checkOpenGLError("ERROR: Failed to attached the Textures to the FrameBuffer.");

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->depthBufferHandler);
		
		Utility::checkOpenGLError("ERROR: Failed to attached the Depth Buffer to the FrameBuffer.");

		GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 }; 
		glDrawBuffers(4, drawBuffers);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->renderCudaGraphicsResource, this->renderTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->rayOriginCudaGraphicsResource, this->rayOriginTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->rayReflectionCudaGraphicsResource, this->rayReflectionTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->rayRefractionCudaGraphicsResource, this->rayRefractionTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

void FrameBuffer::deleteFrameBuffer() {

	// Delete the FrameBuffer from OpenGL 
	glDeleteFramebuffers(1, &this->frameBufferHandler);
	// Delete the DepthBuffer from OpenGL 
	glDeleteRenderbuffers(1, &this->depthBufferHandler);

	// Delete the Render Texture from OpenGL 
	glDeleteTextures(1, &this->renderTextureHandler);

	// Delete the Ray Origin Texture from OpenGL 
	glDeleteTextures(1, &this->rayOriginTextureHandler);
	// Delete the Ray Reflection Texture from OpenGL 
	glDeleteTextures(1, &this->rayReflectionTextureHandler);
	// Delete the Ray Refraction Texture from OpenGL 
	glDeleteTextures(1, &this->rayRefractionTextureHandler);

	Utility::checkOpenGLError("deleteFrameBuffer()");
}

void FrameBuffer::reshape(GLint width, GLint height) {

	this->width = width;
	this->height = height;

	// Reshape the Render Texture
	glBindTexture(GL_TEXTURE_2D, this->renderTextureHandler);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Origin Texture
	glBindTexture(GL_TEXTURE_2D, this->rayOriginTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Reflection Texture
	glBindTexture(GL_TEXTURE_2D, this->rayReflectionTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Refraction Texture
	glBindTexture(GL_TEXTURE_2D, this->rayRefractionTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
 
	// Reshape the Depth Buffer
	glBindRenderbuffer(GL_RENDERBUFFER, this->depthBufferHandler);

		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, this->width, this->height);

	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void FrameBuffer::bindCudaResources() {

	// Bind the CUDA Arrays to the Resources
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->renderCudaArray, this->renderCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->rayOriginCudaArray, this->rayOriginCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->rayReflectionCudaArray, this->rayReflectionCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->rayRefractionCudaArray, this->rayRefractionCudaGraphicsResource, 0, 0));
}

void FrameBuffer::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->renderCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->rayOriginCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->rayReflectionCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->rayRefractionCudaGraphicsResource, 0));

	this->bindCudaResources();
}

void FrameBuffer::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->renderCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->rayOriginCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->rayReflectionCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->rayRefractionCudaGraphicsResource, 0));
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
		
cudaArray* FrameBuffer::getRenderCudaArray() {

	return this->renderCudaArray;
}

cudaArray* FrameBuffer::getRayOriginCudaArray() {

	return this->rayOriginCudaArray;
}

cudaArray* FrameBuffer::getRayReflectionCudaArray() {

	return this->rayReflectionCudaArray;
}

cudaArray* FrameBuffer::getRayRefractionCudaArray() {

	return this->rayRefractionCudaArray;
}

cudaGraphicsResource* FrameBuffer::getRenderCudaGraphicsResource() {

	return this->renderCudaGraphicsResource;
}

cudaGraphicsResource* FrameBuffer::getRayOriginCudaGraphicsResource() {

	return this->rayOriginCudaGraphicsResource;
} 

cudaGraphicsResource* FrameBuffer::getRayReflectionCudaGraphicsResource() {

	return this->rayReflectionCudaGraphicsResource;
}

cudaGraphicsResource* FrameBuffer::getRayRefractionCudaGraphicsResource() {

	return this->rayRefractionCudaGraphicsResource;
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

void FrameBuffer::setRayOriginTextureHandler(GLuint rayOriginTextureHandler) {

	this->rayOriginTextureHandler = rayOriginTextureHandler;
}
		
void FrameBuffer::setRayReflectionTextureHandler(GLuint rayReflectionTextureHandler) {

	this->rayReflectionTextureHandler = rayReflectionTextureHandler;
}

void FrameBuffer::setRayRefractionTextureHandler(GLuint rayRefractionTextureHandler) {

	this->rayRefractionTextureHandler = rayRefractionTextureHandler;
}

void FrameBuffer::setRenderCudaArray(cudaArray* renderCudaArray) {

	this->renderCudaArray = renderCudaArray;
}

void FrameBuffer::setRayOriginCudaArray(cudaArray* rayOriginCudaArray) {

	this->rayOriginCudaArray = rayOriginCudaArray;
}

void FrameBuffer::setRayReflectionCudaArray(cudaArray* rayReflectionCudaArray) {

	this->rayReflectionCudaArray = rayReflectionCudaArray;
}

void FrameBuffer::setRayRefractionCudaArray(cudaArray* rayRefractionCudaArray) {

	this->rayRefractionCudaArray = rayRefractionCudaArray;
}
			
void FrameBuffer::setRenderCudaGraphicsResource(cudaGraphicsResource* renderCudaGraphicsResource) {

	this->renderCudaGraphicsResource = renderCudaGraphicsResource;
}

void FrameBuffer::setRayOriginCudaGraphicsResource(cudaGraphicsResource* rayOriginCudaGraphicsResource) {

	this->rayOriginCudaGraphicsResource = rayOriginCudaGraphicsResource;
}

void FrameBuffer::setRayReflectionCudaGraphicsResource(cudaGraphicsResource* rayReflectionCudaGraphicsResource) {

	this->rayReflectionCudaGraphicsResource = rayReflectionCudaGraphicsResource;
}

void FrameBuffer::setRayRefractionCudaGraphicsResource(cudaGraphicsResource* rayRefractionCudaGraphicsResource) {

	this->rayRefractionCudaGraphicsResource = rayRefractionCudaGraphicsResource;
}