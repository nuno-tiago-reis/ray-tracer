#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(GLint width, GLint height) {

	this->width = width;
	this->height = height;
	
	// Initialize the OpenGL Texture Handlers
	this->frameBufferHandler = UINT_MAX;
	this->depthBufferHandler = UINT_MAX;

	this->diffuseTextureHandler = UINT_MAX;
	this->specularTextureHandler = UINT_MAX;

	this->fragmentPositionTextureHandler = UINT_MAX;
	this->fragmentNormalTextureHandler = UINT_MAX;

	// Initialize the CUDA Array references
	this->diffuseTextureCudaArray = NULL;
	this->specularTextureCudaArray = NULL;

	this->fragmentPositionCudaArray = NULL;
	this->fragmentNormalCudaArray = NULL;
	
	// Initialize the CUDA Graphics Resource references
	this->diffuseTextureCudaGraphicsResource = NULL;
	this->specularTextureCudaGraphicsResource = NULL;
		
	this->fragmentPositionCudaGraphicsResource = NULL;
	this->fragmentNormalCudaGraphicsResource = NULL;
}

FrameBuffer::~FrameBuffer() {
	
	// Delete the FrameBuffer in case it already exists.
	this->deleteFrameBuffer();
}

void FrameBuffer::createFrameBuffer() {

	// Delete the FrameBuffer in case it already exists.
	this->deleteFrameBuffer();
	
	// Generate the Diffuse Texture
	glGenTextures(1, &this->diffuseTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Diffuse Texture.");

    glBindTexture(GL_TEXTURE_2D, this->diffuseTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Diffuse Texture
	glGenTextures(1, &this->specularTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Specular Texture.");

    glBindTexture(GL_TEXTURE_2D, this->specularTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Fragment Position Texture
	glGenTextures(1, &this->fragmentPositionTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Fragment Position Texture.");

    glBindTexture(GL_TEXTURE_2D, this->fragmentPositionTextureHandler);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

	// Generate the Fragment Normal Texture
	glGenTextures(1, &this->fragmentNormalTextureHandler);
	Utility::checkOpenGLError("ERROR: Failed to create the Fragment Normal Texture.");

    glBindTexture(GL_TEXTURE_2D, this->fragmentNormalTextureHandler);

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

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->diffuseTextureHandler, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, this->specularTextureHandler, 0);		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, this->fragmentPositionTextureHandler, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, this->fragmentNormalTextureHandler, 0);

		Utility::checkOpenGLError("ERROR: Failed to attached the Textures to the FrameBuffer.");

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->depthBufferHandler);
		
		Utility::checkOpenGLError("ERROR: Failed to attached the Depth Buffer to the FrameBuffer.");

		GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 }; 
		glDrawBuffers(4, drawBuffers);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->diffuseTextureCudaGraphicsResource, this->diffuseTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->specularTextureCudaGraphicsResource, this->specularTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->fragmentPositionCudaGraphicsResource, this->fragmentPositionTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&this->fragmentNormalCudaGraphicsResource, this->fragmentNormalTextureHandler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

void FrameBuffer::deleteFrameBuffer() {

	// Delete the FrameBuffer from OpenGL 
	glDeleteFramebuffers(1, &this->frameBufferHandler);
	// Delete the DepthBuffer from OpenGL 
	glDeleteRenderbuffers(1, &this->depthBufferHandler);

	// Delete the Diffuse Texture from OpenGL 
	glDeleteTextures(1, &this->diffuseTextureHandler);
	// Delete the Specular Texture from OpenGL 
	glDeleteTextures(1, &this->specularTextureHandler);
	// Delete the Fragment Position Texture from OpenGL 
	glDeleteTextures(1, &this->fragmentPositionTextureHandler);
	// Delete the Fragment Normal Texture from OpenGL 
	glDeleteTextures(1, &this->fragmentNormalTextureHandler);

	Utility::checkOpenGLError("deleteFrameBuffer()");
}

void FrameBuffer::reshape(GLint width, GLint height) {

	this->width = width;
	this->height = height;

	// Reshape the Render Texture
	glBindTexture(GL_TEXTURE_2D, this->diffuseTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Origin Texture
	glBindTexture(GL_TEXTURE_2D, this->specularTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Reflection Texture
	glBindTexture(GL_TEXTURE_2D, this->fragmentPositionTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Reshape the Ray Refraction Texture
	glBindTexture(GL_TEXTURE_2D, this->fragmentNormalTextureHandler);
	
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
 
	// Reshape the Depth Buffer
	glBindRenderbuffer(GL_RENDERBUFFER, this->depthBufferHandler);

		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, this->width, this->height);

	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void FrameBuffer::bindCudaResources() {

	// Bind the CUDA Arrays to the Resources
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->diffuseTextureCudaArray, this->diffuseTextureCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->specularTextureCudaArray, this->specularTextureCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->fragmentPositionCudaArray, this->fragmentPositionCudaGraphicsResource, 0, 0));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&this->fragmentNormalCudaArray, this->fragmentNormalCudaGraphicsResource, 0, 0));
}

void FrameBuffer::mapCudaResource() {

	// Map the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->diffuseTextureCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->specularTextureCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->fragmentPositionCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &this->fragmentNormalCudaGraphicsResource, 0));

	this->bindCudaResources();
}

void FrameBuffer::unmapCudaResource() {

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->diffuseTextureCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->specularTextureCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->fragmentPositionCudaGraphicsResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &this->fragmentNormalCudaGraphicsResource, 0));
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

GLuint FrameBuffer::getDiffuseTextureHandler() {

	return this->diffuseTextureHandler;
}

GLuint FrameBuffer::getSpecularTextureHandler() {

	return this->specularTextureHandler;
}

GLuint FrameBuffer::getFragmentPositionTextureHandler() {

	return this->fragmentPositionTextureHandler;
}

GLuint FrameBuffer::getFragmentNormalTextureHandler() {

	return this->fragmentNormalTextureHandler;
}
		
cudaArray* FrameBuffer::getDiffuseTextureCudaArray() {

	return this->diffuseTextureCudaArray;
}
cudaArray* FrameBuffer::getSpecularTextureCudaArray() {

	return this->specularTextureCudaArray;
}

cudaArray* FrameBuffer::getFragmentPositionCudaArray() {

	return this->fragmentPositionCudaArray;
}

cudaArray* FrameBuffer::getFragmentNormalCudaArray() {

	return this->fragmentNormalCudaArray;
}

cudaGraphicsResource* FrameBuffer::getDiffuseTextureCudaGraphicsResource() {

	return this->diffuseTextureCudaGraphicsResource;
}

cudaGraphicsResource* FrameBuffer::getSpecularTextureCudaGraphicsResource() {

	return this->specularTextureCudaGraphicsResource;
}

cudaGraphicsResource* FrameBuffer::getFragmentPositionCudaGraphicsResource() {

	return this->fragmentPositionCudaGraphicsResource;
}

cudaGraphicsResource* FrameBuffer::getFragmentNormalCudaGraphicsResource() {

	return this->fragmentNormalCudaGraphicsResource;
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

void FrameBuffer::setDiffuseTextureHandler(GLuint diffuseTextureHandler) {

	this->diffuseTextureHandler = diffuseTextureHandler;
}

void FrameBuffer::setSpecularTextureHandler(GLuint specularTextureHandler) {

	this->specularTextureHandler = specularTextureHandler;
}
	
void FrameBuffer::setFragmentPositionTextureHandler(GLuint fragmentPositionTextureHandler) {

	this->fragmentPositionTextureHandler = fragmentPositionTextureHandler;
}

void FrameBuffer::setFragmentNormalTextureHandler(GLuint fragmentNormalTextureHandler) {

	this->fragmentNormalTextureHandler = fragmentNormalTextureHandler;
}
		
void FrameBuffer::setDiffuseTextureCudaArray(cudaArray* diffuseTextureCudaArray) {

	this->diffuseTextureCudaArray = diffuseTextureCudaArray;
}

void FrameBuffer::setSpecularTextureCudaArray(cudaArray* specularTextureCudaArray) {

	this->specularTextureCudaArray = specularTextureCudaArray;
}

void FrameBuffer::setFragmentPositionCudaArray(cudaArray* fragmentPositionCudaArray) {

	this->fragmentPositionCudaArray = fragmentPositionCudaArray;
}

void FrameBuffer::setFragmentNormalCudaArray(cudaArray* fragmentNormalCudaArray) {

	this->fragmentNormalCudaArray = fragmentNormalCudaArray;
}
			
void FrameBuffer::setDiffuseTextureCudaGraphicsResource(cudaGraphicsResource* diffuseTextureCudaGraphicsResource) {

	this->diffuseTextureCudaGraphicsResource = diffuseTextureCudaGraphicsResource;
}

void FrameBuffer::setSpecularTextureCudaGraphicsResource(cudaGraphicsResource* specularTextureCudaGraphicsResource) {

	this->specularTextureCudaGraphicsResource = specularTextureCudaGraphicsResource;
}

void FrameBuffer::setFragmentPositionCudaGraphicsResource(cudaGraphicsResource* fragmentPositionCudaGraphicsResource) {

	this->fragmentPositionCudaGraphicsResource = fragmentPositionCudaGraphicsResource;
}

void FrameBuffer::setFragmentNormalCudaGraphicsResource(cudaGraphicsResource* fragmentNormalCudaGraphicsResource) {

	this->fragmentNormalCudaGraphicsResource = fragmentNormalCudaGraphicsResource;
}