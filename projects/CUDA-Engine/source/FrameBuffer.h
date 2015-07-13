#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"

/* CUDA definitions */
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

/* Error Check */
#include "Utility.h"

class FrameBuffer {

	private:

		/* FrameBuffers Dimensions */
		GLint width;
		GLint height;
		
		// FrameBuffers Handler
		GLuint frameBufferHandler;
		GLuint depthBufferHandler;
		
		// FrameBuffers Rendering Textures
		GLuint diffuseTextureHandler;
		GLuint specularTextureHandler;

		// FrameBuffers Ray-Tracing Textures
		GLuint fragmentPositionTextureHandler;
		GLuint fragmentNormalTextureHandler;
		
		/* CUDA Array references that store the Textures in CUDA */
		cudaArray *diffuseTextureCudaArray;
		cudaArray *specularTextureCudaArray;

		cudaArray *fragmentPositionCudaArray;
		cudaArray *fragmentNormalCudaArray;

		/* CUDA Graphics Resource references that bind the Textures to CUDA */
		cudaGraphicsResource *diffuseTextureCudaGraphicsResource;
		cudaGraphicsResource *specularTextureCudaGraphicsResource;
		
		cudaGraphicsResource *fragmentPositionCudaGraphicsResource;
		cudaGraphicsResource *fragmentNormalCudaGraphicsResource;

	public:

		/* Constructors & Destructors */
		FrameBuffer(GLint width, GLint height);
		~FrameBuffer();

		/* Scene Methods */
		void createFrameBuffer();
		void deleteFrameBuffer();

		void reshape(GLint width, GLint height);

		// CUDA-OpenGL Interop Methods
		void bindCudaResources();

		void mapCudaResource();
		void unmapCudaResource();

		/* Getters */
		GLint getWidth();
		GLint getHeight();

		GLuint getFrameBufferHandler();
		GLuint getDepthBufferHandler();

		GLuint getDiffuseTextureHandler();
		GLuint getSpecularTextureHandler();

		GLuint getFragmentPositionTextureHandler();
		GLuint getFragmentNormalTextureHandler();
		
		cudaArray* getDiffuseTextureCudaArray();
		cudaArray* getSpecularTextureCudaArray();

		cudaArray* getFragmentPositionCudaArray();
		cudaArray* getFragmentNormalCudaArray();

		cudaGraphicsResource* getDiffuseTextureCudaGraphicsResource();
		cudaGraphicsResource* getSpecularTextureCudaGraphicsResource();

		cudaGraphicsResource* getFragmentPositionCudaGraphicsResource();
		cudaGraphicsResource* getFragmentNormalCudaGraphicsResource();

		/* Setters */
		void setWidth(GLint width);
		void setHeight(GLint height);

		void setFrameBufferHandler(GLuint frameBufferHandler);
		void setDepthBufferHandler(GLuint depthBufferHandler);

		void setDiffuseTextureHandler(GLuint diffuseTextureHandler);
		void setSpecularTextureHandler(GLuint specularTextureHandler);
	
		void setFragmentPositionTextureHandler(GLuint fragmentPositionTextureHandler);
		void setFragmentNormalTextureHandler(GLuint fragmentNormalTextureHandler);
		
		void setDiffuseTextureCudaArray(cudaArray* diffuseTextureCudaArray);
		void setSpecularTextureCudaArray(cudaArray* specularTextureCudaArray);

		void setFragmentPositionCudaArray(cudaArray* fragmentPositionCudaArray);
		void setFragmentNormalCudaArray(cudaArray* fragmentNormalCudaArray);
			
		void setDiffuseTextureCudaGraphicsResource(cudaGraphicsResource* diffuseTextureCudaGraphicsResource);
		void setSpecularTextureCudaGraphicsResource(cudaGraphicsResource* specularTextureCudaGraphicsResource);

		void setFragmentPositionCudaGraphicsResource(cudaGraphicsResource* fragmentPositionCudaGraphicsResource);
		void setFragmentNormalCudaGraphicsResource(cudaGraphicsResource* fragmentNormalCudaGraphicsResource);
};

#endif