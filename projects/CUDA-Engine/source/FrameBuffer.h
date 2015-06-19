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

		GLuint renderTextureHandler;

		// FrameBuffers Ray-Tracing Textures
		GLuint originTexture;
		GLuint reflectionDirectionTexture;
		GLuint refractionDirectionTexture;
		
		/* CUDA Array that stores the Texture */
		cudaArray *cudaArrayReference;
		/* CUDA Graphics Resource that binds the FrameBuffer */
		cudaGraphicsResource *cudaGraphicsResourceReference;

	public:

		/* Constructors & Destructors */
		FrameBuffer(GLint width, GLint height);
		~FrameBuffer();

		/* Scene Methods */
		void createFrameBuffer();
		void deleteFrameBuffer();

		void reshape(GLint width, GLint height);

		// CUDA-OpenGL Interop Methods
		void mapCudaResource();
		void unmapCudaResource();

		cudaArray* getArrayPointer();

		/* Getters */
		GLint getWidth();
		GLint getHeight();

		GLuint getFrameBufferHandler();
		GLuint getDepthBufferHandler();

		GLuint getRenderTextureHandler();
		
		cudaArray* getCudaArray();
		cudaGraphicsResource* getCudaGraphicsResource();

		/* Setters */
		void setWidth(GLint width);
		void setHeight(GLint height);

		void setFrameBufferHandler(GLuint frameBufferHandler);
		void setDepthBufferHandler(GLuint depthBufferHandler);

		void setRenderTextureHandler(GLuint renderTextureHandler);
		
		void setCudaArray(cudaArray* cudaArrayReference);
		void setCudaGraphicsResource(cudaGraphicsResource* cudaGraphicsResourceReference);
};

#endif