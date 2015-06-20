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
		GLuint renderTextureHandler;

		// FrameBuffers Ray-Tracing Textures
		GLuint rayOriginTextureHandler;
		GLuint rayReflectionTextureHandler;
		GLuint rayRefractionTextureHandler;
		
		/* CUDA Array references that store the Textures in CUDA */
		cudaArray *renderCudaArray;

		cudaArray *rayOriginCudaArray;
		cudaArray *rayReflectionCudaArray;
		cudaArray *rayRefractionCudaArray;

		/* CUDA Graphics Resource references that bind the Textures to CUDA */
		cudaGraphicsResource *renderCudaGraphicsResource;
		
		cudaGraphicsResource *rayOriginCudaGraphicsResource;
		cudaGraphicsResource *rayReflectionCudaGraphicsResource;
		cudaGraphicsResource *rayRefractionCudaGraphicsResource;

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

		GLuint getRenderTextureHandler();

		GLuint getRayOriginTextureHandler();
		GLuint getRayReflectionTextureHandler();
		GLuint getRayRefractionTextureHandler();
		
		cudaArray* getRenderCudaArray();

		cudaArray* getRayOriginCudaArray();
		cudaArray* getRayReflectionCudaArray();
		cudaArray* getRayRefractionCudaArray();

		cudaGraphicsResource* getRenderCudaGraphicsResource();

		cudaGraphicsResource* getRayOriginCudaGraphicsResource();
		cudaGraphicsResource* getRayReflectionCudaGraphicsResource();
		cudaGraphicsResource* getRayRefractionCudaGraphicsResource();

		/* Setters */
		void setWidth(GLint width);
		void setHeight(GLint height);

		void setFrameBufferHandler(GLuint frameBufferHandler);
		void setDepthBufferHandler(GLuint depthBufferHandler);

		void setRenderTextureHandler(GLuint renderTextureHandler);
	
		void setRayOriginTextureHandler(GLuint rayOriginTextureHandler);
		void setRayReflectionTextureHandler(GLuint rayReflectionTextureHandler);
		void setRayRefractionTextureHandler(GLuint rayRefractionTextureHandler);
		
		void setRenderCudaArray(cudaArray* renderCudaArray);

		void setRayOriginCudaArray(cudaArray* rayOriginCudaArray);
		void setRayReflectionCudaArray(cudaArray* rayReflectionCudaArray);
		void setRayRefractionCudaArray(cudaArray* rayRefractionCudaArray);
			
		void setRenderCudaGraphicsResource(cudaGraphicsResource* renderCudaGraphicsResource);

		void setRayOriginCudaGraphicsResource(cudaGraphicsResource* rayOriginCudaGraphicsResource);
		void setRayReflectionCudaGraphicsResource(cudaGraphicsResource* rayReflectionCudaGraphicsResource);
		void setRayRefractionCudaGraphicsResource(cudaGraphicsResource* rayRefractionCudaGraphicsResource);
};

#endif