#ifndef PIXEL_BUFFER_H
#define PIXEL_BUFFER_H

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

class PixelBuffer {

	protected:

		/* OpenGL PixelBuffer Handler */
		unsigned int handler;

		/* PixelBuffer Dimensions */
		unsigned int width;
		unsigned int height;

		/* CUDA Graphics Resource that binds the PixelBuffer */
		cudaGraphicsResource *cudaGraphicsResourceReference;

	public:

		/* Constructors & Destructors */
		PixelBuffer(unsigned int width, unsigned int height);

		~PixelBuffer();

		/* Loading Methods */
		void createPixelBuffer();
		void deletePixelBuffer();

		/* CUDA-OpenGL Interop Methods */
		void mapCudaResource();
		void unmapCudaResource();

		unsigned int* getDevicePointer();

		/* Getters */
		unsigned int getHandler();

		unsigned int getWidth();
		unsigned int getHeight();

		cudaGraphicsResource* getCudaGraphicsResourceReference();

		/* Setters */
		void setHandler(unsigned int handler);

		void setWidth(unsigned int width);
		void setHeight(unsigned int height);

		void setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference);

		/* Debug Methods */
		void dump();
};

#endif