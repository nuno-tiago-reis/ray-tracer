#ifndef BUFFER_OBJECT_H
#define BUFFER_OBJECT_H

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

class BufferObject {

	protected:

		/* OpenGL BufferObject Handler */
		unsigned int handler;

		/* BufferObject Dimensions */
		unsigned int width;
		unsigned int height;

		/* CUDA Graphics Resource that binds the BufferObject */
		cudaGraphicsResource *cudaGraphicsResourceReference;

	public:

		/* Constructors & Destructors */
		BufferObject(unsigned int width, unsigned int height);

		~BufferObject();

		/* Loading Methods */
		void createBufferObject();
		void deleteBufferObject();

		/* Copy to OpenGL Method */
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