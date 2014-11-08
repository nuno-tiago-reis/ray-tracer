#ifndef SHADING_TEXTURE_H
#define SHADING_TEXTURE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Super-Class definitions */
#include "Texture.h"

class ShadingTexture : public Texture {

	protected:

		/* Texture FileName */
		string fileName;

		/* CUDA Array that stores the Texture */
		cudaArray *cudaArrayReference;
		/* CUDA Graphics Resource that binds the texture */
		cudaGraphicsResource *cudaGraphicsResourceReference;

	public:

		/* Constructors & Destructors */
		ShadingTexture(string name, string fileName);

		~ShadingTexture();

		/* Loading Methods */
		void createTexture();
		void deleteTexture();

		/* Map & Unmap to CUDA Methods */
		void mapResources();
		void unmapResources();

		/* Getters */
		string getFileName();

		cudaArray* getCudaArrayReference();
		cudaGraphicsResource* getCudaGraphicsResourceReference();

		/* Setters */
		void setFileName(string fileName);

		void setCudaArrayReference(cudaArray* cudaArrayReference);
		void setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference);

		/* Debug Methods */
		void dump();
};

#endif