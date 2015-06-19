#ifndef TEXTURE_H
#define TEXTURE_H

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

/* Texture Library */
#include "soil.h"

/* C++ Includes */
#include <string>

class Texture {

	protected:

		/* Texture Name */
		string name;
		/* Texture Filename */
		string filename;

		/* OpenGL Texture Handler */
		GLuint handler;
		/* OpenGL Texture Format (ex. GL_TEXTURE_2D) */
		GLenum format;
		/* OpenGL Texture Shader Uniform */
		string uniform;

		/* CUDA Array that stores the Texture */
		cudaArray *cudaArrayReference;
		/* CUDA Graphics Resource that binds the texture */
		cudaGraphicsResource *cudaGraphicsResourceReference;

	public:

		/* Constructors & Destructors */
		Texture(string name, GLuint format, string uniform, string fileName);
		Texture(string name, GLuint format, string uniform);

		~Texture();

		/* Loading Methods */
		virtual void createTexture();
		virtual void deleteTexture();

		virtual void loadUniforms(GLuint programID, GLuint textureID);

		/* Bind & Unbind to OpenGL Methods */
		virtual void bind(GLuint textureID);
		virtual void unbind(GLuint textureID);

		/* Map & Unmap to CUDA Methods */
		void mapCudaResource();
		void unmapCudaResource();

		cudaArray* getArrayPointer();

		/* Getters */
		string getName();
		string getFilename();

		GLuint getHandler();
		GLenum getFormat();
		string getUniform();
		
		cudaArray* getCudaArrayReference();
		cudaGraphicsResource* getCudaGraphicsResourceReference();

		/* Setters */
		void setName(string name);
		void setFilename(string fileName);

		void setHandler(GLuint handler);
		void setFormat(GLenum format);
		void setUniform(string unifor);

		void setCudaArrayReference(cudaArray* cudaArrayReference);
		void setCudaGraphicsResourceReference(cudaGraphicsResource* cudaGraphicsResourceReference);

		/* Debug Methods */
		void dump();
};

#endif