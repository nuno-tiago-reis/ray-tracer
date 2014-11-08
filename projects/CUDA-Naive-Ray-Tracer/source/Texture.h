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

/* Texture Library */
#include "soil.h"

/* C++ Includes */
#include <string>

/* Error Check */
#include "Utility.h"

class Texture {

	protected:

		/* Texture Name */
		string name;

		/* OpenGL Texture Handler */
		GLuint handler;

	public:

		/* Constructors & Destructors */
		Texture(string name);

		~Texture();

		/* Loading Methods */
		virtual void createTexture() = 0;
		virtual void deleteTexture();

		/* Getters */
		string getName();

		GLuint getHandler();

		/* Setters */
		void setName(string name);

		void setHandler(GLuint handler);

		/* Debug Methods */
		void dump();
};

#endif