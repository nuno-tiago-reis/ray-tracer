#ifndef SCREEN_TEXTURE_H
#define SCREEN_TEXTURE_H

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

class ScreenTexture {

	protected:

		/* Texture Name */
		string name;

		/* OpenGL Texture Handler */
		unsigned int handler;

		/* Texture Dimensions */
		unsigned int width;
		unsigned int height;

	public:

		/* Constructors & Destructors */
		ScreenTexture(string name, unsigned int width, unsigned int height);

		~ScreenTexture();

		/* Loading Methods */
		void createTexture();
		void deleteTexture();

		/* Getters */
		string getName();

		unsigned int getHandler();

		unsigned int getWidth();
		unsigned int getHeight();

		/* Setters */
		void setName(string name);

		void setHandler(unsigned int handler);

		void setWidth(unsigned int width);
		void setHeight(unsigned int height);

		/* Debug Methods */
		void dump();
};

#endif