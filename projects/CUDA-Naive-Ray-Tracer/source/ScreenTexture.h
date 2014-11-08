#ifndef SCREEN_TEXTURE_H
#define SCREEN_TEXTURE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Super-Class definitions */
#include "Texture.h"

class ScreenTexture : public Texture {

	protected:

		/* Texture Dimensions */
		unsigned int width;
		unsigned int height;

	public:

		/* Constructors & Destructors */
		ScreenTexture(string name, unsigned int width, unsigned int height);

		~ScreenTexture();

		/* Loading Methods */
		void createTexture();

		/* Copy to OpenGL Method */
		void replaceTexture();

		/* Getters */
		unsigned int getWidth();
		unsigned int getHeight();

		/* Setters */
		void setWidth(unsigned int width);
		void setHeight(unsigned int height);

		/* Debug Methods */
		void dump();
};

#endif