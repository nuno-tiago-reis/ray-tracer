#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

#include "Utility.h"

class PerlinNoise {

	private:

		static PerlinNoise* instance;

		PerlinNoise();
		~PerlinNoise();

		static void normalize2(GLfloat v[2]);
		static void normalize3(GLfloat v[3]);

	public:

		/* Singleton Methods */
		static PerlinNoise* getInstance();
		static void destroyInstance();

		void init();

		GLfloat noise1(GLfloat arg);
		GLfloat noise2(GLfloat vec[2]);
		GLfloat noise3(GLfloat vec[3]);

		GLfloat PerlinNoise2D(GLfloat x,GLfloat y,GLfloat alpha,GLfloat beta,GLint n);
		GLfloat PerlinNoise3D(GLfloat x,GLfloat y,GLfloat z,GLfloat alpha,GLfloat beta,GLint n);
};

#endif
