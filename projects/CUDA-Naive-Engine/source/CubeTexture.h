#ifndef CUBE_TEXTURE_H
#define CUBE_TEXTURE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Texture */
#include "Texture.h"

/* C++ Includes */
#include <map>

/* Cube Texture Definitions */
#define POSITIVE_X "Positive X"
#define NEGATIVE_X "Negative X"

#define POSITIVE_Y "Positive Y"
#define NEGATIVE_Y "Negative Y"

#define POSITIVE_Z "Positive Z"
#define NEGATIVE_Z "Negative Z"

class CubeTexture : public Texture {

	private:

		/* Texture FileName */
		map<string,string> textureFilenameMap;

	public:

		/* Constructors & Destructors */
		CubeTexture(
			string name,
			string positiveXFileName, string negativeXFileName, string positiveYFileName,
			string negativeYFileName, string positiveZFileName,string negativeZFileName,
			string uniform);

		~CubeTexture();

		/* Loading Methods */
		void createTexture();

		/* Bind & Unbind to OpenGL Methods */
		void bind(GLuint textureID);
		void unbind(GLuint textureID);
};

#endif