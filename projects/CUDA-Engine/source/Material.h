#ifndef MATERIAL_H
#define MATERIAL_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/freeglut.h"
/* OpenGL Error check */
#include "Utility.h"

/* C++ Includes */
#include <string>
#include <map>

/* Math Library */
#include "Vector.h"

/* Generic Shader */
#include "ShaderProgram.h"

/* Generic Texture */
#include "Texture.h"

using namespace std;

class Material {

	protected:

		/* Material Identifier */
		string name;

		/* Material Shader Program */
		ShaderProgram* shaderProgram;

		/* Material Texture Map */
		map<string,Texture*> textureMap;

	public:

		/* Constructors & Destructors */
		Material(string name, ShaderProgram* shaderProgram);
		~Material();

		/* Scene Methods */
		void bind();
		void unbind();

		/* Getters */
		string getName();

		ShaderProgram* getShaderProgram();

		/* Setters */
		void setName(string name);

		void setShaderProgram(ShaderProgram* shaderProgram);

		/* Texture Map Manipulation Methods*/
		void addTexture(Texture* texture);
		void removeTexture(string textureName);

		Texture* getTexture(string textureName);

		/* Debug Methods */
		void dump();
};

#endif