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

/* Generic Texture */
#include "Texture.h"
/* Generic Shader */
#include "ShaderProgram.h"

/* Mesh Reader */
#include "OBJ_Reader.h"

using namespace std;

typedef struct {

	GLfloat ambient[4];
	GLfloat diffuse[4];
	GLfloat specular[4];
	GLfloat specularConstant;

	GLint materialID;

} MaterialStructure;

class Material {

	protected:

		/* Material Identifier */
		string name;

		/* Material Ambient, Diffuse and Specular Properties */
		Vector ambient;
		Vector diffuse;
		Vector specular;

		/* Material Specular Constant */
		float specularConstant;
		
		/* Material Shader Program */
		ShaderProgram* shaderProgram;

		/* Material Texture Map */
		map<string,Texture*> textureMap;

	public:

		/* Constructors & Destructors */
		Material(string name, string materialFilename, ShaderProgram* shaderProgram);
		~Material();

		/* Scene Methods */
		void bind();
		void unbind();

		/* Texture Map Manipulation Methods*/
		void addTexture(Texture* texture);
		void removeTexture(string textureName);

		Texture* getTexture(string textureName);
		
		/* GPU Methods */
		MaterialStructure getMaterialStructure();

		/* Getters */
		string getName();

		Vector getAmbient();
		Vector getDiffuse();
		Vector getSpecular();

		float getSpecularConstant();

		ShaderProgram* getShaderProgram();

		/* Setters */
		void setName(string name);

		void setAmbient(Vector ambient);
		void setDiffuse(Vector diffuse);
		void setSpecular(Vector specular);

		void setSpecularConstant(float specularConstant);

		void setShaderProgram(ShaderProgram* shaderProgram);

		/* Debug Methods */
		void dump();
};

#endif