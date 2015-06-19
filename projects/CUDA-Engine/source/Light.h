#ifndef LIGHT_H
#define	LIGHT_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* C++ Includes */
#include <string>

/* Math Library */
#include "Vector.h"

/* Generic Shader Program */
#include "ShaderProgram.h"

/* Light Source Identifiers */
#define LIGHT_SOURCE_0 0
#define LIGHT_SOURCE_1 1
#define LIGHT_SOURCE_2 2
#define LIGHT_SOURCE_3 3
#define LIGHT_SOURCE_4 4

class Light {

	protected:

		/* Light Name */
		string name;

		/* Light Identifier (0-4) */
		GLint identifier;

		/* Light Attributes */
		Vector position;
		Vector color;

		GLfloat ambientIntensity;
		GLfloat diffuseIntensity;
		GLfloat specularIntensity;

		/* Shader Uniform Buffer Index */
		GLuint uniformBufferIndex;

	public:

		/* Constructors & Destructors */
		Light(string name);
		~Light();

		/* Scene Methods */
		virtual void loadUniforms();

		/* Getters & Setters */
		string getName();

		GLint getIdentifier();

		Vector getPosition();
		Vector getColor();

		GLfloat getAmbientIntensity();
		GLfloat getDiffuseIntensity();
		GLfloat getSpecularIntensity();

		GLuint getUniformBufferIndex();

		/* Setters */
		void setName(string name);

		void setIdentifier(GLint identifier);

		void setPosition(Vector position);
		void setColor(Vector color);

		void setAmbientIntensity(GLfloat ambientIntensity);
		void setDiffuseIntensity(GLfloat diffuseIntensity);
		void setSpecularIntensity(GLfloat specularIntensity);

		void setUniformBufferIndex(GLuint uniformBufferIndex);

		void dump();
};

#endif