#ifndef POSITIONAL_LIGHT_H
#define	POSITIONAL_LIGHT_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Light Source */
#include "Light.h"

/* Positional Light Identifier */
#define POSITIONAL_LIGHT 2

/* Positional Light Names */
#define POSITIONAL_LIGHT_0 "Positional Light 0"
#define POSITIONAL_LIGHT_1 "Positional Light 1"
#define POSITIONAL_LIGHT_2 "Positional Light 2"
#define POSITIONAL_LIGHT_3 "Positional Light 3"
#define POSITIONAL_LIGHT_4 "Positional Light 4"

class PositionalLight: public Light {

	private:

		/* Positional Light Attributes */
		GLfloat constantAttenuation;
		GLfloat linearAttenuation;
		GLfloat exponentialAttenuation;

	public:

		/* Constructors & Destructors */
		PositionalLight(string name);
		~PositionalLight();

		/* Scene Methods */
		void loadUniforms();

		/* Getters */
		GLfloat getConstantAttenuation();
		GLfloat getLinearAttenuation();
		GLfloat getExponentinalAttenuation();

		/* Setters */
		void setConstantAttenuation(GLfloat constantAttenuation);
		void setLinearAttenuation(GLfloat linearAttenuation);
		void setExponentialAttenuation(GLfloat exponentialAttenuation);

		void dump();
};

#endif