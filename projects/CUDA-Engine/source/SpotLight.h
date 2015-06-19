#ifndef SPOT_LIGHT_H
#define	SPOT_LIGHT_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Light Source */
#include "Light.h"

/* Spotlight Identifier */
#define SPOT_LIGHT 1

/* Spotlight Names */
#define SPOT_LIGHT_0 "Spot Light 0"
#define SPOT_LIGHT_1 "Spot Light 1"
#define SPOT_LIGHT_2 "Spot Light 2"
#define SPOT_LIGHT_3 "Spot Light 3"
#define SPOT_LIGHT_4 "Spot Light 4"

class SpotLight: public Light {

	private:

		/* Spot Light Attributes */
		Vector direction;

		GLfloat cutOff;

		GLfloat constantAttenuation;
		GLfloat linearAttenuation;
		GLfloat exponentialAttenuation;

	public:

		/* Constructors & Destructors */
		SpotLight(string name);
		~SpotLight();

		/* Scene Methods */
		void loadUniforms();

		/* Getters */
		Vector getDirection();

		GLfloat getCutOff();

		GLfloat getConstantAttenuation();
		GLfloat getLinearAttenuation();
		GLfloat getExponentinalAttenuation();

		/* Setters */
		void setDirection(Vector direction);

		void setCutOff(GLfloat cutOff);

		void setConstantAttenuation(GLfloat constantAttenuation);
		void setLinearAttenuation(GLfloat linearAttenuation);
		void setExponentialAttenuation(GLfloat exponentialAttenuation);

		void dump();
};

#endif