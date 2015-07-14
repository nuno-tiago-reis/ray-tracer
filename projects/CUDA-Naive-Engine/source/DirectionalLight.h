#ifndef DIRECTIONAL_LIGHT_H
#define	DIRECTIONAL_LIGHT_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Light Source */
#include "Light.h"

/* Directional Light Identifier */
#define DIRECTIONAL_LIGHT 3

/* Directional Light Names */
#define DIRECTIONAL_LIGHT_0 "Directional Light 0"
#define DIRECTIONAL_LIGHT_1 "Directional Light 1"
#define DIRECTIONAL_LIGHT_2 "Directional Light 2"
#define DIRECTIONAL_LIGHT_3 "Directional Light 3"
#define DIRECTIONAL_LIGHT_4 "Directional Light 4"

class DirectionalLight: public Light {

	private:

		/* Directional Light Attributes */
		Vector direction;

	public:

		/* Constructors & Destructors */
		DirectionalLight(string name);
		~DirectionalLight();

		/* Scene Methods */
		void loadUniforms();

		/* Getters */
		Vector getDirection();

		/* Setters */
		void setDirection(Vector direction);

		void dump();
};

#endif