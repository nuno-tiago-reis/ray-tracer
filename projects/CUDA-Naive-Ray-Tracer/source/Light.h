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

class Light {

	protected:

		/* Light Name */
		string name;

		/* Spot Light and Positional Light Position */
		Vector position;
		/* Spot Light and Directional Light Direction */
		Vector direction;
		/* Spot Light Cut-Off */
		GLfloat cutOff;

		/* Light Color and Intensity */
		Vector color;

		float diffuseIntensity;
		float specularIntensity;

		/* Spot Light and Positional Light Attenuatin */
		float constantAttenuation;
		float linearAttenuation;
		float exponentialAttenuation;

	public:

		/* Constructors & Destructors */
		Light(string name);
		~Light();

		/* Getters & Setters */
		string getName();

		Vector getPosition();
		Vector getDirection();

		float getCutOff();

		Vector getColor();

		float getDiffuseIntensity();
		float getSpecularIntensity();

		float getConstantAttenuation();
		float getLinearAttenuation();
		float getExponentinalAttenuation();

		/* Setters */
		void setName(string name);

		void setPosition(Vector position);
		void setDirection(Vector direction);

		void setCutOff(GLfloat cutOff);

		void setColor(Vector color);

		void setDiffuseIntensity(float diffuseIntensity);
		void setSpecularIntensity(float specularIntensity);

		void setConstantAttenuation(float constantAttenuation);
		void setLinearAttenuation(float linearAttenuation);
		void setExponentialAttenuation(float exponentialAttenuation);

		void dump();
};

#endif