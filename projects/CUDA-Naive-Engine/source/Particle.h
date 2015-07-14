#ifndef PARTICLE_H
#define PARTICLE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Object */
#include "Object.h"

/* Scene Manager */
#include "SceneManager.h"

using namespace std;

class Particle : public Object {

	protected:

		/* Particles Lifetime */
		GLfloat lifetime;
		GLfloat maximumLifetime;

	public:

		/* Constructors & Destructors */
		Particle(string name);
		~Particle();

		/* Scene Methods */
		void draw();

		void reset(Vector scale);

		void update();
		void update(GLfloat elapsedTime);

		/* Getters */
		GLfloat getLifetime();
		GLfloat getMaximumLifetime();

		/* Setters */
		void setLifetime(GLfloat lifetime);
		void setMaximumLifetime(GLfloat maximumLifetime);

		/* Debug Methods */
		void dump();
};

#endif