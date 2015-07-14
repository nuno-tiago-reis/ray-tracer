#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Object */
#include "Object.h"

/* Generic Particle */
#include "Particle.h"

using namespace std;

class ParticleSystem : public Object {

	private:

		/* Particle Attributes */
		GLint particleNumber;
		GLfloat particleMaximumLifetime;

		/* Particle Map */
		map<string,Particle*> particleMap;

	public:
		
		/* Constructors & Destructors */
		ParticleSystem(string name, GLint particleNumber, GLfloat particleMaximumLifetime);
		~ParticleSystem();

		/* Scene Methods */
		void init();

		void draw();

		void update();
		void update(GLfloat elapsedTime);

		/* Getters */
		GLint getParticleNumber();
		GLfloat getParticleMaximumLifetime();

		/* Setters */
		void setParticleNumber(GLint particleNumber);
		void setParticleMaximumLifetime(GLfloat particleLifetime);

		/* Particle Map Manipulation Methods */
		void addParticle(Particle* particle);
		void removeParticle(string particleName);

		Particle* getParticle(string particleName);		

		/* Debug Methods */
		void dump();
};

#endif