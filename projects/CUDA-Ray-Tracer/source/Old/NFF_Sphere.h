#ifndef NFF_SPHERE_H
#define NFF_SPHERE_H

#include "Object.h"

using namespace std;

class NFF_Sphere : public Object {

	protected:

		GLfloat radius;

		Vector position;

	public:

		/* Constructors & Destructors */
		NFF_Sphere(GLint identifier);
		~NFF_Sphere();

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		void createBoundingBox();

		/* Getters */
		GLfloat getRadius();
		Vector getPosition();

		/* Setters */
		void setRadius(GLfloat radius);
		void setPosition(Vector position);

		void dump();
};

#endif