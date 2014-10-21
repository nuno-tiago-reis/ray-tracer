#ifndef NFF_PLANE_H
#define NFF_PLANE_H

#include "Object.h"

using namespace std;

class NFF_Plane : public Object {

	protected:

		GLint identifier;

		Vector vertices[3];

	public:

		/* Constructors & Destructors */
		NFF_Plane(GLint identifier);
		~NFF_Plane();

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		void createBoundingBox();

		/* Getters */
		GLint getIdentifier();
		Vector getVertex(int index);

		/* Setters */
		void setIdentifier(GLint identifier);
		void setVertex(Vector vertex, int index);

		void dump();
};

#endif