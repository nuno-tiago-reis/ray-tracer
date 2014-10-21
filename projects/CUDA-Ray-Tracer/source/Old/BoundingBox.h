#ifndef BOUNDING_BOX_H
#define	BOUNDING_BOX_H

#include "GL/glew.h"
#include "GL/freeglut.h"

#include "Vector.h"
#include <map>

class BoundingBox{

	private:

		Vector maximum;
		Vector minimum;

	public:

		BoundingBox();
		~BoundingBox();

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *closestHit, Vector *farthestHit, Vector *normalHit);

		/* Getters */
		Vector getMaximum();
		Vector getMinimum();

		/* Setters */
		void setMaximum(Vector maximum);
		void setMinimum(Vector minimum);

		void dump();
};

#endif