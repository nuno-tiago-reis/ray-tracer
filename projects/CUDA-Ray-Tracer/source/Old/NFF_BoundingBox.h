#ifndef NFF_BOUNDING_BOX_H
#define	NFF_BOUNDING_BOX_H

#include "Object.h"

#include <map>

using namespace std;

class NFF_BoundingBox : public Object {

	private:

		Vector maximum;
		Vector minimum;

	public:

		NFF_BoundingBox(int identifier);
		~NFF_BoundingBox();

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		void createBoundingBox();

		/* Getters */
		Vector getMaximum();
		Vector getMinimum();

		/* Setters */
		void setMaximum(Vector maximum);
		void setMinimum(Vector minimum);

		void dump();
};

#endif