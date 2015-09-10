#ifndef BOUNDING_BOX_H
#define	BOUNDING_BOX_H

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/freeglut.h"
/* OpenGL Error check */
#include "Utility.h"

// Vector
#include "Vector.h"

class BoundingBox {

	private:

		Vector maximum;
		Vector minimum;

	public:

		BoundingBox();
		~BoundingBox();

		// Getters
		Vector getMaximum();
		Vector getMinimum();

		// Setters
		void setMaximum(Vector maximum);
		void setMinimum(Vector minimum);

		void dump();
};

#endif