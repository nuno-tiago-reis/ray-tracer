#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

// OpenGL definitions
#include "GL/glew.h"
#include "GL/freeglut.h"
// OpenGL Error check
#include "Utility.h"

// Miniball
#include "Miniball.hpp"

// Mesh
#include "Mesh.h"
// Vector
#include "Vector.h"

class Mesh;

class BoundingSphere {

	private:

		Vector center;

		GLfloat radius;

	public:

		BoundingSphere();
		~BoundingSphere();

		// Miniball
		void calculateMiniball(Mesh* mesh);

		// Getters
		Vector getCenter();
		float getRadius();

		// Setters
		void setCenter(Vector center);
		void setRadius(float radius);

		void dump();
};

#endif