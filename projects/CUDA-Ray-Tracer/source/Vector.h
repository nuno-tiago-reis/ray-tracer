#ifndef VECTOR_H
#define VECTOR_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"

/* C++ Includes */
#include <iostream>

/* Engine Constants */
#include "Constants.h"

/* Coordinate Constants */
#define VX 0
#define VY 1
#define VZ 2
#define VW 3

using namespace std;

class Vector {

	private:

		/* Matrix Content */
		GLfloat vector[4];

	public:

		static const GLfloat threshold;
		
		/* Constructors & Destructors */
		Vector();
		Vector(GLfloat initialValue);
		Vector(const GLfloat initialValue[4]);
		Vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w);

		~Vector();

		/* Vector Operations */
		GLfloat length();

		void loadIdentity();

		void clean();
		void negate();
		void normalize();

		static Vector projection(Vector u, Vector v);
		static Vector crossProduct(Vector u, Vector v);
		static GLfloat dotProduct(Vector u, Vector v);		

		/* Getters  */
		void getValue(GLfloat* vector);

		/* Setters */
		void setValue(const GLfloat value[4]);

		/* Basic Operations */
		GLfloat& operator[] (int position);

		Vector operator -  ();

		Vector operator +  (Vector vector);
		Vector operator += (Vector vector);
		Vector operator -  (Vector vector);
		Vector operator -= (Vector vector);

		Vector operator *  (GLfloat scalar);
		Vector operator *= (GLfloat scalar);

		bool operator == (Vector vector);	

		/* Debug */
		void dump();
};

#endif