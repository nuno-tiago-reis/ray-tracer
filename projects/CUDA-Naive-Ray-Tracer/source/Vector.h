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
		float vector[4];

	public:

		static const float threshold;
		
		/* Constructors & Destructors */
		Vector();
		Vector(float initialValue);
		Vector(const float initialValue[4]);
		Vector(float x, float y, float z, float w);

		~Vector();

		/* Vector Operations */
		float length();

		void loadIdentity();

		void clean();
		void negate();
		void normalize();

		static Vector projection(Vector u, Vector v);
		static Vector crossProduct(Vector u, Vector v);
		static float dotProduct(Vector u, Vector v);		

		/* Getters  */
		void getValue(float* vector);

		/* Setters */
		void setValue(const float value[4]);

		/* Basic Operations */
		float& operator[] (int position);

		Vector operator -  ();

		Vector operator +  (Vector vector);
		Vector operator += (Vector vector);
		Vector operator -  (Vector vector);
		Vector operator -= (Vector vector);

		Vector operator *  (float scalar);
		Vector operator *= (float scalar);

		bool operator == (Vector vector);	

		/* Debug */
		void dump();
};

#endif