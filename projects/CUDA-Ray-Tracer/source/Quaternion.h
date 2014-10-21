#ifndef QUATERNION_H
#define QUATERNION_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"

/* C++ Includes */
#include <math.h>

/* Math Library */
#include "Vector.h"
#include "Quaternion.h"

/* Engine Constants */
#include "Constants.h"

/* Coordinate Constants */
#define QT 0
#define QX 1
#define QY 2
#define QZ 3

using namespace std;

class Quaternion {

	private:

		/* Quaternion Content */
		GLfloat quaternion[4];

	public:

		static const GLfloat threshold;
		
		/* Constructors & Destructors */
		Quaternion();
		Quaternion(GLfloat* initialValue);
		Quaternion(GLfloat theta, Vector axis);

		~Quaternion();

		/* Quaternion Conversions */
		void toAngle(GLfloat* theta, Vector* axis);		

		/* Quaternion Transformations */
		void clean();

		void invert();

		void conjugate();

		void normalize();

		GLfloat norm();

		GLfloat quadrance();

		/* Getters */
		void getValue(GLfloat* quaternion);

		/* Setters */
		void setValue(const GLfloat value[4]);

		/* Quaternion Operations */
		void lerp(Quaternion quaternion, GLfloat k);
		void slerp(Quaternion quaternion, GLfloat k);

		GLfloat& operator[] (int position);

		Quaternion operator +  (Quaternion quaternion);
		Quaternion operator += (Quaternion quaternion);
		Quaternion operator *  (Quaternion quaternion);
		Quaternion operator *= (Quaternion quaternion);

		Quaternion operator *  (GLfloat scalar);
		Quaternion operator *= (GLfloat scalar);

		bool operator == (Quaternion quaternion);

		/* Debug */
		void dump();							
};

#endif