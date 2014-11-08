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
		float quaternion[4];

	public:

		static const float threshold;
		
		/* Constructors & Destructors */
		Quaternion();
		Quaternion(float* initialValue);
		Quaternion(float theta, Vector axis);

		~Quaternion();

		/* Quaternion Conversions */
		void toAngle(float* theta, Vector* axis);		

		/* Quaternion Transformations */
		void clean();

		void invert();

		void conjugate();

		void normalize();

		float norm();

		float quadrance();

		/* Getters */
		void getValue(float* quaternion);

		/* Setters */
		void setValue(const float value[4]);

		/* Quaternion Operations */
		void lerp(Quaternion quaternion, float k);
		void slerp(Quaternion quaternion, float k);

		float& operator[] (int position);

		Quaternion operator +  (Quaternion quaternion);
		Quaternion operator += (Quaternion quaternion);
		Quaternion operator *  (Quaternion quaternion);
		Quaternion operator *= (Quaternion quaternion);

		Quaternion operator *  (float scalar);
		Quaternion operator *= (float scalar);

		bool operator == (Quaternion quaternion);

		/* Debug */
		void dump();							
};

#endif