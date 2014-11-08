#ifndef MATRIX_H
#define MATRIX_H

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

/* Translation Matrix Positions */
#define T_X 3
#define T_Y 7
#define T_Z 11

/* Scale Matrix Positions */
#define S_X 0
#define S_Y 5
#define S_Z 10

/* Rotation Matrix Positions */
#define R1_X 5
#define R2_X 6
#define R3_X 9
#define R4_X 10

#define R1_Y 0
#define R2_Y 2
#define R3_Y 8
#define R4_Y 10

#define R1_Z 0
#define R2_Z 1
#define R3_Z 4
#define R4_Z 5

using namespace std;

class Matrix {

	private:

		/* Matrix Content */
		float matrix[4][4];

	public:

		static const float threshold;
		
		/* Constructors & Destructors */
		Matrix();
		Matrix(Quaternion quaternion);
		Matrix(float initialValue);
		Matrix(const float initialValue[16]);
		Matrix(const float initialValue[4][4]);

		/* View Matrix Constructor */
		Matrix(const float xAxis[3], const float yAxis[3], const float zAxis[3]);	

		~Matrix();

		/* Identity Transformation */
		void loadIdentity();
		void clean();

		/* Graphical Transformations */
		void scale(Vector scaleVector);
		void scale(float xScale, float yScale, float zScale);

		void rotate(float angle, float xRotation, float yRotation, float zRotation);

		void translate(Vector transationVector);
		void translate(float xTranslation, float yTranslation, float zTranslation);

		void removeTranslation();

		void quaternionRotate(Quaternion quaternion);
		void transpose();
		void invert();

		/* Camera Transformations */
		void setView(Vector eye, Vector center, Vector userUp);
		void setOrthogonalProjection(float left, float right, float top, float bottom, float nearZ, float farZ);
		void setPerspectiveProjection(float fieldOfView, float aspectRatio, float nearZ, float farZ);

		/* Getters */
		void getValue(float* matrix);
		float getValue(int row, int column);

		/* Setters */
		void setValue(const float value[16]);
		void setValue(int row, int column, float value);

		/* Operators */
		float& operator[] (int position);

		Matrix operator +  (Matrix matrix);
		Matrix operator += (Matrix matrix);
		Matrix operator *  (Matrix matrix);
		Matrix operator *= (Matrix matrix);

		Vector operator *  (Vector vector);

		bool operator == (Matrix matrix);

		/* Debug */
		void dump();
};

#endif