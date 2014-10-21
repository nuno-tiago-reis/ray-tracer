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
		GLfloat matrix[4][4];

	public:

		static const GLfloat threshold;
		
		/* Constructors & Destructors */
		Matrix();
		Matrix(Quaternion quaternion);
		Matrix(GLfloat initialValue);
		Matrix(const GLfloat initialValue[16]);
		Matrix(const GLfloat initialValue[4][4]);

		/* View Matrix Constructor */
		Matrix(const GLfloat xAxis[3], const GLfloat yAxis[3], const GLfloat zAxis[3]);	

		~Matrix();

		/* Identity Transformation */
		void loadIdentity();
		void clean();

		/* Graphical Transformations */
		void scale(Vector scaleVector);
		void scale(GLfloat xScale, GLfloat yScale, GLfloat zScale);

		void rotate(GLfloat angle, GLfloat xRotation, GLfloat yRotation, GLfloat zRotation);

		void translate(Vector transationVector);
		void translate(GLfloat xTranslation, GLfloat yTranslation, GLfloat zTranslation);

		void quaternionRotate(Quaternion quaternion);
		void transpose();
		void invert();

		/* Camera Transformations */
		void setView(Vector eye, Vector center, Vector userUp);
		void setOrthogonalProjection(GLfloat left, GLfloat right, GLfloat top, GLfloat bottom, GLfloat nearZ, GLfloat farZ);
		void setPerspectiveProjection(GLfloat fieldOfView, GLfloat aspectRatio, GLfloat nearZ, GLfloat farZ);

		/* Getters */
		void getValue(GLfloat* matrix);
		GLfloat getValue(int row, int column);

		/* Setters */
		void setValue(const GLfloat value[16]);
		void setValue(int row, int column, GLfloat value);

		/* Operators */
		GLfloat& operator[] (int position);

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