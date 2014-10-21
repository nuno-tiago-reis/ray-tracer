#ifndef TRANSFORM_H
#define TRANSFORM_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* C++ Includes */
#include <string>

/* Math Library */
#include "Matrix.h"

using namespace std;

class Transform {

	protected:

		/* Transforms Name */
		string name;

		/* Transforms Model Matrix */
		Matrix modelMatrix;

		/* Transforms Position Vector (Px,Py,Pz,1.0) */
		Vector position;
		/* Transforms Velocity Vector*/
		Vector velocity;

		/* Transforms Scale Factors (Sx,Sy,Sz,1.0) */
		Vector scale;

		/* Transforms Rotation (Rx,Ry,Rz,1.0) */
		Vector rotation;
		/* Transforms Rotation Quaternion */
		Quaternion rotationQuaternion;

	public:

		/* Constructors & Destructors */
		Transform(string name);
		~Transform();

		/* Scene Methods */
		virtual void update();
		virtual void update(GLfloat elapsedTime);

		/* Getters */
		string getName();

		Matrix getModelMatrix();

		Vector getPosition();
		Vector getVelocity();

		Vector getScale();

		Vector getRotation();
		Quaternion getRotationQuaternion();

		/* Setters */
		void setName(string name);

		void setModelMatrix(Matrix modelMatrix);

		void setPosition(Vector position);
		void setVelocity(Vector velocity);

		void setScale(Vector scale);

		void setRotation(Vector rotation);
		void setRotationQuaternion(Quaternion rotationQuaternion);

		/* Debug Methods */
		void dump();
};

#endif