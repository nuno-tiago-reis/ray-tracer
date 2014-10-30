#ifndef MATH_H
#define MATH_H

#include "GL/glew.h"
#include "GL/glut.h"

#include <iostream>

using namespace std;

class Math {

	private:

	public:

		static int clamp(GLfloat value, GLfloat floor, GLfloat ceiling);
};

#endif