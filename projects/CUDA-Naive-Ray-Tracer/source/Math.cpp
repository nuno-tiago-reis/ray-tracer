#include "Math.h"

int Math::clamp(GLfloat value, GLfloat floor, GLfloat ceiling) {

	if(value < floor)
	   return (GLint)floor;
	if(value > ceiling)
	   return (GLint)ceiling;

	return (GLint)value;
}