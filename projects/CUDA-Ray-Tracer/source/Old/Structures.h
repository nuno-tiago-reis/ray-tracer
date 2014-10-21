#ifndef STRUCTURE_H
#define	STRUCTURE_H

#include "GL/glew.h"
#include "GL/glut.h"

typedef struct {

	float r;
	float g;
	float b;

	float Kd;
	float Ks;

	float shine;

	float transmittance;

	float indexOfRefraction;

} MaterialProperty;

#endif