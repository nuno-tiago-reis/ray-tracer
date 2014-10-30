#include "NFF_Sphere.h"

NFF_Sphere::NFF_Sphere(GLint identifier) : Object(identifier) {

	this->identifier = identifier;
}

NFF_Sphere::~NFF_Sphere() {
}


bool NFF_Sphere::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {

	Vector distance = position - rayOrigin;

	GLfloat d = pow(position[VX] - rayOrigin[VX],2) + pow(position[VY] - rayOrigin[VY],2) + pow(position[VZ] - rayOrigin[VZ],2);
	
	if(d == pow(radius,2))
		return false;
	
	GLfloat B = Vector::dotProduct(distance, rayDirection);

	if(d > pow(radius,2) && B < 0.0f)
		return false;

	GLfloat C = d - pow(radius,2);

	GLfloat R = pow(B,2) - C;

	if(R < 0.0f)
		return false;

	GLfloat Ti = 0.0f;

	if(d > pow(radius,2))
		Ti = B - sqrt(R);
	else if(d < pow(radius,2))
		Ti = B + sqrt(R);

	Vector Ri = rayOrigin + rayDirection * Ti;

	Vector Ni = Ri - position;
	Ni.normalize();

	if(pointHit != NULL && normalHit != NULL) {

		if(d < pow(radius,2))
			Ni.negate();

		*pointHit = Ri;
		*normalHit = Ni;
	}

	return true;
}

void NFF_Sphere::createBoundingBox() {

	boundingBox = new BoundingBox();

	Vector maximum = position + Vector(radius);
	Vector minimum = position - Vector(radius);

	boundingBox->setMaximum(maximum);
	boundingBox->setMinimum(minimum);
}

GLfloat NFF_Sphere::getRadius() {

	return radius;
}

Vector NFF_Sphere::getPosition() {

	return position;
}

void NFF_Sphere::setRadius(GLfloat radius) {

	this->radius = radius;
}

void NFF_Sphere::setPosition(Vector position) {

	this->position = position;
}

void NFF_Sphere::dump() {

	cout << "Debugging NFF_Sphere " << identifier << endl;

	cout << "\tPosition "; position.dump();

	cout << "\tRadius  " << radius << endl;

}