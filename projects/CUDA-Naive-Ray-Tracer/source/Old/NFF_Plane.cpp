#include "NFF_Plane.h"

NFF_Plane::NFF_Plane(GLint identifier) : Object(identifier) {

	this->identifier = identifier;
}

NFF_Plane::~NFF_Plane() {
}

bool NFF_Plane::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {

	Vector u = getVertex(0);
	Vector v = getVertex(1);
	Vector w = getVertex(2);

	Vector vu = v - u;
	vu.normalize();

	Vector wu = w - u;
	wu.normalize();

	Vector normal = Vector::crossProduct(vu,wu);
	normal.normalize();

	GLfloat d = - Vector::dotProduct(normal,u);

	if(Vector::dotProduct(normal, rayDirection) == 0)
		return false;

	GLfloat Ti = -((Vector::dotProduct(normal,rayOrigin) + d) / Vector::dotProduct(normal, rayDirection));

	if(Ti < 0.0f)
		return false;

	Vector Ri = rayOrigin + rayDirection * Ti;

	if(pointHit != NULL && normalHit != NULL) {

		*pointHit = Ri;
		*normalHit = normal;		
	}

	return true;
}

void NFF_Plane::createBoundingBox() {

}

GLint NFF_Plane::getIdentifier() {
	return identifier;
}

Vector NFF_Plane::getVertex(int index) {

	return vertices[index];
}

void NFF_Plane::setVertex(Vector vertex, int index) {

	this->vertices[index] = vertex;
}

void NFF_Plane::dump() {

	cout << "Debugging NFF_Plane " << identifier << endl;

	for(int i=0;i<3;i++) {
		cout << "\tVectors: "; 
		vertices[i].dump();
	}
}