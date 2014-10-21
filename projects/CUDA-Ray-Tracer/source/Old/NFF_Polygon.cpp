#include "NFF_Polygon.h"

NFF_Polygon::NFF_Polygon(int identifier) : Object(identifier) {
}


NFF_Polygon::~NFF_Polygon() {
}

bool NFF_Polygon::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {
	
	Vector v0 = getVertex(0);
	Vector v1 = getVertex(1);
	Vector v2 = getVertex(2);

	/* Intersecting Embedding Plane */
	Vector v1v0 = v1 - v0;
	v1v0.normalize();

	Vector v2v0 = v2 - v0;
	v2v0.normalize();

	Vector normal = Vector::crossProduct(v1v0,v2v0);

	/* Early rejection test */
	if(Vector::dotProduct(normal, rayDirection) == 0.0f)
		return false;

	/* Plane representation:   dot(normal, PointOfPlane) + d = 0   */
	GLfloat d = - Vector::dotProduct(normal, v0);

	GLfloat t = - ((Vector::dotProduct(normal,rayOrigin) + d) / Vector::dotProduct(normal, rayDirection));

	if(t <= 0.0f)
		return false;

	/* Intersecting the Polygon */
	Vector point = rayOrigin + rayDirection * t;

	GLint i0, i1, i2;
	GLfloat m = max(abs(normal[VX]),max(abs(normal[VY]),abs(normal[VZ])));

	if(abs(normal[VX]) == m) {

		i0 = VX;
		i1 = VY;
		i2 = VZ;
	}
	else if(abs(normal[VY]) == m) {

		i0 = VY;
		i1 = VX;
		i2 = VZ;
	}
	else {

		i0 = VZ;
		i1 = VX;
		i2 = VY;
	}

	Vector u = Vector(point[i1] - v0[i1], v1[i1] - v0[i1], v2[i1] - v0[i1],	1.0f);
	Vector v = Vector(point[i2] - v0[i2], v1[i2] - v0[i2], v2[i2] - v0[i2], 1.0f);

	GLfloat alfa = (u[VX] * v[VZ] - u[VZ] * v[VX]) / (u[VY] * v[VZ] - u[VZ] * v[VY]);
	GLfloat beta = (u[VY] * v[VX] - u[VX] * v[VY]) / (u[VY] * v[VZ] - u[VZ] * v[VY]);

	if(alfa < 0 || beta < 0 || (alfa + beta > 1))
		return false;

	if(pointHit != NULL && normalHit != NULL) {

		*pointHit = point;
		*normalHit = normal;		
	}
	
	return true;
}

void NFF_Polygon::createBoundingBox() {

	boundingBox = new BoundingBox();

	Vector maximum = Vector(FLT_MIN);
	Vector minimum = Vector(FLT_MAX);

	for(map<GLint, Vector>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++) {
	
		Vector  vertex = vertexIterator->second;

		if(maximum[VX] < vertex[VX])
			maximum[VX] = vertex[VX];

		if(maximum[VY] < vertex[VY])
			maximum[VY] = vertex[VY];

		if(maximum[VZ] < vertex[VZ])
			maximum[VZ] = vertex[VZ];

		if(minimum[VX] > vertex[VX])
			minimum[VX] = vertex[VX];

		if(minimum[VY] > vertex[VY])
			minimum[VY] = vertex[VY];

		if(minimum[VZ] > vertex[VZ])
			minimum[VZ] = vertex[VZ];
	}

	boundingBox->setMaximum(maximum);
	boundingBox->setMinimum(minimum);
}

void NFF_Polygon::addVertex(Vector vertex) {

	vertexMap[vertexMap.size()] = vertex;
}

void NFF_Polygon::removeVertex(int index) {
	
	vertexMap[index] = Vector();
}

Vector NFF_Polygon::getVertex(int index) {

	return vertexMap[index];
}

map<int,Vector> NFF_Polygon::getVertexMap() {

	return vertexMap;
}

void NFF_Polygon::dump() {

	//Object::dump();

	cout << "\tDebugging Polygon [" << identifier << "]" << endl;

	for(int i=0; i<3; i++) {
		cout << "\tVertex [" << i << "] = "; vertexMap[i].dump();
	}
}