#include "BoundingSphere.h"

BoundingSphere::BoundingSphere() {
}


BoundingSphere::~BoundingSphere() {
}

void BoundingSphere::calculateMiniball(Mesh* mesh) {

	typedef float mytype;

	int dimension = 3;
	int pointNumber = 0;

	// Create the Point List
	list<vector<mytype> > pointList;

	// Load the Mesh
	map<int, Vertex*> vertexMap = mesh->getVertexMap();

	// Initialize the Point List
	for(map<int, Vertex*>::iterator vertex = vertexMap.begin(); vertex != vertexMap.end(); vertex++) {
	
		std::vector<mytype> point(dimension);

		for(int j=0; j<3; j++)
			point[j] = vertex->second->getPosition()[j];

		pointNumber++;
		pointList.push_back(point);
	}

	// Create the Iterators
	typedef list<vector<mytype>>::const_iterator PointIterator; 
	typedef vector<mytype>::const_iterator CoordinateIterator;

	// Create the Miniball Type
	typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordinateIterator>> Miniball;

	// Create the Miniball
	Miniball miniball(dimension, pointList.begin(), pointList.end());

	this->center = Vector(miniball.center()[0], miniball.center()[1], miniball.center()[2], 1.0f);
	this->radius = (miniball.squared_radius() * miniball.squared_radius());

	for(map<int, Vertex*>::iterator vertex = vertexMap.begin(); vertex != vertexMap.end(); vertex++) {

		float distance = (this->center - vertex->second->getPosition()).magnitude();

		if(distance > this->radius) {

			//printf("[REVISED] Vertex = %02.020f %02.020f %02.020f\n", vertex->second->getPosition()[VX], vertex->second->getPosition()[VY], vertex->second->getPosition()[VZ]);
			//printf("[REVISED] Radius = %02.010f => %02.010f \n", this->radius, distance);

			this->radius = distance;
		}
	}

	this->radius = min(100000.0f, this->radius);
}

Vector BoundingSphere::getCenter() {

	return center;
}

float BoundingSphere::getRadius() {

	return radius;
}

void BoundingSphere::setCenter(Vector center) {

	this->center = center;
}

void BoundingSphere::setRadius(float radius) {

	this->radius = radius;
}

void BoundingSphere::dump() {

	//Object::dump();

	cout << "\tDebugging BoundingSphere" << endl;

	cout << "\tCenter = "; center.dump();
	cout << "\tRadius = " << radius << endl;
}
