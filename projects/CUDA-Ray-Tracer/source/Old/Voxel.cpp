#include "Voxel.h"

Voxel::Voxel(int indexX, int indexY, int indexZ) {

	this->indexX = indexX;
	this->indexY = indexY;
	this->indexZ = indexZ;
}

Voxel::~Voxel() {
}

Object* Voxel::intersect(Vector rayOrigin, Vector rayDirection, Vector* hitPoint, Vector* hitNormal, GLfloat* hitDistance) {

	Object* objectHit = NULL;

	float minimumDistance = FLT_MAX;

	/* Test Ray intersection wih the Voxel Objects */
	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Vector point;
		Vector normal;
		Object* object = objectIterator->second;

		if(dynamic_cast<NFF_Plane*>(object))
			continue;

		if(object->rayIntersection(rayOrigin, rayDirection, &point, &normal) == true) {

			float distance = (point - rayOrigin).length();

			if(minimumDistance > distance) {

				objectHit = object;

				*hitPoint = point;
				*hitNormal = normal;
				*hitDistance = distance;

				minimumDistance = distance;
			}
		}
	}

	return objectHit;
}

void Voxel::addObject(Object* object) {

	objectMap[ object->getIdentifier() ] = object;
}

void Voxel::removeObject(int identifier) {

	objectMap.erase(identifier);
}

map<GLint, Object*> Voxel::getObjectMap() {

	return objectMap;
}

/* Getters and Setters */
int Voxel::getIndexX() {

	return indexX;
}

int Voxel::getIndexY() {

	return indexY;
}

int Voxel::getIndexZ() {

	return indexZ;
}

void Voxel::setIndexX(int indexX) {

	this->indexX = indexX;
}

void Voxel::setIndexY(int indexY) {

	this->indexY = indexY;
}

void Voxel::setIndexZ(int indexZ) {

	this->indexZ = indexZ;
}

void Voxel::dump() {

	cout << "Debugging Voxel " << indexX << "," << indexY << "," << indexZ << endl;

	cout << "[Voxel] Contains: ";
	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++)
		cout << objectIterator->second->getIdentifier() << ", ";

	cout << endl;
}