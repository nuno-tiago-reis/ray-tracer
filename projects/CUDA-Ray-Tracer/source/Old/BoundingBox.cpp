#include "BoundingBox.h"

BoundingBox::BoundingBox() {
}


BoundingBox::~BoundingBox() {
}

bool BoundingBox::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *closestHit, Vector *farthestHit, Vector *normalHit) {

	GLfloat tClosest = -FLT_MAX;
	GLfloat tFarthest = FLT_MAX;

	for(int i=0; i<3; i++) {

		GLfloat vMinimum = minimum[i];
		GLfloat vMaximum = maximum[i];
	
		GLfloat vOrigin = rayOrigin[i];
		GLfloat vDirection = rayDirection[i];
		
		if(vDirection == 0.0f) {
			
			if(vOrigin < vMinimum || vOrigin > vMaximum)
				return false;
			else
				continue;
		}

		GLfloat tMinimum = (vMinimum - vOrigin) / vDirection;
		GLfloat tMaximum = (vMaximum - vOrigin) / vDirection;

		if(tMinimum > tMaximum)
			swap(tMinimum,tMaximum);

		if(tMinimum > tClosest)
			tClosest = tMinimum;

		if(tMaximum < tFarthest)
			tFarthest = tMaximum;

		if(tClosest > tFarthest)
			return false;

		if(tFarthest < 0)
			return false;
	}

	if(closestHit != NULL && farthestHit != NULL) {

		*closestHit = rayOrigin + rayDirection * tClosest;
		*farthestHit = rayOrigin + rayDirection * tFarthest;
	}

	return true;
}

bool BoundingBox::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {
	
	GLint closestPlane = INT_MAX; 
	GLint farthestPlane = INT_MAX;

	GLfloat tClosest = -FLT_MAX;
	GLfloat tFarthest = FLT_MAX;

	for(int i=0; i<3; i++) {

		GLfloat vMinimum = minimum[i];
		GLfloat vMaximum = maximum[i];
	
		GLfloat vOrigin = rayOrigin[i];
		GLfloat vDirection = rayDirection[i];
		
		if(vDirection == 0.0f) {
			
			if(vOrigin < vMinimum || vOrigin > vMaximum)
				return false;
			else
				continue;
		}

		GLfloat tMinimum = (vMinimum - vOrigin) / vDirection;
		GLfloat tMaximum = (vMaximum - vOrigin) / vDirection;

		if(tMinimum > tMaximum)
			swap(tMinimum,tMaximum);

		if(tMinimum > tClosest) {

			tClosest = tMinimum;

			closestPlane = i;
		}

		if(tMaximum < tFarthest) {

			tFarthest = tMaximum;

			farthestPlane = i;
		}

		if(tClosest > tFarthest)
			return false;

		if(tFarthest < 0)
			return false;
	}

	if(pointHit != NULL && normalHit != NULL) {

		Vector point = rayOrigin + rayDirection * tClosest;
		Vector normal;

		switch(closestPlane) {

			case VX:	normal = Vector(1.0f,0.0f,0.0f,1.0f);

						if(abs(point[VX] - minimum[VX]) < abs(point[VX] - maximum[VX]))
							normal.negate();
						
						break;

			case VY:	normal = Vector(0.0f,1.0f,0.0f,1.0f);
				
						if(abs(point[VY] - minimum[VY]) < abs(point[VY] - maximum[VY]))
							normal.negate();

						break;

			case VZ:	normal = Vector(0.0f,0.0f,1.0f,1.0f);

						if(abs(point[VZ] - minimum[VZ]) < abs(point[VZ] - maximum[VZ]))
							normal.negate();

						break;
		}

		*pointHit = point;
		*normalHit = normal;
	}

	return true;
}

Vector BoundingBox::getMaximum() {

	return maximum;
}

Vector BoundingBox::getMinimum() {

	return minimum;
}

void BoundingBox::setMaximum(Vector maximum) {

	this->maximum = maximum;
}

void BoundingBox::setMinimum(Vector minimum) {

	this->minimum = minimum;
}

void BoundingBox::dump() {

	//Object::dump();

	cout << "\tDebugging BoundingBox" << endl;

	cout << "\tMaximum = "; maximum.dump();
	cout << "\tMinimum = "; minimum.dump();
}
