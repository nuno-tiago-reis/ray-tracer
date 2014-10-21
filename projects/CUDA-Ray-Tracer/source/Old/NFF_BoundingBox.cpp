#include "NFF_BoundingBox.h"

NFF_BoundingBox::NFF_BoundingBox(int identifier) : Object(identifier) {
}

NFF_BoundingBox::~NFF_BoundingBox() {
}

bool NFF_BoundingBox::rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {
	
	return boundingBox->rayIntersection(rayOrigin, rayDirection, pointHit, normalHit);
}

void NFF_BoundingBox::createBoundingBox() {

	boundingBox = new BoundingBox();

	boundingBox->setMaximum(maximum);
	boundingBox->setMinimum(minimum);
}

Vector NFF_BoundingBox::getMaximum() {

	return maximum;
}

Vector NFF_BoundingBox::getMinimum() {

	return minimum;
}

void NFF_BoundingBox::setMaximum(Vector maximum) {

	this->maximum = maximum;
}

void NFF_BoundingBox::setMinimum(Vector minimum) {

	this->minimum = minimum;
}

void NFF_BoundingBox::dump() {

	//Object::dump();

	cout << "\tDebugging AxisAlignedBoundingBox [" << identifier << "]" << endl;

	cout << "\tMaximum = "; maximum.dump();
	cout << "\tMinimum = "; minimum.dump();
}
