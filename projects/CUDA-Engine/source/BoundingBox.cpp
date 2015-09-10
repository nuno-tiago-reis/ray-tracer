#include "BoundingBox.h"

BoundingBox::BoundingBox() {
}


BoundingBox::~BoundingBox() {
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
