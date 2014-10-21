#include "Vector.h"

const GLfloat Vector::threshold = (GLfloat)1.0e-5;

Vector::Vector() {

	for(int i=0; i<3; i++)
		this->vector[i] = 0.0f;

	this->vector[VW] = 1.0f;
}

Vector::Vector(GLfloat initialValue) {

	for(int i=0; i<3; i++)
		this->vector[i] = initialValue;

	this->vector[VW] = 1.0f;
}

Vector::Vector(const GLfloat initialValue[4]) {

	for(int i=0; i<4; i++)
		this->vector[i] = initialValue[i];
}

Vector::Vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w) {

	this->vector[VX] = x;
	this->vector[VY] = y;
	this->vector[VZ] = z;
	this->vector[VW] = w;
}

Vector::~Vector() {
}

void Vector::loadIdentity() {

	this->vector[VX] = 0.0f;
	this->vector[VY] = 0.0f;
	this->vector[VZ] = 0.0f;
	this->vector[VW] = 1.0f;
}

void Vector::clean() {

	for(int i=0; i<3; i++)
		if(fabs(this->vector[i]) < threshold) 
			this->vector[i] = 0.0f;

	this->vector[VW] = 1.0f;
}

void Vector::negate() {
	
	for(int i=0; i<4; i++)
		this->vector[i] = -this->vector[i];

	this->vector[VW] = 1.0f;
}

void Vector::normalize() {
	
	GLfloat magnitude = sqrt(dotProduct(*this,*this));

	if(magnitude == 0)
		return;

	for(int i=0; i<3; i++)
		this->vector[i] /= magnitude;

	this->vector[VW] = 1.0f;
}

GLfloat Vector::length() {

	return sqrt(dotProduct(*this,*this));
}

Vector Vector::projection(Vector u, Vector v) {

	v.normalize();

	Vector projection = v * dotProduct(u,v); 

	projection[VW] = 1.0f;

	return projection;
}

Vector Vector::crossProduct(Vector u, Vector v) {

	Vector crossProduct;

	crossProduct[VX] = u[VY] * v[VZ] - u[VZ] * v[VY];
	crossProduct[VY] = u[VZ] * v[VX] - u[VX] * v[VZ];
	crossProduct[VZ] = u[VX] * v[VY] - u[VY] * v[VX];

	crossProduct[VW] = 1.0f;

	return crossProduct; 
}

GLfloat Vector::dotProduct(Vector u, Vector v) {

	GLfloat internalProduct = 0.0f;

	for(int i=0; i<3; i++)
		internalProduct += u[i] * v[i];

	return internalProduct; 
}

void Vector::getValue(GLfloat* vector) {

	for(int i=0; i<4; i++)
		vector[i] = this->vector[i];
}

void Vector::setValue(const GLfloat value[4]) {

	for(int i=0; i<3; i++)
		this->vector[i] = value[i];
}

GLfloat& Vector::operator [] (int position){

	return this->vector[position];
}

Vector Vector::operator - () {

	Vector result;

	for(int i=0; i<3; i++)
		result[i] = -this->vector[i];

	result[VW] =  1.0f;

	return result;
}

Vector Vector::operator + (Vector vector){

	Vector result;

	for(int i=0; i<3; i++)
		result[i] = this->vector[i] + vector[i];

	result[VW] =  1.0f;

	return result;
}

Vector Vector::operator += (Vector vector) {

	for(int i=0; i<3; i++)
		this->vector[i] +=  vector[i];

	this->vector[VW] =  1.0f;

	return (*this);
}

Vector Vector::operator - (Vector vector) {

	Vector result;

	for(int i=0; i<3; i++)
		result[i] = this->vector[i] - vector[i];

	result[VW] =  1.0f;

	return result;
}

Vector Vector::operator-= (Vector vector) {

	for(int i=0; i<3; i++)
		this->vector[i] -=  vector[i];

	this->vector[VW] =  1.0f;

	return (*this);
}

Vector Vector::operator * (GLfloat scalar) {

	Vector result;

	for(int i=0; i<3; i++)
		result[i] = this->vector[i] * scalar;

	this->vector[VW] = 1.0f;

	return result;
}

Vector Vector::operator *= (GLfloat scalar) {

	for(int i=0; i<3; i++)
		this->vector[i] *= scalar;

	this->vector[VW] = 1.0f;

	return (*this);
}

bool Vector::operator== (Vector vector) {

	for(int i=0; i<3; i++)
		if(!(fabs(this->vector[i] - vector[i]) < threshold))
			return false;

	return true;
}

void Vector::dump() {

	cout.precision(5);

	cout << "<Vector Dump> = ";

	for(int i=0; i<4; i++)
		cout << " [" << this->vector[i] << "]";

	cout << endl;
}