#include "Quaternion.h"

const GLfloat Quaternion::threshold = (GLfloat)1.0e-5;

Quaternion::Quaternion() {

	this->quaternion[QT] = 1.0f;
	this->quaternion[QX] = 0.0f;
	this->quaternion[QY] = 0.0f;
	this->quaternion[QZ] = 0.0f;
}

Quaternion::Quaternion(GLfloat* initialValue) {

	for(int i=0; i<4; i++)
		this->quaternion[i] = initialValue[i];

	clean();
	normalize();
}

Quaternion::Quaternion(GLfloat theta, Vector axis) {

	axis.normalize();

	GLfloat a = theta * (GLfloat)DEGREES_TO_RADIANS;
	GLfloat scalar = sin(a / 2.0f);

	this->quaternion[QT] = cos(a / 2.0f);

	this->quaternion[QX] = axis[VX] * scalar;
	this->quaternion[QY] = axis[VY] * scalar;
	this->quaternion[QZ] = axis[VZ] * scalar;

	clean();
	normalize();
}

Quaternion::~Quaternion() {
}

void Quaternion::toAngle(GLfloat* theta, Vector* axis) {		//const void qToAngleAxix(const Quaternion& q, float& theta, Vector& axis);

	normalize();

	*theta = 2.0f * (GLfloat)acos((GLfloat)this->quaternion[QT]) * (GLfloat)RADIANS_TO_DEGREES;

	GLfloat s = sqrt(1.0f - this->quaternion[QT] * this->quaternion[QT]);

	if(s < threshold) {

		(*axis)[VX] = 1.0f;
		(*axis)[VY] = 0.0f;
		(*axis)[VZ] = 0.0f;
		(*axis)[VW] = 1.0f;

	} else {

		(*axis)[VX] = this->quaternion[QX]/s;
		(*axis)[VY] = this->quaternion[QY]/s;
		(*axis)[VZ] = this->quaternion[QZ]/s;
		(*axis)[VW] = 1.0f;
	}
}

void Quaternion::clean() {							

	if(fabs(this->quaternion[QT]) < threshold)
		this->quaternion[QT] = 0.0f;

	if(fabs(this->quaternion[QX]) < threshold)
		this->quaternion[QX] = 0.0f;

	if(fabs(this->quaternion[QY]) < threshold)
		this->quaternion[QY] = 0.0f;

	if(fabs(this->quaternion[QZ]) < threshold)
		this->quaternion[QZ] = 0.0f;
}

void Quaternion::invert() {		

	GLfloat q = quadrance();

	conjugate();

	(*this) *= (1.0f / q);
}

void Quaternion::conjugate() {								

	this->quaternion[QX] = -this->quaternion[QX];
	this->quaternion[QY] = -this->quaternion[QY];
	this->quaternion[QZ] = -this->quaternion[QZ];
}

void Quaternion::normalize() {								

	float scalar = 1 / norm();

	(*this) *= scalar;
}

GLfloat Quaternion::norm() {								

	return sqrt(quadrance());
}

GLfloat Quaternion::quadrance() {							

	GLfloat quadrance = 0.0f;

	for(int i=0; i<4; i++)
		quadrance += this->quaternion[i]*this->quaternion[i];

	return quadrance;
}

void Quaternion::getValue(GLfloat* quaternion) {
	
	for(int i=0; i<4; i++)
		quaternion[i] = this->quaternion[i];
}

void Quaternion::setValue(const GLfloat value[4]) {

	for(int i=0; i<4; i++)
		this->quaternion[i] = value[i];
}

void Quaternion::lerp(Quaternion quaternion, GLfloat k) {	

	GLfloat angle =	  this->quaternion[QT] * quaternion[QT] 
					+ this->quaternion[QX] * quaternion[QX]
					+ this->quaternion[QY] * quaternion[QY]
					+ this->quaternion[QZ] * quaternion[QZ];

	GLfloat k0 = 1.0f - k;
	GLfloat k1 = (angle > 0) ? k : -k;

	Quaternion temporary = quaternion;

	(*this) *= k0;
	temporary *= k1;

	(*this) += temporary;
}

void Quaternion::slerp(Quaternion quaternion, GLfloat k) {

	GLfloat angle =	acos(     this->quaternion[QT] * quaternion[QT] 
							+ this->quaternion[QX] * quaternion[QX]
							+ this->quaternion[QY] * quaternion[QY]
							+ this->quaternion[QZ] * quaternion[QZ]);

	GLfloat k0 = sin((1-k)*angle) / sin(angle);
	GLfloat k1 = sin(k*angle) / sin(angle);

	Quaternion temporary = quaternion;

	(*this) *= k0;
	temporary *= k1;

	(*this) += temporary;
}

GLfloat& Quaternion::operator [] (int position) {

	return this->quaternion[position];
}

Quaternion Quaternion::operator + (Quaternion quaternion) {

	Quaternion result;

	for(int i=0; i<4; i++)
		result[i] = this->quaternion[i] + quaternion[i];

	return result;
}

Quaternion Quaternion::operator += (Quaternion quaternion) {

	for(int i=0; i<4; i++)
		this->quaternion[i] += quaternion[i];

	return (*this);
}

Quaternion Quaternion::operator * (Quaternion quaternion) {

	Quaternion result;

	Quaternion temporary = *this;

	result[QT] =	  temporary[QT] * quaternion[QT] 
					- temporary[QX] * quaternion[QX]
					- temporary[QY] * quaternion[QY]
					- temporary[QZ] * quaternion[QZ];

	result[QX] =	  temporary[QT] * quaternion[QX]
					+ temporary[QX] * quaternion[QT]
					+ temporary[QY] * quaternion[QZ]
					- temporary[QZ] * quaternion[QY];

	result[QY] =	  temporary[QT] * quaternion[QY]
					+ temporary[QY] * quaternion[QT]
					+ temporary[QZ] * quaternion[QX]
					- temporary[QX] * quaternion[QZ];

	result[QZ] =	  temporary[QT] * quaternion[QZ]
					+ temporary[QZ] * quaternion[QT]
					+ temporary[QX] * quaternion[QY]
					- temporary[QY] * quaternion[QX];
	return result;
}

Quaternion Quaternion::operator *= (Quaternion quaternion) {

	Quaternion temporary = *this;

	this->quaternion[QT] =	  temporary[QT] * quaternion[QT] 
							- temporary[QX] * quaternion[QX]
							- temporary[QY] * quaternion[QY]
							- temporary[QZ] * quaternion[QZ];

	this->quaternion[QX] =	  temporary[QT] * quaternion[QX]
							+ temporary[QX] * quaternion[QT]
							+ temporary[QY] * quaternion[QZ]
							- temporary[QZ] * quaternion[QY];

	this->quaternion[QY] =	  temporary[QT] * quaternion[QY]
							+ temporary[QY] * quaternion[QT]
							+ temporary[QZ] * quaternion[QX]
							- temporary[QX] * quaternion[QZ];

	this->quaternion[QZ] =	  temporary[QT] * quaternion[QZ]
							+ temporary[QZ] * quaternion[QT]
							+ temporary[QX] * quaternion[QY]
							- temporary[QY] * quaternion[QX];
	return (*this);
}

Quaternion Quaternion::operator * (GLfloat scalar) {

	Quaternion result;

	for(int i=0; i<4; i++)
		result[i] = this->quaternion[i] * scalar;

	return result;
}

Quaternion Quaternion::operator *= (GLfloat scalar) {

	for(int i=0; i<4; i++)
		this->quaternion[i] *= scalar;

	return (*this);
}

bool Quaternion::operator == (Quaternion quaternion) {

	for(int i=0; i<4; i++)
		if(!(fabs(this->quaternion[i] - quaternion[i]) < threshold)) 
			return false;

	return true;
}

void Quaternion::dump() {

	cout << "<Quaternion Dump> = ";

	cout.precision(5);

	for(int i=0; i<4; i++)
		cout << " [" << this->quaternion[i] << "]";

	cout << endl;
}
