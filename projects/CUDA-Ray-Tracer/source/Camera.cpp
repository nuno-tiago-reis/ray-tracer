#include "Camera.h"

Camera::Camera(int width, int height) {

	this->width = width;
	this->height = height;

	this->fieldOfView = 0.0f;

	this->zoom = 1.0f;
	this->longitude = 90.0f;
	this->latitude = 89.99f;
}

Camera::~Camera() {
}

void Camera::update(GLint zoom, GLint longitude, GLint latitude, GLfloat elapsedTime) {

	//if(longitude == 0 && latitude == 0 && zoom == 0)
		//return;

	/* Update the Zoom */
	this->zoom -= zoom * 0.05f;

	if(this->zoom < 0.1f)
		this->zoom = 0.1f;
	
	/* Update the Longitude */
	this->longitude += longitude * elapsedTime * 5.0f;

	if(this->longitude > 360.0f)
		this->longitude -= 360.0f;
	else if(this->longitude < -360.0f)
		this->longitude += 360.0f;

	/* Update the Latitude */
	this->latitude += latitude * elapsedTime * 5.0f;

	if(this->latitude > 360.0f)
		this->latitude -= 360.0f;
	else if(this->latitude < -360.0f) 
		this->latitude += 360.0f;

	this->eye[VX] = this->position[VX];// + this->zoom *  CAMERA_RADIUS * sin((this->latitude - 90.0f) * DEGREES_TO_RADIANS) * cos(this->longitude * DEGREES_TO_RADIANS);
	this->eye[VY] = this->position[VY] + CAMERA_RADIUS;// + this->zoom *  CAMERA_RADIUS * cos((this->latitude - 90.0f) * DEGREES_TO_RADIANS);
	this->eye[VZ] = this->position[VZ] - CAMERA_RADIUS;//d + this->zoom * -CAMERA_RADIUS * sin((this->latitude - 90.0f) * DEGREES_TO_RADIANS) * sin(this->longitude * DEGREES_TO_RADIANS);
	this->eye[VW] = 1.0f;

	this->target[VX] = this->position[VX];
	this->target[VY] = this->position[VY];
	this->target[VZ] = this->position[VZ];
	this->target[VW] = 1.0f;

	this->up[VX] = 0.0f;
	this->up[VY] = 0.0f;
	this->up[VZ] = 1.0f;
	this->up[VW] = 1.0f;
}

void Camera::reshape(GLint width, GLint height) {

	this->width = width;
	this->height = height;
}

Vector Camera::getPrimaryRay(GLfloat x, GLfloat y) {

	Vector ze = target - eye;
	GLfloat d =  ze.length();

	GLfloat h = 2.0f * d * tan(fieldOfView * DEGREES_TO_RADIANS /2.0f);

	GLfloat w = ((GLfloat)width / (GLfloat)height) * h;

	ze.normalize();

	Vector xe = Vector::crossProduct(ze,up);
	xe.normalize();

	Vector ye = Vector::crossProduct(xe, ze);
	ye.normalize();

	Vector direction =  ze * d + ye * h * (y / ((GLfloat)height) - 1.0f/2.0f)  + xe * w * (x / ((GLfloat)width) - 1.0f/2.0f);
	direction.normalize();

	return direction;
}

GLint Camera::getWidth() {

	return width;
}

GLint Camera::getHeight() {

	return height;
}

GLfloat Camera::getFieldOfView() {

	return fieldOfView;
}

Vector Camera::getPosition() {

	return position;
}

Vector Camera::getTarget() {

	return target;
}

Vector Camera::getEye() {

	return eye;
}

Vector Camera::getUp() {

	return up;
}

void Camera::setWidth(GLint width) {
	
	this->width = width;
}

void Camera::setHeight(GLint height) {

	this->height = height;
}

void Camera::setFieldOfView(GLfloat fieldOfView) {

	this->fieldOfView = fieldOfView;
}

void Camera::setPosition(Vector position) {

	this->position = position;
}

void Camera::setTarget(Vector target) {

	this->target = target;
}

void Camera::setEye(Vector eye) {

	this->eye = eye;
}

void Camera::setUp(Vector up) {

	this->up = up;
}

void Camera::dump() {

	cout << "Debugging Camera" << endl;
 
	cout << "\tWidth = " << width << endl;
	cout << "\tHeight = " << height << endl;

	cout << "\tField of View = " << fieldOfView << endl;
 
	cout << "Target = "; target.dump();
	cout << "Eye = "; eye.dump();
	cout << "Up = "; up.dump();
}