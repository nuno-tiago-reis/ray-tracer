#include "Camera.h"

Camera::Camera(unsigned int width, unsigned int height) {

	this->width = width;
	this->height = height;

	this->fieldOfView = 0.0f;

	this->zoom = 0.75f;
	this->longitude = 180.0f;
	this->latitude = 45.00f;
}

Camera::~Camera() {
}

void Camera::update(int zoom, int longitude, int latitude, float elapsedTime) {

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

	// Re-Calculate the Cameras Eye Vector
	this->eye[VX] = this->position[VX] + this->zoom * (CAMERA_RADIUS * cos(this->longitude * DEGREES_TO_RADIANS) - CAMERA_RADIUS * sin(this->longitude * DEGREES_TO_RADIANS));
	this->eye[VY] = this->position[VY] + this->zoom *  CAMERA_RADIUS * cos(this->latitude * DEGREES_TO_RADIANS);
	this->eye[VZ] = this->position[VZ] + this->zoom * (CAMERA_RADIUS * sin(this->longitude * DEGREES_TO_RADIANS) + CAMERA_RADIUS * cos(this->longitude * DEGREES_TO_RADIANS));
	this->eye[VW] = 1.0f;
	this->eye.clean();

	// Re-Calculate the Cameras Target Vector
	this->target[VX] = this->position[VX];
	this->target[VY] = this->position[VY];
	this->target[VZ] = this->position[VZ];
	this->target[VW] = 1.0f;
	this->target.clean();

	// Re-Calculate the Cameras Up Vector
	this->up[VX] = 0.0f;
	this->up[VY] = 1.0f;
	this->up[VZ] = 0.0f;
	this->up[VW] = 1.0f;
	this->up.clean();

	// Images Aspect Ratio 
	float aspectRatio = (float)this->width / (float)this->height;
	// Cameras distance to the target 
	float distance = (this->target - this->eye).length();

	// Cameras Field of View 
	float fieldOfView = this->fieldOfView;
	// Projection Frustum Half-Width 
	float theta = (fieldOfView * 0.5f) * DEGREES_TO_RADIANS;
	float halfHeight = 2.0f * distance * tanf(theta);
	float halfWidth = halfHeight * aspectRatio;

	// Re-Calculate the Cameras Direction Vector
	direction = Vector(target[VX] - eye[VX], target[VY] - eye[VY], target[VZ] - eye[VZ], 1.0f);
	direction.normalize();
	direction = direction * distance;

	// Re-Calculate the Cameras Right Vector
	right = Vector::crossProduct(direction, up);
	right.normalize();
	right = right * halfWidth;

	// Re-Calculate the Cameras Up Vector
	up = Vector::crossProduct(right, direction);
	up.normalize();
	up = up * halfHeight;
}

unsigned int Camera::getWidth() {

	return width;
}

unsigned int Camera::getHeight() {

	return this->height;
}

float Camera::getFieldOfView() {

	return this->fieldOfView;
}

Vector Camera::getPosition() {

	return this->position;
}

Vector Camera::getTarget() {

	return this->target;
}

Vector Camera::getEye() {

	return this->eye;
}

Vector Camera::getUp() {

	return this->up;
}

Vector Camera::getRight() {

	return this->right;
}

Vector Camera::getDirection() {

		return this->direction;
}

void Camera::setWidth(unsigned int width) {
	
	this->width = width;
}

void Camera::setHeight(unsigned int height) {

	this->height = height;
}

void Camera::setFieldOfView(float fieldOfView) {

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

void Camera::setRight(Vector right) {

	this->right = right;
}

void Camera::setDirection(Vector direction) {

	this->direction = direction;
}

void Camera::dump() {

	cout << "<Camera Dump>" << endl;
 
	/* Viewports Dimensions */
	cout << "<Camera Width> = " << width << endl;
	cout << "<Camera Height> = " << height << endl;

	/* Cameras Field of View */
	cout << "\tField of View = " << fieldOfView << endl;

	/* Cameras Position Vector */
	cout << "<Camera Position > = "; position.dump();

	/* Cameras Direction Vectors */
	cout << "<Camera Target > = "; target.dump();
	cout << "<Camera Eye > = "; eye.dump();

	/* Cameras Plane Vectors */
	cout << "<Camera Up > = "; up.dump();
	cout << "<Camera Right > = "; right.dump();
	cout << "<Camera Direction > = "; direction.dump();
}