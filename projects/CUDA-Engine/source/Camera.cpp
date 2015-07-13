#include "Camera.h"

Camera::Camera(string name) {

	this->name = name;

	this->viewMatrix.loadIdentity();
	this->projectionMatrix.loadIdentity();

	this->projectionMode = ORTHOGONAL;

	this->uniformBufferIndex = UINT_MAX;

	this->width = WIDTH;
	this->height = HEIGHT;

	this->zoom = 0.75f;
	this->longitude = 180.0f;
	this->latitude = 45.0f;

	loadView();
	loadOrthogonalProjection();
}

Camera::~Camera() {
}

void Camera::update(GLint zoom, GLint longitude, GLint latitude, GLfloat elapsedTime) {

	if(longitude == 0 && latitude == 0 && zoom == 0)
		return;

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

	/* Update the Shader Uniforms */
	loadView();
	loadUniforms();
}

void Camera::reshape(GLint width, GLint height) {

	this->width = width;
	this->height = height;

	if(this->projectionMode == PERSPECTIVE)
		loadPerspectiveProjection();
}

void Camera::loadUniforms() {

	glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBufferIndex);

	/* Get the View and Projection Matrix */
	GLfloat viewMatrix[16];
	this->viewMatrix.getValue(viewMatrix);

	GLfloat projectionMatrix[16];
	this->projectionMatrix.getValue(projectionMatrix);

	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(GLfloat[16]), viewMatrix);
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(GLfloat[16]), sizeof(GLfloat[16]), projectionMatrix);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Camera::loadView() {

	this->eye[VX] = this->position[VX] + this->zoom * CAMERA_RADIUS * sin((this->latitude - 90.0f) * DEGREES_TO_RADIANS) * cos(this->longitude * DEGREES_TO_RADIANS);
	this->eye[VY] = this->position[VY] + this->zoom * CAMERA_RADIUS * cos((this->latitude - 90.0f) * DEGREES_TO_RADIANS);
	this->eye[VZ] = this->position[VZ] + this->zoom * -CAMERA_RADIUS * sin((this->latitude - 90.0f) * DEGREES_TO_RADIANS) * sin(this->longitude * DEGREES_TO_RADIANS);
	this->eye[VW] = 1.0f;

	this->target[VX] = this->position[VX];
	this->target[VY] = this->position[VY];
	this->target[VZ] = this->position[VZ];
	this->target[VW] = 1.0f;

	this->up[VX] = 0.0f;
	this->up[VY] = cos(this->latitude * DEGREES_TO_RADIANS);
	this->up[VZ] = 0.0f;
	this->up[VW] = 1.0f;
	
	this->viewMatrix.loadIdentity();
	this->viewMatrix.setView(this->eye,this->target,this->up);
	this->viewMatrix.transpose();

	// Images Aspect Ratio 
	float aspectRatio = (float)this->width / (float)this->height;
	// Cameras distance to the target 
	float distance = (this->target - this->eye).magnitude();

	// Cameras Field of View 
	float fieldOfView = this->fieldOfView;
	// Projection Frustum Half-Width 
	float theta = (fieldOfView * 0.5f) * DEGREES_TO_RADIANS;
	float halfHeight = 2.0f * distance * tanf(theta);
	float halfWidth = halfHeight * aspectRatio;

	// Re-Calculate the Cameras Direction Vector
	this->direction = Vector(target[VX] - eye[VX], target[VY] - eye[VY], target[VZ] - eye[VZ], 1.0f);
	this->direction.normalize();
	this->direction = direction * distance;

	// Re-Calculate the Cameras Right Vector
	this->right = Vector::crossProduct(direction, up);
	this->right.normalize();
	this->right = right * halfWidth;

	// Re-Calculate the Cameras Up Vector
	this->up = Vector::crossProduct(right, direction);
	this->up.normalize();
	this->up = up * halfHeight;
}

void Camera::loadOrthogonalProjection() {

	this->projectionMode = ORTHOGONAL;

	this->projectionMatrix.loadIdentity();
	this->projectionMatrix.setOrthogonalProjection(ORTHO_LEFT,ORTHO_RIGHT,ORTHO_TOP,ORTHO_BOTTOM,ORTHO_NEAR,ORTHO_FAR);
}

void Camera::loadOrthogonalProjection(GLfloat left, GLfloat right, GLfloat top, GLfloat bottom, GLfloat nearZ, GLfloat farZ) {

	this->projectionMode = ORTHOGONAL;

	this->projectionMatrix.loadIdentity();
	this->projectionMatrix.setOrthogonalProjection(left,right,top,bottom,nearZ,farZ);
}

void Camera::loadPerspectiveProjection() {

	this->projectionMode = PERSPECTIVE;

	this->projectionMatrix.loadIdentity();
	this->projectionMatrix.setPerspectiveProjection(PERS_ANGLE,(GLfloat)this->width/this->height,PERS_NEAR,PERS_FAR);

	this->fieldOfView = PERS_ANGLE;
}

void Camera::loadPerspectiveProjection(GLfloat angle, GLfloat aspectRatio, GLfloat nearZ, GLfloat farZ) {

	this->projectionMode = PERSPECTIVE;

	this->projectionMatrix.loadIdentity();
	this->projectionMatrix.setPerspectiveProjection(angle,aspectRatio,nearZ,farZ);

	this->fieldOfView = angle;
}

string Camera::getName() {

	return this->name;
}

Matrix Camera::getViewMatrix() {

	return this->viewMatrix;
}
Matrix Camera::getProjectionMatrix() {

	return this->projectionMatrix;
}

GLint Camera::getProjectionMode() {

	return this->projectionMode;
}

GLuint Camera::getUniformBufferIndex() {

	return this->uniformBufferIndex;
}

GLint Camera::getWidth() {

	return this->width;
}

GLint Camera::getHeight() {

	return this->height;
}

GLfloat Camera::getZoom() {

	return this->zoom;
}

GLfloat Camera::getLongitude() {

	return this->longitude;
}

GLfloat Camera::getLatitude() {
	
	return this->latitude;
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

void Camera::setName(string name) {

	this->name = name;
}
		
void Camera::setViewMatrix(Matrix viewMatrix) {

	this->viewMatrix = viewMatrix;
}

void Camera::setProjectionMatrix(Matrix projectionMatrix) {

	this->projectionMatrix = projectionMatrix;
}

void Camera::setProjectionMode(GLint projectionMode) {

	this->projectionMode = projectionMode;
}

void Camera::setUniformBufferIndex(GLuint uniformBufferIndex) {

	this->uniformBufferIndex = uniformBufferIndex;
}

void Camera::setWidth(GLint width) {
	
	this->width = width;
}

void Camera::setHeight(GLint height) {

	this->height = height;
}

void Camera::setZoom(GLfloat zoom) {

	this->zoom = zoom;
}

void Camera::setLongitude(GLfloat longitude) {

	this->longitude = longitude;
}

void Camera::setLatitude(GLfloat latitude) {

	this->latitude = latitude;
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

	/* Camera View Matrix */
	cout << "<Camera View Matrix> = " << endl;
	this->viewMatrix.dump();

	/* Camera View Matrix */
	cout << "<Camera Projection Matrix> = " << endl;
	this->projectionMatrix.dump();

	/* Camera Projection Mode (Perspective or Orthogonal)*/
	if(this->projectionMode == ORTHOGONAL)
		cout << "<Camera Projection Mode> = " << "ORTHOGONAL ;" << endl;
	else
		cout << "<Camera Projection Mode> = " << "PERSPECTIVE ;" << endl;

	/* Viewport Width & Height */
	cout << "<Camera Viewport Width> = " << this->width << " ;" << endl;
	cout << "<Camera Viewport Height> = " << this->height << " ;" << endl;

	/* Camera Zoom */
	cout << "<Camera Zoom> = " << this->zoom << " ;" << endl;

	/* Camera Spherical Coordinates */
	cout << "<Camera Longitude> = " << this->longitude << " ;" << endl;
	cout << "<Camera Latitude> = " << this->latitude << " ;" << endl;

	/* Camera Position */
	cout << "<Camera Position> = ";
	this->position.dump();
	
	/* Camera LookAt Vectors */
	cout << "<Camera Eye> = "; 
	this->eye.dump();
	cout << "<Camera Target> = ";
	this->target.dump();
	cout << "<Camera Up> = ";
	this->up.dump();
}