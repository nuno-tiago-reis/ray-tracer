#include "Transform.h"

Transform::Transform(string name) {

	/* Initialize the Transforms Name */
	this->name = name;

	/* Initialize the Transforms Model Matrix */
	this->modelMatrix.loadIdentity();
	
	/* Initialize the Transforms Position and Velocity */
	this->position = Vector(0.0f, 0.0f, 0.0f, 1.0f);
	this->velocity = Vector(0.0f, 0.0f, 0.0f, 1.0f);

	/* Initialize the Transforms Scale */
	this->scale = Vector(1.0f, 1.0f, 1.0f, 1.0f);

	/* Initialize the Transforms Rotation and Rotation Quaternion */
	this->rotation = Vector(0.0f, 0.0f, 0.0f, 1.0f);

	float xAxis[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	float yAxis[] = { 0.0f, 1.0f, 0.0f, 1.0f };
	float zAxis[] = { 0.0f, 0.0f, 1.0f, 1.0f };

	Quaternion xQuaternion(this->rotation[0],xAxis);
	Quaternion yQuaternion(this->rotation[1],yAxis);
	Quaternion zQuaternion(this->rotation[2],zAxis);

	this->rotationQuaternion = xQuaternion;
	this->rotationQuaternion *= yQuaternion;
	this->rotationQuaternion *= zQuaternion;
}

Transform::~Transform() {
}

void Transform::update() {

	/* Update the Model Matrix */
	this->modelMatrix.loadIdentity();

	this->modelMatrix.translate(this->position[0],this->position[1],this->position[2]);

	this->modelMatrix.quaternionRotate(this->rotationQuaternion);

	this->modelMatrix.scale(this->scale);
}

void Transform::update(float elapsedTime) {

	/* Update Position */
	for(int i=0; i<3; i++)
		this->position[i] += this->velocity[i]*elapsedTime;

	/* Update the Model Matrix */
	this->modelMatrix.translate(this->position);
		
	this->modelMatrix.quaternionRotate(this->rotationQuaternion);

	this->modelMatrix.scale(this->scale);
}

string Transform::getName() {

	return this->name;
}

Matrix Transform::getModelMatrix() {

	return this->modelMatrix;
}

Vector Transform::getPosition() {

	return this->position;
}

Vector Transform::getVelocity() {

	return this->velocity;
}

Vector Transform::getScale() {

	return this->scale;
}

Vector Transform::getRotation() {

	return this->rotation;
}

Quaternion Transform::getRotationQuaternion() {

	return this->rotationQuaternion;
}

void Transform::setName(string name) {

	this->name = name;
}

void Transform::setModelMatrix(Matrix modelMatrix) {

	this->modelMatrix = modelMatrix;
}

void Transform::setPosition(Vector position) {

	this->position = position;

	update();
}
void Transform::setVelocity(Vector velocity) {

	this->velocity = velocity;

	update();
}

void Transform::setScale(Vector scale) {

	this->scale = scale;

	update();
}

void Transform::setRotation(Vector rotation) {

	this->rotation = rotation;

	float xAxis[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	float yAxis[] = { 0.0f, 1.0f, 0.0f, 1.0f };
	float zAxis[] = { 0.0f, 0.0f, 1.0f, 1.0f };

	Quaternion xQuaternion(this->rotation[0],xAxis);
	Quaternion yQuaternion(this->rotation[1],yAxis);
	Quaternion zQuaternion(this->rotation[2],zAxis);

	this->rotationQuaternion = xQuaternion;
	this->rotationQuaternion *= yQuaternion;
	this->rotationQuaternion *= zQuaternion;

	update();
}

void Transform::setRotationQuaternion(Quaternion rotationQuaternion) {

	this->rotationQuaternion = rotationQuaternion;

	update();
}

void Transform::dump() {

	cout << "<Transform \"" << this->name << "\" Dump>" << endl;

	/* Transform Model Matrix */
	cout << "<Transform ModelMatrix> = " << endl;
	this->modelMatrix.dump();

	/* Transform Position & Velocity*/
	cout << "<Transform Position> = ";
	this->position.dump();
	cout << "<Transform Velocity> = ";
	this->velocity.dump();

	/* Transform Scale Factor*/
	cout << "<Transform Rotation> = ";
	this->scale.dump();

	/* Transform Rotation & Rotation Quaternion  */
	cout << "<Transform Rotation> = ";
	this->rotation.dump();
	cout << "<Transform Rotation Quaternion> = ";
	this->rotationQuaternion.dump();
}