#include "Particle.h"

Particle::Particle(string name) : Object(name) {

	this->lifetime = 0.0f;
	this->maximumLifetime = 0.0f;
}

Particle::~Particle() {
}

void Particle::draw() {

	Object::draw();
}

void Particle::reset(Vector scale) {
	
	this->lifetime = 0.0f;

	this->transform->setScale(scale);
	this->transform->setPosition(Vector(0.0f, 0.0f, 0.0f, 1.0f));
}

void Particle::update() {

	if(this->transform->getPosition().magnitude() > 2.5f)
		this->lifetime = this->maximumLifetime;

	/* Update the Generic Object */
	Object::update();
}

void Particle::update(GLfloat elapsedTime) {

	/* Get the Camera */
	/*Camera* camera = SceneManager::getInstance()->getActiveCamera();

	Vector lookAt = camera->getPosition() - this->transform->getPosition();
	lookAt.normalize();

	Vector right = Vector::crossProduct(camera->getUp(), lookAt);
	right.normalize();

	Vector up = Vector::crossProduct(lookAt, right);
	up.normalize();

	Matrix rotationMatrix;
	
	for(int i=0; i<3; i++) {

		rotationMatrix.setValue(0,i, right[i]);
		rotationMatrix.setValue(1,i, up[i]);
		rotationMatrix.setValue(2,i, lookAt[i]);
		rotationMatrix.setValue(3,i, lookAt[i]);
	}*/

	/* Increment the Particles lifetime */
	this->lifetime += elapsedTime;

	this->transform->setScale(this->transform->getScale() * (1 - elapsedTime / 10.0f));

	if(this->transform->getPosition().magnitude() > 2.5f)
		this->lifetime = this->maximumLifetime;

	/* Update the Generic Object */
	Object::update(elapsedTime);
}

GLfloat Particle::getLifetime() {

	return this->lifetime;
}

GLfloat Particle::getMaximumLifetime() {

	return this->maximumLifetime;
}

void Particle::setLifetime(GLfloat lifetime) {

	this->lifetime = lifetime;
}

void Particle::setMaximumLifetime(GLfloat maximumLifetime) {

	this->maximumLifetime = maximumLifetime;
}

void Particle::dump() {

	cout << "<Particle \"" << name << "\" Dump>" << endl;

	/* Particle Lifetime & Maximum Lifetime */
	cout << "<Particle Lifetime> = " << this->lifetime << endl;
	cout << "<Particle MaximumLifetime> = " << this->maximumLifetime << endl;

	Object::dump();
}