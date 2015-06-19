#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(string name, GLint particleNumber, GLfloat particleMaximumLifetime) : Object(name) {

	/* Initialize the Particle Attributes */
	this->particleNumber = particleNumber;

	this->particleMaximumLifetime = particleMaximumLifetime;
}

ParticleSystem::~ParticleSystem() {
}

void ParticleSystem::init() {

	/* Initialize all the Particles */
	for(int i=0; i < this->particleNumber; i++) {

		GLfloat radius = (rand() % 8000 + 2000) / 5000.0f;
		GLfloat rho = (rand() % 2000 - 1000) / 2000.0f * PI * 0.5f + PI * 0.5f;
		GLfloat phi = (rand() % 2000 - 1000) / 2000.0f * (2.0f * PI);

		ostringstream stringStream;
		stringStream << "Particle " << i;
		
		//cout << "Particle " << i << " radius = " << radius << " rho = " << rho << " phi = " << phi << endl;

		Particle* particle = new Particle(stringStream.str());

			/* Set the Particles Mesh - Same for all Particles */
			particle->setMesh(this->mesh);

			/* Set the Particles Transform */
			Transform* particleTransform = new Transform(stringStream.str());

				/* Particles Velocity - Random for each particle */
				particleTransform->setVelocity(Vector(radius * cos(phi) * 0.25f, radius * sin(rho), radius * sin(phi) * 0.25f, 1.0f) );

				/* Particles Scale - Same for each particle */
				particleTransform->setScale(Vector(0.25f, 0.25f, 0.25f, 0.0f));

			particle->setTransform(particleTransform);

			/* Set the Particles Material - Same for all Particles */
			particle->setMaterial(this->material);

			/* Set the Particles Maximum Lifetime - Random for each particle */
			GLfloat lifetime = this->particleMaximumLifetime + (rand() % 10) / 10.0f;

			particle->setMaximumLifetime(lifetime);

		addParticle(particle);
	}
}

void ParticleSystem::draw() {

	for(map<string,Particle*>::const_iterator particleIterator = this->particleMap.begin(); particleIterator != this->particleMap.end(); particleIterator++)
		particleIterator->second->draw();
}

void ParticleSystem::update() {

	/* Update the Generic Object */
	Object::update();

	/* Update Particles */
	for(map<string,Particle*>::const_iterator particleIterator = this->particleMap.begin(); particleIterator != this->particleMap.end(); particleIterator++) {
	
		Transform* particleTransform = particleIterator->second->getTransform();
		particleTransform->setModelMatrix(this->transform->getModelMatrix());

		if(particleIterator->second->getMaximumLifetime() <= particleIterator->second->getLifetime())
			particleIterator->second->reset(Vector(0.25f,0.25f,0.25f,1.0f));

		particleIterator->second->update();
	}
}

void ParticleSystem::update(GLfloat elapsedTime) {

	/* Update the Generic Object */
	Object::update(elapsedTime);

	/* Update Particles */
	for(map<string,Particle*>::const_iterator particleIterator = this->particleMap.begin(); particleIterator != this->particleMap.end(); particleIterator++) {
	
		Transform* particleTransform = particleIterator->second->getTransform();
		particleTransform->setModelMatrix(this->transform->getModelMatrix());

		if(particleIterator->second->getMaximumLifetime() <= particleIterator->second->getLifetime())
			particleIterator->second->reset(Vector(0.25f,0.25f,0.25f,1.0f));

		particleIterator->second->update(elapsedTime);
	}
}

GLint ParticleSystem::getParticleNumber() {

	return this->particleNumber;
}

GLfloat ParticleSystem::getParticleMaximumLifetime() {

	return this->particleMaximumLifetime;
}

void ParticleSystem::setParticleNumber(GLint particleNumber) {

	this->particleNumber = particleNumber;
}

void ParticleSystem::setParticleMaximumLifetime(GLfloat particleMaximumLifetime) {

	this->particleMaximumLifetime = particleMaximumLifetime;
}

void ParticleSystem::addParticle(Particle* particle) {

	this->particleMap[particle->getName()] = particle;
}

void ParticleSystem::removeParticle(string particleName) {

	this->particleMap.erase(particleName);
}

Particle* ParticleSystem::getParticle(string particleName) {

	return this->particleMap[particleName];
}

void ParticleSystem::dump() {

	cout << "<ParticleSystem Dump> = " << name << endl;

	/* Particle Attributes */
	cout << "<ParticleSystem Particle Number> = " << this->particleNumber << endl;
	cout << "<ParticleSystem Particle Maximum Lifetime> = " << this->particleMaximumLifetime << endl;

	/* Particle Map */
	cout << "<ParticleSystem Particle Map> = " << endl;
	for(map<string,Particle*>::const_iterator particleIterator = this->particleMap.begin(); particleIterator != this->particleMap.end(); particleIterator++)
		particleIterator->second->dump();
}