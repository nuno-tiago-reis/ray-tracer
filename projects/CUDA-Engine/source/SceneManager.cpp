#include "SceneManager.h"

SceneManager* SceneManager::instance = NULL;

SceneManager::SceneManager() {

	this->rotationAxis = 0;
	this->currentObject = 0;

	this->objectID = 0;

	_activeCamera = NULL;

	/* FMOD Sound System Initialization */
	FMOD::System_Create(&this->fmodSystem);

	this->fmodSystem->init(MAX_SOUND_CHANNELS,FMOD_INIT_NORMAL,0);
}

SceneManager::~SceneManager() {

	/* Destroy Scene Objects */
	map<string,SceneNode*>::const_iterator sceneNodeIterator;
	for(sceneNodeIterator = this->sceneNodeMap.begin(); sceneNodeIterator != this->sceneNodeMap.end(); sceneNodeIterator++)
		delete sceneNodeIterator->second;

	/* Destroy Shaders */
	map<string,ShaderProgram*>::const_iterator shaderProgramIterator;
	for(shaderProgramIterator = this->shaderProgramMap.begin(); shaderProgramIterator != this->shaderProgramMap.end(); shaderProgramIterator++)
		delete shaderProgramIterator->second;

	/* Destroy Camera */
	map<string,Camera*>::const_iterator cameraIterator;
	for(cameraIterator = this->cameraMap.begin(); cameraIterator != this->cameraMap.end(); cameraIterator++)
		delete cameraIterator->second;

	/* Destroy Light */
	map<string,Light*>::const_iterator lightIterator;
	for(lightIterator = this->lightMap.begin(); lightIterator != this->lightMap.end(); lightIterator++)
		delete lightIterator->second;

	/* Destroy Sound */
	map<string,Sound*>::const_iterator soundIterator;
	for(soundIterator = this->soundMap.begin(); soundIterator != this->soundMap.end(); soundIterator++)
		delete soundIterator->second;

	/* Sound System Shutdown */
	this->fmodSystem->close();
	this->fmodSystem->release();

	/* Destroy Matrix Stack */
	MatrixStack::destroyInstance();

	/* Destroy User Interaction Handlers */
	MouseHandler::destroyInstance();
	KeyboardHandler::destroyInstance();
}

SceneManager* SceneManager::getInstance() {

	if(instance == NULL)
		instance = new SceneManager();

	return instance;
}

void SceneManager::destroyInstance() {

	delete instance;

	instance = NULL;
}

void SceneManager::init() {

	loadUniforms();

	/* Load Sounds */
	map<string,Sound*>::const_iterator soundIterator;
	for(soundIterator = this->soundMap.begin(); soundIterator != this->soundMap.end(); soundIterator++)
		soundIterator->second->createSound(this->fmodSystem);
}

void SceneManager::loadUniforms() {

	if(_activeCamera == NULL) {
	
		cerr << "Camera not initialized." << endl;
		return;
	}

	/* Load Light Uniforms */
	map<string,Light*>::const_iterator lightIterator;
	for(lightIterator = this->lightMap.begin(); lightIterator != this->lightMap.end(); lightIterator++) {
		lightIterator->second->setUniformBufferIndex(this->shaderProgramMap.begin()->second->getUniformBufferIndex(LIGHT_SOURCES_UNIFORM));
		lightIterator->second->loadUniforms();
	}

	/* Load Camera Uniforms */
	map<string,Camera*>::const_iterator cameraIterator;
	for(cameraIterator = this->cameraMap.begin(); cameraIterator != this->cameraMap.end(); cameraIterator++)
		cameraIterator->second->setUniformBufferIndex(this->shaderProgramMap.begin()->second->getUniformBufferIndex(MATRICES_UNIFORM));

	_activeCamera->loadUniforms();
}

void SceneManager::draw() {

	map<string,SceneNode*>::const_iterator sceneNodeIterator;

	for(sceneNodeIterator = this->sceneNodeMap.begin(); sceneNodeIterator != this->sceneNodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->draw();
}

void SceneManager::update(GLfloat elapsedTime) {
	
	/* Scene Update */
	readMouse(elapsedTime);
	readKeyboard(elapsedTime);

	Object* activeTeapot;

	switch(this->currentObject) {
	
		case 0:	_activeCamera->setPosition(Vector(0.0f,0.0f,0.0f,1.0f));
				_activeCamera->loadPerspectiveProjection();
				_activeCamera->loadView();

				_activeCamera->loadUniforms();

				activeTeapot = this->objectMap[TABLE_SURFACE];
				break;

		case 1:	_activeCamera->setPosition(this->objectMap[BUMP_MAPPING_OBJECT]->getTransform()->getPosition());
				_activeCamera->loadPerspectiveProjection();
				_activeCamera->loadView();

				_activeCamera->loadUniforms();

				activeTeapot = this->objectMap[BUMP_MAPPING_OBJECT];
				break;

		case 2:	_activeCamera->setPosition(this->objectMap[SPHERE_MAPPING_OBJECT]->getTransform()->getPosition());
				_activeCamera->loadPerspectiveProjection();
				_activeCamera->loadView();

				_activeCamera->loadUniforms();

				activeTeapot = this->objectMap[SPHERE_MAPPING_OBJECT];
				break;

		case 3:	_activeCamera->setPosition(this->objectMap[CUBE_MAPPING_OBJECT]->getTransform()->getPosition());
				_activeCamera->loadPerspectiveProjection();
				_activeCamera->loadView();

				_activeCamera->loadUniforms();

				activeTeapot = this->objectMap[CUBE_MAPPING_OBJECT];
				break;

		case 4:	_activeCamera->setPosition(this->objectMap[BLINN_PHONG_OBJECT]->getTransform()->getPosition());
				_activeCamera->loadPerspectiveProjection();
				_activeCamera->loadView();

				_activeCamera->loadUniforms();

				activeTeapot = this->objectMap[BLINN_PHONG_OBJECT];
				break;
	}

	Vector rotation;

	switch(this->rotationAxis) {

		case 1:	rotation = activeTeapot->getTransform()->getRotation();
				rotation[VX] += 50.00f * elapsedTime;
				activeTeapot->getTransform()->setRotation(rotation);
				break;

		case 2:	rotation = activeTeapot->getTransform()->getRotation();
				rotation[VY] += 50.00f * elapsedTime;
				activeTeapot->getTransform()->setRotation(rotation);
				break;

		case 3:	rotation = activeTeapot->getTransform()->getRotation();
				rotation[VZ] += 50.00f * elapsedTime;
				activeTeapot->getTransform()->setRotation(rotation);
				break;
	}

	this->fmodSystem->update();

	/* Update Scene Graph */
	map<string,SceneNode*>::const_iterator sceneNodeIterator;
	for(sceneNodeIterator = this->sceneNodeMap.begin(); sceneNodeIterator != this->sceneNodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->update(elapsedTime);
}

void SceneManager::reshape(GLint width, GLint height) {

	map<string,Camera*>::const_iterator cameraIterator;

	for(cameraIterator = this->cameraMap.begin(); cameraIterator != this->cameraMap.end(); cameraIterator++)
		cameraIterator->second->reshape(width,height);

	loadUniforms();
}

void SceneManager::readKeyboard(GLfloat elapsedTime) {
	
	KeyboardHandler* handler = KeyboardHandler::getInstance();

	if(!handler->isKeyboardEnabled())
		return;	

	handler->disableKeyboard();

	/* Light Buttons */
	if(handler->isSpecialKeyPressed(GLUT_KEY_UP) && handler->wasSpecialKeyPressed(GLUT_KEY_UP) == true) {

		Vector position = this->lightMap["Positional Light 1"]->getPosition();

		position[VY] += 5.0f * elapsedTime;

		this->lightMap["Positional Light 1"]->setPosition(position);
		this->lightMap["Positional Light 1"]->loadUniforms();
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_DOWN) && handler->wasSpecialKeyPressed(GLUT_KEY_DOWN) == true) {

		Vector position = this->lightMap["Positional Light 1"]->getPosition();

		position[VY] -= 5.0f * elapsedTime;

		this->lightMap["Positional Light 1"]->setPosition(position);
		this->lightMap["Positional Light 1"]->loadUniforms();
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_RIGHT) && handler->wasSpecialKeyPressed(GLUT_KEY_RIGHT) == true) {

		Vector position = this->lightMap["Spot Light 2"]->getPosition();

		position[VY] += 5.0f * elapsedTime;

		this->lightMap["Spot Light 2"]->setPosition(position);
		this->lightMap["Spot Light 2"]->loadUniforms();
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_LEFT) && handler->wasSpecialKeyPressed(GLUT_KEY_LEFT) == true) {

		Vector position = this->lightMap["Spot Light 2"]->getPosition();

		position[VY] -= 5.0f * elapsedTime;

		this->lightMap["Spot Light 2"]->setPosition(position);
		this->lightMap["Spot Light 2"]->loadUniforms();
	}

	/* Sound Buttons */
	if(handler->isSpecialKeyPressed(GLUT_KEY_F1) && handler->wasSpecialKeyPressed(GLUT_KEY_F1) == false) {

		this->fmodSystem->playSound(FMOD_CHANNEL_FREE, this->soundMap[RAYQUAZA_SOUND_NAME]->getFmodSound(), false, &channel[0]);
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_F2) && handler->wasSpecialKeyPressed(GLUT_KEY_F2) == false) {

		this->fmodSystem->playSound(FMOD_CHANNEL_FREE, this->soundMap[MUSIC_SOUND_NAME]->getFmodSound(), false, &channel[1]);
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_F3) && handler->wasSpecialKeyPressed(GLUT_KEY_F3) == false) {

			bool paused;

			channel[1]->getPaused(&paused);

			if(paused)
				channel[1]->setPaused(false);
			else
				channel[1]->setPaused(true);
	}

	if(handler->isSpecialKeyPressed(GLUT_KEY_F4) && handler->wasSpecialKeyPressed(GLUT_KEY_F4) == false) {

		channel[1]->stop();
	}

	/* Teapot Selection Button */
	if(handler->isKeyPressed('n'))
		if(!handler->wasKeyPressed('n'))
			this->currentObject = ++this->currentObject % 5;

	/* Teapot Rotation Button */
	if(handler->isKeyPressed('x'))
		if(!handler->wasKeyPressed('x'))
			this->rotationAxis = 1;

	if(handler->isKeyPressed('y'))
		if(!handler->wasKeyPressed('y'))
			this->rotationAxis = 2;

	if(handler->isKeyPressed('z'))
		if(!handler->wasKeyPressed('z'))
			this->rotationAxis = 3;

	if(handler->isKeyPressed('s'))
		if(!handler->wasKeyPressed('s'))
			this->rotationAxis = 0;

	if(handler->wasKeyPressedThisFrame('q'))
		exit(0);
	
	handler->enableKeyboard();
}

void SceneManager::readMouse(GLfloat elapsedTime) {

	MouseHandler* handler = MouseHandler::getInstance();

	if(!handler->isMouseEnabled())
		return;	

	handler->disableMouse();

	GLint zoom = handler->getMouseWheelPosition();
	GLint longitude = handler->getLongitude(GLUT_RIGHT_BUTTON);
	GLint latitude = handler->getLatitude(GLUT_RIGHT_BUTTON);

	_activeCamera->update(zoom,longitude,latitude,elapsedTime);

	if(handler->isButtonPressed(GLUT_LEFT_BUTTON)) {

		GLint* mousePosition = handler->getMouseClickPosition(GLUT_LEFT_BUTTON);

		//rayCast(mousePosition,elapsedTime);

		delete mousePosition;

	}
	else if(handler->wasButtonPressed(GLUT_LEFT_BUTTON)) {

		//_malletPicked = false;
	}

	handler->enableMouse();
}

void SceneManager::rayCast(GLint* mousePosition, GLfloat elapsedTime) {

	/*Matrix invertedProjectionMatrix = _activeCamera->getProjectionMatrix();
	invertedProjectionMatrix.invert();
	invertedProjectionMatrix.transpose();

	Matrix invertedViewMatrix = _activeCamera->getViewMatrix();
	invertedViewMatrix.invert();
	invertedViewMatrix.transpose();

	Vector rayOrigin;
	Vector rayTarget;

	if(_activeCamera->getProjectionMode() == PERSPECTIVE) {
	
		rayOrigin = _activeCamera->getEye();

		rayTarget[VX] = 2.0f * mousePosition[0] / _activeCamera->getWidth() - 1.0f;
		rayTarget[VY] = 1.0f - (2.0f * mousePosition[1]) / _activeCamera->getHeight();
		rayTarget[VZ] = 1.0f;
		rayTarget[VW] = 1.0f;

		rayTarget = invertedProjectionMatrix * rayTarget;

		rayTarget[VW] = 1.0f;

		rayOrigin.clean();
		rayTarget.clean();

		rayTarget = invertedViewMatrix * rayTarget;

		rayTarget.clean();
	}
	else {

		rayOrigin[VX] = 2.0f * mousePosition[0] / _activeCamera->getWidth() - 1.0f;
		rayOrigin[VY] = 1.0f - (2.0f * mousePosition[1]) / _activeCamera->getHeight();
		rayOrigin[VZ] =-1.0f;
		rayOrigin[VW] = 1.0f;

		rayTarget[VX] = 2.0f * mousePosition[0] / _activeCamera->getWidth() - 1.0f;
		rayTarget[VY] = 1.0f - (2.0f * mousePosition[1]) / _activeCamera->getHeight();
		rayTarget[VZ] = 1.0f;
		rayTarget[VW] = 1.0f;

		rayOrigin = invertedProjectionMatrix * rayOrigin;
		rayTarget = invertedProjectionMatrix * rayTarget;

		rayOrigin.clean();
		rayTarget.clean();

		rayOrigin = invertedViewMatrix * rayOrigin;
		rayTarget = invertedViewMatrix * rayTarget;

		rayOrigin.clean();
		rayTarget.clean();
	}

	Vector rayDirection = rayTarget;
	rayDirection -= rayOrigin;
	rayDirection.normalize();
	rayDirection.clean();

	rayOrigin.clean();

	Object* mallet = this->objectMap[MALLET];
	Object* platform = this->objectMap[PLATFORM];

	GLfloat malletIntersectionPoint = mallet->isIntersecting(rayOrigin,rayDirection);
	GLfloat platformIntersectionPoint = platform->isIntersecting(rayOrigin,rayDirection);

	if(malletIntersectionPoint == NULL && platformIntersectionPoint == NULL)
		_malletPicked = false;

	if(malletIntersectionPoint != NULL)
		_malletPicked = true;
				
	if(platformIntersectionPoint != NULL && _malletPicked == true) {

		_malletDepth = platformIntersectionPoint;

		Vector nextPosition = rayOrigin + rayDirection * _malletDepth;
		Vector lastPosition = mallet->getPosition();

		nextPosition[VZ] = 0.0f;

		Vector velocity = nextPosition - lastPosition;

		velocity = velocity * (1.0f / elapsedTime);
		velocity[VW] = 1.0f;

		mallet->setVelocity(velocity);
	}*/

	/*if(platformIntersectionPoint != NULL)
		cout << "Intersected " << platform->getName() << " " << platformIntersectionPoint << " " << rand()%100 << endl;*/
}

int SceneManager::getObjectID() {

	return this->objectID++;
}

void SceneManager::setActiveCamera(Camera* camera) {

	_activeCamera = camera;
}

Camera* SceneManager::getActiveCamera() {

	return _activeCamera;
}

void SceneManager::addCamera(Camera* camera) {

	this->cameraMap[camera->getName()] = camera;
}

void SceneManager::removeCamera(string cameraName) {

	this->cameraMap.erase(cameraName);
}

Camera* SceneManager::getCamera(string cameraName) {

	if(this->cameraMap.find(cameraName) == this->cameraMap.end())
		return NULL;

	return this->cameraMap[cameraName];
}

void SceneManager::addSound(Sound* sound) {

	this->soundMap[sound->getName()] = sound;
}

void SceneManager::removeSound(string soundName) {

	this->soundMap.erase(soundName);
}

Sound* SceneManager::getSound(string soundName) {

	if(this->soundMap.find(soundName) == this->soundMap.end())
		return NULL;

	return this->soundMap[soundName];
}

void SceneManager::addLight(Light* light) {

	this->lightMap[light->getName()] = light;
}

void SceneManager::removeLight(string lightName) {

	this->lightMap.erase(lightName);
}

Light* SceneManager::getLight(string lightName) {

	if(this->lightMap.find(lightName) == this->lightMap.end())
		return NULL;

	return this->lightMap[lightName];
}

void SceneManager::addShaderProgram(ShaderProgram* shaderProgram) {

	this->shaderProgramMap[shaderProgram->getName()] = shaderProgram;
}

void SceneManager::removeShaderProgram(string shaderProgramName) {

	this->shaderProgramMap.erase(shaderProgramName);
}

ShaderProgram* SceneManager::getShaderProgram(string shaderProgramName) {

	if(this->shaderProgramMap.find(shaderProgramName) == this->shaderProgramMap.end())
		return NULL;

	return this->shaderProgramMap[shaderProgramName];
}

void SceneManager::addObject(Object* graphicObject) {

	this->objectMap[graphicObject->getName()] = graphicObject;
}

void SceneManager::removeObject(string graphicObjectName) {

	this->objectMap.erase(graphicObjectName);
}

Object* SceneManager::getObject(string graphicObjectName) {

	if(this->objectMap.find(graphicObjectName) == this->objectMap.end())
		return NULL;

	return this->objectMap[graphicObjectName];
}

void SceneManager::addSceneNode(SceneNode* sceneNode) {

	this->sceneNodeMap[sceneNode->getName()] = sceneNode;
}

void SceneManager::removeSceneNode(string sceneNodeName) {

	this->sceneNodeMap.erase(sceneNodeName);
}

SceneNode* SceneManager::getSceneNode(string sceneNodeName) {

	if(this->sceneNodeMap.find(sceneNodeName) == this->sceneNodeMap.end())
		return NULL;

	return this->sceneNodeMap[sceneNodeName];
}

void SceneManager::dump() {

	cout << "<SceneManager Dump>" << endl;

	/* Active Camera*/
	cout << "<SceneManager Active Camera> = " << endl;
	_activeCamera->dump();

	/* Sound Map */
	cout << "<SceneManager Sound List> = " << endl;
	for(map<string,Sound*>::const_iterator soundMap = this->soundMap.begin(); soundMap != this->soundMap.end(); soundMap++)
		soundMap->second->dump();

	/* Light Map */
	cout << "<SceneManager Light List> = " << endl;
	for(map<string,Light*>::const_iterator lightIterator = this->lightMap.begin(); lightIterator != this->lightMap.end(); lightIterator++)
		lightIterator->second->dump();

	/* Camera Map */
	cout << "<SceneManager Camera List> = " << endl;
	for(map<string,Camera*>::const_iterator cameraIterator = this->cameraMap.begin(); cameraIterator != this->cameraMap.end(); cameraIterator++)
		cameraIterator->second->dump();

	/* Shader Program Map */
	cout << "<SceneManager Shader List> = " << endl;
	for(map<string,ShaderProgram*>::const_iterator shaderProgramIterator = this->shaderProgramMap.begin(); shaderProgramIterator != this->shaderProgramMap.end(); shaderProgramIterator++)
		shaderProgramIterator->second->dump();

	/* Graphic Object Map */
	cout << "<SceneManager Object List> = " << endl;
	for(map<string,Object*>::const_iterator graphicObjectIterator = this->objectMap.begin(); graphicObjectIterator != this->objectMap.end(); graphicObjectIterator++)
		graphicObjectIterator->second->dump();

	/* Scene Node Map */
	cout << "<SceneManager Scene Node List> = " << endl;
	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->sceneNodeMap.begin(); sceneNodeIterator != this->sceneNodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->dump();
}