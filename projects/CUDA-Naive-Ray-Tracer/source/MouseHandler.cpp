#include "MouseHandler.h"

MouseHandler* MouseHandler::instance = NULL;

MouseHandler::MouseHandler() {

	this->mouseEnabled = false;

	this->mouseWheelPosition = 0;
}

MouseHandler::~MouseHandler() {

}

MouseHandler* MouseHandler::getInstance() {

	if(instance == NULL)
		instance = new MouseHandler();

	return instance;
}

void MouseHandler::destroyInstance() {

	delete instance;

	instance = NULL;
}

void MouseHandler::enableMouse() {

	this->previousButtonMap.clear();
	this->previousButtonMap.insert(this->currentButtonMap.begin(),this->currentButtonMap.end());

	this->mouseEnabled = true;
}

void MouseHandler::disableMouse() {

	this->mouseEnabled = false;
}

void MouseHandler::mouseMovementListener(int x, int y) {

	if(!this->mouseEnabled)
		return;

	if(this->currentButtonMap[GLUT_LEFT_BUTTON] == true) {
		
		this->currentLongitudeMap[GLUT_LEFT_BUTTON] = x;
		this->currentLatitudeMap[GLUT_LEFT_BUTTON] = y;
	}

	if(this->currentButtonMap[GLUT_RIGHT_BUTTON] == true) {
		
		this->currentLongitudeMap[GLUT_RIGHT_BUTTON] = x;
		this->currentLatitudeMap[GLUT_RIGHT_BUTTON] = y;
	}

	this->currentLongitudeMap[DEFAULT] = x;
	this->currentLatitudeMap[DEFAULT] = y;
}

void MouseHandler::mousePassiveMovementListener(int x, int y) {

	if(!this->mouseEnabled)
		return;

	this->currentLongitudeMap[DEFAULT] = x;
	this->currentLatitudeMap[DEFAULT] = y;
}

void MouseHandler::mouseEventListener(int button, int state, int x, int y) {  

	if(!this->mouseEnabled)
		return;

	if(state == GLUT_UP)
		this->currentButtonMap[button] = false;
	else if(state == GLUT_DOWN)
		this->currentButtonMap[button] = true;

	this->currentLongitudeMap[button] = this->oldLongitudeMap[button] = x;
	this->currentLatitudeMap[button] = this->oldLatitudeMap[button] = y;

	this->currentLongitudeMap[DEFAULT] = x;
	this->currentLatitudeMap[DEFAULT] = y;
}

void MouseHandler::mouseWheelListener(int button, int direction, int x, int y) {

	if(!this->mouseEnabled)
		return;

	if(direction > 0)
		this->mouseWheelPosition++;
	else
		this->mouseWheelPosition--;
}

bool MouseHandler::isButtonPressed(int button) {

	return this->currentButtonMap[button];
}

bool MouseHandler::wasButtonPressed(int button) {

	return this->previousButtonMap[button];
}

int MouseHandler::getLongitude(int button) {

	GLint longitude = this->currentLongitudeMap[button] - this->oldLongitudeMap[button];

	this->oldLongitudeMap[button] = this->currentLongitudeMap[button];

	return longitude;
}

int MouseHandler::getLatitude(int button) {

	GLint latitude =  this->currentLatitudeMap[button] - this->oldLatitudeMap[button];

	this->oldLatitudeMap[button] = this->currentLatitudeMap[button];

	return latitude;
}

int MouseHandler::getMouseWheelPosition() { 

	GLint position = this->mouseWheelPosition;

	this->mouseWheelPosition = 0;

	return position;
}

int* MouseHandler::getMouseClickPosition(int button) {

	GLint *position = new int[2];

	position[0] = this->currentLongitudeMap[button];
	position[1] = this->currentLatitudeMap[button];

	return position;
}

bool MouseHandler::isOutsideViewport() {

	//cout << "Longitude: " << this->currentLongitudeMap[DEFAULT] / 640 - 1.0f << endl;
	//cout << "Latitude: " << 1.0f - (2.0f * this->currentLatitudeMap[DEFAULT]) / 640 << endl;

	//cout << this->currentLongitudeMap[GLUT_LEFT_BUTTON]  << this->currentLatitudeMap[GLUT_LEFT_BUTTON] << endl;
	//cout << this->currentLongitudeMap[GLUT_RIGHT_BUTTON]  << this->currentLatitudeMap[GLUT_RIGHT_BUTTON] << endl;
	//cout << this->currentLongitudeMap[DEFAULT]  << this->currentLatitudeMap[DEFAULT] << endl;

	return false;
}

bool MouseHandler::isMouseEnabled() {

	return this->mouseEnabled;
}