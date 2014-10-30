#include "KeyboardHandler.h"

KeyboardHandler* KeyboardHandler::instance = NULL;

KeyboardHandler::KeyboardHandler() {

	this->keyboardEnabled = false;
}

KeyboardHandler::~KeyboardHandler() {

}

KeyboardHandler* KeyboardHandler::getInstance() {

	if(instance == NULL)
		instance = new KeyboardHandler();

	return instance;
}

void KeyboardHandler::destroyInstance() {

	delete instance;

	instance = NULL;
}

void KeyboardHandler::enableKeyboard() {

	this->previousKeyMap.clear();
	this->previousKeyMap.insert(this->currentKeyMap.begin(),this->currentKeyMap.end());

	this->previousSpecialKeyMap.clear();
	this->previousSpecialKeyMap.insert(this->currentSpecialKeyMap.begin(),this->currentSpecialKeyMap.end());

	this->keyboardEnabled = true;
}

void KeyboardHandler::disableKeyboard() {

	this->keyboardEnabled = false;
}

void KeyboardHandler::releasedNormalKeyListener(unsigned char key, int x, int y) {  

	if(!this->keyboardEnabled)
		return;

	this->currentKeyMap[key] = false;
}

void KeyboardHandler::releasedSpecialKeyListener(int key, int x, int y) {  
	
	if(!this->keyboardEnabled)
		return;

	this->currentSpecialKeyMap[key] = false;
}

void KeyboardHandler::normalKeyListener(unsigned char key, int x, int y) {  

	if(!this->keyboardEnabled)
		return;

	if(this->currentKeyMap[key] == true)
		this->previousKeyMap[key] = true;

	this->currentKeyMap[key] = true;
}  

void KeyboardHandler::specialKeyListener(int key, int x, int y) {  

	if(!this->keyboardEnabled)
		return;

	if(this->currentSpecialKeyMap[key] == true)
		this->previousSpecialKeyMap[key] = true;

	this->currentSpecialKeyMap[key] = true;
}  

bool KeyboardHandler::isKeyPressed(unsigned char key) {

	return this->currentKeyMap[key];
}

bool KeyboardHandler::wasKeyPressed(unsigned char key) {

	return this->previousKeyMap[key];
}

bool KeyboardHandler::wasKeyPressedThisFrame(unsigned char key){
	return(isKeyPressed(key) && !wasKeyPressed(key));	
}
		
bool KeyboardHandler::isSpecialKeyPressed(int key) {

	return this->currentSpecialKeyMap[key];
}

bool KeyboardHandler::wasSpecialKeyPressed(int key) {

	return this->previousSpecialKeyMap[key];
}

bool KeyboardHandler::wasSpecialKeyPressedThisFrame(int key){
	return(isSpecialKeyPressed(key) && !wasSpecialKeyPressed(key));
}

bool KeyboardHandler::isKeyboardEnabled() {

	return this->keyboardEnabled;
}