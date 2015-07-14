#include "Sound.h"

Sound::Sound(string name, string fileName) {

	_name = name;
	_fileName = fileName;

	_fmodSound = NULL;
}

Sound::~Sound() {

	//TODO
}

void Sound::createSound(FMOD::System* fmodSystem) {

	fmodSystem->createSound(_fileName.c_str(), FMOD_HARDWARE, 0, &_fmodSound);
}

void Sound::setMode(FMOD_MODE fmodMode) {

	if(_fmodSound == NULL) {
	
		cerr << "FMOD Sound not initialized." << endl;
		return;
	}
	
	_fmodSound->setMode(fmodMode);
}

void Sound::destroySound() {

	_fmodSound->release();
}

string Sound::getName() {
	
	return _name;
}

string Sound::getFileName() {
	
	return _fileName;
}

FMOD::Sound* Sound::getFmodSound() {

	return _fmodSound;
}

void Sound::setName(string name) {
	
	_name = name;
}

void Sound::setFileName(string fileName) {

	_fileName = fileName;
}

void Sound::setFmodSound(FMOD::Sound* fmodSound) {
	
	_fmodSound = fmodSound;
}

void Sound::dump() {

	cout << "<Sound \"" << _name << "\" Dump>" << endl;
}
