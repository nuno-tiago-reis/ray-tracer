#ifndef SOUND_H
#define	SOUND_H

/* FMODEX Sound Library */
#include "fmod.h"
#include "fmod.hpp"
#include "fmod_errors.h"

#include "Utility.h"

using namespace std;

class Sound {

	protected:

		/* Sound Name */
		string _name;
		/* Sound FileName */
		string _fileName;

		/* FMOD::Sound Handler */
		FMOD::Sound* _fmodSound;

	public:

		/* Constructors & Destructors */
		Sound(string name, string fileName);
		~Sound();

		/* Scene Methods */
		void createSound(FMOD::System* fmodSystem);

		void setMode(FMOD_MODE fmodMode);

		void destroySound();

		/* Getters & Setters */
		string getName();
		string getFileName();

		FMOD::Sound* getFmodSound();

		/* Setters */
		void setName(string name);
		void setFileName(string fileName);

		void setFmodSound(FMOD::Sound* fmodSound);

		/* Debug Methods */
		void dump();
};

#endif