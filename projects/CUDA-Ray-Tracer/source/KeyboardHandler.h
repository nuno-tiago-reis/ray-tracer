#ifndef KEYBOARD_HANDLER_H
#define KEYBOARD_HANDLER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

#include "GL/glew.h"
#include "GL/freeglut.h"

#include <map>

using namespace std;

class KeyboardHandler {

	private:

		/* Singleton Instance */
		static KeyboardHandler *instance;

		/* Keyboard Enabling Toggle */
		bool keyboardEnabled;

		/* Pressed Key Maps */
		map<unsigned char, bool> currentKeyMap;
		map<unsigned char, bool> previousKeyMap;

		map<int, bool> currentSpecialKeyMap;
		map<int, bool> previousSpecialKeyMap;

		/* Constructors & Destructors - Private due to Singleton */
		KeyboardHandler();
		~KeyboardHandler();

	public:

		/* Singleton Methods */
		static KeyboardHandler* getInstance();
		static void destroyInstance();

		/* Keyboard Enabling/Disabling Methods */
		void enableKeyboard();
		void disableKeyboard();

		/* Listeners */
		void releasedNormalKeyListener(unsigned char key, int x, int y);
		void releasedSpecialKeyListener (int key, int x, int y);

		void normalKeyListener(unsigned char key, int x, int y);
		void specialKeyListener(int key, int x, int y);
		
		/* Getters */
		bool isKeyPressed(unsigned char key);
		bool wasKeyPressed(unsigned char key);
		bool wasKeyPressedThisFrame(unsigned char key);

		bool isSpecialKeyPressed(int key);
		bool wasSpecialKeyPressed(int key);
		bool wasSpecialKeyPressedThisFrame(int key);

		bool isKeyboardEnabled();
};

#endif