#ifndef MOUSE_HANDLER_H
#define MOUSE_HANDLER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

#include "GL/glew.h"
#include "GL/freeglut.h"

#include <map>
#include <iostream>

using namespace std;

#define DEFAULT 0xFFFFFFFF

class MouseHandler {

	private:

		/* Singleton Instance */
		static MouseHandler* instance;

		/* Mouse Enabling Toggle */
		bool mouseEnabled;

		/* Mouse Wheel Position */
		GLint mouseWheelPosition;

		/* Pressed Button Maps */
		map<int, bool> currentButtonMap;
		map<int, bool> previousButtonMap;

		/* Mouse Movement Attributes */
		map<int, int> oldLongitudeMap;
		map<int, int> oldLatitudeMap;

		map<int, int> currentLongitudeMap;
		map<int, int> currentLatitudeMap;

		/* Constructors & Destructors - Private due to Singleton */
		MouseHandler();
		~MouseHandler();

	public:

		/* Singleton Methods */
		static MouseHandler* getInstance();
		static void destroyInstance();

		/* Mousing Enabling/Disabling Methods */
		void enableMouse();
		void disableMouse();

		/* Listeners */
		void mouseMovementListener(int x, int y);
		void mousePassiveMovementListener(int x, int y);

		void mouseEventListener(int button, int state, int x, int y);
		void mouseWheelListener(int button, int direction, int x, int y);

		/* Getters */
		bool isButtonPressed(int button);
		bool wasButtonPressed(int button);

		int getLongitude(int button);
		int getLatitude(int button);
		
		int getMouseWheelPosition();

		int* getMouseClickPosition(int button);

		bool isOutsideViewport();

		bool isMouseEnabled();
};

#endif