#ifndef CAMERA_H
#define CAMERA_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL Includes */
#include "GL/glew.h"
#include "GL/glut.h"

/* Math Library */
#include "Vector.h"

/* Camera Radius */
#define CAMERA_RADIUS 25.0f

class Camera {

	private:

		/* Viewports Dimensions */
		unsigned int width;
		unsigned int height;

		/* Cameras Field of View */
		float fieldOfView;

		/* Cameras Zoom */
		float zoom;
		/* Camera Spherical Coordinates */
		float longitude;
		float latitude;

		/* Camera Position */
		Vector position;

		/* Camera Direction Vectors */
		Vector target;
		Vector eye;

		/* Camera Plane Vectors */
		Vector up;
		Vector right;
		Vector direction;

	public:

		/* Constructors & Destructors */
		Camera(unsigned int  width, unsigned int  height);
		~Camera();

		/* Camera Methods */
		void update(int zoom, int longitude, int latitude, float elapsedTime);

		/* Getters */
		unsigned int getWidth();
		unsigned int getHeight();

		float getFieldOfView();

		Vector getPosition();

		Vector getTarget();
		Vector getEye();

		Vector getUp();
		Vector getRight();
		Vector getDirection();

		/* Setters */
		void setWidth(unsigned int width);
		void setHeight(unsigned int height);

		void setFieldOfView(float fieldOfView);

		void setPosition(Vector position);

		void setTarget(Vector target);
		void setEye(Vector eye);

		void setUp(Vector up);
		void setRight(Vector right);
		void setDirection(Vector direction);

		/* Debug Methods */
		void dump();
};

#endif