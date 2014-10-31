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

		/* Viewport Width & Height */
		GLint width;
		GLint height;

		/* Camera Position */
		GLfloat fieldOfView;

		/* Camera Zoom */
		GLfloat zoom;

		/* Camera Spherical Coordinates */
		GLfloat longitude;
		GLfloat latitude;

		/* Camera Position */
		Vector position;

		/* Look At Vectors */
		Vector target;
		Vector eye;
		Vector up;

	public:

		/* Constructors & Destructors */
		Camera(int width, int height);
		~Camera();

		/* Camera Methods */
		void update(GLint zoom, GLint longitude, GLint latitude, GLfloat elapsedTime);

		void reshape(GLint width, GLint height);

		/* Ray-Tracing Methods */
		Vector getPrimaryRay(GLfloat x, GLfloat y);

		/* Getters */
		GLint getWidth();
		GLint getHeight();

		GLfloat getFieldOfView();

		Vector getPosition();

		Vector getTarget();
		Vector getEye();
		Vector getUp();

		/* Setters */
		void setWidth(GLint width);
		void setHeight(GLint height);

		void setFieldOfView(GLfloat fov);

		void setPosition(Vector position);

		void setTarget(Vector target);
		void setEye(Vector eye);
		void setUp(Vector up);

		/* Debug Methods */
		void dump();
};

#endif