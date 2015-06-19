#ifndef CAMERA_H
#define CAMERA_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"

/* C++ Includes */
#include <string>

/* Math Library */
#include "Matrix.h"

/* Orthogonal Camera */
#define ORTHOGONAL_NAME "ORTHOGONAL"
#define ORTHOGONAL 0

#define ORTHO_LEFT -10.0f
#define ORTHO_RIGHT 10.0f
#define ORTHO_TOP 10.0f
#define ORTHO_BOTTOM -10.0f
#define ORTHO_FAR 30.0f
#define ORTHO_NEAR 1.0f

/* Perspective Camera */
#define PERSPECTIVE_NAME "PERSPECTIVE"
#define PERSPECTIVE 1

#define PERS_ANGLE 30.0f
#define PERS_NEAR 1.0f
#define PERS_FAR 100.0f

/* Camera Radius */
#define CAMERA_RADIUS 22.5f

class Camera {

	private:

		/* Camera Name */
		string name;

		/* Scene View and Projection Matrix */
		Matrix viewMatrix;
		Matrix projectionMatrix;

		/* Scene Projection Mode (Perspective or Orthogonal) */
		GLint projectionMode;

		/* Shader Buffer Index for View & Projection Matrix */
		GLuint uniformBufferIndex;

		/* Viewport Width & Height */
		GLint width;
		GLint height;

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
		Camera(string name);
		~Camera();

		/* Camera Methods */
		void update(GLint zoom, GLint longitude, GLint latitude, GLfloat elapsedTime);

		void reshape(GLint width, GLint height);

		/* View & Projection Methods */
		void loadUniforms();

		void loadView();

		void loadOrthogonalProjection();
		void loadOrthogonalProjection(GLfloat left, GLfloat right, GLfloat top, GLfloat bottom, GLfloat nearZ, GLfloat farZ);

		void loadPerspectiveProjection();
		void loadPerspectiveProjection(GLfloat angle, GLfloat aspectRatio, GLfloat nearZ, GLfloat farZ);

		/* Getters */
		string getName();
		
		Matrix getViewMatrix();
		Matrix getProjectionMatrix();

		GLint getProjectionMode();
		GLuint getUniformBufferIndex();

		GLint getWidth();
		GLint getHeight();

		GLfloat getZoom();
		GLfloat getLongitude();
		GLfloat getLatitude();

		Vector getPosition();

		Vector getTarget();
		Vector getEye();
		Vector getUp();

		/* Setters */
		void setName(string name);
		
		void setViewMatrix(Matrix viewMatrix);
		void setProjectionMatrix(Matrix projectionMatrix);

		void setProjectionMode(GLint projectionMode);
		void setUniformBufferIndex(GLuint uniformBufferIndex);

		void setWidth(GLint width);
		void setHeight(GLint height);

		void setZoom(GLfloat zoom);
		void setLongitude(GLfloat longitude);
		void setLatitude(GLfloat latitude);

		void setPosition(Vector position);

		void setTarget(Vector target);
		void setEye(Vector eye);
		void setUp(Vector up);

		/* Debug Methods */
		void dump();
};

#endif