#include "GL/glew.h"
#include "GL/freeglut.h"

#include <iostream> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "Constants.h"
#include "Matrix.h"
#include "Vector.h"

#include "NFF_Reader.h"

/* Regular Ray Tracing Constants */
#define DEPTH 6
#define AIR_REFRACTION_INDEX 1.0f

/* Monte-Carlo Anti-Aliasing Constants */
#define MONTE_CARLO_DEPTH 3
#define MONTE_CARLO_THRESHOLD 0.3f

/* Depth of Field Constants */
#define DOF true

#define DOF_FOCAL_DISTANCE 2.0f
#define DOF_APERTURE 0.1f
#define DOF_HALF_APERTURE (DOF_APERTURE / 2.0f)

#define DOF_SAMPLING 4

#define DOF_CAMERA_POINTS 4

//#define MODEL_FILE "balls_medium.nff"
//#define MODEL_FILE "aabb_low.nff"
//#define MODEL_FILE "balls_medium.nff"
//#define MODEL_FILE "dof_low.nff"
//#define MODEL_FILE "grid_low.nff"
//#define MODEL_FILE "mount_high.nff"
#define MODEL_FILE "balls_low.nff"

#define BOTTOM 0
#define RIGHT 1
#define TOP 2
#define LEFT 3

#define BOTTOM_LEFT 0
#define BOTTOM_RIGHT 1
#define TOP_RIGHT 2
#define TOP_LEFT 3

int width = 0;
int height = 0;

Scene* scene = NULL;

void reshape(int w, int h) { 

	glClearColor(0.0, 0.0, 0.0, 0.0); 
	glClear(GL_COLOR_BUFFER_BIT); 
	glViewport(0, 0, w, h); 
	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity(); 
 
	gluOrtho2D(0, width, 0, height); 
	glMatrixMode (GL_MODELVIEW); 
	glLoadIdentity(); 
}

const std::string currentDateTime() {

	time_t     now = time(0);
	struct tm  tstruct;
	char       buffer[80];

	tstruct = *localtime(&now);

	strftime(buffer, sizeof(buffer), "%Y-%m-%d.%X", &tstruct);

	return buffer;
}

/* Monte Carlo Anti-Aliasing */
Vector monteCarloSubDivision(Vector rayOrigin, int depth, float ior, float x, float y, float size, int monteDepth,
								Vector* cornerColors) {

	/* Check if Anti-Aliasing is necessary */
	bool subDivide = false;

	/* Comparing every other corner with the bottom-left corner */
	Vector colorDifference = cornerColors[BOTTOM_LEFT] - cornerColors[BOTTOM_RIGHT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	colorDifference = cornerColors[BOTTOM_LEFT] - cornerColors[TOP_RIGHT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	colorDifference = cornerColors[BOTTOM_LEFT] - cornerColors[TOP_LEFT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	/* Comparing every other corner with the bottom-right corner */
	colorDifference = cornerColors[BOTTOM_RIGHT] - cornerColors[TOP_RIGHT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	colorDifference = cornerColors[BOTTOM_RIGHT] - cornerColors[TOP_LEFT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	/* Comparing the top-right with the top-left corner */
	colorDifference = cornerColors[TOP_RIGHT] - cornerColors[TOP_LEFT];
	if(subDivide == false && abs(colorDifference[VX]) + abs(colorDifference[VY]) + abs(colorDifference[VZ]) > MONTE_CARLO_THRESHOLD)
		subDivide = true;

	Vector color;

	/* If Anti-Aliasing isn't necessary or if maximum depth was reached */
	if(subDivide == false || monteDepth == 0) {

		for(int i=0;i<4;i++)
			color += cornerColors[i] * (1.0f/4.0f);
	}
	else {

		Vector centerColor;
		Vector edgeColors[4];

		/* Calculate the Pixels center color */
		Vector rayDirection = scene->getCamera()->getPrimaryRay(x, y);
		centerColor = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);

		/* Calculate the Pixels edge colors */
		rayDirection = scene->getCamera()->getPrimaryRay(x,y - size/2.0f);
		edgeColors[BOTTOM] = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);

		rayDirection = scene->getCamera()->getPrimaryRay(x + size/2.0f,y);
		edgeColors[RIGHT] = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);

		rayDirection = scene->getCamera()->getPrimaryRay(x,y + size/2.0f);
		edgeColors[TOP] = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);

		rayDirection = scene->getCamera()->getPrimaryRay(x - size/2.0f,y);
		edgeColors[LEFT] = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);

		/* Bottom Left Corner */
		float bottomLeftCenterX = x - size/4.0f;
		float bottomLeftCenterY = y - size/4.0f;

		Vector bottomLeftCorners[4];
		bottomLeftCorners[BOTTOM_LEFT] = cornerColors[BOTTOM_LEFT];
		bottomLeftCorners[BOTTOM_RIGHT] = edgeColors[BOTTOM];
		bottomLeftCorners[TOP_RIGHT] = centerColor;
		bottomLeftCorners[TOP_LEFT] = edgeColors[LEFT];

		color += monteCarloSubDivision(rayOrigin,DEPTH, AIR_REFRACTION_INDEX,
			bottomLeftCenterX,bottomLeftCenterY,size/2.0f,monteDepth-1,bottomLeftCorners) * (1.0f / 4.0f);

		/* Bottom Right Corner */
		float bottomRightCenterX = x + size/4.0f;
		float bottomRightCenterY = y - size/4.0f;

		Vector bottomRightCorners[4];
		bottomRightCorners[BOTTOM_LEFT] = edgeColors[BOTTOM];
		bottomRightCorners[BOTTOM_RIGHT] = cornerColors[BOTTOM_RIGHT];
		bottomRightCorners[TOP_RIGHT] = edgeColors[RIGHT];
		bottomRightCorners[TOP_LEFT] = centerColor;

		color += monteCarloSubDivision(rayOrigin,DEPTH, AIR_REFRACTION_INDEX,
			bottomRightCenterX,bottomRightCenterY,size/2.0f,monteDepth-1,bottomRightCorners) * (1.0f / 4.0f);

		/* Top Right Corner */
		float topRightCenterX = x + size/4.0f;
		float topRightCenterY = y + size/4.0f;

		Vector topRightCorners[4];
		topRightCorners[BOTTOM_LEFT] = centerColor;
		topRightCorners[BOTTOM_RIGHT] = edgeColors[RIGHT];
		topRightCorners[TOP_RIGHT] = cornerColors[TOP_RIGHT];
		topRightCorners[TOP_LEFT] = edgeColors[TOP];

		color += monteCarloSubDivision(rayOrigin,DEPTH, AIR_REFRACTION_INDEX,
			topRightCenterX,topRightCenterY,size/2.0f,monteDepth-1,topRightCorners) * (1.0f / 4.0f);

		/* Top Left Corner */
		float topLeftCenterX = x - size/4.0f;
		float topLeftCenterY = y + size/4.0f;

		Vector topLeftCorners[4];
		topLeftCorners[BOTTOM_LEFT] = edgeColors[LEFT];
		topLeftCorners[BOTTOM_RIGHT] = centerColor;
		topLeftCorners[TOP_RIGHT] = edgeColors[TOP];
		topLeftCorners[TOP_LEFT] = cornerColors[TOP_LEFT];

		color += monteCarloSubDivision(rayOrigin,DEPTH, AIR_REFRACTION_INDEX,
			topLeftCenterX,topLeftCenterY,size/2.0f,monteDepth-1,topLeftCorners) * (1.0f / 4.0f);
	}

	return color;
}

void drawScene() {

	/* Time Measurement */
	cout << currentDateTime() << endl;

	/* Create a Buffer so we don't calculate the same pixel twice */
	map<int,Vector> colorBuffer;

	/* Camera Eye */
	Vector rayOrigin = scene->getCamera()->getEye();

	for(int y=0; y < height; y++) {

		for(int x=0; x < width; x++) {

			/* Calculate the Pixels corner colors */
			for(int i=0;i<2;i++) {

				for(int j=0;j<2;j++) {

					if(colorBuffer.find((x + j) + (y + i) * width) == colorBuffer.end()) {

						Vector rayDirection = scene->getCamera()->getPrimaryRay((float)(x + i),(float)(y + j));
						colorBuffer[(x + j) + (y + i) * width] = scene->rayTracing(rayOrigin, rayDirection, DEPTH, AIR_REFRACTION_INDEX);
					}
				}
			}

			/* Pixel Corners */
			Vector corners[4];

			corners[BOTTOM_LEFT] = colorBuffer[x + y * width];
			corners[BOTTOM_RIGHT] = colorBuffer[(x + 1) + y * width];
			corners[TOP_RIGHT] = colorBuffer[(x + 1) + (y + 1) * width];
			corners[TOP_LEFT] = colorBuffer[x + (y + 1) * width];

			/* Monte-Carlo Anti-Aliasing */
			Vector color = monteCarloSubDivision(rayOrigin,DEPTH, AIR_REFRACTION_INDEX,x+0.5f,y+0.5f,1.0f,MONTE_CARLO_DEPTH,corners);

			/* Paint the Pixel */
			glBegin(GL_POINTS); 
				glColor3f(color[VX], color[VY], color[VZ]); 
				glVertex2i(x, y); 
			glEnd();
		}

		glFlush();
	}

	cout << "Terminou! " << endl; 

	cout << currentDateTime() << endl;
}

void drawSceneDOF() {

	/* Camera */
	Camera* camera = scene->getCamera();

	/* Ray Origin */
	Vector rayOrigin = camera->getEye();

	GLfloat focalDistance = DOF_FOCAL_DISTANCE;

	/* Camera Vectors */
	Vector cameraUp = camera->getUp();
	cameraUp.normalize();

	Vector cameraLookAt = camera->getTarget() - rayOrigin;
	cameraLookAt.normalize();

	Vector cameraRight = Vector::crossProduct(cameraUp,cameraLookAt);
	cameraRight.normalize();

	Matrix viewMatrix;
	viewMatrix.setView(camera->getEye(),camera->getTarget(),camera->getUp());

	viewMatrix.setValue(0,3,0.0f);
	viewMatrix.setValue(1,3,0.0f);
	viewMatrix.setValue(2,3,0.0f);

	viewMatrix.setValue(3,0,0.0f);
	viewMatrix.setValue(3,1,0.0f);
	viewMatrix.setValue(3,2,0.0f);

	for (int y=0; y < height; y++) {

		for (int x=0; x < width; x++) {

			Vector color;

			for(int p=0;p<DOF_SAMPLING;p++) {

				for(int q=0;q<DOF_SAMPLING;q++) {		

					float xCoordinate = (float)x + ((float)p+0.5f)/(float)DOF_SAMPLING;
					float yCoordinate = (float)y + ((float)q+0.5f)/(float)DOF_SAMPLING;

					Vector rayDirection = scene->getCamera()->getPrimaryRay((float)xCoordinate,(float)yCoordinate);
					rayDirection.normalize();

					Vector viewSpaceRayDirection = viewMatrix * rayDirection;
					viewSpaceRayDirection.normalize();

					/* Calculate Time of Impact with the Focal Plane */
					GLfloat t = focalDistance / fabs(viewSpaceRayDirection[VZ]);

					/* Calculate the Focal Point using the Time of Impact */
					Vector focalPoint = rayOrigin + rayDirection * t;

					/* Sampling with Interpolated Points in a Grid with the Camera Eye as the Center */
					for(int i=0;i<DOF_CAMERA_POINTS;i++) {

						for(int j=0;j<DOF_CAMERA_POINTS;j++) {

							/* Calculate the Interpolated Points */ 
							Vector interpolatedOrigin = rayOrigin - cameraRight * DOF_HALF_APERTURE - cameraUp * DOF_HALF_APERTURE;
							interpolatedOrigin += cameraRight * (i * DOF_APERTURE / DOF_CAMERA_POINTS) + cameraUp * (j * DOF_APERTURE / DOF_CAMERA_POINTS);

							rayDirection = focalPoint - interpolatedOrigin;
							rayDirection.normalize();

							color += scene->rayTracing(interpolatedOrigin,rayDirection, DEPTH, AIR_REFRACTION_INDEX) * (1.0f / (DOF_CAMERA_POINTS * DOF_CAMERA_POINTS)) * (1.0f/(pow((float)DOF_SAMPLING,2)));
						}
					}
				}
			}

			glBegin(GL_POINTS); 
				glColor3f(color[VX], color[VY], color[VZ]); 
				glVertex2i(x, y); 
			glEnd();
		}

		glFlush();
	}

	printf("Terminou!\n"); 
}

int main(int argc, char**argv) {

	scene = new Scene();

	/* Parse the NFF File */
	NFF_Reader* nffReader = NFF_Reader::getInstance();
	nffReader->parseNFF(MODEL_FILE, scene);

	/* Initialized the Grid */
	scene->initializeGrid();

	/* Scene Camera - Defined in the NFF File */
	Camera* camera = scene->getCamera();

	/* Viewport Dimensions */
	width = camera->getWidth();
	height = camera->getHeight();

	/* Initialize GLUT */
	glutInit(&argc, argv); 
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA); 
 
	glutInitWindowSize(width, height); 
	glutInitWindowPosition(50, 50); 

	glutCreateWindow("Ray Tracing"); 
	glClearColor(0, 0, 0, 1); 
	glClear(GL_COLOR_BUFFER_BIT); 
 
	/* Initialize Callbacks */
	glutReshapeFunc(reshape); 
	if(DOF == false)
		glutDisplayFunc(drawScene); 
	else
		glutDisplayFunc(drawSceneDOF); 
	glDisable(GL_DEPTH_TEST); 
 
	glutMainLoop();

	return 0;
} 