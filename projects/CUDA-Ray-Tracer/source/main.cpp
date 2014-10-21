/* OpenGL definitions */
#include <GL/glew.h>
#include <GL/glut.h>

/* CUDA definitions */
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "vector_types.h"
#include "vector_functions.h"
//Custom
#include "helper_cuda.h"
#include "helper_math.h"

/* C++ Includes */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

/* Custom Includes */
#include "Object.h"
#include "Camera.h"

/* Math Library */
#include "Matrix.h"

/* Texture Library */
#include <soil.h>

/* Error Checking */
#include "Utility.h"

/* User Interaction */
#include "MouseHandler.h"
#include "KeyboardHandler.h"

using namespace std;

// the interface between C++ and CUDA -----------
#define DEPTH 3

// the implementation of RayTraceImage is in the "raytracer.cu" file
extern "C" void RayTraceImage(unsigned int *outputPixelBufferObject, 
								int width, int height, 
								int triangleTotal,
								float3 cameraRight, float3 cameraUp, float3 cameraDirection,
								float3 cameraPosition,
								float3 lightPosition,
								float3 lightColor);

// the implementation of bindTrianglePositions is in the "raytracer.cu" file
extern "C" void bindTrianglePositions(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleNormals is in the "raytracer.cu" file
extern "C" void bindTriangleNormals(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleTangents is in the "raytracer.cu" file
extern "C" void bindTriangleTangents(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleTextureCoordinates is in the "raytracer.cu" file
extern "C" void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleDiffuseProperties is in the "raytracer.cu" file
extern "C" void bindTriangleDiffuseProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleSpecularProperties is in the "raytracer.cu" file
extern "C" void bindTriangleSpecularProperties(float *cudaDevicePointer, unsigned int triangleTotal);

/* Global Variables */
unsigned int windowWidth  = 1024;
unsigned int windowHeight = 1024;

unsigned int imageWidth   = 1024;
unsigned int imageHeight  = 1024;

int frameCount = 0;
int windowHandle = 0;

/* Camera */
Camera* camera;

/* Objects */
Object* spheres[4];
Object* platform;

/* Object TextureIDs */
GLuint chessTextureID;

				// Lighting ---------------------------- TODO
				float lightPosition[3] = { 0.0f, 5.0f, 0.0f };
				float lightColor[3] = { 1.0f, 1.0f, 1.0f };

/* Scene Time Management */
int lastFrameTime = 0;
float deltaTime = 0.0f;

/* PixelBufferObjects ID and Cuda Resource */
GLuint pixelBufferObjectID;
cudaGraphicsResource *pixelBufferObjectResource = NULL;
/* Screens Texture ID and Cuda Resource */
GLuint screenTextureID;
cudaGraphicsResource *screenTextureResource = NULL;

/* Total number of Triangles - Used for the memory necessary to allocate */
int triangleTotal = 0;

/* CudaDevicePointers to the uploaded Triangles Positions */
float *cudaTrianglePositionsDP = NULL; 
/* CudaDevicePointers to the uploaded Triangles Normals and Tangents */
float *cudaTriangleNormalsDP = NULL;
float *cudaTriangleTangentsDP = NULL;
/* CudaDevicePointers to the uploaded Triangles Texture Coordinates */
float *cudaTriangleTextureCoordinatesDP = NULL;
/* CudaDevicePointers to the uploaded Triangles Materials */
float *cudaTriangleDiffusePropertiesDP = NULL;
float *cudaTriangleSpecularPropertiesDP = NULL;

/* Initialization Declarations */
void initCamera();
void initObjects();

bool initGLUT(int argc, char **argv);
bool initGLEW();
bool initOpenGL();

bool initCUDA(int argc, char **argv);
void initCUDAmemory();

/* User Input Functions */
void normalKeyListener(unsigned char key, int x, int y);
void releasedNormalKeyListener(unsigned char key, int x, int y);

void specialKeyListener(int key, int x, int y);
void releasedSpecialKeyListener(int key, int x, int y);

void mouseEventListener(int button, int state, int x, int y);
void mouseMovementListener(int x, int y);
void mousePassiveMovementListener(int x, int y);
void mouseWheelListener(int button, int direction, int x, int y);

void readMouse(float elapsedTime);
void readKeyboard(float elapsedTime);

/* Run-time Function Declarations */
void display();
void displayFrames(int time);
void reshape(int width, int height);
void cleanup();

void rayTrace();

/* Initialize the Scene */
void initCamera() {

	camera = new Camera(windowWidth, windowHeight);
	
	camera->setFieldOfView(60.0f);

	camera->setPosition(Vector(0.0f, 0.0f, 0.0f, 1.0f));
}

void initObjects() {

	// Create 4 Objects that will contain the Meshes
	spheres[0] = new Object("Sphere 0");
	spheres[1] = new Object("Sphere 1");
	spheres[2] = new Object("Sphere 2");
	spheres[3] = new Object("Sphere 3");

	platform = new Object("Platform");

	/* Load the Spheres Mesh from the OBJ file */
	Mesh* sphere0Mesh = new Mesh("Sphere", "emeraldsphere.obj", "emerald.mtl");
	Mesh* sphere1Mesh = new Mesh("Sphere", "rubysphere.obj", "ruby.mtl");
	Mesh* sphere2Mesh = new Mesh("Sphere", "goldsphere.obj", "gold.mtl");
	Mesh* sphere3Mesh = new Mesh("Sphere", "silversphere.obj", "silver.mtl");
	/* Load the Platforms Mesh from the OBJ file */
	Mesh* platformMesh = new Mesh("Platform", "cube.obj", "cube.mtl");

	/* Load Sphere0s Transform */
 	Transform* sphere0Transform = new Transform("Sphere 0 Transform");
	sphere0Transform->setPosition(Vector(-10.0f, -2.5f, 10.0f, 1.0f));
	sphere0Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	/* Load Sphere1s Transform */
	Transform* sphere1Transform = new Transform("Sphere 1 Transform");
	sphere1Transform->setPosition(Vector(-15.0f, -2.5f,-10.0f, 1.0f));
	sphere1Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	/* Load Sphere2s Transform */
	Transform* sphere2Transform = new Transform("Sphere 2 Transform");
	sphere2Transform->setPosition(Vector( 15.0f, -2.5f,-10.0f, 1.0f));
	sphere2Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	/* Load Sphere3s Transform */
	Transform* sphere3Transform = new Transform("Sphere 3 Transform");
	sphere3Transform->setPosition(Vector( 10.0f, -2.5f, 10.0f, 1.0f));
	sphere3Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	/* Load Platforms Transform */
	Transform* platformTransform = new Transform("Platform Transform");
	platformTransform->setPosition(Vector( 0.0f,-15.0f, 0.0f, 1.0f));
	platformTransform->setScale(Vector( 50.0f, 0.75f, 50.0f, 1.0f));

	/* Set the Mesh and Transform of the created Objects */
	spheres[0]->setMesh(sphere0Mesh);
	spheres[0]->setTransform(sphere0Transform);

	spheres[1]->setMesh(sphere1Mesh);
	spheres[1]->setTransform(sphere1Transform);

	spheres[2]->setMesh(sphere2Mesh);
	spheres[2]->setTransform(sphere2Transform);

	spheres[3]->setMesh(sphere3Mesh);
	spheres[3]->setTransform(sphere3Transform);

	platform->setMesh(platformMesh);
	platform->setTransform(platformTransform);
}
/* Initialize GLUT */
bool initGLUT(int argc, char** argv) {

	glutInit(&argc, argv);

	/* Setup the Minimum OpenGL version */
	glutInitContextVersion(2,0);

	/* Setup the Display */
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth,windowHeight);

	/* Setup the Window */
	windowHandle = glutCreateWindow("CUDA Ray Tracer");

	if(windowHandle < 1) {

		fprintf(stderr, "ERROR: Could not create a new rendering window.\n");
		fflush(stderr);

		exit(EXIT_FAILURE);
	}

	/* Setup the Callback Functions */
	glutDisplayFunc(display);
	glutTimerFunc(0, displayFrames, 0);
		
	glutReshapeFunc(reshape);

	glutKeyboardFunc(normalKeyListener); 
	glutKeyboardUpFunc(releasedNormalKeyListener); 
	glutSpecialFunc(specialKeyListener);
	glutSpecialUpFunc(releasedSpecialKeyListener);

	glutMouseFunc(mouseEventListener);
	glutMotionFunc(mouseMovementListener);
	glutPassiveMotionFunc(mousePassiveMovementListener);
	glutMouseWheelFunc(mouseWheelListener);

	glutCloseFunc(cleanup);

	return true;
}

/* Initialize GLEW */
bool initGLEW() {

	GLenum error = glewInit();

	/* Check if the Initialization went ok. */
	if(error != GLEW_OK) {

		fprintf(stderr, "ERROR: %s\n", glewGetErrorString(error));
		fflush(stderr);

		exit(EXIT_FAILURE);
	}


	/* Check if OpenGL 2.0 is supported */
	if(!glewIsSupported("GL_VERSION_2_0")) {

		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);

		exit(EXIT_FAILURE);
	}

	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	return true;
}

/* Initialize OpenGL */
bool initOpenGL() {

	/* Initialize the State */
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	/* Initialize the Viewport */
	glViewport(0, 0, windowWidth, windowHeight);

	fprintf(stdout, "Status: Using OpenGL v%s",glGetString(GL_VERSION));

	return true;
}

/* Initialize CUDA */
bool initCUDA() {

	int device = gpuGetMaxGflopsDeviceId();

	cudaSetDevice(device);
	Utility::checkCUDAError("cudaSetDevice()");
	cudaGLSetGLDevice(device);
	Utility::checkCUDAError("cudaGLSetGLDevice()");

	return true;
}

/* Initialize CUDA Memory with the necessary space for the Meshes */
void initCUDAmemory() {

	// Initialize the PixelBufferObject for transferring data from CUDA to OpenGL (as a texture).
	unsigned int texelNumber = imageWidth * imageHeight;
	unsigned int pixelBufferObjectSize = sizeof(GLubyte) * texelNumber * 4;

	// Create the PixelBufferObject.
	glGenBuffers(1, &pixelBufferObjectID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// Register the PixelBufferObject with CUDA.
	cudaGraphicsGLRegisterBuffer(&pixelBufferObjectResource, pixelBufferObjectID, cudaGraphicsRegisterFlagsWriteDiscard);
	Utility::checkCUDAError("cudaGraphicsGLRegisterBuffer()");

	// Create the Texture to output the Ray-Tracing result.
	glGenTextures(1, &screenTextureID);
	glBindTexture(GL_TEXTURE_2D, screenTextureID);

	// Set the basic Texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Define the basic Texture parameters
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	Utility::checkOpenGLError("glTexImage2D()");

	/* Load the Triangles to an Array */
	vector<float4> trianglePositions;
	vector<float4> triangleNormals;
	vector<float4> triangleTangents;
	vector<float2> triangleTextureCoordinates;

	vector<float4> triangleDiffuseProperties;
	vector<float4> triangleSpecularProperties;

	// Sphere Vertices
	for(unsigned int i = 0; i < 4; i++) {
	
		// Used for the position transformations
		Matrix modelMatrix = spheres[i]->getTransform()->getModelMatrix();
		// Used for the normal transformations
		Matrix modelMatrixInverseTranspose = modelMatrix;
		//modelMatrixInverseTranspose.removeTranslation();
		modelMatrixInverseTranspose.transpose();
		modelMatrixInverseTranspose.invert();

		for(int j = 0; j < spheres[0]->getMesh()->getVertexCount(); j++)	{

			// Get the original vertex from the mesh 
			Vertex originalVertex = spheres[i]->getMesh()->getVertex(j);

			// Position: Multiply the original vertex using the objects model matrix
			Vector modifiedPosition = modelMatrix * Vector(originalVertex.position[VX], originalVertex.position[VY], originalVertex.position[VZ], 1.0f);
			float4 position = { modifiedPosition[VX], modifiedPosition[VY], modifiedPosition[VZ], 1.0f };
			trianglePositions.push_back(position);

			// Normal: Multiply the original normal using the objects inverted transposed model matrix	
			Vector modifiedNormal = modelMatrixInverseTranspose * Vector(originalVertex.normal[VX], originalVertex.normal[VY], originalVertex.normal[VZ], 0.0f);
			modifiedNormal.normalize();
			float4 normal = { modifiedNormal[VX], modifiedNormal[VY], modifiedNormal[VZ], 0.0f };
			triangleNormals.push_back(normal);

			// Tangent: Multiply the original tangent using the objects inverted transposed model matrix
			Vector modifiedTangent = modelMatrixInverseTranspose * Vector(originalVertex.tangent[VX], originalVertex.tangent[VY], originalVertex.tangent[VZ], 0.0f);
			float4 tangent = { modifiedTangent[VX], modifiedTangent[VY], modifiedTangent[VZ], originalVertex.tangent[VW] };
			triangleTangents.push_back(tangent);

			// Texture Coordinate: Same as the original values
			float2 textureCoordinates = { originalVertex.textureUV[VX], originalVertex.textureUV[VY] };
			triangleTextureCoordinates.push_back(textureCoordinates);

			// Material: Same as the original values
			float4 diffuseProperty = { originalVertex.diffuse[VX], originalVertex.diffuse[VY], originalVertex.diffuse[VZ], 1.0f };
			float4 specularProperty = { originalVertex.specular[VX], originalVertex.specular[VY], originalVertex.specular[VZ], originalVertex.specularConstant };

			triangleDiffuseProperties.push_back(diffuseProperty);
			triangleSpecularProperties.push_back(specularProperty);
		}
	}
	
	// Platform Vertices

	// Used for the position transformations
	Matrix modelMatrix = platform->getTransform()->getModelMatrix();
	// Used for the normal transformations
	Matrix modelMatrixInverseTranspose = modelMatrix;
	//modelMatrixInverseTranspose.removeTranslation();
	modelMatrixInverseTranspose.transpose();
	modelMatrixInverseTranspose.invert();

	for(int j = 0; j < platform->getMesh()->getVertexCount(); j++) {

		// Get the original vertex from the mesh 
		Vertex originalVertex = platform->getMesh()->getVertex(j);
			
		// Position: Multiply the original vertex using the objects model matrix
		Vector modifiedPosition = modelMatrix * Vector(originalVertex.position[VX], originalVertex.position[VY], originalVertex.position[VZ], 1.0f);
		float4 position = { modifiedPosition[VX], modifiedPosition[VY], modifiedPosition[VZ], 1.0f };
		trianglePositions.push_back(position);

		// Normal: Multiply the original normal using the objects inverted transposed model matrix	
		Vector modifiedNormal = modelMatrixInverseTranspose * Vector(originalVertex.normal[VX], originalVertex.normal[VY], originalVertex.normal[VZ], 0.0f);
		modifiedNormal.normalize();
		float4 normal = { modifiedNormal[VX], modifiedNormal[VY], modifiedNormal[VZ], 0.0f };
		triangleNormals.push_back(normal);

		// Tangent: Multiply the original tangent using the objects inverted transposed model matrix
		Vector modifiedTangent = modelMatrixInverseTranspose * Vector(originalVertex.tangent[VX], originalVertex.tangent[VY], originalVertex.tangent[VZ], 0.0f);
		float4 tangent = { modifiedTangent[VX], modifiedTangent[VY], modifiedTangent[VZ], originalVertex.tangent[VW] };
		triangleTangents.push_back(tangent);

		// Texture Coordinate: Same as the original values
		float2 textureCoordinates = { originalVertex.textureUV[VX], originalVertex.textureUV[VY] };
		triangleTextureCoordinates.push_back(textureCoordinates);

		// Material: Same as the original values
		float4 diffuseProperty = { originalVertex.diffuse[VX], originalVertex.diffuse[VY], originalVertex.diffuse[VZ], 1.0f };
		float4 specularProperty = { originalVertex.specular[VX], originalVertex.specular[VY], originalVertex.specular[VZ], originalVertex.specularConstant };

		triangleDiffuseProperties.push_back(diffuseProperty);
		triangleSpecularProperties.push_back(specularProperty);
	}

	// Total number of Triangles should be the number of loaded vertices divided by 3
	triangleTotal = trianglePositions.size() / 3;

	cout << "Total number of triangles:" << 4 * spheres[0]->getMesh()->getVertexCount() / 3 +  platform->getMesh()->getVertexCount() / 3 << " == " << triangleTotal << endl;

	// Each triangle contains Position, Normal, Tangent, Texture UV and Material Properties for 3 vertices
	size_t trianglePositionsSize = trianglePositions.size() * sizeof(float4);
	cout << "Triangle Positions Storage Size:" << trianglePositionsSize << "(" << trianglePositions.size() << " values)" << endl;

	size_t triangleNormalsSize = triangleNormals.size() * sizeof(float4);
	cout << "Triangle Normals Storage Size:" << triangleNormalsSize << "(" << triangleNormals.size() << " values)" << endl;
	
	size_t triangleTangentsSize = triangleTangents.size() * sizeof(float4);
	cout << "Triangle Tangents Storage Size:" << triangleTangentsSize << "(" << triangleTangents.size() << " values)" << endl;

	size_t triangleTextureCoordinatesSize = triangleTextureCoordinates.size() * sizeof(float2);
	cout << "Triangle Texture Coordinates Storage Size:" << triangleTextureCoordinatesSize << "(" << triangleTextureCoordinates.size() << " values)" << endl;

	size_t triangleDiffusePropertiesSize = triangleDiffuseProperties.size() * sizeof(float4);
	cout << "Triangle Diffuse Properties Storage Size:" << triangleDiffusePropertiesSize << "(" << triangleDiffuseProperties.size() << " values)" << endl;
	size_t triangleSpecularPropertiesSize = triangleSpecularProperties.size() * sizeof(float4);
	cout << "Triangle Specular Properties Storage Size:" << triangleSpecularPropertiesSize << "(" << triangleSpecularProperties.size() << " values)" << endl;

	// Allocate the required CUDA Memory
	if(triangleTotal > 0) {

		// Load the Triangle Positions
		cudaMalloc((void **)&cudaTrianglePositionsDP, trianglePositionsSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTrianglePositionsDP, &trianglePositions[0], trianglePositionsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTrianglePositions(cudaTrianglePositionsDP, triangleTotal);

		// Load the Triangle Normals
		cudaMalloc((void **)&cudaTriangleNormalsDP, triangleNormalsSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleNormalsDP, &triangleNormals[0], triangleNormalsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTriangleNormals(cudaTriangleNormalsDP, triangleTotal);

		/*// Load the Triangle Tangents
		cudaMalloc((void **)&cudaTriangleTangentsDP, triangleTangentsSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleTangentsDP, &triangleTangents[0], triangleTangentsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTriangleTangents(cudaTriangleTangentsDP, triangleTotal);

		// Load the Triangle Texture Coordinates
		cudaMalloc((void **)&cudaTriangleTextureCoordinatesDP, triangleTextureCoordinatesSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleTextureCoordinatesDP, &triangleTextureCoordinates[0], triangleTextureCoordinatesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTriangleTextureCoordinates(cudaTriangleTextureCoordinatesDP, triangleTotal);*/

		// Load the Triangle Diffuse Properties
		cudaMalloc((void **)&cudaTriangleDiffusePropertiesDP, triangleDiffusePropertiesSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleDiffusePropertiesDP, &triangleDiffuseProperties[0], triangleDiffusePropertiesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTriangleDiffuseProperties(cudaTriangleDiffusePropertiesDP, triangleTotal);

		// Load the Triangle Specular Properties
		cudaMalloc((void **)&cudaTriangleSpecularPropertiesDP, triangleSpecularPropertiesSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleSpecularPropertiesDP, &triangleSpecularProperties[0], triangleSpecularPropertiesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");

		bindTriangleSpecularProperties(cudaTriangleSpecularPropertiesDP, triangleTotal);
	}

	/* Load a Sample Texture */
	//chessTextureID = SOIL_load_OGL_texture("textures/fieldstone_diffuse.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y);

	/* Check for an error during the load process */
	//if(chessTextureID == 0)
		//cout << "SOIL loading error (\"" << "textures/fieldstone_diffuse.jpg" << "\": " << SOIL_last_result() << std::endl;

	//Utility::checkOpenGLError("SOIL_load_OGL_texture()");

	/* Register the necessary Textures */
	/*cudaGraphicsGLRegisterImage(cudaTextureHandle, chessTextureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()");

	cudaGraphicsMapResources(1,cudaTextureHandle,0);*/
}

/* User Input Functions */
void normalKeyListener(unsigned char key, int x, int y) {

	KeyboardHandler::getInstance()->normalKeyListener(key,x,y);
}

void releasedNormalKeyListener(unsigned char key, int x, int y) {

	KeyboardHandler::getInstance()->releasedNormalKeyListener(key,x,y);
}

void specialKeyListener(int key, int x, int y) {

	KeyboardHandler::getInstance()->specialKeyListener(key,x,y);
}

void releasedSpecialKeyListener(int key, int x, int y) {

	KeyboardHandler::getInstance()->releasedSpecialKeyListener(key,x,y);
}

void mouseEventListener(int button, int state, int x, int y) {

	MouseHandler::getInstance()->mouseEventListener(button,state,x,y);
}

void mouseMovementListener(int x, int y) {

	MouseHandler::getInstance()->mouseMovementListener(x,y);
}

void mousePassiveMovementListener(int x, int y) {

	MouseHandler::getInstance()->mouseMovementListener(x,y);
}

void mouseWheelListener(int button, int direction, int x, int y)  {

	MouseHandler::getInstance()->mouseWheelListener(button,direction,x,y);
} 

/* Manages the Mouse Input */
void readMouse(GLfloat elapsedTime) {

	MouseHandler* handler = MouseHandler::getInstance();

	handler->disableMouse();

	GLint zoom = handler->getMouseWheelPosition();
	GLint longitude = handler->getLongitude(GLUT_RIGHT_BUTTON);
	GLint latitude = handler->getLatitude(GLUT_RIGHT_BUTTON);

	//camera->update(zoom,longitude,latitude,elapsedTime);

	handler->enableMouse();
}

/* Manages the Keyboard Input */
void readKeyboard(GLfloat elapsedTime) {

	elapsedTime;
}

void displayFrames(int time) {

	std::ostringstream oss;
	oss << "CUDA Ray Tracer" << ": " << frameCount << " FPS @ (" << windowWidth << "x" << windowHeight << ")";
	std::string s = oss.str();

	glutSetWindow(windowHandle);
	glutSetWindowTitle(s.c_str());

    frameCount = 0;

    glutTimerFunc(1000, displayFrames, 0);
}

void display() {

	++frameCount;

	/* Time Management */
	if(lastFrameTime == 0)
		lastFrameTime = glutGet(GLUT_ELAPSED_TIME);

	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = (float)(currentTime - lastFrameTime) / 1000.0f;
	lastFrameTime = currentTime;

	/* Update the Scenes Objects */
	readMouse(deltaTime);
	readKeyboard(deltaTime);

	camera->update(0,0,0,deltaTime); //TODO

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Call the Ray-Tracing CUDA Implementation */
	rayTrace();

	/* Draw the resulting Texture on a Quad covering the Screen */
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);

	glPushMatrix();

		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glMatrixMode( GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, windowWidth, windowHeight);

		glBegin(GL_QUADS);
			glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,-1.0, 0.5);
			glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,-1.0, 0.5);
			glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, 1.0, 0.5);
			glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glMatrixMode(GL_PROJECTION);

	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
	
	Utility::checkOpenGLError("display()");

	/* Swap the Screen Buffer and Call the Display function again. */
	glutSwapBuffers();
	glutPostRedisplay();
}

/* Callback function called by GLUT when window size changes */
void reshape(int width, int height) {

	/* Set OpenGL view port and camera */
	glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Update the Camera */
	camera->reshape(width, height);

	windowWidth = width;
	windowHeight = height;

	imageWidth = width;
	imageHeight = height;
}

/* Callback function called by GLUT when the program exits */
void cleanup() {

	/* Delete the PixelBufferObject from CUDA */
	cudaGraphicsUnregisterResource(pixelBufferObjectResource);
	Utility::checkCUDAError("cudaGraphicsUnregisterResource()");

	/* Delete the PixelBufferObject from OpenGL */
    glBindBuffer(1, pixelBufferObjectID);
    glDeleteBuffers(1, &pixelBufferObjectID);
	Utility::checkOpenGLError("glDeleteBuffers()");

	/* Force CUDA to flush profiling information */
	cudaDeviceReset();
}

void rayTrace() {

	/* Camera defining Vectors */
	Vector up = camera->getUp();
	up.clean();
	Vector eye = camera->getEye();
	eye.clean();
	Vector target = camera->getTarget();
	target.clean();

	/* Images Aspect Ratio */
	float aspectRatio = (float)imageWidth / (float)imageHeight;
	/* Cameras distance to the target */
	float distance = (target - eye).length();
	/* Cameras Field of View */
	float fieldOfView = camera->getFieldOfView();
	/* Projection Frustum Half-Width */
	float theta = (fieldOfView * 0.5f) * DEGREES_TO_RADIANS;
	float halfHeight = 2.0f * distance * tanf(theta);
	float halfWidth = halfHeight * aspectRatio;

	/* Camera Position and Direction */
	float3 cameraPosition = make_float3(eye[VX], eye[VY], eye[VZ]);
	float3 cameraDirection = make_float3(target[VX] - eye[VX], target[VY] - eye[VY], target[VZ] - eye[VZ]);
	cameraDirection = normalize(cameraDirection);
	cameraDirection = distance * cameraDirection;

	/* Camera Right and Up */
	float3 cameraUp = make_float3(up[VX], up[VY], up[VZ]);
	cameraUp = normalize(cameraUp);

	float3 cameraRight = cross(cameraDirection, cameraUp);
	cameraRight = normalize(cameraRight);
	cameraRight = halfWidth * cameraRight;

	cameraUp = cross(cameraRight, cameraDirection);
	cameraUp = normalize(cameraUp);
	cameraUp = halfHeight * cameraUp;

	unsigned int* outData;

	/* Map the PixelBufferObject and Ray-Trace */
    cudaGraphicsMapResources(1, &pixelBufferObjectResource, 0);
	Utility::checkCUDAError("cudaGraphicsMapResources()");
    cudaGraphicsResourceGetMappedPointer((void **)&outData, NULL, pixelBufferObjectResource);
	Utility::checkCUDAError("cudaGraphicsResourceGetMappedPointer()");

	RayTraceImage(outData, imageWidth, imageHeight, triangleTotal, 
		cameraRight, cameraUp, cameraDirection, 
		cameraPosition,
		make_float3(lightPosition[0], lightPosition[1], lightPosition[2]),
		make_float3(lightColor[0], lightColor[1], lightColor[2]));

	cudaGraphicsUnmapResources(1, &pixelBufferObjectResource, 0);
	Utility::checkCUDAError("cudaGraphicsUnmapResources()");

	/* Copy the Output to the Texture */
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectID);

		glBindTexture(GL_TEXTURE_2D, screenTextureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	Utility::checkOpenGLError("glTexSubImage2D()");
}

int main(int argc, char** argv) {

	/* Initialize the Scene */
	initCamera();
	initObjects();

	/* Initialize GLUT, GLEW and OpenGL */
	initGLUT(argc,argv);
	initGLEW();
	initOpenGL();

	/* Initialize CUDA */
	initCUDA();
	initCUDAmemory();

	// start rendering main-loop
	glutMainLoop();

	cudaThreadExit();

	return 0;
}