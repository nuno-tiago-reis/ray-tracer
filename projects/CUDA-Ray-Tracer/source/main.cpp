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

/* Error Checking */
#include "Utility.h"

/* User Interaction */
#include "MouseHandler.h"
#include "KeyboardHandler.h"

using namespace std;

// the interface between C++ and CUDA -----------

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
// the implementation of bindTriangleAmbientProperties is in the "raytracer.cu" file
extern "C" void bindTriangleAmbientProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleDiffuseProperties is in the "raytracer.cu" file
extern "C" void bindTriangleDiffuseProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleSpecularProperties is in the "raytracer.cu" file
extern "C" void bindTriangleSpecularProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// the implementation of bindTriangleSpecularConstants is in the "raytracer.cu" file
extern "C" void bindTriangleSpecularConstants(float *cudaDevicePointer, unsigned int triangleTotal);

/* Global Variables */
unsigned int windowWidth  = 1024;
unsigned int windowHeight = 1024;

unsigned int imageWidth   = 1024;
unsigned int imageHeight  = 1024;

int frameCount = 0;
int windowHandle = 0;

/* PixelBufferObjectID containing the resulting image */
GLuint pixelBufferObjectID;
/* TextureID of the texture where the Ray-Tracing result will be stored */
GLuint textureID;

/* Objects */
Object* spheres[4];
Object* platform;

/* Camera */
Camera* camera;

/* Scene Time Management */
int lastFrameTime = 0;
float deltaTime = 0.0f;

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
float *cudaTriangleAmbientPropertiesDP = NULL;
float *cudaTriangleDiffusePropertiesDP = NULL;
float *cudaTriangleSpecularPropertiesDP = NULL;
float *cudaTriangleSpecularConstantsDP = NULL;

	// Lighting ---------------------------- TODO
	float lightPosition[3] = { 0.0f, 5.0f, 0.0f };
	float lightColor[3] = { 1.0f, 1.0f, 1.0f };

/* Initialization Declarations */
void initCamera();
void initObjects();

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
void cleanup();
void display();
void reshape(int width, int height);

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
	Mesh* sphereMesh = new Mesh("Sphere", "sphere.obj", "sphere.mtl");
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
	platformTransform->setScale(Vector( 50.0f, 0.15f, 50.0f, 1.0f));

	/* Set the Mesh and Transform of the created Objects */
	spheres[0]->setMesh(sphereMesh);
	spheres[0]->setTransform(sphere0Transform);

	spheres[1]->setMesh(sphereMesh);
	spheres[1]->setTransform(sphere1Transform);

	spheres[2]->setMesh(sphereMesh);
	spheres[2]->setTransform(sphere2Transform);

	spheres[3]->setMesh(sphereMesh);
	spheres[3]->setTransform(sphere3Transform);

	platform->setMesh(platformMesh);
	platform->setTransform(platformTransform);
}

/* Initialize OpenGL */
bool initOpenGL() {

	glewInit();

	if(!glewIsSupported("GL_VERSION_2_0")) {

		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);

		exit(0);
	}

	/* Initialize the State */
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	/* Initialize the Viewport */
	glViewport(0, 0, windowWidth, windowHeight);

	return true;
}

/* Initialize CUDA */
bool initCUDA() {

	int device = gpuGetMaxGflopsDeviceId();

	cudaSetDevice(device);
	Utility::checkCUDAError("cudaSetDevice");
	cudaGLSetGLDevice(device);
	Utility::checkCUDAError("cudaGLSetGLDevice");

	return true;
}

/* Initialize CUDA Memory with the necessary space for the Meshes */
void initCUDAmemory() {

	// Initialize the PixelBufferObject for transferring data from CUDA to OpenGL (as a texture).
	unsigned int texelNumber = imageWidth * imageHeight;
	unsigned int pixelBufferObjectSize = sizeof(GLubyte) * texelNumber * 4;

	void *pixelBufferObjectData = malloc(pixelBufferObjectSize);

	//size_t freeSize, totalSize;
	//cuMemGetInfo(&freeSize,&totalSize);

	// Create the PixelBufferObject.
	glGenBuffers(1, &pixelBufferObjectID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	free(pixelBufferObjectData);

	// Register the PixelBufferObject with CUDA.
	cudaGLRegisterBufferObject(pixelBufferObjectID);
	//cudaGraphicsResource *pixelBufferObjectResource[1];
	//cudaGraphicsGLRegisterBuffer(pixelBufferObjectResource, pixelBufferObjectID, cudaGraphicsRegisterFlagsWriteDiscard);
	Utility::checkCUDAError("cudaGLRegisterBufferObject");

	// Create the Texture to output the Ray-Tracing result.
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Set the basic Texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Define the basic Texture parameters
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	Utility::checkOpenGLError("ERROR: Texture Generation failed.");

	/* Load the Triangles to an Array */
	vector<float4> trianglePositions;
	vector<float4> triangleNormals;
	vector<float4> triangleTangents;
	vector<float2> triangleTextureCoordinates;

	vector<float4> triangleAmbientProperties;
	vector<float4> triangleDiffuseProperties;
	vector<float4> triangleSpecularProperties;
	vector<float> triangleSpecularConstants;

	// Sphere Vertices
	for(unsigned int i = 0; i < 4; i++) {
	
		// Used for the position transformations
		Matrix modelMatrix = spheres[i]->getTransform()->getModelMatrix();
		// Used for the normal transformations
		Matrix modelMatrixInverseTranspose = modelMatrix;
		modelMatrixInverseTranspose.removeTranslation();
		modelMatrixInverseTranspose.transpose();
		modelMatrixInverseTranspose.invert();

		for(int j = 0; j < spheres[0]->getMesh()->getVertexCount(); j++)	{

			// Get the original vertex from the mesh 
			Vertex originalVertex = spheres[0]->getMesh()->getVertex(j);

			// Position: Multiply the original vertex using the objects model matrix
			Vector modifiedPosition = modelMatrix * Vector(originalVertex.position[VX], originalVertex.position[VY], originalVertex.position[VZ], 1.0f);
			float4 position = { modifiedPosition[VX], modifiedPosition[VY], modifiedPosition[VZ], 1.0f };
			trianglePositions.push_back(position);

			// Normal: Multiply the original normal using the objects inverted transposed model matrix	
			Vector modifiedNormal = modelMatrixInverseTranspose * Vector(originalVertex.normal[VX], originalVertex.normal[VY], originalVertex.normal[VZ], 0.0f);
			modifiedNormal.normalize();
			float4 normal = { modifiedNormal[VX], modifiedNormal[VY], modifiedNormal[VZ], 0.0f };
			triangleNormals.push_back(normal);

			/*
			// Tangent: Multiply the original tangent using the objects inverted transposed model matrix
			Vector modifiedTangent = modelMatrixInverseTranspose * Vector(originalVertex.tangent[VX], originalVertex.tangent[VY], originalVertex.tangent[VZ], originalVertex.tangent[VW]);
			float4 tangent = { modifiedTangent[VX], modifiedTangent[VY], modifiedTangent[VZ], originalVertex.tangent[VW] };
			triangleTangents.push_back(tangent);

			// Texture Coordinate: Same as the original values
			float2 textureCoordinates = { originalVertex.textureUV[VX], originalVertex.textureUV[VY] };
			triangleTextureCoordinates.push_back(textureCoordinates);

			// Material: Same as the original values
			float4 ambientProperty = { originalVertex.ambient[VX], originalVertex.ambient[VY], originalVertex.ambient[VZ], 1.0f };
			float4 diffuseProperty = { originalVertex.diffuse[VX], originalVertex.diffuse[VY], originalVertex.diffuse[VZ], 1.0f };
			float4 specularProperty = { originalVertex.specular[VX], originalVertex.specular[VY], originalVertex.specular[VZ], 1.0f };
			float specularConstant = { originalVertex.specularConstant };

			triangleAmbientProperties.push_back(ambientProperty);
			triangleDiffuseProperties.push_back(diffuseProperty);
			triangleSpecularProperties.push_back(specularProperty);
			triangleSpecularConstants.push_back(specularConstant);
			*/
		}
	}
	
	// Platform Vertices

	// Used for the position transformations
	Matrix modelMatrix = platform->getTransform()->getModelMatrix();
	// Used for the normal transformations
	Matrix modelMatrixInverseTranspose = modelMatrix;
	modelMatrixInverseTranspose.removeTranslation();
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

		/*
		// Tangent: Multiply the original tangent using the objects inverted transposed model matrix
		Vector modifiedTangent = modelMatrixInverseTranspose * Vector(originalVertex.tangent[VX], originalVertex.tangent[VY], originalVertex.tangent[VZ], originalVertex.tangent[VW]);
		float4 tangent = { modifiedTangent[VX], modifiedTangent[VY], modifiedTangent[VZ], originalVertex.tangent[VW] };
		triangleTangents.push_back(tangent);

		// Texture Coordinate: Same as the original values
		float2 textureCoordinates = { originalVertex.textureUV[VX], originalVertex.textureUV[VY] };
		triangleTextureCoordinates.push_back(textureCoordinates);

		// Material: Same as the original values
		float4 ambientProperty = { originalVertex.ambient[VX], originalVertex.ambient[VY], originalVertex.ambient[VZ], 1.0f };
		float4 diffuseProperty = { originalVertex.diffuse[VX], originalVertex.diffuse[VY], originalVertex.diffuse[VZ], 1.0f };
		float4 specularProperty = { originalVertex.specular[VX], originalVertex.specular[VY], originalVertex.specular[VZ], 1.0f };
		float specularConstant = { originalVertex.specularConstant };

		triangleAmbientProperties.push_back(ambientProperty);
		triangleDiffuseProperties.push_back(diffuseProperty);
		triangleSpecularProperties.push_back(specularProperty);
		triangleSpecularConstants.push_back(specularConstant);
		*/
	}

	// Total number of Triangles should be the number of loaded vertices divided by 3
	triangleTotal = trianglePositions.size() / 3;

	cout << "Total number of triangles:" << 4 * spheres[0]->getMesh()->getVertexCount() / 3 +  platform->getMesh()->getVertexCount() / 3 << " == " << triangleTotal << endl;

	// Each triangle contains Position, Normal, Tangent, Texture UV and Material Properties for 3 vertices
	size_t trianglePositionsSize = trianglePositions.size() * sizeof(float4);
	cout << "Triangle Positions Storage Size:" << trianglePositionsSize << "(" << trianglePositions.size() << " values)" << endl;

	size_t triangleNormalsSize = triangleNormals.size() * sizeof(float4);
	cout << "Triangle Normals Storage Size:" << triangleNormalsSize << "(" << triangleNormals.size() << " values)" << endl;
	/*
	size_t triangleTangentsSize = triangleTangents.size() * sizeof(float4);
	cout << "Triangle Tangents Storage Size:" << triangleTangentsSize << "(" << triangleTangents.size() << " values)" << endl;

	size_t triangleTextureCoordinatesSize = triangleTextureCoordinates.size() * sizeof(float2);
	cout << "Triangle Texture Coordinates Storage Size:" << triangleTextureCoordinatesSize << "(" << triangleTextureCoordinates.size() << " values)" << endl;

	size_t triangleAmbientPropertiesSize = triangleAmbientProperties.size() * sizeof(float4);
	cout << "Triangle Ambient Properties Storage Size:" << triangleAmbientPropertiesSize << "(" << triangleAmbientProperties.size() << " values)" << endl;
	size_t triangleDiffusePropertiesSize = triangleDiffuseProperties.size() * sizeof(float4);
	cout << "Triangle Diffuse Properties Storage Size:" << triangleDiffusePropertiesSize << "(" << triangleDiffuseProperties.size() << " values)" << endl;
	size_t triangleSpecularPropertiesSize = triangleSpecularProperties.size() * sizeof(float4);
	cout << "Triangle Specular Properties Storage Size:" << triangleSpecularPropertiesSize << "(" << triangleSpecularProperties.size() << " values)" << endl;
	size_t triangleSpecularConstantsSize = triangleSpecularConstants.size() * sizeof(float);
	cout << "Triangle Specular Constant Properties Storage Size:" << triangleSpecularConstantsSize << "(" << triangleSpecularConstants.size() << " values)" << endl;
	*/

	// Allocate the required CUDA Memory
	if(triangleTotal > 0) {

		// Load the Triangle Positions
		cudaMalloc((void **)&cudaTrianglePositionsDP, trianglePositionsSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTrianglePositionsDP, &trianglePositions[0], trianglePositionsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTrianglePositions(cudaTrianglePositionsDP, triangleTotal);

		// Load the Triangle Normals
		cudaMalloc((void **)&cudaTriangleNormalsDP, triangleNormalsSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleNormalsDP, &triangleNormals[0], triangleNormalsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleNormals(cudaTriangleNormalsDP, triangleTotal);

		/*// Load the Triangle Tangents
		cudaMalloc((void **)&cudaTriangleTangentsDP, triangleTangentsSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleTangentsDP, &triangleTangents[0], triangleTangentsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleTangents(cudaTriangleTangentsDP, triangleTotal);

		// Load the Triangle Texture Coordinates
		cudaMalloc((void **)&cudaTriangleTextureCoordinatesDP, triangleTextureCoordinatesSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleTextureCoordinatesDP, &triangleTextureCoordinates[0], triangleTextureCoordinatesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleTextureCoordinates(cudaTriangleTextureCoordinatesDP, triangleTotal);

		// Load the Triangle Ambient Properties
		cudaMalloc((void **)&cudaTriangleAmbientPropertiesDP, triangleAmbientPropertiesSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleAmbientPropertiesDP, &triangleAmbientProperties[0], triangleAmbientPropertiesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleAmbientProperties(cudaTriangleAmbientPropertiesDP, triangleTotal);

		// Load the Triangle Diffuse Properties
		cudaMalloc((void **)&cudaTriangleDiffusePropertiesDP, triangleDiffusePropertiesSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleDiffusePropertiesDP, &triangleDiffuseProperties[0], triangleDiffusePropertiesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleDiffuseProperties(cudaTriangleDiffusePropertiesDP, triangleTotal);

		// Load the Triangle Specular Properties
		cudaMalloc((void **)&cudaTriangleSpecularPropertiesDP, triangleSpecularPropertiesSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleSpecularPropertiesDP, &triangleSpecularProperties[0], triangleSpecularPropertiesSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleSpecularProperties(cudaTriangleSpecularPropertiesDP, triangleTotal);

		// Load the Triangle Specular Constants
		cudaMalloc((void **)&cudaTriangleSpecularConstantsDP, triangleSpecularConstantsSize);
		Utility::checkCUDAError("cudaMalloc");
		cudaMemcpy(cudaTriangleSpecularConstantsDP, &triangleSpecularConstants[0], triangleSpecularConstantsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy");

		bindTriangleSpecularConstants(cudaTriangleSpecularConstantsDP, triangleTotal);
		*/
	}
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

/* Callback function called by GLUT when the program exits */
void cleanup() {

    //glDeleteBuffers(1, &pixelBufferObjectID);
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
	cudaGLMapBufferObject((void**)&outData, pixelBufferObjectID);
	Utility::checkCUDAError("cudaGLMapBufferObject()");

	RayTraceImage(outData, imageWidth, imageHeight, triangleTotal, 
		cameraRight, cameraUp, cameraDirection, 
		cameraPosition, 
		make_float3(lightPosition[0], lightPosition[1], lightPosition[2]),
		make_float3(lightColor[0], lightColor[1], lightColor[2]));

	cudaGLUnmapBufferObject(pixelBufferObjectID);
	Utility::checkCUDAError("cudaGLUnmapBufferObject()");

	/* Copy the Output to the Texture */
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObjectID);

		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	Utility::checkOpenGLError("glTexSubImage2D()");
}

int main(int argc, char** argv) {

	// Create GL context
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth,windowHeight);

	windowHandle = glutCreateWindow("CUDA Ray Tracer");

	/* Initialize the Scene */
	initCamera();
	initObjects();

	/* Initialize OpenGL */
	initOpenGL();

	/* Initialize CUDA */
	initCUDA();
	initCUDAmemory();

	// register callbacks
	glutCloseFunc(cleanup);

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

	// start rendering main-loop
	glutMainLoop();

	cudaThreadExit();

	return 0;
}