// OpenGL definitions
#include <GL/glew.h>
#include <GL/glut.h>

// CUDA definitions
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "vector_types.h"
#include "vector_functions.h"
//Custom
#include "helper_cuda.h"
#include "helper_math.h"

// C++ Includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Custom Includes
#include "Object.h"
#include "Camera.h"

#include "ScreenTexture.h"
#include "ShadingTexture.h"

// Math Library
#include "Matrix.h"

// Texture Library
#include <soil.h>

// Error Checking
#include "Utility.h"

// User Interaction
#include "MouseHandler.h"
#include "KeyboardHandler.h"

using namespace std;

// The interface between C++ and CUDA

// Implementation of RayTraceWrapper is in the "RayTracer.cu" file
extern "C" void RayTraceWrapper(unsigned int *outputPixelBufferObject, 
								int width, int height, 
								int triangleTotal,
								float3 cameraRight, float3 cameraUp, float3 cameraDirection,
								float3 cameraPosition);

// Implementation of bindTrianglePositions is in the "RayTracer.cu" file
extern "C" void bindTrianglePositions(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTriangleNormals is in the "RayTracer.cu" file
extern "C" void bindTriangleNormals(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTriangleTangents is in the "RayTracer.cu" file
extern "C" void bindTriangleTangents(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTriangleTextureCoordinates is in the "RayTracer.cu" file
extern "C" void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTriangleDiffuseProperties is in the "RayTracer.cu" file
extern "C" void bindTriangleDiffuseProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTriangleSpecularProperties is in the "RayTracer.cu" file
extern "C" void bindTriangleSpecularProperties(float *cudaDevicePointer, unsigned int triangleTotal);
// Implementation of bindTextureArray is in the "RayTracer.cu" file
extern "C" void bindTextureArray(cudaArray *cudaArray);

// Global Variables

// Window
unsigned int windowWidth  = 640;
unsigned int windowHeight = 640;

int frameCount = 0;
int windowHandle = 0;

// Camera
Camera* camera;

// Objects
#define PLATFORM 0
#define SPHERE_0 1
#define SPHERE_1 2
#define SPHERE_2 3
#define SPHERE_3 4

map<int,Object*> objectMap;
map<int,Object*> materialMap;

// Scene Time Management
int lastFrameTime = 0;
float deltaTime = 0.0f;

//  PixelBufferObjects ID and Cuda Resource
GLuint pixelBufferObjectID;
cudaGraphicsResource *pixelBufferObjectResource = NULL;

// Object TextureIDs and  and Cuda Resources
GLuint chessTextureID;
cudaArray *chessTextureArray = NULL;
cudaGraphicsResource *chessTextureResource = NULL;

//  Screens Texture ID
GLuint screenTextureID;

//  Screens Textures Wrapper
ScreenTexture *screenTexture;

// Shading Textures Wrappers
ShadingTexture* chessTexture;
//ShadingTexture* rockTexture;
//ShadingTexture* brickTexture;

// Total number of Triangles - Used for the memory necessary to allocate
int triangleTotal = 0;

// CudaDevicePointers to the uploaded Triangles Positions 
float *cudaTrianglePositionsDP = NULL; 
// CudaDevicePointers to the uploaded Triangles Normals and Tangents 
float *cudaTriangleNormalsDP = NULL;
float *cudaTriangleTangentsDP = NULL;
// CudaDevicePointers to the uploaded Triangles Texture Coordinates 
float *cudaTriangleTextureCoordinatesDP = NULL;
// CudaDevicePointers to the uploaded Triangles Materials 
float *cudaTriangleDiffusePropertiesDP = NULL;
float *cudaTriangleSpecularPropertiesDP = NULL;

// Initialization Declarations
void createBufferObject(unsigned int *bufferObjectID, cudaGraphicsResource **bufferObjectResource, unsigned int width, unsigned int height);
void deleteBufferObject(unsigned int *bufferObjectID);

void createScreenTexture(unsigned int *textureID, unsigned int width, unsigned int height);
void deleteScreenTexture(unsigned int *textureID);

void createShadingTexture(unsigned int *textureID, cudaGraphicsResource **shadingTextureResource, char* fileName);
void deleteShadingTexture(unsigned int *textureID);

bool initGLUT(int argc, char **argv);
bool initGLEW();
bool initOpenGL();

void initCamera();
void initObjects();

bool initCUDA(int argc, char **argv);
void initCUDAmemory();

// User Input Functions 
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

// Run-time Function Declarations 
void display();
void displayFrames(int time);

void userInteraction(int time);

void reshape(int width, int height);
void cleanup();

void rayTrace();
void draw();

void createBufferObject(unsigned int *bufferObjectID, cudaGraphicsResource **bufferObjectResource, unsigned int width, unsigned int height) {

	unsigned int pixelBufferObjectSize = sizeof(GLubyte) * width * height * 4;

	// Delete the PixelBufferObject in case it already exists.
	glDeleteBuffers(1, bufferObjectID);
	Utility::checkOpenGLError("glDeleteBuffers()");

	// Create the PixelBufferObject to output the Ray-Tracing result.
	glGenBuffers(1, bufferObjectID);
	Utility::checkOpenGLError("glGenBuffers()");

	glBindBuffer(GL_ARRAY_BUFFER, *bufferObjectID);

		glBufferData(GL_ARRAY_BUFFER, pixelBufferObjectSize, NULL, GL_DYNAMIC_DRAW);
		Utility::checkOpenGLError("glBufferData()");

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register the PixelBufferObject with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterBuffer()", cudaGraphicsGLRegisterBuffer(bufferObjectResource, *bufferObjectID, cudaGraphicsMapFlagsWriteDiscard));
}

void deleteBufferObject(unsigned int *bufferObjectID) {

	// Delete the BufferObject from OpenGL 
    glDeleteBuffers(1, bufferObjectID);
	Utility::checkOpenGLError("glDeleteBuffers()");
}

void createScreenTexture(unsigned int *textureID, unsigned int width, unsigned int height) {

	// Delete the Texture in case it already exists.
	glDeleteTextures(1, textureID);

	// Create the Texture to output the Ray-Tracing result.
	glGenTextures(1, textureID);
	glBindTexture(GL_TEXTURE_2D, *textureID);

		// Set the basic Texture parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// Define the basic Texture parameters
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		Utility::checkOpenGLError("glTexImage2D()");

	glBindTexture(GL_TEXTURE_2D, 0);
}

void deleteScreenTexture(unsigned int *textureID) {

	// Delete the Texture from OpenGL 
	glDeleteTextures(1, textureID);
	Utility::checkOpenGLError("glDeleteTextures()");
}

void createShadingTexture(unsigned int *textureID, cudaGraphicsResource **textureResource, char* fileName) {

	// Delete the Texture in case it already exists.
	glDeleteTextures(1, textureID);

	// Create the Texture to store the Chess Image.
	glGenTextures(1, textureID);

	// Load a Sample Texture
	*textureID = SOIL_load_OGL_texture(fileName, SOIL_LOAD_RGBA, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y);

	// Check for an error during the loading process
	if(*textureID == 0) {

		cout << "[SOIL Error] Loading failed. (\"" << fileName << "\": " << SOIL_last_result() << std::endl;

		exit(1);
	}

	// Set the basic Texture parameters
	glBindTexture(GL_TEXTURE_2D, *textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(textureResource, *textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

void deleteShadingTexture(unsigned int *textureID) {

	// Delete the Texture from OpenGL 
	glDeleteTextures(1, textureID);
	Utility::checkOpenGLError("glDeleteTextures()");
}

// Initialize GLUT 
bool initGLUT(int argc, char** argv) {

	glutInit(&argc, argv);

	// Setup the Minimum OpenGL version 
	glutInitContextVersion(4,0);

	glutInitContextFlags( GLUT_FORWARD_COMPATIBLE );
	glutInitContextProfile( GLUT_COMPATIBILITY_PROFILE );

	// Setup the Display 
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth,windowHeight);

	// Setup the Window 
	windowHandle = glutCreateWindow("CUDA Ray Tracer");

	if(windowHandle < 1) {

		fprintf(stderr, "[GLUT Error] Could not create a new rendering window.\n");
		fflush(stderr);

		exit(EXIT_FAILURE);
	}

	// Setup the Callback Functions 
	glutDisplayFunc(display);
	glutTimerFunc(0, displayFrames, 0);
	glutTimerFunc(0, userInteraction, 0);
		
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

	cout << "[Initialization] GLUT Initialization Successfull" << endl << endl;

	return true;
}

// Initialize GLEW 
bool initGLEW() {

	GLenum error = glewInit();

	// Check if the Initialization went ok. 
	if(error != GLEW_OK) {

		fprintf(stderr, "[GLEW Error] %s\n", glewGetErrorString(error));
		fflush(stderr);

		exit(EXIT_FAILURE);
	}


	// Check if OpenGL 2.0 is supported 
	if(!glewIsSupported("GL_VERSION_2_0")) {

		fprintf(stderr, "[GLEW Error] Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);

		exit(EXIT_FAILURE);
	}

	fprintf(stdout, "[Initialization] Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	cout << "[Initialization] GLEW Initialization Successfull" << endl << endl;

	return true;
}

// Initialize OpenGL 
bool initOpenGL() {

	// Initialize the State 
	glClearColor(0, 0, 0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	// Initialize the Viewport 
	glViewport(0, 0, windowWidth, windowHeight);

	fprintf(stdout, "[Initialization] Status: Using OpenGL v%s\n", glGetString(GL_VERSION));

		cout << "[Initialization] OpenGL Initialization Successfull" << endl << endl;

	return true;
}

// Initialize the Scene 
void initCamera() {

	camera = new Camera(windowWidth, windowHeight);
	
	camera->setFieldOfView(60.0f);

	camera->setPosition(Vector(0.0f, 0.0f, 0.0f, 1.0f));

	cout << "[Initialization] Camera Initialization Successfull" << endl << endl;
}

void initObjects() {

	// Create 5 Objects that will contain the Meshes
	objectMap[SPHERE_0] = new Object("Sphere 0");
	objectMap[SPHERE_1] = new Object("Sphere 1");
	objectMap[SPHERE_2] = new Object("Sphere 2");
	objectMap[SPHERE_3] = new Object("Sphere 3");

	objectMap[PLATFORM] = new Object("Platform");

	// Load the Spheres Mesh from the OBJ file 
	Mesh* sphere0Mesh = new Mesh("Sphere", "emeraldsphere.obj", "emerald.mtl");
	Mesh* sphere1Mesh = new Mesh("Sphere", "rubysphere.obj", "ruby.mtl");
	Mesh* sphere2Mesh = new Mesh("Sphere", "goldsphere.obj", "gold.mtl");
	Mesh* sphere3Mesh = new Mesh("Sphere", "silversphere.obj", "silver.mtl");

	// Load the Platforms Mesh from the OBJ file 
	//Mesh* platformMesh = new Mesh("Platform", "surface.obj", "surface.mtl");
	Mesh* platformMesh = new Mesh("Platform", "cube.obj", "cube.mtl");

	// Load Sphere0s Transform 
 	Transform* sphere0Transform = new Transform("Sphere 0 Transform");
	sphere0Transform->setPosition(Vector(-10.0f, -2.5f, 10.0f, 1.0f));
	sphere0Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	// Load Sphere1s Transform 
	Transform* sphere1Transform = new Transform("Sphere 1 Transform");
	sphere1Transform->setPosition(Vector(-10.0f, -2.5f,-10.0f, 1.0f));
	sphere1Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	// Load Sphere2s Transform 
	Transform* sphere2Transform = new Transform("Sphere 2 Transform");
	sphere2Transform->setPosition(Vector( 10.0f, -2.5f,-10.0f, 1.0f));
	sphere2Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	// Load Sphere3s Transform 
	Transform* sphere3Transform = new Transform("Sphere 3 Transform");
	sphere3Transform->setPosition(Vector( 10.0f, -2.5f, 10.0f, 1.0f));
	sphere3Transform->setScale(Vector( 5.0f, 5.0f, 5.0f, 1.0f));
	// Load Platforms Transform 
	Transform* platformTransform = new Transform("Platform Transform");
	platformTransform->setPosition(Vector( 0.0f,-15.0f, 0.0f, 1.0f));
	platformTransform->setScale(Vector( 50.0f, 0.75f, 50.0f, 1.0f));

	// Set the Mesh and Transform of the created Objects 
	objectMap[SPHERE_0]->setMesh(sphere0Mesh);
	objectMap[SPHERE_0]->setTransform(sphere0Transform);

	objectMap[SPHERE_1]->setMesh(sphere1Mesh);
	objectMap[SPHERE_1]->setTransform(sphere1Transform);

	objectMap[SPHERE_2]->setMesh(sphere2Mesh);
	objectMap[SPHERE_2]->setTransform(sphere2Transform);

	objectMap[SPHERE_3]->setMesh(sphere3Mesh);
	objectMap[SPHERE_3]->setTransform(sphere3Transform);

	objectMap[PLATFORM]->setMesh(platformMesh);
	objectMap[PLATFORM]->setTransform(platformTransform);

	cout << "[Initialization] Object Initialization Successfull" << endl << endl;
}

// Initialize CUDA 
bool initCUDA() {

	int device = gpuGetMaxGflopsDeviceId();

	// Force CUDA to use the Highest performance GPU
	Utility::checkCUDAError("cudaSetDevice()",		cudaSetDevice(device));
	Utility::checkCUDAError("cudaGLSetGLDevice()",	cudaGLSetGLDevice(device));

	cout << "[Initialization] CUDA Initialization Successfull" << endl << endl;

	return true;
}

// Initialize CUDA Memory with the necessary space for the Meshes 
void initCUDAmemory() {

	// Create the PixelBufferObject to output the Ray-Tracing result.
	createBufferObject(&pixelBufferObjectID, &pixelBufferObjectResource, windowWidth, windowHeight);

	// Create the Texture to output the Ray-Tracing result.
	createScreenTexture(&screenTextureID, windowWidth, windowHeight);

	//screenTexture = new ScreenTexture("Screen Texture", windowWidth, windowHeight);
	//screenTexture->createTexture();

	// Create the Chess Texture
	createShadingTexture(&chessTextureID, &chessTextureResource, "textures/fieldstone_diffuse.jpg");

	//chessTexture = new ShadingTexture("Chess Texture", "textures/fieldstone_diffuse.jpg");
	//chessTexture->createTexture();

	// Load the Triangles to an Array
	vector<float4> trianglePositions;
	vector<float4> triangleNormals;
	vector<float4> triangleTangents;
	vector<float2> triangleTextureCoordinates;

	vector<float4> triangleDiffuseProperties;
	vector<float4> triangleSpecularProperties;

	for(map<int,Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		if(object->getName() != "Platform")
			continue;
	
		// Used for the position transformations
		Matrix modelMatrix = object->getTransform()->getModelMatrix();
		// Used for the normal transformations
		Matrix modelMatrixInverseTranspose = modelMatrix;
		//modelMatrixInverseTranspose.removeTranslation();
		modelMatrixInverseTranspose.transpose();
		modelMatrixInverseTranspose.invert();

		for(int j = 0; j < object->getMesh()->getVertexCount(); j++)	{

			// Get the original vertex from the mesh 
			Vertex originalVertex = object->getMesh()->getVertex(j);

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

	// Total number of Triangles should be the number of loaded vertices divided by 3
	triangleTotal = trianglePositions.size() / 3;

	cout << "[Initialization] Total number of triangles:" << triangleTotal << endl;

	// Each triangle contains Position, Normal, Tangent, Texture UV and Material Properties for 3 vertices
	size_t trianglePositionsSize = trianglePositions.size() * sizeof(float4);
	cout << "[Initialization] Triangle Positions Storage Size: " << trianglePositionsSize << " (" << trianglePositions.size() << " floats)" << endl;

	size_t triangleNormalsSize = triangleNormals.size() * sizeof(float4);
	cout << "[Initialization] Triangle Normals Storage Size: " << triangleNormalsSize << " (" << triangleNormals.size() << " floats)" << endl;
	size_t triangleTangentsSize = triangleTangents.size() * sizeof(float4);
	cout << "[Initialization] Triangle Tangents Storage Size: " << triangleTangentsSize << " (" << triangleTangents.size() << " floats)" << endl;

	size_t triangleTextureCoordinatesSize = triangleTextureCoordinates.size() * sizeof(float2);
	cout << "[Initialization] Triangle Texture Coordinates Storage Size: " << triangleTextureCoordinatesSize << " (" << triangleTextureCoordinates.size() << " floats)" << endl;

	size_t triangleDiffusePropertiesSize = triangleDiffuseProperties.size() * sizeof(float4);
	cout << "[Initialization] Triangle Diffuse Properties Storage Size: " << triangleDiffusePropertiesSize << " (" << triangleDiffuseProperties.size() << " floats)" << endl;
	size_t triangleSpecularPropertiesSize = triangleSpecularProperties.size() * sizeof(float4);
	cout << "[Initialization] Triangle Specular Properties Storage Size: " << triangleSpecularPropertiesSize << " (" << triangleSpecularProperties.size() << " floats)" << endl;

	// Allocate the required CUDA Memory
	if(triangleTotal > 0) {

		// Load the Triangle Positions
		Utility::checkCUDAError("cudaMalloc()",	cudaMalloc((void **)&cudaTrianglePositionsDP, trianglePositionsSize));
		Utility::checkCUDAError("cudaMemcpy()",	cudaMemcpy(cudaTrianglePositionsDP, &trianglePositions[0], trianglePositionsSize, cudaMemcpyHostToDevice));

		bindTrianglePositions(cudaTrianglePositionsDP, triangleTotal);

		// Load the Triangle Normals
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleNormalsDP, triangleNormalsSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleNormalsDP, &triangleNormals[0], triangleNormalsSize, cudaMemcpyHostToDevice));

		bindTriangleNormals(cudaTriangleNormalsDP, triangleTotal);

		/*// Load the Triangle Tangents
		cudaMalloc((void **)&cudaTriangleTangentsDP, triangleTangentsSize);
		Utility::checkCUDAError("cudaMalloc()");
		cudaMemcpy(cudaTriangleTangentsDP, &triangleTangents[0], triangleTangentsSize, cudaMemcpyHostToDevice);
		Utility::checkCUDAError("cudaMemcpy()");*/

		//bindTriangleTangents(cudaTriangleTangentsDP, triangleTotal);

		// Load the Triangle Texture Coordinates
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleTextureCoordinatesDP, triangleTextureCoordinatesSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleTextureCoordinatesDP, &triangleTextureCoordinates[0], triangleTextureCoordinatesSize, cudaMemcpyHostToDevice));

		bindTriangleTextureCoordinates(cudaTriangleTextureCoordinatesDP, triangleTotal);

		// Load the Triangle Diffuse Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleDiffusePropertiesDP, triangleDiffusePropertiesSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleDiffusePropertiesDP, &triangleDiffuseProperties[0], triangleDiffusePropertiesSize, cudaMemcpyHostToDevice));

		bindTriangleDiffuseProperties(cudaTriangleDiffusePropertiesDP, triangleTotal);

		// Load the Triangle Specular Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleSpecularPropertiesDP, triangleSpecularPropertiesSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleSpecularPropertiesDP, &triangleSpecularProperties[0], triangleSpecularPropertiesSize, cudaMemcpyHostToDevice));

		bindTriangleSpecularProperties(cudaTriangleSpecularPropertiesDP, triangleTotal);
	}

	cout << "[Initialization] CUDA Memory Initialization Successfull" << endl << endl;
}

// User Input Functions 
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

// Manages the Mouse Input 
void readMouse(GLfloat elapsedTime) {

	MouseHandler* handler = MouseHandler::getInstance();

	if(!handler->isMouseEnabled())
		return;

	handler->disableMouse();

	handler->enableMouse();
}

// Manages the Keyboard Input 
void readKeyboard(GLfloat elapsedTime) {

	KeyboardHandler* handler = KeyboardHandler::getInstance();

	if(!handler->isKeyboardEnabled())
		return;

	handler->disableKeyboard();

	GLint zoom = 0;
	GLint longitude = 0;
	GLint latitude = 0;

	/* Camera Buttons */
	if(handler->isSpecialKeyPressed(GLUT_KEY_LEFT) && handler->wasSpecialKeyPressed(GLUT_KEY_LEFT) == true) {

		longitude+=10;		
	}
	if(handler->isSpecialKeyPressed(GLUT_KEY_RIGHT) && handler->wasSpecialKeyPressed(GLUT_KEY_RIGHT) == true) {

		longitude-=10;		
	}

	/* Camera Buttons */
	if(handler->isSpecialKeyPressed(GLUT_KEY_UP) && handler->wasSpecialKeyPressed(GLUT_KEY_UP) == true) {

		zoom+=10;		
	}
	if(handler->isSpecialKeyPressed(GLUT_KEY_DOWN) && handler->wasSpecialKeyPressed(GLUT_KEY_DOWN) == true) {

		zoom-=10;		
	}

	camera->update(zoom, longitude, latitude, elapsedTime);

	handler->enableKeyboard();
}

void displayFrames(int time) {

	std::ostringstream oss;
	oss << "CUDA Ray Tracer" << ": " << frameCount << " FPS @ (" << windowWidth << "x" << windowHeight << ")";
	std::string s = oss.str();

	// Update the Window
	glutSetWindow(windowHandle);
	glutSetWindowTitle(s.c_str());

    frameCount = 0;

    glutTimerFunc(1000, displayFrames, 0);
}

void userInteraction(int time) {

	// Update the Scenes Objects
	readMouse(deltaTime);
	readKeyboard(deltaTime);

	glutTimerFunc(1000/60, userInteraction, 0);
}

void display() {

	++frameCount;

	// Time Management
	int currentFrameTime = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = (float)(currentFrameTime - lastFrameTime) / 1000.0f;
	lastFrameTime = currentFrameTime;

	camera->update(0, 5, 0, deltaTime);

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Call the Ray-Tracing CUDA Implementation 
	rayTrace();
	// Call the OpenGL Rendering
	draw();

	// Swap the Screen Buffer and Call the Display function again. 
	glutSwapBuffers();
	glutPostRedisplay();

	cout << "[Callback] Display Successfull" << endl;
}

// Callback function called by GLUT when window size changes - TODO
void reshape(int width, int height) {

	windowWidth = width;
	windowHeight = height;

	// Set OpenGL Viewport and Projection
	glViewport(0, 0, windowWidth, windowHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Update the Camera
	camera->reshape(windowWidth, windowHeight);

	// Create the PixelBufferObject to output the Ray-Tracing result.
	createBufferObject(&pixelBufferObjectID, &pixelBufferObjectResource, windowWidth, windowHeight);

	// Create the Texture to output the Ray-Tracing result.
	createScreenTexture(&screenTextureID, windowWidth, windowHeight);

	//screenTexture->setWidth(windowWidth);
	//screenTexture->setHeight(windowHeight);

	//screenTexture->createTexture();

	cout << "[Callback] Reshape Successfull" << endl;
}

// Callback function called by GLUT when the program exits
void cleanup() {

	// Delete the PixelBufferObject
	deleteBufferObject(&pixelBufferObjectID);

	// Delete the Screen Texture 
	deleteScreenTexture(&screenTextureID);

	//screenTexture->deleteTexture();

	// Delete the Shading Textures
	deleteShadingTexture(&chessTextureID);

	//chessTexture->deleteTexture();

	// Delete the CudaDevicePointers to the uploaded Triangle Information 
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTrianglePositionsDP));
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTriangleNormalsDP));
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTriangleTangentsDP));
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTriangleTextureCoordinatesDP));
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTriangleDiffusePropertiesDP));
	Utility::checkCUDAError("cudaFree()",  cudaFree(cudaTriangleSpecularPropertiesDP));

	// Force CUDA to flush profiling information 
	cudaDeviceReset();

	cout << "[Callback] Cleanup Successfull" << endl;
}

void rayTrace() {

	// Camera defining Vectors 
	Vector up = camera->getUp();
	up.clean();
	Vector eye = camera->getEye();
	eye.clean();
	Vector target = camera->getTarget();
	target.clean();

	// Images Aspect Ratio 
	float aspectRatio = (float)windowWidth / (float)windowHeight;
	// Cameras distance to the target 
	float distance = (target - eye).length();
	// Cameras Field of View 
	float fieldOfView = camera->getFieldOfView();
	// Projection Frustum Half-Width 
	float theta = (fieldOfView * 0.5f) * DEGREES_TO_RADIANS;
	float halfHeight = 2.0f * distance * tanf(theta);
	float halfWidth = halfHeight * aspectRatio;

	// Camera Position and Direction 
	float3 cameraPosition = make_float3(eye[VX], eye[VY], eye[VZ]);
	float3 cameraDirection = make_float3(target[VX] - eye[VX], target[VY] - eye[VY], target[VZ] - eye[VZ]);
	cameraDirection = normalize(cameraDirection);
	cameraDirection = distance * cameraDirection;

	// Camera Right and Up 
	float3 cameraUp = make_float3(up[VX], up[VY], up[VZ]);
	cameraUp = normalize(cameraUp);

	float3 cameraRight = cross(cameraDirection, cameraUp);
	cameraRight = normalize(cameraRight);
	cameraRight = halfWidth * cameraRight;

	cameraUp = cross(cameraRight, cameraDirection);
	cameraUp = normalize(cameraUp);
	cameraUp = halfHeight * cameraUp;

	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &pixelBufferObjectResource, 0));
	Utility::checkCUDAError("cudaGraphicsMapResources()", cudaGraphicsMapResources(1, &chessTextureResource, 0));

	unsigned int* screenTextureDP = NULL;
	size_t screenTextureSize = 0;

	// Map the PixelBufferObject and Textures
	Utility::checkCUDAError("cudaGraphicsResourceGetMappedPointer()", cudaGraphicsResourceGetMappedPointer((void**)&screenTextureDP, &screenTextureSize, pixelBufferObjectResource));
	Utility::checkCUDAError("cudaGraphicsSubResourceGetMappedArray()", cudaGraphicsSubResourceGetMappedArray(&chessTextureArray, chessTextureResource, 0, 0));

	bindTextureArray(chessTextureArray);

	//chessTexture->mapResources();

	//cudaArray* chessTextureCudaArray = chessTexture->getCudaArrayReference();

	//bindTextureArray(chessTextureCudaArray);

	// Kernel Launch
	RayTraceWrapper(screenTextureDP,
		windowWidth, windowHeight, 
		triangleTotal,
		cameraRight, cameraUp, cameraDirection, 
		cameraPosition);

	// Unmap the used CUDA Resources
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &pixelBufferObjectResource, 0));
	Utility::checkCUDAError("cudaGraphicsUnmapResources()", cudaGraphicsUnmapResources(1, &chessTextureResource, 0));

	//chessTexture->unmapResources();

	// Copy the Output to the Texture 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, screenTextureID);

		glActiveTexture(GL_TEXTURE0);

		glBindTexture(GL_TEXTURE_2D, screenTextureID);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//screenTexture->replaceTexture();

	Utility::checkOpenGLError("glTexSubImage2D()");
}

void draw() {

	// Draw the resulting Texture on a Quad covering the Screen 
	glEnable(GL_TEXTURE_2D);

		//glBindTexture(GL_TEXTURE_2D, screenTexture->getHandler());
		glBindTexture(GL_TEXTURE_2D, screenTextureID);

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

		glBindTexture(GL_TEXTURE_2D, 0);

	glDisable(GL_TEXTURE_2D);
	
	Utility::checkOpenGLError("drawPixels()");
}

int main(int argc, char** argv) {

	// Initialize the Scene 
	initCamera();
	initObjects();

	// Initialize GLUT, GLEW and OpenGL 
	initGLUT(argc,argv);
	initGLEW();
	initOpenGL();

	// Initialize CUDA 
	initCUDA();
	initCUDAmemory();

	/* Enable User Interaction */
	KeyboardHandler::getInstance()->enableKeyboard();
	MouseHandler::getInstance()->enableMouse();

	/* Start the Clock */
	lastFrameTime = glutGet(GLUT_ELAPSED_TIME);

	// start rendering main-loop
	glutMainLoop();

	cudaThreadExit();

	return 0;
}