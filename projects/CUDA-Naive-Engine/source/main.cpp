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

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

// Object
#include "Object.h"

// Particle Systems
#include "ParticleSystem.h"

// Lighting
#include "SpotLight.h"
#include "PositionalLight.h"
#include "DirectionalLight.h"

// Shaders
#include "BlinnPhongShader.h"
#include "BumpMappingShader.h"

#include "SphereMappingShader.h"
#include "CubeMappingShader.h"

#include "FireShader.h"
#include "WoodShader.h"

// Textures
#include "Texture.h"
#include "CubeTexture.h"
#include "GeneratedTexture.h"

// Scene Manager
#include "SceneManager.h"

// CUDA-OpenGL Interop
#include "FrameBuffer.h"
#include "PixelBuffer.h"

#include "ScreenTexture.h"

// Utility
#include "XML_Reader.h"
#include "OBJ_Reader.h"

#define FPS_60	1000/60

#define CAPTION	"OpenGL-CUDA Engine 2015"

// Frame Count Global Variable
int frameCount = 0;

// Window Handling Global Variables
int windowHandle = 0;

int windowWidth = WIDTH;
int windowHeight = HEIGHT;

// Clock Handling Global Variables
GLint startTime = 0;
GLfloat time = 0;
GLfloat elapsedTime = 0;

// Scene Manager
SceneManager* sceneManager = SceneManager::getInstance();

// FrameBuffer Wrapper
FrameBuffer *frameBuffer;
// PixelBuffer Wrapper
PixelBuffer *pixelBuffer;
// Screens Textures Wrapper
ScreenTexture *screenTexture;

// Total number of Triangles - Used for the memory necessary to allocate
int triangleTotal = 0;
// Total number of Materials - Used for the memory necessary to allocate
int materialTotal = 0;
// Total number of Lights - Used for the memory necessary to allocate
int lightTotal = 0;

// CUDA DevicePointers to the uploaded Triangles Positions and Normals
float *cudaTrianglePositionsDP = NULL; 
float *cudaTriangleNormalsDP = NULL;
// CUDA DevicePointers to the uploaded Triangles Texture Coordinates 
float *cudaTriangleTextureCoordinatesDP = NULL;
// CUDA DevicePointers to the uploaded Triangles Object and Material IDs
float *cudaTriangleObjectIDsDP = NULL;
float *cudaTriangleMaterialIDsDP = NULL;

// CUDA DevicePointers to the uploaded Triangles Materials 
float *cudaMaterialDiffusePropertiesDP = NULL;
float *cudaMaterialSpecularPropertiesDP = NULL;

// CUDA DevicePointers to the uploaded Lights
float *cudaLightPositionsDP = NULL;
float *cudaLightColorsDP = NULL;
float *cudaLightIntensitiesDP = NULL;

// CUDA DevicePointers Arrays
float4* cudaUpdatedTrianglePositionsDP = NULL;
float4* cudaUpdatedTriangleNormalsDP = NULL;

float* cudaUpdatedModelMatricesDP = NULL;
float* cudaUpdatedNormalMatricesDP = NULL;

// [CUDA-OpenGL Interop] 
extern "C" {

	// Implementation of RayTraceWrapper is in the "RayTracer.cu" file
	void RayTraceWrapper(	unsigned int *pixelBufferObject,
							// Screen Dimensions
							int width, int height, 			
							// Updated Normal Matrices Array
							float* modelMatricesArray,
							// Updated Normal Matrices Array
							float* normalMatricesArray,
							// Updated Triangle Positions Array
							float4* trianglePositionArray,
							// Updated Triangle Normals Array
							float4* triangleNormalArray,
							// Total Number of Triangles in the Scene
							int triangleTotal,
							// Total Number of Lights in the Scene
							int lightTotal,
							// Camera Definitions
							float3 cameraPosition);

	// Implementation of bindRenderTextureArray is in the "RayTracer.cu" file
	void bindDiffuseTextureArray(cudaArray *diffuseTextureArray);
	// Implementation of bindRayOriginTextureArray is in the "RayTracer.cu" file
	void bindSpecularTextureArray(cudaArray *specularTextureArray);
	// Implementation of bindRayReflectionTextureArray is in the "RayTracer.cu" file
	void bindFragmentPositionArray(cudaArray *fragmentPositionArray);
	// Implementation of bindRayRefractionTextureArray is in the "RayTracer.cu" file
	void bindFragmentNormalArray(cudaArray *fragmentNormalArray);

	// Implementation of bindTrianglePositions is in the "RayTracer.cu" file
	void bindTrianglePositions(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of bindTriangleNormals is in the "RayTracer.cu" file
	void bindTriangleNormals(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of bindTriangleTextureCoordinates is in the "RayTracer.cu" file
	void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of bindTriangleObjectIDs is in the "RayTracer.cu" file
	void bindTriangleObjectIDs(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of bindTriangleMaterialIDs is in the "RayTracer.cu" file
	void bindTriangleMaterialIDs(float *cudaDevicePointer, unsigned int triangleTotal);

	// Implementation of bindMaterialDiffuseProperties is in the "RayTracer.cu" file
	void bindMaterialDiffuseProperties(float *cudaDevicePointer, unsigned int materialTotal);
	// Implementation of bindMaterialSpecularProperties is in the "RayTracer.cu" file
	void bindMaterialSpecularProperties(float *cudaDevicePointer, unsigned int materialTotal);

	// Implementation of bindLightPositions is in the "RayTracer.cu" file
	void bindLightPositions(float *cudaDevicePointer, unsigned int lightTotal);
	// Implementation of bindLightColors is in the "RayTracer.cu" file
	void bindLightColors(float *cudaDevicePointer, unsigned int lightTotal);
	// Implementation of bindLightIntensities is in the "RayTracer.cu" file
	void bindLightIntensities(float *cudaDevicePointer, unsigned int lightTotal);
}

// [Scene Functions]

	void update(int value);

	void display();
	void reshape(int weight, int height);

	void cleanup();
	void timer(int value);

// [Scene Listeners]

	void normalKeyListener(unsigned char key, int x, int y);
	void releasedNormalKeyListener(unsigned char key, int x, int y);

	void specialKeyListener(int key, int x, int y);
	void releasedSpecialKeyListener(int key, int x, int y);

	void mouseEventListener(int button, int state, int x, int y);

	void mouseMovementListener(int x, int y);
	void mousePassiveMovementListener(int x, int y);

	void mouseWheelListener(int button, int direction, int x, int y);

// [OpenGL Initialization]

	void initializeCallbacks();
	
	void initializeGLUT(int argc, char **argv);
	void initializeGLEW();
	void initializeOpenGL();

	void initializeCUDA();
	void initializeCUDAmemory();

// [Scene Initialization]

	void initializeShaders();

	void initializeLights();
	void initializeCameras();

// [Scene] Updates the Scene
void update(int value) {

	/* Update the Timer Variables */
	elapsedTime = (GLfloat)(glutGet(GLUT_ELAPSED_TIME) - startTime)/1000;
	startTime = glutGet(GLUT_ELAPSED_TIME);
	time += elapsedTime;

	/* Update the Scene */
	sceneManager->update(elapsedTime);
	sceneManager->getCamera(PERSPECTIVE_NAME)->update(0, 5, 0, elapsedTime);

	glutTimerFunc(FPS_60, update, 0);

	glutPostRedisplay();

	cout << "[Callback] Update Successfull" << endl;
}

// [Scene] Displays the Scene
void display() {

	++frameCount;
	
	// Bind the FrameBuffer
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer->getFrameBufferHandler());

		// Clear the Buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Draw to the Screen
		sceneManager->draw();

	// Unbind the Buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Map the necessary CUDA Resources
	frameBuffer->mapCudaResource();
	pixelBuffer->mapCudaResource();
	
    // Get the CUDA arrays references
	cudaArray* diffuseTextureArray = frameBuffer->getDiffuseTextureCudaArray();
	cudaArray* specularTextureArray = frameBuffer->getSpecularTextureCudaArray();
	cudaArray* fragmentPositionArray = frameBuffer->getFragmentPositionCudaArray();
	cudaArray* fragmentNormalArray = frameBuffer->getFragmentNormalCudaArray();

    // Bind the textures to CUDA arrays
    bindDiffuseTextureArray(diffuseTextureArray);
	bindSpecularTextureArray(specularTextureArray);
	bindFragmentPositionArray(fragmentPositionArray);
	bindFragmentNormalArray(fragmentNormalArray);

	// Get the Device Pointer References
	unsigned int* pixelBufferDevicePointer = pixelBuffer->getDevicePointer();

	// Get the Camera Positions 
	Vector position = sceneManager->getActiveCamera()->getEye();

	// Get the Updated Model and Normal Matrices
	map<string, Object*> objectMap = sceneManager->getObjectMap();

	float* modelMatrices = new float[objectMap.size() * 16];
	float* normalMatrices = new float[objectMap.size() * 16];

	for(map<string,Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		// Model Matrix/
		Matrix modelMatrix = object->getTransform()->getModelMatrix();
		modelMatrix.getValue(&modelMatrices[object->getID() * 16]);
		
		// Normal Matrix
		Matrix normalMatrix = object->getTransform()->getModelMatrix();
		normalMatrix.removeTranslation();
		normalMatrix.transpose();
		normalMatrix.invert();
		normalMatrix.getValue(&normalMatrices[object->getID() * 16]);

		/*map<int, Vertex*> vertexMap = object->getMesh()->getVertexMap();

		printf("\nFirst List (%s) (%d)\n\n", object->getName().c_str(), vertexMap.size());

		for(map<int, Vertex*>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++) {

			// Get the vertex from the mesh 
			Vertex* vertex = vertexIterator->second;

			// Position: Multiply the original vertex using the objects model matrix
			Vector originalPosition = vertex->getPosition();
			Vector modifiedPosition = modelMatrix * originalPosition;
			
			printf("1 Vertex[%d] = [%.2f] [%.2f] [%.2f]\n", vertexIterator->first, modifiedPosition[VX], modifiedPosition[VY], modifiedPosition[VZ]);

			// Normal: Multiply the original normal using the objects inverted transposed model matrix	
			Vector originalNormal = vertex->getNormal();
			Vector modifiedNormal = normalMatrix * originalNormal;
			modifiedNormal.normalize();
			
			printf("1 Normal[%d] = [%.2f] [%.2f] [%.2f]\n", vertexIterator->first, modifiedNormal[VX], modifiedNormal[VY], modifiedNormal[VZ]);
		}*/
	}

	// Copy the Matrices to CUDA	
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedModelMatricesDP, &modelMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedNormalMatricesDP, &normalMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));

	// Kernel Launch
	RayTraceWrapper(
		pixelBufferDevicePointer,
		windowWidth, windowHeight,
		cudaUpdatedModelMatricesDP, cudaUpdatedNormalMatricesDP,
		cudaUpdatedTrianglePositionsDP, cudaUpdatedTriangleNormalsDP,
		triangleTotal,
		lightTotal,
		make_float3(position[VX], position[VY], position[VZ]));

	/*float* models = new float[objectMap.size() * 16];
	float* normals = new float[objectMap.size() * 16];
	
	// Copy the Matrices to CUDA	
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&models[0], cudaUpdatedModelMatricesDP, objectMap.size() * 16 * sizeof(float), cudaMemcpyDeviceToHost));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&normals[0], cudaUpdatedNormalMatricesDP, objectMap.size() * 16 * sizeof(float), cudaMemcpyDeviceToHost));

	printf("\nSecond Matrices List (%d)\n\n", objectMap.size());

	for(int i=0; i<objectMap.size(); i++) {
		
		int offset = i*16;

		printf("Model Matrix\n");
		for(int j=0; j<4; j++)
			printf("[%.2f][%.2f][%.2f][%.2f]\n", models[offset + j * 4],  models[offset + j * 4 + 1],  models[offset + j * 4 + 2],  models[offset + j * 4 + 3]);
	
		printf("Normal Matrix\n");
		for(int j=0; j<4; j++)
			printf("[%.2f][%.2f][%.2f][%.2f]\n", normals[offset + j * 4],  normals[offset + j * 4 + 1],  normals[offset + j * 4 + 2],  normals[offset + j * 4 + 3]);
	}*/

	/*float4* trianglePositions = new float4[triangleTotal * 3];
	float4* triangleNormals = new float4[triangleTotal * 3];
	
	// Copy the Matrices to CUDA	
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&trianglePositions[0], cudaUpdatedTrianglePositionsDP, triangleTotal * 3 * sizeof(float4), cudaMemcpyDeviceToHost));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&triangleNormals[0], cudaUpdatedTriangleNormalsDP, triangleTotal * 3 * sizeof(float4), cudaMemcpyDeviceToHost));

	printf("\nSecond Vertices List (%d)\n\n", triangleTotal*3);

	for(int i=0; i<triangleTotal*3; i++) {

		float4 position = trianglePositions[i];
	
		printf("2 Vertex[%d] = [%.2f] [%.2f] [%.2f] [%.2f]\n", i, position.x, position.y, position.z, position.w);

		float4 normal = triangleNormals[i];
	
		printf("2 Normal[%d] = [%.2f] [%.2f] [%.2f] [%.2f]\n", i, normal.x, normal.y, normal.z, normal.w);
	}*/
		
	// Unmap the used CUDA Resources
	frameBuffer->unmapCudaResource();
	pixelBuffer->unmapCudaResource();

	// Copy the Output to the Texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pixelBuffer->getHandler());
		glBindTexture(GL_TEXTURE_2D, screenTexture->getHandler());
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	// Clear the Buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Draw the resulting Texture on a Quad covering the Screen 
	glEnable(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, screenTexture->getHandler());

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

	// Swap the Buffers
	glutSwapBuffers();
	
	cout << "[Callback] Display Successfull" << endl;
}

// [Scene] Reshapes up the Scene
void reshape(int weight, int height) {

	windowWidth = weight;
	windowHeight = height;

	glViewport(0, 0, windowWidth, windowHeight);
	
	// Update the SceneManager
	sceneManager->reshape(windowWidth,windowHeight);

	// Update the FrameBuffer
	frameBuffer->setWidth(windowWidth);
	frameBuffer->setHeight(windowHeight);

	frameBuffer->createFrameBuffer();

	// Update the PixelBuffers
	pixelBuffer->setWidth(windowWidth);
	pixelBuffer->setHeight(windowHeight);

	pixelBuffer->createPixelBuffer();

	// Update the Screen Texture
	screenTexture->setWidth(windowWidth);
	screenTexture->setHeight(windowHeight);

	screenTexture->createTexture();

	cout << "[Callback] Reshape Successfull" << endl;
}

// [Scene] Cleans up the Scene
void cleanup() {

	// Delete the Scene
	sceneManager->destroyInstance();

	// Delete the FrameBuffer
	frameBuffer->deleteFrameBuffer();
	// Delete the PixelBuffer
	pixelBuffer->deletePixelBuffer();
	// Delete the Screen Texture 
	screenTexture->deleteTexture();

	// Force CUDA to flush profiling information 
	cudaDeviceReset();

	cout << "[Callback] Cleanup Successfull" << endl;
}

// [Scene] Updates the Scenes Timers
void timer(int value) {

	std::ostringstream oss;
	oss << CAPTION << ": " << frameCount << " FPS @ (" << windowWidth << "x" << windowHeight << ")";
	std::string s = oss.str();

	glutSetWindow(windowHandle);
	glutSetWindowTitle(s.c_str());

    frameCount = 0;

    glutTimerFunc(1000, timer, 0);
}

// [Scene] Updates the Scenes Normal Key Variables
void normalKeyListener(unsigned char key, int x, int y) {

	KeyboardHandler::getInstance()->normalKeyListener(key,x,y);
}

// [Scene] Updates the Scenes Released Normal Key Variables
void releasedNormalKeyListener(unsigned char key, int x, int y) {

	KeyboardHandler::getInstance()->releasedNormalKeyListener(key,x,y);
}

// [Scene] Updates the Scenes Special Key Variables
void specialKeyListener(int key, int x, int y) {

	KeyboardHandler::getInstance()->specialKeyListener(key,x,y);
}

// [Scene] Updates the Scenes Released Special Key Variables
void releasedSpecialKeyListener(int key, int x, int y) {

	KeyboardHandler::getInstance()->releasedSpecialKeyListener(key,x,y);
}

// [Scene] Updates the Scenes Mouse Click Event Variables
void mouseEventListener(int button, int state, int x, int y) {

	MouseHandler::getInstance()->mouseEventListener(button,state,x,y);
}

// [Scene] Updates the Scenes Mouse Movement Event Variables
void mouseMovementListener(int x, int y) {

	MouseHandler::getInstance()->mouseMovementListener(x,y);
}

// [Scene] Updates the Scenes Mouse Passive Movement Event Variables
void mousePassiveMovementListener(int x, int y) {

	MouseHandler::getInstance()->mouseMovementListener(x,y);
}

// [Scene] Updates the Scenes Mouse Wheel Event Variables
void mouseWheelListener(int button, int direction, int x, int y)  {

	MouseHandler::getInstance()->mouseWheelListener(button,direction,x,y);
} 

// [OpenGL] Initializes the Callbacks
void initializeCallbacks() {

	glutCloseFunc(cleanup);

	glutDisplayFunc(display);

	glutReshapeFunc(reshape);

	glutKeyboardFunc(normalKeyListener); 
	glutKeyboardUpFunc(releasedNormalKeyListener); 
	glutSpecialFunc(specialKeyListener);
	glutSpecialUpFunc(releasedSpecialKeyListener);

	glutMouseFunc(mouseEventListener);
	glutMotionFunc(mouseMovementListener);
	glutPassiveMotionFunc(mousePassiveMovementListener);
	glutMouseWheelFunc(mouseWheelListener);

	glutTimerFunc(0,timer,0);
	glutTimerFunc(0,update,0);
}

// [OpenGL] Initializes Glut
void initializeGLUT(int argc, char **argv) {

	glutInit(&argc, argv);
	
	// Setup the Minimum OpenGL version 
	glutInitContextVersion(4,0);

	glutInitContextFlags( GLUT_FORWARD_COMPATIBLE );
	glutInitContextProfile( GLUT_COMPATIBILITY_PROFILE );

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	
	// Setup the Display 
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	
	// Setup the Window 
	windowHandle = glutCreateWindow(CAPTION);

	if(windowHandle < 1) {

		fprintf(stderr, "[GLUT Error] Could not create a new rendering window.\n");
		fflush(stderr);

		exit(EXIT_FAILURE);
	}

	cout << "[Initialization] GLUT Initialization Successfull" << endl << endl;
}

// [OpenGL] Initializes Glew
void initializeGLEW() {

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
}

// [OpenGL] Initializes OpenGL
void initializeOpenGL() {

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glDepthRange(0.0f,1.0f);
	glClearDepth(1.0f);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	fprintf(stdout, "[Initialization] Status: Using OpenGL v%s\n", glGetString(GL_VERSION));

	cout << "[Initialization] OpenGL Initialization Successfull" << endl << endl;
}

// [CUDA] Initializes CUDA 
void initializeCUDA() {

	int device = gpuGetMaxGflopsDeviceId();

	// Force CUDA to use the Highest performance GPU
	Utility::checkCUDAError("cudaSetDevice()",		cudaSetDevice(device));
	Utility::checkCUDAError("cudaGLSetGLDevice()",	cudaGLSetGLDevice(device));

	cout << "[Initialization] CUDA Initialization Successfull" << endl << endl;
}

// Initialize CUDA Memory with the necessary space for the Meshes 
void initializeCUDAmemory() {
		
	// Create the FrameBuffer to output the Rendering result.
	frameBuffer = new FrameBuffer(windowWidth, windowHeight);
	frameBuffer->createFrameBuffer();

	// Create the PixelBufferObject to output the Ray-Tracing result.
	pixelBuffer = new PixelBuffer(windowWidth, windowHeight);
	pixelBuffer->createPixelBuffer();

	// Create the Texture to output the Ray-Tracing result.
	screenTexture = new ScreenTexture("Screen Texture", windowWidth, windowHeight);
	screenTexture->createTexture();

	cout << "[Initialization] CUDA Memory Initialization Successfull" << endl << endl;
}

// [Scene] Initializes the Scenes Shaders
void initializeShaders() {

	/* Create Blinn Phong Shader */
	BlinnPhongShader* blinnPhongShader = new BlinnPhongShader(BLINN_PHONG_SHADER);
	blinnPhongShader->createShaderProgram();
	blinnPhongShader->bindAttributes();
	blinnPhongShader->linkShaderProgram();
	blinnPhongShader->bindUniforms();

	sceneManager->addShaderProgram(blinnPhongShader);

	/* Create Bump Map Shader*/
	BumpMappingShader* bumpMappingShader = new BumpMappingShader(BUMP_MAPPING_SHADER);
	bumpMappingShader->createShaderProgram();
	bumpMappingShader->bindAttributes();
	bumpMappingShader->linkShaderProgram();
	bumpMappingShader->bindUniforms();

	sceneManager->addShaderProgram(bumpMappingShader);

	/* Create Sphere Map Shader */
	SphereMappingShader* sphereMappingShader = new SphereMappingShader(SPHERE_MAPPING_SHADER);
	sphereMappingShader->createShaderProgram();
	sphereMappingShader->bindAttributes();
	sphereMappingShader->linkShaderProgram();
	sphereMappingShader->bindUniforms();

	sceneManager->addShaderProgram(sphereMappingShader);

	/* Create Cube Map Shader */
	CubeMappingShader* cubeMappingShader = new CubeMappingShader(CUBE_MAPPING_SHADER);
	cubeMappingShader->createShaderProgram();
	cubeMappingShader->bindAttributes();
	cubeMappingShader->linkShaderProgram();
	cubeMappingShader->bindUniforms();

	sceneManager->addShaderProgram(cubeMappingShader);

	/* Create Real Wood Shader */
	WoodShader* woodShader = new WoodShader(WOOD_SHADER);
	woodShader->createShaderProgram();
	woodShader->bindAttributes();
	woodShader->linkShaderProgram();
	woodShader->bindUniforms();

	sceneManager->addShaderProgram(woodShader);

	/* Create Fire Shader */
	FireShader* fireShader = new FireShader(FIRE_SHADER);
	fireShader->createShaderProgram();
	fireShader->bindAttributes();
	fireShader->linkShaderProgram();
	fireShader->bindUniforms();

	sceneManager->addShaderProgram(fireShader);

	cout << "[Initialization] Shader Initialization Successfull" << endl << endl;
}

// [Scene] Initializes the Scenes Lights
void initializeLights() {

	// Light Map
	map<int, Light*> lightMap;

	// Light Source 1
	PositionalLight* positionalLight1 = new PositionalLight(POSITIONAL_LIGHT_1);

	positionalLight1->setIdentifier(LIGHT_SOURCE_1);

	positionalLight1->setPosition(Vector(0.0f, 10.0f, 0.0f, 1.0f));
	positionalLight1->setColor(Vector(1.0f, 1.0f, 1.0f, 1.0f));

	positionalLight1->setAmbientIntensity(0.25f);
	positionalLight1->setDiffuseIntensity(0.75f);
	positionalLight1->setSpecularIntensity(0.75f);

	positionalLight1->setConstantAttenuation(0.025f);
	positionalLight1->setLinearAttenuation(0.0075f);
	positionalLight1->setExponentialAttenuation(0.00075f);
	
	lightMap[positionalLight1->getIdentifier()] = positionalLight1;
	sceneManager->addLight(positionalLight1);

	// Stores the Lights Information in the form of Arrays
	vector<float4> lightPositionList;
	vector<float4> lightColorList;
	vector<float2> lightIntensityList;

	for(map<int,Light*>::const_iterator lightIterator = lightMap.begin(); lightIterator != lightMap.end(); lightIterator++) {

		Light* light = lightIterator->second;

		// Position
		Vector originalPosition = light->getPosition();
		float4 position = { originalPosition[VX], originalPosition[VY], originalPosition[VZ], 1.0f};
		lightPositionList.push_back(position);

		// Color
		Vector originalColor = light->getColor();
		float4 color = { originalColor[VX], originalColor[VY], originalColor[VZ], originalColor[VW]};
		lightColorList.push_back(color);

		// Intensity
		GLfloat originalDiffuseIntensity = light->getDiffuseIntensity();
		GLfloat originalSpecularIntensity = light->getSpecularIntensity();

		float2 intensity = { originalDiffuseIntensity, originalSpecularIntensity };
		lightIntensityList.push_back(intensity);

		lightTotal++;
	}

	// Total number of Lights 
	cout << "[Initialization] Total number of lights:" << lightTotal << endl;

	// Each light has a Position, Color and Intensity
	size_t lightPositionListSize = lightPositionList.size() * sizeof(float4);
	cout << "[Initialization] Light Positions Storage Size: " << lightPositionListSize << " (" << lightPositionList.size() << " float4s)" << endl;
	size_t lightColorListSize = lightColorList.size() * sizeof(float4);
	cout << "[Initialization] Light Color Storage Size: " << lightColorListSize << " (" << lightColorList.size() << " float4s)" << endl;
	size_t lightIntensityListSize = lightIntensityList.size() * sizeof(float2);
	cout << "[Initialization] Light Intensity Storage Size: " << lightIntensityListSize << " (" << lightIntensityList.size() << " float2s)" << endl;

	// Allocate the required CUDA Memory for the Lights
	if(lightTotal > 0) {

		// Load the Light Positions
		Utility::checkCUDAError("cudaMalloc()",	cudaMalloc((void **)&cudaLightPositionsDP, lightPositionListSize));
		Utility::checkCUDAError("cudaMemcpy()",	cudaMemcpy(cudaLightPositionsDP, &lightPositionList[0], lightPositionListSize, cudaMemcpyHostToDevice));

		bindLightPositions(cudaLightPositionsDP, lightTotal);

		// Load the Light Colors
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaLightColorsDP, lightColorListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaLightColorsDP, &lightColorList[0], lightColorListSize, cudaMemcpyHostToDevice));

		bindLightColors(cudaLightColorsDP, lightTotal);

		// Load the Light Intensities
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaLightIntensitiesDP, lightIntensityListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaLightIntensitiesDP, &lightIntensityList[0], lightIntensityListSize, cudaMemcpyHostToDevice));

		bindLightIntensities(cudaLightIntensitiesDP, lightTotal);
	}

	cout << "[Initialization] Light Initialization Successfull" << endl << endl;
}

// [Scene] Initializes the Scenes Cameras
void initializeCameras() {

	/* Create Orthogonal Camera */
	Camera* orthogonalCamera = new Camera(ORTHOGONAL_NAME);

	orthogonalCamera->loadOrthogonalProjection();
	orthogonalCamera->loadView();

	sceneManager->addCamera(orthogonalCamera);

	/* Create Perspective Camera */
	Camera* perspectiveCamera = new Camera(PERSPECTIVE_NAME);
	perspectiveCamera->setPosition(Vector(0.0f,0.0f, 0.0f,1.0f));
	perspectiveCamera->loadPerspectiveProjection();
	perspectiveCamera->loadView();

	sceneManager->addCamera(perspectiveCamera);

	/* Set Active Camera */
	sceneManager->setActiveCamera(perspectiveCamera);

	cout << "[Initialization] Camera Initialization Successfull" << endl << endl;
}

void init(int argc, char* argv[]) {

	//freopen("output.txt","w",stderr);
	//freopen("output.txt","w",stdout);

	// Initialize OpenGL
	initializeGLUT(argc, argv);
	initializeGLEW();
	initializeOpenGL();

	// Initialize CUDA 
	initializeCUDA();
	initializeCUDAmemory();

	// Initialize the Scene
	initializeShaders();

	initializeLights();
	initializeCameras();

	// Object Map
	map<int, Object*> objectMap;

	// Table Surface
	Object* tableSurface = new Object(TABLE_SURFACE);

		// Set the Objects Mesh
		Mesh* tableSurfaceMesh = new Mesh(TABLE_SURFACE, "cube.obj");
		tableSurface->setMesh(tableSurfaceMesh);

		// Set the Objects Transform
		Transform* tableSurfaceTransform = new Transform(TABLE_SURFACE);
		tableSurfaceTransform->setPosition(Vector(0.0f,-7.5f, 0.0f, 1.0f));
		tableSurfaceTransform->setScale(Vector(50.0f, 0.5f, 50.0f, 1.0f));

		tableSurface->setTransform(tableSurfaceTransform);

		// Set the Objects Material
		Material* tableSurfaceMaterial = new Material(TABLE_SURFACE, "cube.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		tableSurface->setMaterial(tableSurfaceMaterial);

		// Initialize the Object
		tableSurface->createMesh();
		tableSurface->setID(sceneManager->getObjectID());

	// Add the Object to the Scene Manager
	sceneManager->addObject(tableSurface);
	// Add the Object to the Map (CUDA Loading)
	objectMap[tableSurface->getID()] = tableSurface;

	// Blinn-Phong Sphere 0
	Object* sphere0Object = new Object(SPHERE_0);

		// Set the Objects Mesh
		Mesh* sphere0Mesh = new Mesh(SPHERE_0, "sphere.obj");
		sphere0Object->setMesh(sphere0Mesh);

		// Set the Objects Transform
		Transform* sphere0Transform = new Transform(SPHERE_0);
		sphere0Transform->setPosition(Vector(5.0f, 0.5f,0.0f,1.0f));
		sphere0Transform->setRotation(Vector(0.0f, 0.0f,0.0f,1.0f));
		sphere0Transform->setScale(Vector(2.5f,2.5f,2.5f,1.0f));

		sphere0Object->setTransform(sphere0Transform);

		// Set the Objects Material
		Material* sphere0Material = new Material(SPHERE_0, "teapot/GoldTeapot.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphere0Object->setMaterial(sphere0Material);

		// Initialize the Object
		sphere0Object->createMesh();
		sphere0Object->setID(sceneManager->getObjectID());

	// Add the Object to the Scene Manager
	sceneManager->addObject(sphere0Object);
	// Add the Object to the Map (CUDA Loading)
	objectMap[sphere0Object->getID()] = sphere0Object;

	// Blinn-Phong Sphere 1
	Object* sphere1Object = new Object(SPHERE_1);

		// Set the Objects Mesh
		Mesh* sphere1Mesh = new Mesh(SPHERE_1, "sphere.obj");
		sphere1Object->setMesh(sphere1Mesh);

		// Set the Objects Transform
		Transform* sphere1Transform = new Transform(SPHERE_1);
		sphere1Transform->setPosition(Vector(-5.0f,0.5f,5.0f,1.0f));
		sphere1Transform->setRotation(Vector(0.0f,90.0f,0.0f,1.0f));
		sphere1Transform->setScale(Vector(2.5f,2.5f,2.5f,1.0f));

		sphere1Object->setTransform(sphere1Transform);

		// Set the Objects Material
		Material* sphere1Material = new Material(SPHERE_1, "teapot/GoldTeapot.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphere1Object->setMaterial(sphere1Material);

		// Initialize the Object
		sphere1Object->createMesh();
		sphere1Object->setID(sceneManager->getObjectID());

	// Add the Object to the Scene Manager
	sceneManager->addObject(sphere1Object);
	// Add the Object to the Map (CUDA Loading)
	objectMap[sphere1Object->getID()] = sphere1Object;

	// Blinn-Phong Sphere 2
	Object* sphere2Object = new Object(SPHERE_2);

		// Set the Objects Mesh
		Mesh* sphere2Mesh = new Mesh(SPHERE_2, "sphere.obj");
		sphere2Object->setMesh(sphere2Mesh);

		// Set the Objects Transform
		Transform* sphere2Transform = new Transform(SPHERE_2);
		sphere2Transform->setPosition(Vector(-5.0f,0.5f,-5.0f,1.0f));
		sphere2Transform->setRotation(Vector(0.0f,0.0f,0.0f,1.0f));
		sphere2Transform->setScale(Vector(2.5f,2.5f,2.5f,1.0f));

		sphere2Object->setTransform(sphere2Transform);

		// Set the Objects Material
		Material* sphere2Material = new Material(SPHERE_2, "teapot/GoldTeapot.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphere2Object->setMaterial(sphere2Material);

		// Initialize the Object
		sphere2Object->createMesh();
		sphere2Object->setID(sceneManager->getObjectID());

	// Add the Object to the Scene Manager
	sceneManager->addObject(sphere2Object);
	// Add the Object to the Map (CUDA Loading)
	objectMap[sphere2Object->getID()] = sphere2Object;

	cout << endl;

	// Destroy the Readers
	OBJ_Reader::destroyInstance();
	XML_Reader::destroyInstance();

	// Create the Scene Graph Nodes
	SceneNode* tableSurfaceNode = new SceneNode(TABLE_SURFACE);
	tableSurfaceNode->setObject(tableSurface);

	SceneNode* sphere0ObjectNode = new SceneNode(SPHERE_0);
	sphere0ObjectNode->setObject(sphere0Object);
	SceneNode* sphere1ObjectNode = new SceneNode(SPHERE_1);
	sphere1ObjectNode->setObject(sphere1Object);
	SceneNode* sphere2ObjectNode = new SceneNode(SPHERE_2);
	sphere2ObjectNode->setObject(sphere2Object);

	// Add the Root Nodes to the Scene
	sceneManager->addSceneNode(tableSurfaceNode);
	sceneManager->addSceneNode(sphere0ObjectNode);
	sceneManager->addSceneNode(sphere1ObjectNode);
	sceneManager->addSceneNode(sphere2ObjectNode);

	// Init the SceneManager
	sceneManager->init();

	// Setup GLUT Callbacks 
	initializeCallbacks();

	// Stores the Triangles Information in the form of Arrays
	vector<float4> trianglePositionList;
	vector<float4> triangleNormalList;
	vector<float2> triangleTextureCoordinateList;

	// References which Material each Triangle is using
	vector<int1> triangleObjectIDList;
	vector<int1> triangleMaterialIDList;

	// Stores the Materials Information in the form of Arrays
	vector<float4> materialDiffusePropertyList;
	vector<float4> materialSpecularPropertyList;

	for(map<int,Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		// Used to store the Objects vertex data
		map<int, Vertex*> vertexMap = object->getMesh()->getVertexMap();

		for(map<int, Vertex*>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++) {

			// Get the vertex from the mesh 
			Vertex* vertex = vertexIterator->second;

			// Position
			Vector originalPosition = vertex->getPosition();
			float4 position = { originalPosition[VX], originalPosition[VY], originalPosition[VZ], 1.0f};
			trianglePositionList.push_back(position);

			// Normal
			Vector originalNormal = vertex->getNormal();
			float4 normal = { originalNormal[VX], originalNormal[VY], originalNormal[VZ], 0.0f};
			triangleNormalList.push_back(normal);

			// Texture Coordinates
			Vector originalTextureCoordinates = vertex->getTextureCoordinates();
			float2 textureCoordinates = { originalTextureCoordinates[VX], originalTextureCoordinates[VY] };
			triangleTextureCoordinateList.push_back(textureCoordinates);

			// Object ID
			int1 objectID = { object->getID() };
			triangleObjectIDList.push_back(objectID);

			// Material ID
			int1 materialID = { materialTotal };
			triangleMaterialIDList.push_back(materialID);

			triangleTotal++;
		}

		// Get the Material from the mesh 
		Material* material = object->getMaterial();

		// Material: Same as the original values
		Vector originalDiffuseProperty = material->getDiffuse();
		Vector originalSpecularProperty = material->getSpecular();
		float originalSpecularConstant = material->getSpecularConstant();

		float4 diffuseProperty = { originalDiffuseProperty[VX], originalDiffuseProperty[VY], originalDiffuseProperty[VZ], 1.0f };
		float4 specularProperty = { originalSpecularProperty[VX], originalSpecularProperty[VY], originalSpecularProperty[VZ], originalSpecularConstant };

		materialDiffusePropertyList.push_back(diffuseProperty);
		materialSpecularPropertyList.push_back(specularProperty);

		materialTotal++;
	}

	// Total number of Triangles should be the number of loaded vertices divided by 3
	triangleTotal = trianglePositionList.size() / 3;

	cout << "[Initialization] Total number of triangles:" << triangleTotal << endl;

	// Each triangle contains Position, Normal, Texture UV and Material Properties for 3 vertices
	size_t trianglePositionListSize = trianglePositionList.size() * sizeof(float4);
	cout << "[Initialization] Triangle Positions Storage Size: " << trianglePositionListSize << " (" << trianglePositionList.size() << " float4s)" << endl;
	size_t triangleNormalListSize = triangleNormalList.size() * sizeof(float4);
	cout << "[Initialization] Triangle Normals Storage Size: " << triangleNormalListSize << " (" << triangleNormalList.size() << " float4s)" << endl;
	size_t triangleTextureCoordinateListSize = triangleTextureCoordinateList.size() * sizeof(float2);
	cout << "[Initialization] Triangle Texture Coordinates Storage Size: " << triangleTextureCoordinateListSize << " (" << triangleTextureCoordinateList.size() << " float2s)" << endl;
	size_t triangleObjectIDListSize = triangleObjectIDList.size() * sizeof(int1);
	cout << "[Initialization] Triangle Object IDs Storage Size: " << triangleObjectIDListSize << " (" << triangleObjectIDList.size() << " int1s)" << endl;
	size_t triangleMaterialIDListSize = triangleMaterialIDList.size() * sizeof(int1);
	cout << "[Initialization] Triangle Material IDs Storage Size: " << triangleMaterialIDListSize << " (" << triangleMaterialIDList.size() << " int1s)" << endl;

	// Allocate the required CUDA Memory for the Triangles
	if(triangleTotal > 0) {

		// Load the Triangle Positions
		Utility::checkCUDAError("cudaMalloc()",	cudaMalloc((void **)&cudaTrianglePositionsDP, trianglePositionListSize));
		Utility::checkCUDAError("cudaMemcpy()",	cudaMemcpy(cudaTrianglePositionsDP, &trianglePositionList[0], trianglePositionListSize, cudaMemcpyHostToDevice));

		bindTrianglePositions(cudaTrianglePositionsDP, triangleTotal);

		// Load the Triangle Normals
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleNormalsDP, triangleNormalListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleNormalsDP, &triangleNormalList[0], triangleNormalListSize, cudaMemcpyHostToDevice));

		bindTriangleNormals(cudaTriangleNormalsDP, triangleTotal);

		// Load the Triangle Texture Coordinates
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleTextureCoordinatesDP, triangleTextureCoordinateListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleTextureCoordinatesDP, &triangleTextureCoordinateList[0], triangleTextureCoordinateListSize, cudaMemcpyHostToDevice));

		bindTriangleTextureCoordinates(cudaTriangleTextureCoordinatesDP, triangleTotal);

		// Load the Triangle Object IDs
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleObjectIDsDP, triangleObjectIDListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleObjectIDsDP, &triangleObjectIDList[0], triangleObjectIDListSize, cudaMemcpyHostToDevice));

		bindTriangleObjectIDs(cudaTriangleObjectIDsDP, triangleTotal);

		// Load the Triangle Material IDs
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaTriangleMaterialIDsDP, triangleMaterialIDListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaTriangleMaterialIDsDP, &triangleMaterialIDList[0], triangleMaterialIDListSize, cudaMemcpyHostToDevice));

		bindTriangleMaterialIDs(cudaTriangleMaterialIDsDP, triangleTotal);

		// Triangle Positions and Normals Memory Allocation
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaUpdatedTrianglePositionsDP, triangleTotal * sizeof(float4) * 3));
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaUpdatedTriangleNormalsDP, triangleTotal * sizeof(float4) * 3));

		// Model and Normal Matrices Memory Allocation
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaUpdatedModelMatricesDP, objectMap.size() * sizeof(float) * 16));
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaUpdatedNormalMatricesDP, objectMap.size() * sizeof(float) * 16));
	}

	cout << endl;

	// Total number of Materials
	cout << "[Initialization] Total number of Materials:" << materialTotal << endl;

	// Each Material contains Diffuse and Specular Properties
	size_t materialDiffusePropertyListSize = materialDiffusePropertyList.size() * sizeof(float4);
	cout << "[Initialization] Material Diffuse Properties Storage Size: " << materialDiffusePropertyListSize << " (" << materialDiffusePropertyList.size() << " float4s)" << endl;
	size_t materialSpecularPropertyListSize = materialSpecularPropertyList.size() * sizeof(float4);
	cout << "[Initialization] Material Specular Properties Storage Size: " << materialSpecularPropertyListSize << " (" << materialSpecularPropertyList.size() << " float4s)" << endl;

	// Allocate the required CUDA Memory for the Materials
	if(materialTotal > 0) {
	
		// Load the Triangle Diffuse Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaMaterialDiffusePropertiesDP, materialDiffusePropertyListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaMaterialDiffusePropertiesDP, &materialDiffusePropertyList[0], materialDiffusePropertyListSize, cudaMemcpyHostToDevice));

		bindMaterialDiffuseProperties(cudaMaterialDiffusePropertiesDP, materialTotal);

		// Load the Triangle Specular Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaMaterialSpecularPropertiesDP, materialSpecularPropertyListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaMaterialSpecularPropertiesDP, &materialSpecularPropertyList[0], materialSpecularPropertyListSize, cudaMemcpyHostToDevice));

		bindMaterialSpecularProperties(cudaMaterialSpecularPropertiesDP, materialTotal);
	}

	cout << endl;
}

int main(int argc, char* argv[]) {

	/* Init the Animation */
	init(argc, argv);

	/* Enable User Interaction */
	KeyboardHandler::getInstance()->enableKeyboard();
	MouseHandler::getInstance()->enableMouse();

	/* Start the Clock */
	startTime = glutGet(GLUT_ELAPSED_TIME);

	/* Start the Animation! */
	glutMainLoop();

	#ifdef MEMORY_LEAK
		_CrtDumpMemoryLeaks();
	#endif

	exit(EXIT_SUCCESS);
}