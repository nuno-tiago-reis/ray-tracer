// OpenGL definitions
#include <GL/glew.h>
#include <GL/glut.h>

// CUDA definitions
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "vector_types.h"
#include "vector_functions.h"

// Custom
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

// Frame Cap
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

// Total number of Rays - Updated per Frame
int rayTotal;
// Total number of Chunks - Updated per Frame
int chunkTotal;
// Total number of Hierarchy Hits - Updated per Frame
int hierarchyHitTotal;

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

// CUDA DevicePointers to the Update Triangles
float4* cudaUpdatedTrianglePositionsDP = NULL;
float4* cudaUpdatedTriangleNormalsDP = NULL;

// CUDA DevicePointers to the Updated Matrices
float* cudaUpdatedModelMatricesDP = NULL;
float* cudaUpdatedNormalMatricesDP = NULL;

// CUDA DevicePointer to the Hierarchy Array
float4* cudaHierarchyArrayDP = NULL;
// CUDA DevicePointer to the Hierarchy Hits Arrays
int2* cudaPrimaryHierarchyHitsArrayDP = NULL;
int2* cudaSecondaryHierarchyHitsArrayDP = NULL;

// CUDA DevicePointers to the Unsorted Rays
float3* cudaRayArrayDP = NULL;

// CUDA DevicePointers to the Chunk Base and Size Arrays
int* cudaChunkBasesArrayDP = NULL;
int* cudaChunkSizesArrayDP = NULL;

// CUDA DevicePointers to the Trimmed and Sorted Ray Index Arrays
int* cudaPrimaryRayIndexValuesArrayDP = NULL;
int* cudaPrimaryRayIndexKeysArrayDP = NULL;
int* cudaSecondaryRayIndexValuesArrayDP = NULL;
int* cudaSecondaryRayIndexKeysArrayDP = NULL;

// CUDA DevicePointers to the Unsorted and Sorted Ray Index Arrays
int* cudaPrimaryChunkKeysArrayDP = NULL;
int* cudaPrimaryChunkValuesArrayDP = NULL;
int* cudaSecondaryChunkKeysArrayDP = NULL;
int* cudaSecondaryChunkValuesArrayDP = NULL;

// CUDA DevicePointers to the Sorting Auxiliary Arrays
int* cudaHeadFlagsArrayDP = NULL;
int* cudaScanArrayDP = NULL;

// [CUDA-OpenGL Interop] 
extern "C" {

	// Implementation of 'TriangleUpdateWrapper' is in the "RayTracer.cu" file
	void TriangleUpdateWrapper(	
							// Input Array containing the updated Model Matrices.
							float* modelMatricesArray,
							// Input Array containing the updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Triangle Total.
							int triangleTotal,
							// Output Array containing the updated Triangle Positions.
							float4* trianglePositionsArray,
							// Output Array containing the updated Triangle Normals.
							float4* triangleNormalsArray);
	
	// Implementation of 'MemoryPreparationWrapper' is in the "RayTracer.cu" file
	void MemoryPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, int windowHeight,
							// Auxiliary Variable containing the Triangle Total.
							int triangleTotal,
							// Auxiliary Array containing the Head Flags.
							int* headFlagsArray, 
							// Auxiliary Array containing the Scan Output.
							int* scanArray,
							// Auxiliary Arrays containing the Ray Chunks.
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray,
							// Auxiliary Arrays containing the Sorted Ray Chunks.
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray);

	// Implementation of 'ScreenPreparationWrapper' is in the "RayTracer.cu" file
	void ScreenPreparationWrapper(
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);

	// Implementation of 'ShadowRayCreationWrapper' is in the "RayTracer.cu" file
	void ShadowRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, int windowHeight,
							// Auxiliary Variables containing the Light Index.
							int lightIndex,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray);
	
	// Implementation of 'ReflectionRayCreationWrapper' is in the "RayTracer.cu" file
	void ReflectionRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, 
							int windowHeight,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray);

	// Implementation of 'RayTrimmingWrapper' is in the "RayTracer.cu" file
	void RayTrimmingWrapper(	
							// Input Arrays containing the Untrimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, int windowHeight,
							// Auxiliary Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							int* scanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Output Variable containing the Number of Rays.
							int* rayTotal);

	// Implementation of 'RayCompressionWrapper' is in the "RayTracer.cu" file
	void RayCompressionWrapper(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Number of Rays.
							int rayTotal,
							// Auxiliary Array containing the Ray Chunk Head Flags.
							int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							int* scanArray, 
							// Output Arrays containing the Ray Chunk Bases and Sizes.
							int* chunkBasesArray,
							int* chunkSizesArray,
							// Output Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray,
							// Output Variable containing the Number of Chunks.
							int* chunkTotal);
	
	// Implementation of 'RaySortingWrapper' is in the "RayTracer.cu" file
	void RaySortingWrapper(	
							// Input Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray,
							// Auxiliary Variable containing the Number of Chunks.
							int chunkTotal,
							// Output Arrays containing the Sorted Ray Chunks [Keys = Hashes, Values = Indices]
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray);

	// Implementation of 'RayDecompressionWrapper' is in the "RayTracer.cu" file
	void RayDecompressionWrapper(	
							// Input Array containing the Ray Chunk Bases.
							int* chunkBasesArray,
							// Input Array containing the Ray Chunk Sizes.
							int* chunkSizesArray,
							// Input Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray,
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Input Array containing the Sorted Ray Arrays Skeleton.
							int* skeletonArray,
							// Input Array containing the Inclusive Scan Output.
							int* scanArray, 
							// Auxiliary Variable containing the Chunk Total.
							int chunkTotal, 
							// Output Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray);

	// Implementation of HierarchyCreationWrapper is in the "RayTracer.cu" file
	void HierarchyCreationWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray,
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							int rayTotal,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray);

	// Implementation of HierarchyTraversalWrapper is in the "RayTracer.cu" file
	void HierarchyTraversalWrapper(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Ray Total.
							int rayTotal,
							// Auxiliary Variable containing the Triangle Total.
							int triangleTotal,
							// Auxiliary Array containing the Hierarchy Hits Flags.
							int* headFlagsArray,
							// Auxiliary Array containing the Inclusive Scan Output.
							int* scanArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							int2* trimmedHierarchyHitsArray,
							// Output Variable containing the Number of Hits.
							int* hierarchyHitTotal);
	
	// Implementation of LocalIntersectionWrapper is in the "RayTracer.cu" file
	void ShadowRayIntersectionWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Input Array containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Number of Hits.
							int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, 
							int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);
	
	// Implementation of ReflectionRayIntersectionWrapper is in the "RayTracer.cu" file
	void ReflectionRayIntersectionWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Input Array containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Input Array containing the Updated Triangle Normals.
							float4* triangleNormalsArray,
							// Auxiliary Variable containing the Number of Hits.
							int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, 
							int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Auxiliary Array containing the Intersection Times.
							int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);

	// Implementation of bindRenderTextureArray is in the "RayTracer.cu" file
	void bindDiffuseTextureArray(cudaArray *diffuseTextureArray);
	// Implementation of bindRayOriginTextureArray is in the "RayTracer.cu" file
	void bindSpecularTextureArray(cudaArray *specularTextureArray);
	// Implementation of bindRayReflectionTextureArray is in the "RayTracer.cu" file
	void bindFragmentPositionArray(cudaArray *fragmentPositionArray);
	// Implementation of bindRayRefractionTextureArray is in the "RayTracer.cu" file
	void bindFragmentNormalArray(cudaArray *fragmentNormalArray);

	// Implementation of 'bindTrianglePositions' is in the "RayTracer.cu" file
	void bindTrianglePositions(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of 'bindTriangleNormals' is in the "RayTracer.cu" file
	void bindTriangleNormals(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of 'bindTriangleTextureCoordinates' is in the "RayTracer.cu" file
	void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of 'bindTriangleObjectIDs' is in the "RayTracer.cu" file
	void bindTriangleObjectIDs(float *cudaDevicePointer, unsigned int triangleTotal);
	// Implementation of 'bindTriangleMaterialIDs' is in the "RayTracer.cu" file
	void bindTriangleMaterialIDs(float *cudaDevicePointer, unsigned int triangleTotal);

	// Implementation of 'bindMaterialDiffuseProperties' is in the "RayTracer.cu" file
	void bindMaterialDiffuseProperties(float *cudaDevicePointer, unsigned int materialTotal);
	// Implementation of 'bindMaterialSpecularProperties' is in the "RayTracer.cu" file
	void bindMaterialSpecularProperties(float *cudaDevicePointer, unsigned int materialTotal);

	// Implementation of 'bindLightPositions' is in the "RayTracer.cu" file
	void bindLightPositions(float *cudaDevicePointer, unsigned int lightTotal);
	// Implementation of 'bindLightColors' is in the "RayTracer.cu" file
	void bindLightColors(float *cudaDevicePointer, unsigned int lightTotal);
	// Implementation of 'bindLightIntensities' is in the "RayTracer.cu" file
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

// [Ray-Tracing]

// [Ray-Tracing] Creates a Batch of Shadow Rays.
void createShadowRays(bool rasterizer, float3 cameraPosition) {

	if(rasterizer == true) {
	
		// Create the Rays and Index them [DONE]
		ShadowRayCreationWrapper(
			windowWidth, windowHeight, 
			0, 
			cudaRayArrayDP, 
			cudaHeadFlagsArrayDP, 
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP);
	
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif
	}
}

// [Ray-Tracing] Colors a processed Batch of Shadow Rays.
void colorShadowRays(bool rasterizer, float3 cameraPosition, unsigned int* pixelBufferObject) {
	
	if(rasterizer == true) {

		// Traverse the Hierarchy Hits testing each Ray with the corresponding Triangle
		ShadowRayIntersectionWrapper(
			cudaRayArrayDP,
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
			cudaSecondaryHierarchyHitsArrayDP,
			cudaUpdatedTrianglePositionsDP,
			hierarchyHitTotal,
			rayTotal,
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			cudaHeadFlagsArrayDP,
			pixelBufferObject);
		
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif
	}
}

// [Ray-Tracing] Creates a Batch of Reflection Rays.
void createReflectionRays(bool rasterizer, float3 cameraPosition) {
	
	if(rasterizer == true) {

		// Create the Rays and Index them [DONE]
		ReflectionRayCreationWrapper(
			windowWidth, windowHeight,
			cameraPosition,
			cudaRayArrayDP, 
			cudaHeadFlagsArrayDP, 
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP);
	
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif
	}
}

// [Ray-Tracing] Colors a processed Batch of Reflection Rays.
void colorReflectionRays(bool rasterizer, float3 cameraPosition, unsigned int* pixelBufferObject) {
	
	if(rasterizer == true) {

		// Traverse the Hierarchy Hits testing each Ray with the corresponding Triangle
		ReflectionRayIntersectionWrapper(
			cudaRayArrayDP,
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
			cudaSecondaryHierarchyHitsArrayDP,
			cudaUpdatedTrianglePositionsDP,
			cudaUpdatedTriangleNormalsDP,
			hierarchyHitTotal,
			rayTotal,
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			cudaHeadFlagsArrayDP,
			pixelBufferObject);
		
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif
	}
}

// [Ray-Tracing] Processes a Batch of Rays previously created.
void castRays() {

	// Trim the Ray Indices [DONE]
	RayTrimmingWrapper(
		cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP, 
		windowWidth, windowHeight, 
		cudaHeadFlagsArrayDP, 
		cudaScanArrayDP, 
		cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP,
		&rayTotal);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	// Compress the Unsorted Ray Indices into Chunks [DONE]
	RayCompressionWrapper(
		cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP, 
		rayTotal,
		cudaHeadFlagsArrayDP, 
		cudaScanArrayDP, 
		cudaChunkBasesArrayDP, cudaChunkSizesArrayDP, 
		cudaPrimaryChunkKeysArrayDP, cudaPrimaryChunkValuesArrayDP,
		&chunkTotal);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	// Sort the Chunks [DONE]
	RaySortingWrapper(
		cudaPrimaryChunkKeysArrayDP, cudaPrimaryChunkValuesArrayDP, 
		chunkTotal,
		cudaSecondaryChunkKeysArrayDP, cudaSecondaryChunkValuesArrayDP);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	// Decompress the Sorted Chunks into the Sorted Ray Indices [DONE]
	RayDecompressionWrapper(
		cudaChunkBasesArrayDP, cudaChunkSizesArrayDP,
		cudaSecondaryChunkKeysArrayDP, cudaSecondaryChunkValuesArrayDP, 
		cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP,
		cudaHeadFlagsArrayDP,
		cudaScanArrayDP, 
		chunkTotal,
		cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	HierarchyCreationWrapper(
		cudaRayArrayDP,
		cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
		rayTotal,
		cudaHierarchyArrayDP);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	// Traverse the Hierarchy testing each Node against the Triangles Bounding Spheres [DONE]
	HierarchyTraversalWrapper(
		cudaHierarchyArrayDP,
		cudaUpdatedTrianglePositionsDP,
		rayTotal,
		triangleTotal,
		cudaHeadFlagsArrayDP,
		cudaScanArrayDP,
		cudaPrimaryHierarchyHitsArrayDP,
		cudaSecondaryHierarchyHitsArrayDP,
		&hierarchyHitTotal);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("LocalIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
	#endif
}

// [Scene] Updates the Scene
void update(int value) {

	/* Update the Timer Variables */
	elapsedTime = (GLfloat)(glutGet(GLUT_ELAPSED_TIME) - startTime)/1000;
	startTime = glutGet(GLUT_ELAPSED_TIME);
	time += elapsedTime;

	/* Update the Scene */
	sceneManager->update(elapsedTime);

	glutTimerFunc(FPS_60, update, 0);

	glutPostRedisplay();

	//cout << "[Callback] Update Successfull" << endl;
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
	unsigned int* pixelBufferObject = pixelBuffer->getDevicePointer();

	// Get the Camera Positions 
	Vector cameraPosition = sceneManager->getActiveCamera()->getEye();

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
	}

	// Copy the Matrices to CUDA	
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedModelMatricesDP, &modelMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedNormalMatricesDP, &normalMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));

	// Preparation Kernels - [Start]

		// Update the Triangle Positions and Normals
		TriangleUpdateWrapper(cudaUpdatedModelMatricesDP, cudaUpdatedNormalMatricesDP, triangleTotal, cudaUpdatedTrianglePositionsDP, cudaUpdatedTriangleNormalsDP);
		
		Utility::checkCUDAError("TriangleUpdateWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("TriangleUpdateWrapper::cudaGetLastError()", cudaGetLastError());

		// Prepare the Auxiliary memory necessary.
		MemoryPreparationWrapper(
			windowWidth, windowHeight,
			triangleTotal, 
			cudaHeadFlagsArrayDP, cudaScanArrayDP,
			cudaPrimaryChunkKeysArrayDP, cudaPrimaryChunkValuesArrayDP,
			cudaSecondaryChunkKeysArrayDP, cudaSecondaryChunkValuesArrayDP);

		// Prepare the Screen
		ScreenPreparationWrapper(
			cudaRayArrayDP,
			windowWidth, windowHeight,
			lightTotal, 
			make_float3(cameraPosition[VX], cameraPosition[VY], cameraPosition[VZ]),
			pixelBufferObject);

		Utility::checkCUDAError("PreparationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("PreparationWrapper::cudaGetLastError()", cudaGetLastError());

	// Preparation Kernels - [End]

	// Ray-Tracing Kernels - [Start]
	
	for(int i=0; i<DEPTH; i++) {

		float3 cameraEye = make_float3(cameraPosition[VX], cameraPosition[VY], cameraPosition[VZ]);

		// Cast the Shadow Ray Batch
		if(i == 0) {

			// Calculate the Shadow Rays based on the Rasterizer Input for the first Iteration.
			createShadowRays(true, cameraEye);

			// Cast the Generic Ray-Tracing Algorithm.
			castRays();

			// Calculate the Color based on the Rasterizer Input for the first Iteration.
			colorShadowRays(true, cameraEye, pixelBufferObject);
		}
		
		// Cast the Reflection and Refraction Ray Batches
		if(i == 1) {
			
			// Calculate the Reflection Rays based on the Rasterizer Input for the first Iteration.
			createReflectionRays(true, cameraEye);

			// Cast the Generic Ray-Tracing Algorithm.
			castRays();
			
			// Calculate the Color.
			colorReflectionRays(true, cameraEye, pixelBufferObject);
			
			/*// Calculate the Refraction Rays based on the Rasterizer Input for the first Iteration.
			createRefractionRays(true, make_float3(cameraPosition[VX], cameraPosition[VY], cameraPosition[VZ]));

			// Cast the Generic Ray-Tracing Algorithm.
			castRays();

			colorRefractionRays(pixelBufferObject);*/
		}
	}

	// Ray-Tracing Kernels - [End]

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

	//cudaDeviceReset();
	//exit(0);

	//cout << "[Callback] Display Successfull" << endl;
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

	// Update the Array Size 
	int rayMaximum = windowWidth * windowHeight;

	// Update the CUDA Hierarchy Array Size
	int hierarchyMaximum = 0;
	int hierarchyNodeMaximum[HIERARCHY_MAXIMUM_DEPTH];

	// Each Node Maximum is calculated form the Ray Maximum
	hierarchyNodeMaximum[0] = rayMaximum / HIERARCHY_SUBDIVISION + (rayMaximum % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
	// Each Node Maximum adds to the Hierarchy Maximum
	hierarchyMaximum = hierarchyNodeMaximum[0]; 

	for(int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

		// Each Node Maximum is calculated form the Ray Maximum
		hierarchyNodeMaximum[i] = hierarchyNodeMaximum[i-1] / HIERARCHY_SUBDIVISION + (hierarchyNodeMaximum[i-1] % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
		// Each Node Maximum adds to the Hierarchy Maximum
		hierarchyMaximum += hierarchyNodeMaximum[i]; 
	}

	size_t allocated = 0;
	
	// Update the CUDA Hierarchy Array
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaHierarchyArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryHierarchyHitsArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryHierarchyHitsArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaHierarchyArrayDP, hierarchyMaximum * sizeof(float4) * 2));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryHierarchyHitsArrayDP, hierarchyNodeMaximum[0] * triangleTotal * sizeof(int2)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryHierarchyHitsArrayDP, hierarchyNodeMaximum[0] * triangleTotal * sizeof(int2)));

	allocated += hierarchyMaximum * sizeof(float4) * 2;
	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(int2);
	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(int2);

	// Update the CUDA Ray Array
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaRayArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaRayArrayDP, rayMaximum * sizeof(float3) * 2));

	allocated += rayMaximum * sizeof(float3) * 2;

	// Update the CUDA Chunks Base and Size Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaChunkBasesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaChunkSizesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaChunkBasesArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaChunkSizesArrayDP, rayMaximum * sizeof(int)));

	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);

	// Update the CUDA Ray and Sorted Ray Index Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryRayIndexKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryRayIndexValuesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryRayIndexKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryRayIndexValuesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryRayIndexKeysArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryRayIndexValuesArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryRayIndexKeysArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryRayIndexValuesArrayDP, rayMaximum * sizeof(int)));

	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);
	
	// Update the CUDA Chunk and Sorted Chunk Index Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryChunkKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryChunkValuesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryChunkKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryChunkValuesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryChunkKeysArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryChunkValuesArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryChunkKeysArrayDP, rayMaximum * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryChunkValuesArrayDP, rayMaximum * sizeof(int)));

	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);
	allocated += rayMaximum * sizeof(int);

	// Update the CUDA Head Flags and Scan Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaHeadFlagsArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaScanArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaHeadFlagsArrayDP, hierarchyNodeMaximum[0] * triangleTotal * sizeof(int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaScanArrayDP, hierarchyNodeMaximum[0] * triangleTotal  * sizeof(int)));

	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(int);
	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(int);
	
	size_t free, total;
	Utility::checkCUDAError("cudaGetMemInfo()", cudaMemGetInfo(&free, &total));

	cout << endl;

	printf("[Callback] Total Memory:\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		total, total / 1024.0f, total / 1024.0f / 1024.0f, total / 1024.0f / 1024.0f / 1024.0f); 
	printf("[Callback] Free Memory:\t\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		free, total / 1024.0f, free / 1024.0f / 1024.0f, free / 1024.0f / 1024.0f / 1024.0f);
	printf("[Callback] Allocated Memory:\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		allocated, allocated / 1024.0f, allocated / 1024.0f / 1024.0f, allocated / 1024.0f / 1024.0f / 1024.0f);

	cout << endl;

	cout << "[Callback] Reshape Successfull" << endl << endl;
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

	cout << "[Callback] Cleanup Successfull" << endl << endl;
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

	/*cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	cout << "Maximum Grid Size = " << properties.maxGridSize << endl;
	cout << "Maximum Threads per Block = " << properties.maxThreadsPerBlock << endl;*/

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
	positionalLight1->setDiffuseIntensity(1.5f);
	positionalLight1->setSpecularIntensity(1.5f);

	positionalLight1->setConstantAttenuation(0.0025f);
	positionalLight1->setLinearAttenuation(0.00075f);
	positionalLight1->setExponentialAttenuation(0.000075f);
	
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

	/*// Cube
	Object* cubeObject = new Object("Cube");

		// Set the Objects Mesh
		Mesh* cubeMesh = new Mesh(TABLE_SURFACE, "cube.obj");

		cubeObject->setMesh(cubeMesh);

		// Set the Objects Transform
		Transform* cubeTransform = new Transform("Cube");
		cubeTransform->setPosition(Vector(0.0f, 2.5f, 0.0f, 1.0f));
		cubeTransform->setScale(Vector(7.5f, 2.5f, 7.5f, 1.0f));

		cubeObject->setTransform(cubeTransform);

		// Set the Objects Material
		Material* cubeMaterial = new Material("Cube", "cube.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		
		cubeObject->setMaterial(cubeMaterial);

		// Initialize the Object
		cubeObject->createMesh();
		cubeObject->setID(sceneManager->getObjectID());

	// Add the Object to the Scene Manager
	sceneManager->addObject(cubeObject);
	// Add the Object to the Map (CUDA Loading)
	objectMap[cubeObject->getID()] = cubeObject;*/

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

	/*SceneNode* cubeNode = new SceneNode("Cube");
	cubeNode->setObject(cubeObject);*/

	// Add the Root Nodes to the Scene
	sceneManager->addSceneNode(tableSurfaceNode);
	sceneManager->addSceneNode(sphere0ObjectNode);
	sceneManager->addSceneNode(sphere1ObjectNode);
	sceneManager->addSceneNode(sphere2ObjectNode);
	//sceneManager->addSceneNode(cubeNode);

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
}

int main(int argc, char* argv[]) {

	freopen("output.txt","w",stderr);
	freopen("output.txt","w",stdout);
	
	// Init the Animation
	init(argc, argv);

	// Enable User Interaction
	KeyboardHandler::getInstance()->enableKeyboard();
	MouseHandler::getInstance()->enableMouse();

	// Start the Clock
	startTime = glutGet(GLUT_ELAPSED_TIME);

	// Start the Animation!
	glutMainLoop();

	#ifdef MEMORY_LEAK
		_CrtDumpMemoryLeaks();
	#endif

	exit(EXIT_SUCCESS);
}