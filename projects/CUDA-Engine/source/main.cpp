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
#include "TestManager.h"
#include "XML_Reader.h"
#include "OBJ_Reader.h"

// Frame Cap
#define FPS_60	1000/60

#define CAPTION	"OpenGL-CUDA Engine 2015"

// Frame Count Global Variable
int frameCount = 0;

// Window Handling Global Variables
int windowHandle = 0;

unsigned int windowWidth = WIDTH;
unsigned int windowHeight = HEIGHT;

// Clock Handling Global Variables
GLint startTime = 0;
GLfloat currentTime = 0;
GLfloat elapsedTime = 0;

// Scene Manager
SceneManager* sceneManager = SceneManager::getInstance();

// Object Map
map<int, Object*> objectMap;
// Light Map
map<int, Light*> lightMap;

// Algorithm ID
int algorithmID = 0;

// Scene ID
int sceneID = 0;
// Scene Exitor
int sceneExitor = 0;

// Soft Shadows
bool softShadows = false;

// FrameBuffer Wrapper
FrameBuffer *frameBuffer;
// PixelBuffer Wrapper
PixelBuffer *pixelBuffer;
// Screens Textures Wrapper
ScreenTexture *screenTexture;

// Total number of Triangles - Used for the memory necessary to allocate
unsigned int triangleTotal = 0;
// Total number of Materials - Used for the memory necessary to allocate
unsigned int materialTotal = 0;
// Total number of Bounding Spheres - Used for the memory necessary to allocate
unsigned int boundingSphereTotal = 0;

// Total number of Lights - Used for the memory necessary to allocate
unsigned int lightTotal = 0;

// Total number of Rays - Updated per Frame
unsigned int rayTotal;
// Total number of Chunks - Updated per Frame
unsigned int chunkTotal;
// Total number of Hierarchy Hits - Updated per Frame
unsigned int hierarchyHitTotal;

// Total number of Memory Allocated for the Hierarchy Hits
unsigned int hierarchyHitMemoryTotal;

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

// CUDA DevicePointer to the uploaded Bounding Spheres
float* cudaBoundingSpheresDP = NULL;

// CUDA DevicePointers to the uploaded Lights
float *cudaLightPositionsDP = NULL;
float *cudaLightColorsDP = NULL;
float *cudaLightIntensitiesDP = NULL;

// CUDA DevicePointers to the Update Triangles
float4* cudaUpdatedTrianglePositionsDP = NULL;
float4* cudaUpdatedTriangleNormalsDP = NULL;

// CUDA DevicePointers to the Updated Bounding Spheres
float3* cudaUpdatedBoundingSpheresDP = NULL;

// CUDA DevicePointers to the Updated Matrices
float* cudaUpdatedModelMatricesDP = NULL;
float* cudaUpdatedNormalMatricesDP = NULL;

// CUDA DevicePointer to the Hierarchy Array
float4* cudaHierarchyArrayDP = NULL;
// CUDA DevicePointer to the Hierarchy Hits Arrays
unsigned int* cudaPrimaryHierarchyHitsArrayDP = NULL;
unsigned int* cudaSecondaryHierarchyHitsArrayDP = NULL;

// CUDA DevicePointers to the Unsorted Rays
float3* cudaRayArrayDP = NULL;

// CUDA DevicePointers to the Chunk Base and Size Arrays
unsigned int* cudaChunkBasesArrayDP = NULL;
unsigned int* cudaChunkSizesArrayDP = NULL;

// CUDA DevicePointers to the Trimmed and Sorted Ray Index Arrays
unsigned int* cudaPrimaryRayIndexValuesArrayDP = NULL;
unsigned int* cudaPrimaryRayIndexKeysArrayDP = NULL;
unsigned int* cudaSecondaryRayIndexValuesArrayDP = NULL;
unsigned int* cudaSecondaryRayIndexKeysArrayDP = NULL;

// CUDA DevicePointers to the Unsorted and Sorted Ray Index Arrays
unsigned int* cudaPrimaryChunkKeysArrayDP = NULL;
unsigned int* cudaPrimaryChunkValuesArrayDP = NULL;
unsigned int* cudaSecondaryChunkKeysArrayDP = NULL;
unsigned int* cudaSecondaryChunkValuesArrayDP = NULL;

// CUDA DevicePointers to the Sorting Auxiliary Arrays
unsigned int* cudaHeadFlagsArrayDP = NULL;
unsigned int* cudaScanArrayDP = NULL;

// [CUDA-OpenGL Interop] 
extern "C" {

	// Implementation of 'TriangleUpdateWrapper' is in the "RayTracer.cu" file
	void TriangleUpdateWrapper(	
							// Input Array containing the updated Model Matrices.
							float* modelMatricesArray,
							// Input Array containing the updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Output Array containing the updated Triangle Positions.
							float4* trianglePositionsArray,
							// Output Array containing the updated Triangle Normals.
							float4* triangleNormalsArray);
	
	// Implementation of 'BoundingSphereUpdateWrapper' is in the "RayTracer.cu" file
	void BoundingSphereUpdateWrapper(
							// Input Array containing the updated Translation Matrices.
							float* translationMatricesArray,
							// Input Array containing the updated Scale Matrices.
							float* scaleMatricesArray,
							// Auxiliary Variable containing the Bounding Sphere Total.
							const unsigned int boundingSphereTotal,
							// Output Array containing the updated Bounding Boxes.
							float3* boundingBoxArray);
	
	// Implementation of 'MemoryPreparationWrapper' is in the "RayTracer.cu" file
	void MemoryPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Hierarchy Hit Memory Size.
							const unsigned int hierarchyHitMemoryTotal,
							// Auxiliary Array containing the Head Flags.
							unsigned int* headFlagsArray, 
							// Auxiliary Array containing the Scan Output.
							unsigned int* scanArray,
							// Auxiliary Arrays containing the Ray Chunks.
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray,
							// Auxiliary Arrays containing the Sorted Ray Chunks.
							unsigned int* sortedChunkIndexKeysArray, 
							unsigned int* sortedChunkIndexValuesArray);

	// Implementation of 'ScreenPreparationWrapper' is in the "RayTracer.cu" file
	void ScreenPreparationWrapper(
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Input Array containing the updated Bounding Spheres.
							float3* boundingSphereArray,
							// Auxiliary Variable containing the Bounding Sphere Total.
							const unsigned int boundingSphereTotal,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraDirection,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraUp,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraRight,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);

	void ScreenCleaningWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);

	// Implementation of 'ShadowRayCreationWrapper' is in the "RayTracer.cu" file
	void ShadowRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Light Total.
							const unsigned int lightTotal,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray);
	
	// Implementation of 'ReflectionRayCreationWrapper' is in the "RayTracer.cu" file
	void ReflectionRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray);

	// Implementation of 'RayTrimmingWrapper' is in the "RayTracer.cu" file
	void RayTrimmingWrapper(	
							// Input Arrays containing the Untrimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Light Total.
							const unsigned int lightTotal,
							// Auxiliary Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Output Variable containing the Number of Rays.
							unsigned int* rayTotal);

	// Implementation of 'RayCompressionWrapper' is in the "RayTracer.cu" file
	void RayCompressionWrapper(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Number of Rays.
							const unsigned int rayTotal,
							// Auxiliary Array containing the Ray Chunk Head Flags.
							unsigned int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Output Arrays containing the Ray Chunk Bases and Sizes.
							unsigned int* chunkBasesArray,
							unsigned int* chunkSizesArray,
							// Output Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray,
							// Output Variable containing the Number of Chunks.
							unsigned int* chunkTotal);
	
	// Implementation of 'RaySortingWrapper' is in the "RayTracer.cu" file
	void RaySortingWrapper(	
							// Input Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray,
							// Auxiliary Variable containing the Number of Chunks.
							const unsigned int chunkTotal,
							// Output Arrays containing the Sorted Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* sortedChunkIndexKeysArray, 
							unsigned int* sortedChunkIndexValuesArray);

	// Implementation of 'RayDecompressionWrapper' is in the "RayTracer.cu" file
	void RayDecompressionWrapper(	
							// Input Array containing the Ray Chunk Bases.
							unsigned int* chunkBasesArray,
							// Input Array containing the Ray Chunk Sizes.
							unsigned int* chunkSizesArray,
							// Input Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							unsigned int* sortedChunkIndexKeysArray, 
							unsigned int* sortedChunkIndexValuesArray,
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Input Array containing the Sorted Ray Arrays Skeleton.
							unsigned int* skeletonArray,
							// Input Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Auxiliary Variable containing the Chunk Total.
							const unsigned int chunkTotal, 
							// Output Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray);

	// Implementation of 'HierarchyCreationWrapper' is in the "RayTracer.cu" file
	void HierarchyCreationWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray,
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Initial Sphere Radius.
							const float initialRadius,
							// Auxiliary Variable containing the Initial Cone Spread.
							const float initialSpread,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray);

	// Implementation of'HierarchyTraversalWarmUpWrapper' is in the "RayTracer.cu" file
	void HierarchyTraversalWarmUpWrapper(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the updated Bounding Spheres.
							float3* boundingSphereArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Bounding Box Total.
							const unsigned int boundingSphereTotal,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Array containing the Hierarchy Hits Flags.
							unsigned int* headFlagsArray,
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							unsigned int* trimmedHierarchyHitsArray,
							// Output Variable containing the Number of Hits.
							unsigned int* hierarchyHitTotal,
							// Output Variable containing the Hierarchy Hit Memory Size.
							unsigned int *hierarchyHitMemoryTotal);

	// Implementation of'HierarchyTraversalWrapper' is in the "RayTracer.cu" file
	void HierarchyTraversalWrapper(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Array containing the Hierarchy Hits Flags.
							unsigned int* headFlagsArray,
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							unsigned int* trimmedHierarchyHitsArray,
							// Output Variable containing the Number of Hits.
							unsigned int* hierarchyHitTotal,
							// Output Variable containing the Hierarchy Hit Memory Size.
							unsigned int *hierarchyHitMemoryTotal);
	
	// Implementation of'ShadowRayPreparationWrapper' is in the "RayTracer.cu" file
	void ShadowRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray);
	
	// Implementation of 'ShadowRayIntersectionWrapper' is in the "RayTracer.cu" file
	void ShadowRayIntersectionWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Input Array containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Number of Hits.
							const unsigned int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray);
	
	// Implementation of 'ShadowRayColoringWrapper'ShadowRayColoringWrapper is in the "RayTracer.cu" file
	void ShadowRayColoringWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);
	
	// Implementation of 'ReflectionRayPreparationWrapper'ShadowRayColoringWrapper is in the "RayTracer.cu" file
	void ReflectionRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* intersectionTimeArray);
	
	// Implementation of 'ReflectionRayIntersectionWrapper' is in the "RayTracer.cu" file
	void ReflectionRayIntersectionWrapper(
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Input Array containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Number of Hits.
							const unsigned int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliary Array containing the Intersection Times.
							unsigned int* intersectionTimeArray);
	
	// Implementation of 'ReflectionRayColoringWrapper' is in the "RayTracer.cu" file
	void ReflectionRayColoringWrapper(
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Input Array containing the Updated Triangle Normals.
							float4* triangleNormalsArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliary Variables indicating if its the last Iteration.
							const bool createRays,
							// Auxiliary Array containing the Intersection Times.
							unsigned int* intersectionTimeArray,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject);
	
	// Implementation of 'AntiAliasingWrapper' is in the "RayTracer.cu" file
	void AntiAliasingWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Primary Screen Buffer.
							unsigned int *primaryPixelBufferObject,
							// Output Array containing the Secondary Screen Buffer.
							unsigned int *secondaryPixelBufferObject);

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
	
	// Implementation of 'bindBoundingSpheres' is in the "RayTracer.cu" file
	void bindBoundingSpheres(float *cudaDevicePointer, unsigned int boundingSphereTotal);

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
bool createShadowRays(bool rasterizer, float3 cameraPosition) {

	if(rasterizer == true) {

		// If we're using CRSH
		if(algorithmID == 0) {

			// Create the Rays and Index them [DONE]
			ShadowRayCreationWrapper(
				windowWidth, windowHeight, 
				LIGHT_SOURCE_MAXIMUM,
				cudaRayArrayDP, 
				cudaHeadFlagsArrayDP, 
				cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP);
		}
		// If we're using RAH
		else {

			// Create the Rays and Index them [DONE]
			ShadowRayCreationWrapper(
				windowWidth, windowHeight, 
				LIGHT_SOURCE_MAXIMUM,
				cudaRayArrayDP, 
				cudaHeadFlagsArrayDP, 
				cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP);
		}

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ShadowRayCreationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ShadowRayCreationWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		return true;
	}

	return false;
}

// [Ray-Tracing] Colors a processed Batch of Shadow Rays.
bool colorShadowRays(bool rasterizer, float3 cameraPosition, unsigned int* pixelBufferObject) {

	if(rasterizer == true) {

		// Prepare the Shadow Flags Array
		ShadowRayPreparationWrapper(
			windowWidth,  windowHeight,
			LIGHT_SOURCE_MAXIMUM,
			cudaPrimaryChunkKeysArrayDP);

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		// Traverse the Ray Hierarchy once for every Batch
		for(unsigned int i=0; i<(triangleTotal/HIERARCHY_TRIANGLE_MAXIMUM + (triangleTotal % HIERARCHY_TRIANGLE_MAXIMUM ? 1 : 0)); i++) {
			
			#ifdef TRIANGLE_DIVISION_DEBUG
				cout << "Shadow Ray Iteration " << (i+1) << "/" << (triangleTotal/HIERARCHY_TRIANGLE_MAXIMUM + (triangleTotal % HIERARCHY_TRIANGLE_MAXIMUM ? 1 : 0)) << endl;
			#endif
			
			unsigned int triangleOffset = i * HIERARCHY_TRIANGLE_MAXIMUM;
			unsigned int triangleDivisionTotal = HIERARCHY_TRIANGLE_MAXIMUM - max(HIERARCHY_TRIANGLE_MAXIMUM * (i + 1) - triangleTotal, 0);
			
			#ifdef TRIANGLE_DIVISION_DEBUG
				cout << "Shadow Ray Iteration " << triangleOffset << "/" << triangleTotal << endl;
			#endif

			// Traverse the Hierarchy testing each Node against the Triangles Bounding Spheres [DONE]
			HierarchyTraversalWarmUpWrapper(
				cudaHierarchyArrayDP,
				cudaUpdatedBoundingSpheresDP,
				cudaUpdatedTrianglePositionsDP,
				boundingSphereTotal,
				rayTotal,
				triangleDivisionTotal,
				triangleOffset,
				cudaHeadFlagsArrayDP,
				cudaScanArrayDP,
				cudaPrimaryHierarchyHitsArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				&hierarchyHitTotal,
				&hierarchyHitMemoryTotal);

			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaGetLastError()", cudaGetLastError());
			#endif

			// Traverse the Hierarchy testing each Node against the Triangles Bounding Spheres [DONE]
			HierarchyTraversalWrapper(
				cudaHierarchyArrayDP,
				cudaUpdatedTrianglePositionsDP,
				rayTotal,
				triangleDivisionTotal,
				triangleOffset,
				cudaHeadFlagsArrayDP,
				cudaScanArrayDP,
				cudaPrimaryHierarchyHitsArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				&hierarchyHitTotal,
				&hierarchyHitMemoryTotal);

			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("HierarchyTraversalWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("HierarchyTraversalWrapper::cudaGetLastError()", cudaGetLastError());
			#endif

			// Traverse the Hierarchy Hits testing each Ray with the corresponding Triangle
			ShadowRayIntersectionWrapper(
				cudaRayArrayDP,
				cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				cudaUpdatedTrianglePositionsDP,
				hierarchyHitTotal,
				rayTotal,
				triangleOffset,
				windowWidth, windowHeight,
				cameraPosition,
				cudaPrimaryChunkKeysArrayDP);

			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("ShadowRayIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("ShadowRayIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
			#endif
		}

		// Color the Scene according to the Shadow Rays
		ShadowRayColoringWrapper(
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			cudaPrimaryChunkKeysArrayDP,
			pixelBufferObject);

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ShadowRayColoringWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ShadowRayColoringWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		return true;
	}

	return false;
}

// [Ray-Tracing] Creates a Batch of Reflection Rays.
bool createReflectionRays(bool rasterizer, float3 cameraPosition) {
	
	if(rasterizer == true) {

		// If we're using CRSH
		if(algorithmID == 0) {

			// Create the Rays and Index them [DONE]
			ReflectionRayCreationWrapper(
				windowWidth, windowHeight,
				cameraPosition,
				cudaRayArrayDP, 
				cudaHeadFlagsArrayDP, 
				cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP);
		}
		// If we're using RAH
		else {

			// Create the Rays and Index them [DONE]
			ReflectionRayCreationWrapper(
				windowWidth, windowHeight,
				cameraPosition,
				cudaRayArrayDP, 
				cudaHeadFlagsArrayDP, 
				cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP);
		}
	
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ReflectionRayCreationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ReflectionRayCreationWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		return true;
	}

	return false;
}

// [Ray-Tracing] Colors a processed Batch of Reflection Rays.
bool colorReflectionRays(bool rasterizer, bool createRays, float3 cameraPosition, unsigned int* pixelBufferObject) {

	if(rasterizer == true) {

		// Prepare the Intersection Times Array
		ReflectionRayPreparationWrapper(
			windowWidth,  windowHeight,
			cudaPrimaryChunkKeysArrayDP);

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		// Traverse the Ray Hierarchy once for every Batch
		for(unsigned int i=0; i<(triangleTotal/HIERARCHY_TRIANGLE_MAXIMUM + (triangleTotal % HIERARCHY_TRIANGLE_MAXIMUM ? 1 : 0)); i++) {

			#ifdef TRIANGLE_DIVISION_DEBUG
				cout << "Reflection Ray Iteration " << (i+1) << "/" << (triangleTotal/HIERARCHY_TRIANGLE_MAXIMUM + (triangleTotal % HIERARCHY_TRIANGLE_MAXIMUM ? 1 : 0)) << endl;
			#endif

			unsigned int triangleOffset = i * HIERARCHY_TRIANGLE_MAXIMUM;
			unsigned int triangleDivisionTotal = HIERARCHY_TRIANGLE_MAXIMUM - max(HIERARCHY_TRIANGLE_MAXIMUM * (i + 1) - triangleTotal, 0);

			#ifdef TRIANGLE_DIVISION_DEBUG
				cout << "Reflection Ray Iteration " << triangleOffset << "/" << triangleTotal << endl;
			#endif

			// Traverse the Hierarchy testing each Node against the Triangles Bounding Spheres [DONE]
			HierarchyTraversalWarmUpWrapper(
				cudaHierarchyArrayDP,
				cudaUpdatedBoundingSpheresDP,
				cudaUpdatedTrianglePositionsDP,
				boundingSphereTotal,
				rayTotal,
				triangleDivisionTotal,
				triangleOffset,
				cudaHeadFlagsArrayDP,
				cudaScanArrayDP,
				cudaPrimaryHierarchyHitsArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				&hierarchyHitTotal,
				&hierarchyHitMemoryTotal);

			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaGetLastError()", cudaGetLastError());
			#endif

			// Traverse the Hierarchy testing each Node against the Triangles Bounding Spheres [DONE]
			HierarchyTraversalWrapper(
				cudaHierarchyArrayDP,
				cudaUpdatedTrianglePositionsDP,
				rayTotal,
				triangleDivisionTotal,
				triangleOffset,
				cudaHeadFlagsArrayDP,
				cudaScanArrayDP,
				cudaPrimaryHierarchyHitsArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				&hierarchyHitTotal,
				&hierarchyHitMemoryTotal);

			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("HierarchyTraversalWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("HierarchyTraversalWrapper::cudaGetLastError()", cudaGetLastError());
			#endif

			// Traverse the Hierarchy Hits testing each Ray with the corresponding Triangle
			ReflectionRayIntersectionWrapper(
				cudaRayArrayDP,
				cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
				cudaSecondaryHierarchyHitsArrayDP,
				cudaUpdatedTrianglePositionsDP,
				hierarchyHitTotal,
				rayTotal,
				triangleOffset,
				windowWidth, windowHeight,
				lightTotal,
				cameraPosition,
				cudaPrimaryChunkKeysArrayDP);
		
			#ifdef SYNCHRONIZE_DEBUG
				Utility::checkCUDAError("ReflectionRayIntersectionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
				Utility::checkCUDAError("ReflectionRayIntersectionWrapper::cudaGetLastError()", cudaGetLastError());
			#endif
		}

		// If we're using CRSH
		if(algorithmID == 0) {

			// Color the Scene according to the Reflection Rays
			ReflectionRayColoringWrapper(
				cudaUpdatedTrianglePositionsDP, 
				cudaUpdatedTriangleNormalsDP,
				windowWidth, windowHeight,
				lightTotal,
				cameraPosition,
				createRays,
				cudaPrimaryChunkKeysArrayDP,
				cudaRayArrayDP,
				cudaHeadFlagsArrayDP,
				cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
				pixelBufferObject);
		}
		// If we're using RAH
		else {

			// Color the Scene according to the Reflection Rays
			ReflectionRayColoringWrapper(
				cudaUpdatedTrianglePositionsDP, 
				cudaUpdatedTriangleNormalsDP,
				windowWidth, windowHeight,
				lightTotal,
				cameraPosition,
				createRays,
				cudaPrimaryChunkKeysArrayDP,
				cudaRayArrayDP,
				cudaHeadFlagsArrayDP,
				cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP,
				pixelBufferObject);
		}

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("ReflectionRayColoringWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ReflectionRayColoringWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		return true;
	}

	return false;
}

// [Ray-Tracing] Processes a Batch of Rays previously created.
bool castRays(bool shadows) {

	// If we're using CRSH
	if(algorithmID == 0) {

		// Trim the Ray Indices [DONE]
		RayTrimmingWrapper(
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
			windowWidth, windowHeight, 
			LIGHT_SOURCE_MAXIMUM,
			cudaHeadFlagsArrayDP, 
			cudaScanArrayDP, 
			cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP,
			&rayTotal);

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("RayTrimmingWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("RayTrimmingWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		// Early Exit
		if(rayTotal == 0)
			return false;

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
			Utility::checkCUDAError("RayCompressionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("RayCompressionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		// Sort the Chunks [DONE]
		RaySortingWrapper(
			cudaPrimaryChunkKeysArrayDP, cudaPrimaryChunkValuesArrayDP, 
			chunkTotal,
			cudaSecondaryChunkKeysArrayDP, cudaSecondaryChunkValuesArrayDP);
		
		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("RaySortingWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("RaySortingWrapper::cudaGetLastError()", cudaGetLastError());
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
			Utility::checkCUDAError("RayDecompressionWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("RayDecompressionWrapper::cudaGetLastError()", cudaGetLastError());
		#endif
	}
	// If we're using RAH
	else {

		// Trim the Ray Indices [DONE]
		RayTrimmingWrapper(
			cudaSecondaryRayIndexKeysArrayDP, cudaSecondaryRayIndexValuesArrayDP,
			windowWidth, windowHeight, 
			LIGHT_SOURCE_MAXIMUM,
			cudaHeadFlagsArrayDP, 
			cudaScanArrayDP, 
			cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
			&rayTotal);

		#ifdef SYNCHRONIZE_DEBUG
			Utility::checkCUDAError("RayTrimmingWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("RayTrimmingWrapper::cudaGetLastError()", cudaGetLastError());
		#endif

		// Early Exit
		if(rayTotal == 0)
			return false;
	}

	// Create the Ray Hierarchy
	HierarchyCreationWrapper(
		cudaRayArrayDP,
		cudaPrimaryRayIndexKeysArrayDP, cudaPrimaryRayIndexValuesArrayDP,
		rayTotal,
		(shadows == true && softShadows == true) ? SHADOW_RAY_RADIUS : 0.0f,
		(shadows == true && softShadows == true) ? SHADOW_RAY_SPREAD : 0.0f,
		cudaHierarchyArrayDP);
		
	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("HierarchyCreationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("HierarchyCreationWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	return true;
}

// [Scene] Updates the Scene
void update(int value) {

	// Update the Timer Variables
	elapsedTime = (GLfloat)(glutGet(GLUT_ELAPSED_TIME) - startTime)/1000;
	startTime = glutGet(GLUT_ELAPSED_TIME);
	currentTime += elapsedTime;

	// Update the Scene
	sceneManager->update(elapsedTime);

	// Call Update again
	glutTimerFunc(FPS_60, update, 0);

	// Call Display again
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
	Vector cameraUp = sceneManager->getActiveCamera()->getUp();
	Vector cameraRight = sceneManager->getActiveCamera()->getRight();
	Vector cameraDirection  = sceneManager->getActiveCamera()->getDirection();

	// Get the Updated Model and Normal Matrices
	map<string, Object*> objectMap = sceneManager->getObjectMap();

	float* modelMatrices = new float[objectMap.size() * 16];
	float* normalMatrices = new float[objectMap.size() * 16];

	/****************************************************************/
	/*																*/
	/*				Model and Normal Matrix Updating				*/
	/*																*/
	/****************************************************************/

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
	
	// Update the Model and Normal Matrices
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedModelMatricesDP, &modelMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedNormalMatricesDP, &normalMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));

	/****************************************************************/
	/*																*/
	/*						Triangle Updating						*/
	/*																*/
	/****************************************************************/

	// Update the Triangle Positions and Normals
	TriangleUpdateWrapper(cudaUpdatedModelMatricesDP, cudaUpdatedNormalMatricesDP, triangleTotal, cudaUpdatedTrianglePositionsDP, cudaUpdatedTriangleNormalsDP);

	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("TriangleUpdateWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("TriangleUpdateWrapper::cudaGetLastError()", cudaGetLastError());
	#endif

	/****************************************************************/
	/*																*/
	/*				Translation and Scale Matrix Updating			*/
	/*																*/
	/****************************************************************/

	for(map<string,Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		// Translation Matrix/
		Matrix translationMatrix = object->getTransform()->getModelMatrix();
		translationMatrix.getValue(&modelMatrices[object->getID() * 16]);
		
		// Scale Matrix
		Matrix scaleMatrix;
		scaleMatrix.scale(object->getTransform()->getScale());
		scaleMatrix.getValue(&normalMatrices[object->getID() * 16]);
	}
	
	// Update the Model and Normal Matrices
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedModelMatricesDP, &modelMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));
	Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaUpdatedNormalMatricesDP, &normalMatrices[0], objectMap.size() * sizeof(float) * 16, cudaMemcpyHostToDevice));

	/****************************************************************/
	/*																*/
	/*					Bounding Sphere Updating					*/
	/*																*/
	/****************************************************************/

	// Update the Bounding Spheres
	BoundingSphereUpdateWrapper(cudaUpdatedModelMatricesDP, cudaUpdatedNormalMatricesDP, boundingSphereTotal, cudaUpdatedBoundingSpheresDP);
		
	#ifdef BOUNDING_SPHERE_DEBUG

		float3* boundingSpheres = new float3[boundingSphereTotal * 2];

		// Copy the Bounding Spheres from CUDA
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&boundingSpheres[0], cudaUpdatedBoundingSpheresDP, boundingSphereTotal * 2 * sizeof(float3), cudaMemcpyDeviceToHost));

		for(unsigned int i=0; i<boundingSphereTotal; i++) {

			cout << "Bounding Sphere " << i << endl;

			printf("Center = %02.04f %02.04f %02.04f\n", boundingSpheres[i * 2].x, boundingSpheres[i * 2].y, boundingSpheres[i * 2].z);
			printf("Radius = %02.020f\n", boundingSpheres[i * 2 + 1].x);

			cout << "Bounds = " << boundingSpheres[i * 2 + 1].y << " => " << boundingSpheres[i * 2 + 1].z << endl;
		}

	#endif

	#ifdef SYNCHRONIZE_DEBUG
		Utility::checkCUDAError("BoundingSphereUpdate::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("BoundingSphereUpdate::cudaGetLastError()", cudaGetLastError());
	#endif

	/****************************************************************/
	/*																*/
	/*						Ray-Tracing Core						*/
	/*																*/
	/****************************************************************/

	for(int i=0; i<DEPTH; i++) {

		float3 cameraEye = make_float3(cameraPosition[VX], cameraPosition[VY], cameraPosition[VZ]);

		// Cast the Shadow Ray Batch
		if(i == 0) {

			bool result = true;

			// Calculate the Shadow Rays based on the Rasterizer Input for the first Iteration.
			if(result == true)
				result = createShadowRays(true, cameraEye);
			// Cast the Generic Ray-Tracing Algorithm.
			if(result == true)
				result = castRays(true);
			// Calculate the Color based on the Rasterizer Input for the first Iteration.
			if(result == true)
				result = colorShadowRays(true, cameraEye, pixelBufferObject);

			// Debug
			TestManager* testManager = TestManager::getInstance();
			testManager->dump(algorithmID, sceneID, i, rayTotal, triangleTotal);
		}
		
		// Cast the Reflection and Refraction Ray Batches
		if(i == 1 && (sceneID == 1 || sceneID == 3)) {

			bool result = true;

			// Calculate the Reflection Rays based on the Rasterizer Input for the first Iteration.
			if(result == true)
				result = createReflectionRays(true, cameraEye);
			// Cast the Generic Ray-Tracing Algorithm.
			if(result == true)
				result = castRays(false);
			// Calculate the Color.
			if(result == true)
				result = colorReflectionRays(true, true, cameraEye, pixelBufferObject);

			// Debug
			TestManager* testManager = TestManager::getInstance();
			testManager->dump(algorithmID, sceneID, i, rayTotal, triangleTotal);
		}

		// Cast the Reflection and Refraction Ray Batches
		if(i == 2 && (sceneID == 1 || sceneID == 3)) {

			bool result = true;

			// Cast the Generic Ray-Tracing Algorithm.
			if(result == true)
				result = castRays(false);
			// Calculate the Color.
			if(result == true)
				result = colorReflectionRays(true, false, cameraEye, pixelBufferObject);

			// Debug
			TestManager* testManager = TestManager::getInstance();
			testManager->dump(algorithmID, sceneID, i, rayTotal, triangleTotal);
		}
	}

	/****************************************************************/
	/*																*/
	/*					OpenGL Colouring Core						*/
	/*																*/
	/****************************************************************/

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

	if(sceneExitor > 0)
		exit(0);
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
	unsigned int rayMaximum = windowWidth * windowHeight * LIGHT_SOURCE_MAXIMUM;

	// Update the CUDA Hierarchy Array Size
	unsigned int hierarchyMaximum = 0;
	unsigned int hierarchyNodeMaximum[HIERARCHY_MAXIMUM_DEPTH];

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

	// Heuristic Modification
	hierarchyNodeMaximum[0] = (unsigned int)(hierarchyNodeMaximum[0] / 4);

	// Store the Memory Total
	hierarchyHitMemoryTotal = hierarchyNodeMaximum[0] * HIERARCHY_TRIANGLE_ALLOCATION_MAXIMUM;

	size_t allocated = 0;
	
	// Update the CUDA Hierarchy Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaHierarchyArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryHierarchyHitsArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryHierarchyHitsArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaHierarchyArrayDP, hierarchyMaximum * sizeof(float4) * 2));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryHierarchyHitsArrayDP, hierarchyHitMemoryTotal * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryHierarchyHitsArrayDP, hierarchyHitMemoryTotal * sizeof(unsigned int)));

	allocated += hierarchyMaximum * sizeof(float4) * 2;
	allocated += hierarchyHitMemoryTotal * triangleTotal * sizeof(unsigned int);
	allocated += hierarchyHitMemoryTotal * triangleTotal * sizeof(unsigned int);

	// Update the CUDA Ray Array
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaRayArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaRayArrayDP, rayMaximum * sizeof(float3) * 2));

	allocated += rayMaximum * sizeof(float3) * 2;

	// Update the CUDA Chunks Base and Size Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaChunkBasesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaChunkSizesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaChunkBasesArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaChunkSizesArrayDP, rayMaximum * sizeof(unsigned int)));

	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);

	// Update the CUDA Ray and Sorted Ray Index Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryRayIndexKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryRayIndexValuesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryRayIndexKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryRayIndexValuesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryRayIndexKeysArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryRayIndexValuesArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryRayIndexKeysArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryRayIndexValuesArrayDP, rayMaximum * sizeof(unsigned int)));

	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);
	
	// Update the CUDA Chunk and Sorted Chunk Index Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryChunkKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaPrimaryChunkValuesArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryChunkKeysArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaSecondaryChunkValuesArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryChunkKeysArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaPrimaryChunkValuesArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryChunkKeysArrayDP, rayMaximum * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaSecondaryChunkValuesArrayDP, rayMaximum * sizeof(unsigned int)));

	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);
	allocated += rayMaximum * sizeof(unsigned int);

	// Update the CUDA Head Flags and Scan Arrays
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaHeadFlagsArrayDP));
	Utility::checkCUDAError("cudaFree()", cudaFree((void *)cudaScanArrayDP));

	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaHeadFlagsArrayDP, hierarchyHitMemoryTotal * sizeof(unsigned int)));
	Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaScanArrayDP, hierarchyHitMemoryTotal  * sizeof(unsigned int)));

	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(unsigned int);
	allocated += hierarchyNodeMaximum[0] * triangleTotal * sizeof(unsigned int);

	size_t free, total;
	Utility::checkCUDAError("cudaGetMemInfo()", cudaMemGetInfo(&free, &total));

	printf("[Callback] Total Memory:\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		total, total / 1024.0f, total / 1024.0f / 1024.0f, total / 1024.0f / 1024.0f / 1024.0f); 
	printf("[Callback] Free Memory:\t\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		free, free / 1024.0f, free / 1024.0f / 1024.0f, free / 1024.0f / 1024.0f / 1024.0f);
	printf("[Callback] Allocated Memory:\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		allocated, allocated / 1024.0f, allocated / 1024.0f / 1024.0f, allocated / 1024.0f / 1024.0f / 1024.0f);

	cout << endl;

	// Prepare the Auxiliary memory necessary.
	MemoryPreparationWrapper(
		windowWidth, windowHeight,
		hierarchyHitMemoryTotal,
		cudaHeadFlagsArrayDP, cudaScanArrayDP,
		cudaPrimaryChunkKeysArrayDP, cudaPrimaryChunkValuesArrayDP,
		cudaSecondaryChunkKeysArrayDP, cudaSecondaryChunkValuesArrayDP);

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

	// Reset the Device
	 Utility::checkCUDAError("cudaDeviceReset()", cudaDeviceReset());

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

	// Create Blinn Phong Shader 
	BlinnPhongShader* blinnPhongShader = new BlinnPhongShader(BLINN_PHONG_SHADER);
	blinnPhongShader->createShaderProgram();
	blinnPhongShader->bindAttributes();
	blinnPhongShader->linkShaderProgram();
	blinnPhongShader->bindUniforms();

	sceneManager->addShaderProgram(blinnPhongShader);

	// Create Bump Map Shader
	BumpMappingShader* bumpMappingShader = new BumpMappingShader(BUMP_MAPPING_SHADER);
	bumpMappingShader->createShaderProgram();
	bumpMappingShader->bindAttributes();
	bumpMappingShader->linkShaderProgram();
	bumpMappingShader->bindUniforms();

	sceneManager->addShaderProgram(bumpMappingShader);

	// Create Sphere Map Shader 
	SphereMappingShader* sphereMappingShader = new SphereMappingShader(SPHERE_MAPPING_SHADER);
	sphereMappingShader->createShaderProgram();
	sphereMappingShader->bindAttributes();
	sphereMappingShader->linkShaderProgram();
	sphereMappingShader->bindUniforms();

	sceneManager->addShaderProgram(sphereMappingShader);

	// Create Cube Map Shader 
	CubeMappingShader* cubeMappingShader = new CubeMappingShader(CUBE_MAPPING_SHADER);
	cubeMappingShader->createShaderProgram();
	cubeMappingShader->bindAttributes();
	cubeMappingShader->linkShaderProgram();
	cubeMappingShader->bindUniforms();

	sceneManager->addShaderProgram(cubeMappingShader);

	// Create Real Wood Shader 
	WoodShader* woodShader = new WoodShader(WOOD_SHADER);
	woodShader->createShaderProgram();
	woodShader->bindAttributes();
	woodShader->linkShaderProgram();
	woodShader->bindUniforms();

	sceneManager->addShaderProgram(woodShader);

	// Create Fire Shader 
	FireShader* fireShader = new FireShader(FIRE_SHADER);
	fireShader->createShaderProgram();
	fireShader->bindAttributes();
	fireShader->linkShaderProgram();
	fireShader->bindUniforms();

	sceneManager->addShaderProgram(fireShader);

	cout << "[Initialization] Shader Initialization Successfull" << endl;
}

// [Scene] Initializes the Scenes Lights
void initializeLights() {

	// Office
	if(sceneID == 0) {

		// Light Source - White
		PositionalLight* positionalLight1 = new PositionalLight(POSITIONAL_LIGHT_1);

		positionalLight1->setIdentifier(LIGHT_SOURCE_1);

		positionalLight1->setPosition(Vector(-15.0f, 15.0f, 7.5f, 1.0f));
		positionalLight1->setColor(Vector(1.0f, 1.0f, 1.0f, 1.0f));

		positionalLight1->setDiffuseIntensity(0.75f);
		positionalLight1->setSpecularIntensity(0.75f);
	
		lightMap[positionalLight1->getIdentifier()] = positionalLight1;
		sceneManager->addLight(positionalLight1);
	}
	// Kornell
	else if(sceneID == 1) {

		// Light Source - White
		PositionalLight* positionalLight1 = new PositionalLight(POSITIONAL_LIGHT_1);

		positionalLight1->setIdentifier(LIGHT_SOURCE_1);

		positionalLight1->setPosition(Vector( 0.0f, 45.0f, 0.0f, 1.0f));
		positionalLight1->setColor(Vector(0.8f, 0.8f, 0.8f, 1.0f));

		positionalLight1->setDiffuseIntensity(0.75f);
		positionalLight1->setSpecularIntensity(0.75f);
	
		lightMap[positionalLight1->getIdentifier()] = positionalLight1;
		sceneManager->addLight(positionalLight1);
	}
	// Street
	else if(sceneID == 2) {

		// Light Source - White
		PositionalLight* positionalLight1 = new PositionalLight(POSITIONAL_LIGHT_1);

		positionalLight1->setIdentifier(LIGHT_SOURCE_1);

		positionalLight1->setPosition(Vector(0.0f, 25.0f, 0.0f, 1.0f));
		positionalLight1->setColor(Vector(1.0f, 1.0f, 1.0f, 1.0f));

		positionalLight1->setDiffuseIntensity(0.5f);
		positionalLight1->setSpecularIntensity(0.5f);
	
		lightMap[positionalLight1->getIdentifier()] = positionalLight1;
		sceneManager->addLight(positionalLight1);

		/*// Light Source - White
		PositionalLight* positionalLight2 = new PositionalLight(POSITIONAL_LIGHT_2);

		positionalLight2->setIdentifier(LIGHT_SOURCE_2);

		positionalLight2->setPosition(Vector(-100.0f, 25.0f, 0.0f, 1.0f));
		positionalLight2->setColor(Vector(1.0f, 1.0f, 1.0f, 1.0f));

		positionalLight2->setDiffuseIntensity(0.75f);
		positionalLight2->setSpecularIntensity(0.75f);
	
		lightMap[positionalLight2->getIdentifier()] = positionalLight1;
		sceneManager->addLight(positionalLight2);*/
	}
	else {

		// Light Source - White
		PositionalLight* positionalLight1 = new PositionalLight(POSITIONAL_LIGHT_1);

		positionalLight1->setIdentifier(LIGHT_SOURCE_1);

		positionalLight1->setPosition(Vector(0.0f, 15.0f, 0.0f, 1.0f));
		positionalLight1->setColor(Vector(1.0f, 1.0f, 1.0f, 1.0f));

		positionalLight1->setDiffuseIntensity(0.75f);
		positionalLight1->setSpecularIntensity(0.75f);
	
		lightMap[positionalLight1->getIdentifier()] = positionalLight1;
		sceneManager->addLight(positionalLight1);
	}

	cout << "[Initialization] Light Initialization Successfull" << endl;
}

// [Scene] Initializes the Scenes Cameras
void initializeCameras() {

	// Create Orthogonal Camera
	Camera* orthogonalCamera = new Camera(ORTHOGONAL_NAME);
	orthogonalCamera->loadOrthogonalProjection();
	orthogonalCamera->loadView();

	sceneManager->addCamera(orthogonalCamera);

	// Create Perspective Camera
	Camera* perspectiveCamera = new Camera(PERSPECTIVE_NAME);
	perspectiveCamera->loadPerspectiveProjection();
	perspectiveCamera->loadView();

	sceneManager->addCamera(perspectiveCamera);

	// Set Active Camera
	sceneManager->setActiveCamera(perspectiveCamera);

	// Office
	if(sceneID == 0) {
		
		Camera* camera = sceneManager->getActiveCamera();
		
		// Initialize the Zoom
		camera->setZoom(0.5f);

		// Initialize the Latitude and the Longitude
		camera->setLatitude(30.5f);
		camera->setLongitude(215.0f);

		// Initialize the Target
		camera->setTarget(Vector(-4.50f, 7.25f, 1.50f, 1.0f));
	}
	// Kornell
	else if(sceneID == 1) {
	
		Camera* camera = sceneManager->getActiveCamera();
		
		// Initialize the Zoom
		camera->setZoom(4.5f);

		// Initialize the Latitude and the Longitude
		camera->setLatitude(0.5f);
		camera->setLongitude(225.0f);

		// Initialize the Target
		camera->setTarget(Vector(4.35f,-24.35f, 1.50f, 1.0f));
	}
	// Sponza
	else if(sceneID == 2) {

		Camera* camera = sceneManager->getActiveCamera();

		// Initialize the Zoom
		camera->setZoom(2.15f);

		// Initialize the Latitude and the Longitude
		camera->setLatitude(1.5f);
		camera->setLongitude(197.0f);

		// Initialize the Target
		camera->setTarget(Vector(-9.25f, 2.25f,-0.15f, 1.0f));
	}
	else {

		Camera* camera = sceneManager->getActiveCamera();

		// Initialize the Zoom
		camera->setZoom(5.5f);

		// Initialize the Latitude and the Longitude
		camera->setLatitude(49.99f);
		camera->setLongitude(45.0f);
	}

	cout << "[Initialization] Camera Initialization Successfull" << endl;
}

// [Scene] Initializes the Scenes Objects
void initializeObjects() {

	// Office
	if(sceneID == 0) {

		while(OBJ_Reader::getInstance()->canReadMesh("office.obj") == true) {

			// Office Object
			Object* officeObject = new Object("Office Object");

				// Create the Objects Mesh
				Mesh* officeObjectMesh = new Mesh("Office Mesh", "office.obj");

				// Create the Objects Material
				Material* officeObjectMaterial = new Material("Office Material", "office.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

				// Create the Objects Transform
				Transform* officeObjectTransform = new Transform(officeObject->getName());
				officeObjectTransform->setPosition(Vector(0.0f,-15.0f, 0.0f, 1.0f));
				officeObjectTransform->setScale(Vector(10.0f, 10.0f, 10.0f, 1.0f));
		
				// Set the Objects Name
				officeObject->setName(officeObjectMesh->getName());
				// Set the Objects Mesh
				officeObject->setMesh(officeObjectMesh);
				// Set the Objects Material
				officeObject->setMaterial(officeObjectMaterial);
				// Set the Objects Transform
				officeObject->setTransform(officeObjectTransform);

				// Initialize the Object
				officeObject->createMesh();
				officeObject->setID(sceneManager->getObjectID());

				// Add the Object to the Scene Manager
				sceneManager->addObject(officeObject);
				// Add the Object to the Object Map (CUDA Loading)
				objectMap[officeObject->getID()] = officeObject;

				// Create the Objects Scene Node
				SceneNode* officeObjectNode = new SceneNode(officeObject->getName());
				officeObjectNode->setObject(officeObject);

			// Add the Root Nodes to the Scene
			sceneManager->addSceneNode(officeObjectNode);
		}
	}
	// Kornell
	else if(sceneID == 1) {

		Vector positionList[6];
		Vector rotationList[6];
		Vector scaleList[6];

		// Top Face
		positionList[0] = Vector(  0.0f, 50.0f,  0.0f, 1.0f);
		rotationList[0] = Vector(  0.0f,  0.0f,  0.0f, 1.0f);
		scaleList[0] = Vector(200.0f, 0.5f, 200.0f, 1.0f);
		// Bottom Face
		positionList[1] = Vector(  0.0f,-50.0f,  0.0f, 1.0f);
		rotationList[1] = Vector(  0.0f,  0.0f,  0.0f, 1.0f);
		scaleList[1] = Vector(200.0f, 0.5f, 200.0f, 1.0f);
		// Left Face
		positionList[2] = Vector(-100.0f,  0.0f,  0.0f, 1.0f);
		rotationList[2] = Vector(   0.0f,  0.0f, 90.0f, 1.0f);
		scaleList[2] = Vector(100.0f, 0.5f, 200.0f, 1.0f);
		// Right Face
		positionList[3] = Vector( 100.0f,  0.0f,  0.0f, 1.0f);
		rotationList[3] = Vector(   0.0f,  0.0f, 90.0f, 1.0f);
		scaleList[3] = Vector(100.0f, 0.5f, 200.0f, 1.0f);
		// Front Face
		positionList[4] = Vector(  0.0f,  0.0f, 100.0f, 1.0f);
		rotationList[4] = Vector( 90.0f,  0.0f,   0.0f, 1.0f);
		scaleList[4] = Vector(200.0f, 0.5f, 100.0f, 1.0f);
		// Back Face
		positionList[5] = Vector(  0.0f,  0.0f,-100.0f, 1.0f);
		rotationList[5] = Vector( 90.0f,  0.0f,   0.0f, 1.0f);
		scaleList[5] = Vector(200.0f, 0.5f, 100.0f, 1.0f);

		// Create the Objects Mesh
		Mesh* wallObjectMesh = new Mesh(TABLE_SURFACE, "cube.obj");

		// Create the Objects Material
		Material* wallObjectMaterial[6]; 

		wallObjectMaterial[0] = new Material(TABLE_SURFACE, "silver.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		wallObjectMaterial[1] = new Material(TABLE_SURFACE, "silver.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		wallObjectMaterial[2] = new Material(TABLE_SURFACE, "gold.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		wallObjectMaterial[3] = new Material(TABLE_SURFACE, "ruby.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		wallObjectMaterial[4] = new Material(TABLE_SURFACE, "saphire.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		wallObjectMaterial[5] = new Material(TABLE_SURFACE, "emerald.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

		for(int i=0; i<6; i++) {

			// Create the Objects Name
			ostringstream stringStream;
			stringStream << "Wall " << i;

			// Table Surface
			Object* wallObject = new Object(string(stringStream.str()));

				// Create the Objects Transform
				Transform* wallObjectTransform = new Transform(TABLE_SURFACE);
				wallObjectTransform->setScale(scaleList[i]);
				wallObjectTransform->setPosition(positionList[i]);
				wallObjectTransform->setRotation(rotationList[i]);

				// Set the Objects Mesh
				wallObject->setMesh(wallObjectMesh);
				// Set the Objects Material
				wallObject->setMaterial(wallObjectMaterial[i]);
				// Set the Objects Transform
				wallObject->setTransform(wallObjectTransform);

				// Initialize the Object
				wallObject->createMesh();
				wallObject->setID(sceneManager->getObjectID());

				// Add the Object to the Scene Manager
				sceneManager->addObject(wallObject);
				// Add the Object to the Object Map (CUDA Loading)
				objectMap[wallObject->getID()] = wallObject;

				// Create the Objects Scene Node
				SceneNode* wallObjectNode = new SceneNode(wallObject->getName());
				wallObjectNode->setObject(wallObject);

			// Add the Root Nodes to the Scene
			sceneManager->addSceneNode(wallObjectNode);
		}

		// Create the Spheres Mesh
		Mesh* sphereMesh = new Mesh("Sphere Mesh", "sphere/sphere.obj");
		// Create the Spheres Materials
		Material* sphereMaterial = new Material("Sphere Gold Material", "sphere/gold.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

		Object* sphere0Object = new Object("Sphere 0");

			// Create the Objects Transform
			Transform* sphere0Transform = new Transform("Sphere");
			sphere0Transform->setPosition(Vector(0.0f,-32.5, 0.0f, 1.0f));
			sphere0Transform->setScale(Vector(25.0f,25.0f,25.0f,1.0f));
				
			// Set the Objects Mesh
			sphere0Object->setMesh(sphereMesh);
			// Set the Objects Material
			sphere0Object->setMaterial(sphereMaterial);
			// Set the Objects Transform
			sphere0Object->setTransform(sphere0Transform);

			// Initialize the Object
			sphere0Object->createMesh();
			sphere0Object->setID(sceneManager->getObjectID());

			// Add the Object to the Scene Manager
			sceneManager->addObject(sphere0Object);
			// Add the Object to the Object Map (CUDA Loading)
			objectMap[sphere0Object->getID()] = sphere0Object;

			// Create the Objects Scene Node
			SceneNode* sphere0ObjectNode = new SceneNode("Sphere 0");
			sphere0ObjectNode->setObject(sphere0Object);
	
		sceneManager->addSceneNode(sphere0ObjectNode);

		Object* sphere1Object = new Object("Sphere 1");

			// Create the Objects Transform
			Transform* sphere1Transform = new Transform("Sphere");
			sphere1Transform->setPosition(Vector(32.5f,-32.5, 0.0f, 1.0f));
			sphere1Transform->setScale(Vector(25.0f,25.0f,25.0f,1.0f));
				
			// Set the Objects Mesh
			sphere1Object->setMesh(sphereMesh);
			// Set the Objects Material
			sphere1Object->setMaterial(sphereMaterial);
			// Set the Objects Transform
			sphere1Object->setTransform(sphere1Transform);

			// Initialize the Object
			sphere1Object->createMesh();
			sphere1Object->setID(sceneManager->getObjectID());

			// Add the Object to the Scene Manager
			sceneManager->addObject(sphere1Object);
			// Add the Object to the Object Map (CUDA Loading)
			objectMap[sphere1Object->getID()] = sphere1Object;

			// Create the Objects Scene Node
			SceneNode* sphere1ObjectNode = new SceneNode("Sphere 1");
			sphere1ObjectNode->setObject(sphere1Object);
	
		sceneManager->addSceneNode(sphere1ObjectNode);

		Object* sphere2Object = new Object("Sphere 2");

			// Create the Objects Transform
			Transform* sphere2Transform = new Transform("Sphere");
			sphere2Transform->setPosition(Vector(0.0f,-32.5,-32.5f, 1.0f));
			sphere2Transform->setScale(Vector(25.0f,25.0f,25.0f,1.0f));
				
			// Set the Objects Mesh
			sphere2Object->setMesh(sphereMesh);
			// Set the Objects Material
			sphere2Object->setMaterial(sphereMaterial);
			// Set the Objects Transform
			sphere2Object->setTransform(sphere2Transform);

			// Initialize the Object
			sphere2Object->createMesh();
			sphere2Object->setID(sceneManager->getObjectID());

			// Add the Object to the Scene Manager
			sceneManager->addObject(sphere2Object);
			// Add the Object to the Object Map (CUDA Loading)
			objectMap[sphere2Object->getID()] = sphere2Object;

			// Create the Objects Scene Node
			SceneNode* sphere2ObjectNode = new SceneNode("Sphere 2");
			sphere2ObjectNode->setObject(sphere2Object);
	
		sceneManager->addSceneNode(sphere2ObjectNode);
	}
	// Sponza
	else if(sceneID == 2) {
	
		while(OBJ_Reader::getInstance()->canReadMesh("sponza.obj") == true) {

			// Office Object
			Object* sponzaObject = new Object("Street Object");

				// Create the Objects Mesh
				Mesh* sponzaObjectMesh = new Mesh("Street Mesh", "sponza.obj");

				// Create the Objects Material
				Material* sponzaObjectMaterial = new Material("Street Material", "sponza.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

				// Create the Objects Transform
				Transform* sponzaObjectTransform = new Transform(sponzaObject->getName());
				sponzaObjectTransform->setPosition(Vector(0.0f, -25.0f, 0.0f, 1.0f));
				sponzaObjectTransform->setScale(Vector(10.0f, 10.0f, 10.0f, 1.0f));
		
				// Set the Objects Name
				sponzaObject->setName(sponzaObjectMesh->getName());
				// Set the Objects Mesh
				sponzaObject->setMesh(sponzaObjectMesh);
				// Set the Objects Material
				sponzaObject->setMaterial(sponzaObjectMaterial);
				// Set the Objects Transform
				sponzaObject->setTransform(sponzaObjectTransform);

				// Initialize the Object
				sponzaObject->createMesh();
				sponzaObject->setID(sceneManager->getObjectID());

				// Add the Object to the Scene Manager
				sceneManager->addObject(sponzaObject);
				// Add the Object to the Object Map (CUDA Loading)
				objectMap[sponzaObject->getID()] = sponzaObject;

				// Create the Objects Scene Node
				SceneNode* sponzaObjectNode = new SceneNode(sponzaObject->getName());
				sponzaObjectNode->setObject(sponzaObject);

			// Add the Root Nodes to the Scene
			sceneManager->addSceneNode(sponzaObjectNode);
		}
	}
	else {

		// Table Surface
		Object* tableSurface = new Object(TABLE_SURFACE);

			// Create the Objects Mesh
			Mesh* tableSurfaceMesh = new Mesh(TABLE_SURFACE, "cube.obj");

			// Create the Objects Material
			Material* tableSurfaceMaterial = new Material(TABLE_SURFACE, "sphere/silver.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

			// Create the Objects Transform
			Transform* tableSurfaceTransform = new Transform(TABLE_SURFACE);
			tableSurfaceTransform->setPosition(Vector(0.0f,-7.5f, 0.0f, 1.0f));
			tableSurfaceTransform->setScale(Vector(75.0f, 0.5f, 75.0f, 1.0f));
		
			// Set the Objects Mesh
			tableSurface->setMesh(tableSurfaceMesh);
			// Set the Objects Material
			tableSurface->setMaterial(tableSurfaceMaterial);
			// Set the Objects Transform
			tableSurface->setTransform(tableSurfaceTransform);

			// Initialize the Object
			tableSurface->createMesh();
			tableSurface->setID(sceneManager->getObjectID());

			// Add the Object to the Scene Manager
			sceneManager->addObject(tableSurface);
			// Add the Object to the Object Map (CUDA Loading)
			objectMap[tableSurface->getID()] = tableSurface;

			// Create the Objects Scene Node
			SceneNode* tableSurfaceNode = new SceneNode(TABLE_SURFACE);
			tableSurfaceNode->setObject(tableSurface);

		// Add the Root Nodes to the Scene
		sceneManager->addSceneNode(tableSurfaceNode);

		// Create the Spheres Mesh
		Mesh* sphereMesh = new Mesh("Sphere Mesh", "sphere/sphere.obj");

		// Create the Spheres Materials
		Material* sphereMaterial[5];
		sphereMaterial[0] = new Material("Sphere Gold Material", "sphere/gold.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphereMaterial[1] = new Material("Sphere Gold Material", "sphere/silver.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphereMaterial[2] = new Material("Sphere Gold Material", "sphere/ruby.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphereMaterial[3] = new Material("Sphere Gold Material", "sphere/saphire.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));
		sphereMaterial[4] = new Material("Sphere Gold Material", "sphere/emerald.mtl", sceneManager->getShaderProgram(BLINN_PHONG_SHADER));

		for(int i=0; i<7; i++) {

			for(int j=0; j<7; j++) {

				// Create the Objects Name
				ostringstream stringStream;
				stringStream << SPHERE << (i * 7 + j);

				// Create the Objects Name
				string sphereName(stringStream.str());

				Object* sphereObject = new Object(sphereName);

					// Create the Objects Transform
					Transform* sphereTransform = new Transform(sphereName);

					sphereTransform->setPosition(Vector(i * 10.0f - 30.0f, 0.5f, j * 10.0f - 30.0f, 1.0f));
					sphereTransform->setRotation(Vector(0.0f, 0.0f,0.0f,1.0f));
					sphereTransform->setScale(Vector(2.5f,2.5f,2.5f,1.0f));
				
					// Set the Objects Mesh
					sphereObject->setMesh(sphereMesh);
					// Set the Objects Material
					sphereObject->setMaterial(sphereMaterial[i % 5]);
					// Set the Objects Transform
					sphereObject->setTransform(sphereTransform);

					// Initialize the Object
					sphereObject->createMesh();
					sphereObject->setID(sceneManager->getObjectID());

					// Add the Object to the Scene Manager
					sceneManager->addObject(sphereObject);
					// Add the Object to the Object Map (CUDA Loading)
					objectMap[sphereObject->getID()] = sphereObject;

					// Create the Objects Scene Node
					SceneNode* sphereObjectNode = new SceneNode(sphereName);
					sphereObjectNode->setObject(sphereObject);
	
				sceneManager->addSceneNode(sphereObjectNode);
			}
		}
	}

	// Destroy the Readers
	OBJ_Reader::destroyInstance();
	XML_Reader::destroyInstance();

	cout << "[Initialization] Object Initialization Successfull" << endl << endl;
}

// [Scene] Initializes the Scene
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

	initializeObjects();

	// Init the SceneManager
	sceneManager->init();

	// Setup GLUT Callbacks 
	initializeCallbacks();

	/****************************************************************/
	/*																*/
	/*						Triangle Storage						*/
	/*																*/
	/****************************************************************/

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

	// Stores the Bounding Spheres Information in the form of Arrays
	vector<float4> boundingSphereList;

	for(map<int,Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		// Bounding Box Auxiliary Variables
		int startingIndex = trianglePositionList.size() / 3;

		// Used to store the Objects vertex data
		map<int, Vertex*> vertexMap = object->getMesh()->getVertexMap();

		for(map<int, Vertex*>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++) {

			// Get the vertex from the mesh 
			Vertex* vertex = vertexIterator->second;

			// Position
			Vector originalPosition = vertex->getPosition();
			float4 position = make_float4(originalPosition[VX], originalPosition[VY], originalPosition[VZ], 1.0f);
			trianglePositionList.push_back(position);

			// Normal
			Vector originalNormal = vertex->getNormal();
			float4 normal = make_float4(originalNormal[VX], originalNormal[VY], originalNormal[VZ], 0.0f);
			triangleNormalList.push_back(normal);

			// Texture Coordinates
			Vector originalTextureCoordinates = vertex->getTextureCoordinates();
			float2 textureCoordinates = make_float2(originalTextureCoordinates[VX], originalTextureCoordinates[VY]);
			triangleTextureCoordinateList.push_back(textureCoordinates);

			// Object ID
			int1 objectID = make_int1(object->getID());
			triangleObjectIDList.push_back(objectID);

			// Material ID
			int1 materialID = make_int1(materialTotal);
			triangleMaterialIDList.push_back(materialID);
		}

		// Get the Material from the mesh 
		Material* material = object->getMaterial();

		// Material: Same as the original values
		Vector originalDiffuseProperty = material->getDiffuse();
		Vector originalSpecularProperty = material->getSpecular();
		float originalSpecularConstant = material->getSpecularConstant();

		float4 diffuseProperty = make_float4(originalDiffuseProperty[VX], originalDiffuseProperty[VY], originalDiffuseProperty[VZ], 1.0f);
		float4 specularProperty = make_float4(originalSpecularProperty[VX], originalSpecularProperty[VY], originalSpecularProperty[VZ], originalSpecularConstant);

		materialDiffusePropertyList.push_back(diffuseProperty);
		materialSpecularPropertyList.push_back(specularProperty);

		materialTotal++;

		// Bounding Box Auxiliary Variables
		int finalIndex = trianglePositionList.size() / 3 - 1;

		// Get the Bounding Sphere from the mesh 
		BoundingSphere* boundingSphere = object->getMesh()->getBoundingSphere();

		// Cemter and Radius: Same as the original values
		Vector originalCenter = boundingSphere->getCenter();
		float originalRadius = boundingSphere->getRadius();

		float4 center = make_float4(originalCenter[VX], originalCenter[VY], originalCenter[VZ], 1.0f);
		float4 radiusAndBounds = make_float4(originalRadius, (float)startingIndex, (float)finalIndex, 1.0f);

		boundingSphereList.push_back(center);
		boundingSphereList.push_back(radiusAndBounds);

		#ifdef BOUNDING_SPHERE_DEBUG
			cout << "Bounding Sphere " << boundingSphereTotal << endl;

			printf("Center = %02.010f %02.010f %02.010f\n", center.x, center.y, center.z);
			printf("Radius = %02.010f\n", radiusAndBounds.x);

			cout << "Bounds = " << (unsigned int)radiusAndBounds.y << " => " << (unsigned int)radiusAndBounds.z << endl;
		#endif

		boundingSphereTotal++;
	}

	// Total number of Triangles should be the number of loaded vertices divided by 3
	triangleTotal = trianglePositionList.size() / 3;

	cout << "[Initialization] Total number of triangles: " << triangleTotal << endl;

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

	size_t allocated = trianglePositionListSize + triangleNormalListSize + triangleTextureCoordinateListSize + triangleObjectIDListSize + triangleMaterialIDListSize;

	cout << endl;

	printf("[Initialization] Allocated Memory: \t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
		allocated, allocated / 1024.0f, allocated / 1024.0f / 1024.0f, allocated / 1024.0f / 1024.0f / 1024.0f);

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
	cout << "[Initialization] Total number of Materials: " << materialTotal << endl;

	// Each Material contains Diffuse and Specular Properties
	size_t materialDiffusePropertyListSize = materialDiffusePropertyList.size() * sizeof(float4);
	cout << "[Initialization] Material Diffuse Properties Storage Size: " << materialDiffusePropertyListSize << " (" << materialDiffusePropertyList.size() << " float4s)" << endl;
	size_t materialSpecularPropertyListSize = materialSpecularPropertyList.size() * sizeof(float4);
	cout << "[Initialization] Material Specular Properties Storage Size: " << materialSpecularPropertyListSize << " (" << materialSpecularPropertyList.size() << " float4s)" << endl;

	// Allocate the required CUDA Memory for the Materials
	if(materialTotal > 0) {
	
		// Load the Material Diffuse Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaMaterialDiffusePropertiesDP, materialDiffusePropertyListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaMaterialDiffusePropertiesDP, &materialDiffusePropertyList[0], materialDiffusePropertyListSize, cudaMemcpyHostToDevice));

		bindMaterialDiffuseProperties(cudaMaterialDiffusePropertiesDP, materialTotal);

		// Load the Material Specular Properties
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaMaterialSpecularPropertiesDP, materialSpecularPropertyListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaMaterialSpecularPropertiesDP, &materialSpecularPropertyList[0], materialSpecularPropertyListSize, cudaMemcpyHostToDevice));

		bindMaterialSpecularProperties(cudaMaterialSpecularPropertiesDP, materialTotal);
	}

	cout << endl;

	// Total number of Bounding Spheres
	cout << "[Initialization] Total number of Bounding Spheres: " << boundingSphereTotal << endl;

	// Each Bounding Box Contains a Center, Radius and two Bounds
	size_t boundingSphereListSize = boundingSphereList.size() * sizeof(float4);
	cout << "[Initialization] Bounding Sphere Storage Size: " << boundingSphereListSize << " (" << boundingSphereList.size() << " float4s)" << endl;

	// Allocate the required CUDA Memory for the Bounding Spheres
	if(boundingSphereTotal > 0) {
	
		// Load the Bounding Boxes
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaBoundingSpheresDP, boundingSphereListSize));
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(cudaBoundingSpheresDP, &boundingSphereList[0], boundingSphereListSize, cudaMemcpyHostToDevice));

		bindBoundingSpheres(cudaBoundingSpheresDP, boundingSphereTotal);

		// Bounding Boxes Memory Allocation
		Utility::checkCUDAError("cudaMalloc()", cudaMalloc((void **)&cudaUpdatedBoundingSpheresDP, boundingSphereTotal * sizeof(float3) * 2));
	}

	cout << endl;

	/****************************************************************/
	/*																*/
	/*							Light Storage						*/
	/*																*/
	/****************************************************************/

	// Stores the Lights Information in the form of Arrays
	vector<float4> lightPositionList;
	vector<float4> lightColorList;
	vector<float2> lightIntensityList;

	for(map<int,Light*>::const_iterator lightIterator = lightMap.begin(); lightIterator != lightMap.end(); lightIterator++) {
		
		Light* light = lightIterator->second;

		// Position
		Vector originalPosition = light->getPosition();
		float4 position = make_float4(originalPosition[VX], originalPosition[VY], originalPosition[VZ], 1.0f);
		lightPositionList.push_back(position);

		// Color
		Vector originalColor = light->getColor();
		float4 color = make_float4(originalColor[VX], originalColor[VY], originalColor[VZ], originalColor[VW]);
		lightColorList.push_back(color);

		// Intensity
		GLfloat originalDiffuseIntensity = light->getDiffuseIntensity();
		GLfloat originalSpecularIntensity = light->getSpecularIntensity();

		float2 intensity = make_float2(originalDiffuseIntensity, originalSpecularIntensity);
		lightIntensityList.push_back(intensity);

		lightTotal++;
	}

	// Total number of Lights 
	cout << "[Initialization] Total number of lights: " << lightTotal << endl;

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

	cout << "[Initialization] Initialization Successfull" << endl << endl;
}

int main(int argc, char* argv[]) {

	// No Scene selected
	if (argc < 3) {

		cout << "No Scene Selected." << endl;
		cout << "[USAGE] Parameter 1: 0 to 2 (Office: 0, Cornell: 1, Sponza: 2)" << endl;
		cout << "[USAGE] Parameter 2: 0 or 1 (CRSH Algorithm: 0, RAH Algorith: 1)" << endl;
		cout << "[USAGE] Parameter 3: 0 or 1 (Exit after the first frame: 0, Continue after the first frame: 1)" << endl;
	}

	// Scene selected
	if (argc >= 2) {

		int scene = atoi(argv[1]);

		cout << "Scene Selected = " << endl;

		switch(scene) {
		
			case 0:		cout << " Office" << endl;
						break;

			case 1:		cout << " Cornell" << endl;
						break;

			case 2:		cout << " Sponza" << endl;
						break;

			default:	cout << " Invalid (going default)" << endl;
						break;
		}

		sceneID = min(scene, 3);
	}

	// Algorithm selected
	if (argc >= 3) {

		string algorithm(argv[2]);

		cout << "Scene Selected = " << endl;

		if(algorithm.compare("CRSH") == 0)
			algorithmID = 0;

		if(algorithm.compare("RAH") == 0)
			algorithmID = 1;
	}

	// Exitor defined
	if (argc >= 4) {

		int exitor = atoi(argv[3]);

		sceneExitor = exitor;
	}

	// Initialize the Soft Shadows
	#ifdef SOFT_SHADOWS
		softShadows = true;
	#else
		softShadows = false;
	#endif

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