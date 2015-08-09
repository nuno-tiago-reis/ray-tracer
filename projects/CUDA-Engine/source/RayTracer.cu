#define CUB_STDERR

// CUDA definitions
#include <cuda_runtime.h>
// CUB definitions
#include <cub.cuh>

// Math Includes 
#include <helper_math.h>
#include <math_functions.h>
// Vector Includes
#include <vector_types.h>
#include <vector_functions.h>

// C++ Includes
#include <stdio.h>
// Utility Includes
#include "Utility.h"
#include "Constants.h"

// Secondary Ray Depth
static const int depth = 0;
// Air Refraction Index
static const float refractionIndex = 1.0f;

// Ray testing Constant
//static const float epsilon = 0.01f;

// Temporary Storage
static void *scanTemporaryStorage = NULL;
static size_t scanTemporaryStoreBytes = 0;

static void *radixSortTemporaryStorage = NULL;
static size_t radixSortTemporaryStoreBytes = 0;

// Ray indexing Constants
__constant__ __device__ static const float bit_mask_1_4_f = 15.0f;
//__constant__ __device__ static const float bit_mask_1_5_f = 31.0f;

__constant__ __device__ static const float half_bit_mask_1_4_f = 7.0f;
//__constant__ __device__ static const float half_bit_mask_1_5_f = 15.0f;

// OpenGL Diffuse and Specular Textures
texture<float4, cudaTextureType2D, cudaReadModeElementType> diffuseTexture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> specularTexture;
// OpenGL Fragment Position and Normal Textures
texture<float4, cudaTextureType2D, cudaReadModeElementType> fragmentPositionTexture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> fragmentNormalTexture;

// CUDA Triangle Textures
texture<float4, 1, cudaReadModeElementType> trianglePositionsTexture;
texture<float4, 1, cudaReadModeElementType> triangleNormalsTexture;
texture<float2, 1, cudaReadModeElementType> triangleTextureCoordinatesTexture;

// CUDA Triangle ID Textures
texture<int1, 1, cudaReadModeElementType> triangleObjectIDsTexture;
texture<int1, 1, cudaReadModeElementType> triangleMaterialIDsTexture;

// CUDA Material Textures
texture<float4, 1, cudaReadModeElementType> materialDiffusePropertiesTexture;
texture<float4, 1, cudaReadModeElementType> materialSpecularPropertiesTexture;

// CUDA Light Textures
texture<float4, 1, cudaReadModeElementType> lightPositionsTexture;
texture<float4, 1, cudaReadModeElementType> lightColorsTexture;
texture<float2, 1, cudaReadModeElementType> lightIntensitiesTexture;

// Ray structure
struct Ray {

	float3 origin;
	float3 direction;
	float3 inverseDirection;

	__device__ Ray() {};
	__device__ Ray(const float3 &o,const float3 &d) {

		origin = o;
		direction = d;
		direction = normalize(direction);
		inverseDirection = make_float3(1.0/direction.x, 1.0/direction.y, 1.0/direction.z);
	}
};

struct HitRecord {

	float time;

	float3 color;

	float3 point;
	float3 normal;

	int triangleIndex;

	__device__ HitRecord(const float3 &c) {

			time = UINT_MAX;

			color = c;

			point = make_float3(0,0,0);
			normal = make_float3(0,0,0);

			triangleIndex = -1; 
	}

	__device__ void resetTime() {
		
			time = UINT_MAX;

			point = make_float3(0,0,0);
			normal = make_float3(0,0,0);

			triangleIndex = -1;
	}
};

// Converts 8-bit integer to floating point rgb color
__device__ float3 intToRgb(int color) {

	float red	= color & 255;
	float green	= (color >> 8) & 255;
	float blue	= (color >> 16) & 255;

	return make_float3(red, green, blue);
}

// Converts floating point rgb color to 8-bit integer
__device__ int rgbToInt(float red, float green, float blue) {

	red		= clamp(red,	0.0f, 255.0f);
	green	= clamp(green,	0.0f, 255.0f);
	blue	= clamp(blue,	0.0f, 255.0f);

	return (int(red)) | (int(green)<<8) | (int(blue)<<16); // notice switch red and blue to counter the GL_BGRA
}

// Converts a Direction Vector to Spherical Coordinates
__device__ float2 vectorToSpherical(float3 direction) {

	float azimuth = atan(direction.y / direction.x) * 2.0f;
	float polar = acos(direction.z);

	return make_float2(azimuth,polar);
}

// Converts Spherical Coordinates to a Direction Vector
__device__ float3 sphericalToVector(float2 spherical) {

	float x = cos(spherical.x) * sin(spherical.y);
	float y = sin(spherical.x) * sin(spherical.y);
	float z = cos(spherical.y);

	return make_float3(x,y,z);
}

// Converts a ray to an integer hash value
__device__ int rayToIndex(float3 origin, float3 direction) {

	int index = 0;

	// Convert the Direction to Spherical Coordinates
	index = (unsigned int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);
	index = (index << 9) | (unsigned int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);

	// Clamp the Origin to the 0-15 range
	index = (index << 4) | (unsigned int)clamp(origin.x + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.y + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.z + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	//index = (index << 5) | (unsigned int)clamp(origin.y + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);
	//index = (index << 5) | (unsigned int)clamp(origin.z + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);

	index++;

	return index;
}

// Ray - BoundingBox Intersection Code
__device__ int RayBoxIntersection(const float3 &BBMin, const float3 &BBMax, const float3 &RayOrigin, const float3 &RayDirectionInverse, float &tmin, float &tmax) {

	float l1   = (BBMin.x - RayOrigin.x) * RayDirectionInverse.x;
	float l2   = (BBMax.x - RayOrigin.x) * RayDirectionInverse.x;
	tmin = fminf(l1,l2);
	tmax = fmaxf(l1,l2);

	l1   = (BBMin.y - RayOrigin.y) * RayDirectionInverse.y;
	l2   = (BBMax.y - RayOrigin.y) * RayDirectionInverse.y;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	l1   = (BBMin.z - RayOrigin.z) * RayDirectionInverse.z;
	l2   = (BBMax.z - RayOrigin.z) * RayDirectionInverse.z;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	return ((tmax >= tmin) && (tmax >= 0.0f));
}

// Ray - Triangle Intersection Code
__device__ float RayTriangleIntersection(const Ray &ray, const float3 &vertex0, const float3 &edge1, const float3 &edge2) {  

	float3 tvec = ray.origin - vertex0;  
	float3 pvec = cross(ray.direction, edge2);  

	float  determinant  = dot(edge1, pvec);  
	determinant = __fdividef(1.0f, determinant);  

	// First Test
	float u = dot(tvec, pvec) * determinant;  
	if (u < 0.0f || u > 1.0f)  
		return -1.0f;  

	// Second Test
	float3 qvec = cross(tvec, edge1);  

	float v = dot(ray.direction, qvec) * determinant;  
	if (v < 0.0f || (u + v) > 1.0f)  
		return -1.0f;  

	return dot(edge2, qvec) * determinant;  
}  

// Implementation of the Matrix Multiplication
__global__ void UpdateVertex(
							// Updated Normal Matrices Array
							float* modelMatricesArray,
							// Updated Normal Matrices Array
							float* normalMatricesArray,
							// Updated Triangle Positions Array
							float4* trianglePositionsArray,
							// Updated Triangle Normals Array
							float4* triangleNormalsArray,
							// Total Number of Vertices in the Scene
							int vertexTotal) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= vertexTotal)
		return;

	// Matrices ID
	int matrixID = tex1Dfetch(triangleObjectIDsTexture, x).x;

	// Model Matrix - Multiply each Vertex Position by it.
	float modelMatrix[16];

	for(int i=0; i<16; i++)
		modelMatrix[i] = modelMatricesArray[matrixID * 16 + i];
	
	float4 vertex = tex1Dfetch(trianglePositionsTexture, x);

	float updatedVertex[4];

	for(int i=0; i<4; i++) {

		updatedVertex[i] = 0.0f;
		updatedVertex[i] += modelMatrix[i * 4 + 0] * vertex.x;
		updatedVertex[i] += modelMatrix[i * 4 + 1] * vertex.y;
		updatedVertex[i] += modelMatrix[i * 4 + 2] * vertex.z;
		updatedVertex[i] += modelMatrix[i * 4 + 3] * vertex.w;
	}
	
	// Store the updated Vertex Position.
	trianglePositionsArray[x] = make_float4(updatedVertex[0], updatedVertex[1], updatedVertex[2], matrixID);

	// Normal Matrix - Multiply each Vertex Normal by it.
	float normalMatrix[16];

	for(int i=0; i<16; i++)
		normalMatrix[i] = normalMatricesArray[matrixID * 16 + i];

	float4 normal = tex1Dfetch(triangleNormalsTexture, x);

	float updatedNormal[4];

	for(int i=0; i<4; i++) {

		updatedNormal[i] = 0.0f;
		updatedNormal[i] += normalMatrix[i * 4 + 0] * normal.x;
		updatedNormal[i] += normalMatrix[i * 4 + 1] * normal.y;
		updatedNormal[i] += normalMatrix[i * 4 + 2] * normal.z;
		updatedNormal[i] += normalMatrix[i * 4 + 3] * normal.w;
	}

	// Store the updated Vertex Normal.
	triangleNormalsArray[x] = make_float4(normalize(make_float3(updatedNormal[0], updatedNormal[1], updatedNormal[2])), 0.0f);
}

//	Ray index Array	
__global__ void CreateRays(// Input Array containing the unsorted Rays
							float3* rayArray,
							// Screen Dimensions
							int windowWidth, int windowHeight,
							// Total number of Light Sources in the Scene
							int lightTotal,
							// Cameras Position in the Scene
							float3 cameraPosition,
							// Output Array containing the exclusing scan result
							int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	int rayBase = windowWidth * windowHeight;
	int rayOffset = x + y * windowWidth;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {
		
		// Ray Direction Creation
		float3 rayReflectionDirection = reflect(normalize(fragmentPosition-cameraPosition), normalize(fragmentNormal));
		float3 rayRefractionDirection = refract(normalize(fragmentPosition-cameraPosition), normalize(fragmentNormal), 1.0f / 1.52f);
		
		// Light Positions - Sent from the CPU
		float3 shadowRayPositions[LIGHT_SOURCE_MAXIMUM];
		float3 shadowRayDirections[LIGHT_SOURCE_MAXIMUM];

		// Create the Reflection and Refraction Rays and store their directions
		rayArray[(rayOffset * 2)] = fragmentPosition;
		rayArray[(rayOffset * 2) + 1] = rayReflectionDirection;

		rayArray[(rayBase + rayOffset) * 2] = fragmentPosition;
		rayArray[(rayBase + rayOffset) * 2 + 1] = rayRefractionDirection;

		// Create the Shadow Rays
		for(int l = 0; l < lightTotal; l++) {

			// Calculate the Shadow Rays Position and Direction
			shadowRayPositions[l] = make_float3(tex1Dfetch(lightPositionsTexture, l));
			shadowRayDirections[l] = normalize(shadowRayPositions[l] - fragmentPosition);

			// Diffuse Factor
			float diffuseFactor = max(dot(shadowRayDirections[l], fragmentNormal), 0.0f);
			clamp(diffuseFactor, 0.0f, 1.0f);
				
			// Store the Shadow Rays its direction
			if(diffuseFactor <= 0.0f)
				shadowRayDirections[l] = make_float3(0.0f);
			
			rayArray[(rayBase * (2 + l) + rayOffset) * 2] = fragmentPosition;
			rayArray[(rayBase * (2 + l) + rayOffset) * 2 + 1] = shadowRayDirections[l];
		}

		// Store the Reflection and Refraction Ray indices
		rayIndexKeysArray[rayOffset] = rayToIndex(fragmentPosition, rayReflectionDirection);
		rayIndexValuesArray[rayOffset] = rayOffset;

		rayIndexKeysArray[rayBase + rayOffset] = rayToIndex(fragmentPosition, rayRefractionDirection);
		rayIndexValuesArray[rayBase + rayOffset] = rayBase + rayOffset;

		// Store the Shadow Ray Indices
		for(int l = 0; l < lightTotal; l++) {
				
			// Create the Shadow Ray and store its direction
			if(length(shadowRayDirections[l]) > 0.0f) {

				rayIndexKeysArray[rayBase * (2 + l) + rayOffset] = rayToIndex(shadowRayPositions[l], shadowRayDirections[l]);
				rayIndexValuesArray[rayBase * (2 + l) + rayOffset] = rayBase * (2 + l) + rayOffset;
			}
			else {

				rayIndexKeysArray[rayBase * (2 + l) + rayOffset] = 0;
				rayIndexValuesArray[rayBase * (2 + l) + rayOffset] = 0;
			}
		}
		
		// Clean the Shadow Ray Index storage
		for(int l = lightTotal; l < LIGHT_SOURCE_MAXIMUM; l++) {
		
			rayIndexKeysArray[rayBase * (2 + l) + rayOffset] = 0;
			rayIndexValuesArray[rayBase * (2 + l) + rayOffset] = 0;
		}

		// Store the Reflection and Refraction Ray flags
		headFlagsArray[rayOffset] = 0;
		headFlagsArray[rayBase + rayOffset] = 0;

		// Store the Shadow Ray Indices
		for(int l = 0; l < lightTotal; l++) {
			
			// Create the Shadow Ray and store its direction
			if(length(shadowRayDirections[l]) > 0.0f)
				headFlagsArray[rayBase * (2 + l) + rayOffset] = 0;
			else
				headFlagsArray[rayBase * (2 + l) + rayOffset] = 1;
		}
		
		// Clean the Shadow Ray Index storage
		for(int l = lightTotal; l < LIGHT_SOURCE_MAXIMUM; l++) 	
			headFlagsArray[rayBase * (2 + l) + rayOffset] = 1;
	}
	else {		

		// Clear the Reflection and Refraction Ray Indices
		rayIndexKeysArray[rayOffset] = 0;
		rayIndexValuesArray[rayOffset] = 0;
		
		rayIndexKeysArray[rayBase + rayOffset] = 0;
		rayIndexValuesArray[rayBase + rayOffset] = 0;

		// Clean the Shadow Ray Indices		
		for(int l = 0; l < LIGHT_SOURCE_MAXIMUM; l++)  {
		
			rayIndexKeysArray[rayBase * (2 + l) + rayOffset] = 0;
			rayIndexValuesArray[rayBase * (2 + l) + rayOffset] = 0;
		}

		// Clear the Reflection and Refraction Ray Flags
		headFlagsArray[rayOffset] = 1;
		headFlagsArray[rayBase + rayOffset] = 1;

		// Clear the Shadow Ray Flags	
		for(int l = 0; l < LIGHT_SOURCE_MAXIMUM; l++) 
			headFlagsArray[rayBase * (2 + l) + rayOffset] = 1;
	}
}

// Implementation of the Ray Trimming
__global__ void TrimRays(	
							// Input Arrays containing the untrimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Total number of Rays
							int screenDimensions,
							// Auxiliary Array containing the exclusing scan result
							int* inclusiveScanArray, 
							// Output Arrays containing the trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= screenDimensions)
		return;

	int startingPosition = 0;

	// Initial Position
	if(x == 0 && inclusiveScanArray[0] == 0) {

		startingPosition = 1;

		trimmedRayIndexKeysArray[0] = rayIndexKeysArray[0];
		trimmedRayIndexValuesArray[0] = rayIndexValuesArray[0];
	}

	// Remaining Positions
	for(int i=startingPosition; i<RAYS_PER_PIXEL_MAXIMUM; i++) {

		int currentPosition = x * RAYS_PER_PIXEL_MAXIMUM + i;

		// Sum Array Offsets
		int currentOffset = inclusiveScanArray[currentPosition];
		int previousOffset = inclusiveScanArray[currentPosition - 1];

		// If the Current and the Next Scan value are the same then shift the Ray
		if(currentOffset == previousOffset) {
		
			trimmedRayIndexKeysArray[currentPosition - currentOffset] = rayIndexKeysArray[currentPosition];
			trimmedRayIndexValuesArray[currentPosition - currentOffset] = rayIndexValuesArray[currentPosition];
		}
	}
}
	

// Implementation of the Ray Compression
__global__ void CreateChunkFlags(	
							// Input Arrays containing the trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Total number of Rays
							int rayTotal,
							// Output Array containing the Chunk Head Flags
							int* headFlagsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= rayTotal)
		return;

	int startingPosition = 0;

	// Initial Position
	if(x == 0) {

		startingPosition = 1;

		headFlagsArray[x] = 1;
	}

	// Remaining Positions
	for(int i=startingPosition; i<CHUNK_DIVISION; i++) {

		int currentPosition = x * CHUNK_DIVISION + i;

		if(currentPosition >= rayTotal)
			return;
	
		// Ray Hashes
		int currentHash = trimmedRayIndexKeysArray[currentPosition];
		int previousHash = trimmedRayIndexKeysArray[currentPosition - 1];

		// If the Current and Previous Ray Hashes are different, store the Head Flag
		if(currentHash != previousHash)
			headFlagsArray[currentPosition] = 1;
		else
			headFlagsArray[currentPosition] = 0;
	}
}

__global__ void CreateChunkBases(	
							// Input Arrays containing the trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Total number of Rays
							int rayTotal,
							// Auxiliary Array containing the head flags result
							int* headFlagsArray, 
							// Auxiliary Array containing the exclusing scan result
							int* scanArray, 
							// Output Array containing the Ray Chunk Bases
							int* chunkBasesArray,
							// Output Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Remaining Positions
	for(int i=0; i<CHUNK_DIVISION; i++) {

		int currentPosition = x * CHUNK_DIVISION + i ;

		if(currentPosition >= rayTotal)
			return;
		
		// If the Head Flag isn't 1, continue;
		if(headFlagsArray[currentPosition] == 0)
			continue;

		// Store the Position of the Chunk
		int position = scanArray[currentPosition] - 1;

		// Store the Ray Base for the Chunk
		chunkBasesArray[position] = currentPosition; 
	
		// Store the Ray Hash and the Chunk Position for the Chunk
		chunkIndexKeysArray[position] = trimmedRayIndexKeysArray[currentPosition];
		chunkIndexValuesArray[position] = position;
	}
}

__global__ void CreateChunkSizes(
							// Input Array containing the Ray Chunk Bases
							int* chunkBasesArray,
							// Total number of Ray Chunks
							int chunkTotal,
							// Total number of Rays
							int rayTotal,
							// Output Array containing the Ray Chunks Sizes
							int* chunkSizesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	// Final Position
	if(x == chunkTotal - 1) {

		// Chunk Bases
		int currentBase = chunkBasesArray[x];
	
		chunkSizesArray[x] = rayTotal - currentBase;
	}
	else {
		
		// Chunk Bases
		int currentBase = chunkBasesArray[x];
		int nextBase = chunkBasesArray[x+1];

		chunkSizesArray[x] = nextBase - currentBase;
	}
}

__global__ void CreateChunkSkeleton(
							// Input Array containing the Ray Chunk Sizes
							int* chunkSizesArray,
							// Input Array containing the Ray Chunk Values
							int* sortedChunkValuesArray,
							// Total number of Ray Chunks
							int chunkTotal,
							// Output Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	skeletonArray[x] = chunkSizesArray[sortedChunkValuesArray[x]];
}

__global__ void ClearSortedRays(
							// Total number of Rays
							int rayTotal,
							// Output Arrays containing the sorted Ray Indices
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= rayTotal)
		return;

	sortedRayIndexKeysArray[x] = 0;
	sortedRayIndexValuesArray[x] = 0;
}

__global__ void CreateSortedRays(
							// Input Arrays containing the Ray Chunk Bases and Sizes
							int* chunkBasesArray,
							int* chunkSizesArray,
							// Input Array containing the chunk hashes and positions
							int* sortedChunkKeysArray,
							int* sortedChunkValuesArray,
							// Input Array containing the inclusive segmented scan result
							int* scanArray, 
							// Total number of Ray Chunks
							int chunkTotal,
							// Auxiliary Array containing the Ray Chunk Arrays head flags 
							int* headFlagsArray, 
							// Auxiliary Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray,
							// Output Arrays containing the sorted Ray Indices
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	int chunkKey = sortedChunkKeysArray[x];
	int chunkValue = sortedChunkValuesArray[x];

	int chunkBase = chunkBasesArray[chunkValue];
	int chunkSize = chunkSizesArray[chunkValue];

	int startingPosition = scanArray[x];
	int finalPosition = startingPosition + chunkSize;

	sortedRayIndexKeysArray[startingPosition] = chunkKey;
	sortedRayIndexValuesArray[startingPosition] = chunkBase;

	// Remaining Positions
	for(int i=startingPosition+1, j=1; i<finalPosition; i++) {

		sortedRayIndexKeysArray[i] = chunkKey;
		sortedRayIndexValuesArray[i] = chunkBase + j;
	}
}

__global__ void CreateHierarchyLevel1(	
							// Input Array containing the Rays
							float3* rayArray,
							// Input Arrays containing the trimmed Ray Indices
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Input Arrays containing the sorted Ray Indices
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Total number of Nodes
							int nodeTotal,
							// Output Array containing the Ray Hierarchy
							float4* hierarchyArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	float3 coneDirection = rayArray[trimmedRayIndexValuesArray[sortedRayIndexValuesArray[x]] * 2 + 1];
	float coneSpread = 0.0f;

	float3 sphereCenter = rayArray[trimmedRayIndexValuesArray[sortedRayIndexValuesArray[x]] * 2];
	float sphereRadius = 0.0f;
	
	for(int i=1; i<3; i++) {

		float3 currentConeDirection = rayArray[trimmedRayIndexValuesArray[sortedRayIndexValuesArray[x + i]] * 2 + 1];
		float currentConeSpread = acos(dot(coneDirection, currentConeDirection));
	
		coneDirection = normalize(coneDirection + currentConeDirection);
		coneSpread = currentConeSpread + max(coneSpread, currentConeSpread);
		
		float3 currentSphereCenter = rayArray[trimmedRayIndexValuesArray[sortedRayIndexValuesArray[x + i]] * 2];
		float currentSphereRadius = 0.0f;

		sphereCenter = sphereCenter + normalize(sphereCenter - currentSphereCenter) * length(sphereCenter - currentSphereCenter);
		sphereRadius = length(sphereCenter - currentSphereCenter) * 0.5f + max(sphereRadius, currentSphereRadius);
	}

	hierarchyArray[x] = make_float4(coneDirection.x, coneDirection.y, coneDirection.z, coneSpread);
}

__global__ void CreateHierarchyLevelN(	
							// Input and Output Array containing the Ray Hierarchy
							float4* hierarchyArray,
							// Hierarchy current Level
							int hierarchyLevel,
							// Total number of Nodes
							int nodeTotal) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;
}

// Implementation of Whitteds Ray-Tracing Algorithm
__global__ void RayTracePixel(	unsigned int* pixelBufferObject,
								// Screen Dimensions
								const int windowWidth, 
								const int windowHeight,
								// Updated Triangle Position Array
								float4* trianglePositionsArray,
								// Updated Triangle Position Array
								float4* triangleNormalsArray,
								// Input Arrays containing the unsorted Ray Indices
								int* rayIndexKeysArray, 
								int* rayIndexValuesArray,
								// Input Array containing the unsorted Rays
								float3* rayArray,
								// Total Number of Triangles in the Scene
								const int triangleTotal,
								// Total Number of Lights in the Scene
								const int lightTotal,
								// Ray Bounce Depth
								const int depth,
								// Medium Refraction Index
								const float refractionIndex,
								// Camera Definitions
								const float3 cameraPosition) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Ray Creation
	float3 rayOrigin = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 rayDirection = reflect(normalize(rayOrigin-cameraPosition), normalize(make_float3(tex2D(fragmentNormalTexture, x,y))));

	if(length(rayOrigin) != 0.0f) {
			
		// Calculate the Final Color
		float3 finalColor = normalize(rayOrigin);
		//float3 finalColor = rayArray[x + y * windowWidth];

		// Update the Pixel Buffer
		pixelBufferObject[y * windowWidth + x] = rgbToInt(finalColor.x * 255, finalColor.y * 255, finalColor.z * 255);
	}
	else {

		// Update the Pixel Buffer
		pixelBufferObject[y * windowWidth + x] = rgbToInt(0.0f, 0.0f, 0.0f);
	}
}

extern "C" {

	void TriangleUpdateWrapper(	// Array containing the updated Model Matrices
								float* modelMatricesArray,
								// Array containing the updated Normal Matrices
								float* normalMatricesArray,
								// Array containing the updated Triangle Positions
								float4* trianglePositionsArray,
								// Array containing the updated Triangle Normals
								float4* triangleNormalsArray,
								// Total Number of Triangles in the Scene
								int triangleTotal) {
		
		// Grid based on the Triangle Count
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(triangleTotal*3/multiplicationBlock.x + 1);
		
		// Model and Normal Matrix Multiplication
		UpdateVertex<<<multiplicationBlock, multiplicationGrid>>>(modelMatricesArray, normalMatricesArray, trianglePositionsArray, triangleNormalsArray, triangleTotal * 3);
	}

	void RayCreationWrapper(
							// Input Array containing the unsorted Rays
							float3* rayArray,
							// Screen Dimensions
							int windowWidth, int windowHeight,
							// Total number of Light Sources in the Scene
							int lightTotal,
							// Cameras Position in the Scene
							float3 cameraPosition,
							// Output Array containing the exclusing scan result
							int* headFlagsArray, 
							// Output Arrays containing the unsorted Ray Indices
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

		// Grid based on the Pixel Count
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1,windowHeight/block.y + 1);

		// Create the Rays
		CreateRays<<<block, grid>>>(rayArray, windowWidth, windowHeight, lightTotal, cameraPosition, headFlagsArray, rayIndexKeysArray, rayIndexValuesArray);
	}

	void RayTrimmingWrapper(	
							// Input Arrays containing the unsorted Ray Indices
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Screen Dimensions
							int windowWidth, int windowHeight,
							// Auxiliary Array containing the head flags
							int* headFlagsArray, 
							// Auxiliary Array containing the exclusing scan result
							int* scanArray, 
							// Output Arrays containing the sorted Ray Indices
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray) {
	
		// Number of Rays potentialy being cast per Frame
		int rayTotal = windowWidth * windowHeight * RAYS_PER_PIXEL_MAXIMUM;

		// Prepare the Inclusive Scan
		if(scanTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));
			// Allocate temporary storage for exclusive prefix scan
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&scanTemporaryStorage, scanTemporaryStoreBytes));
		}

		// Create the Trim Scan Array
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));

		// Number of Pixels per Frame
		int screenDimensions = windowWidth * windowHeight;

		// Grid based on the Pixel Count
		dim3 block(1024);
		dim3 grid(screenDimensions/block.x + 1);	

		// Trim the Ray Indices Array
		TrimRays<<<block, grid>>>(rayIndexKeysArray, rayIndexValuesArray, screenDimensions, scanArray, trimmedRayIndexKeysArray, trimmedRayIndexValuesArray);
	}

	void RayCompressionWrapper(	
							// Input Arrays containing the trimmed Ray Indices
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Total number of Rays
							int rayTotal,
							// Auxiliary Array containing the head flags result
							int* headFlagsArray, 
							// Auxiliary Array containing the exclusing scan result
							int* scanArray, 
							// Output Arrays containing the Ray Chunk Bases and Sizes
							int* chunkBasesArray,
							int* chunkSizesArray,
							// Output Arrays containing the Ray Chunks
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray) {

		// Grid based on the Ray Count
		dim3 rayBlock(1024);
		dim3 rayGrid(rayTotal/CHUNK_DIVISION/rayBlock.x + 1);

		// Create the Chunk Flags
		CreateChunkFlags<<<rayBlock, rayGrid>>>(trimmedRayIndexKeysArray, trimmedRayIndexValuesArray, rayTotal, headFlagsArray);

		// Prepare the Exclusive Scan
		if(scanTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));
			// Allocate temporary storage for exclusive prefix scan
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&scanTemporaryStorage, scanTemporaryStoreBytes));
		}

		// Update the Scan Array with each Chunks 
		Utility::checkCUDAError("cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));

		int chunkTotal;
		// Check the Ray Total (last position of the scan array)
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&chunkTotal, &scanArray[rayTotal-1], sizeof(int), cudaMemcpyDeviceToHost));

		// Create the Chunk Bases
		CreateChunkBases<<<rayBlock, rayGrid>>>(trimmedRayIndexKeysArray, trimmedRayIndexValuesArray, rayTotal, headFlagsArray, scanArray, chunkBasesArray, chunkIndexKeysArray, chunkIndexValuesArray);

		// Grid based on the Ray Chunk Count
		dim3 chunkBlock(1024);
		dim3 chunkGrid(chunkTotal/chunkBlock.x + 1);
		
		// Create the Chunk Sizes
		CreateChunkSizes<<<chunkBlock, chunkGrid>>>(chunkBasesArray, chunkTotal, rayTotal, chunkSizesArray);
	}

	void RaySortingWrapper(	
							// Input Arrays containing the Ray Chunks
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray,
							// Total number of Ray Chunks
							int chunkTotal,
							// Output Arrays containing the Ray Chunks
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray) {

		// Prepare the Radix Sort by allocating temporary storage
		if(radixSortTemporaryStorage == NULL) {

			int total = 768 * 768 * RAYS_PER_PIXEL_MAXIMUM;

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceRadixSort::SortPairs1()", 
				cub::DeviceRadixSort::SortPairs(radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
				chunkIndexKeysArray, sortedChunkIndexKeysArray,
				chunkIndexValuesArray, sortedChunkIndexValuesArray, 
				total));
			// Allocate the temporary storage
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&radixSortTemporaryStorage, radixSortTemporaryStoreBytes));
		}
					
		// Run sorting operation
		Utility::checkCUDAError("cub::DeviceRadixSort::SortPairs2()", 
			cub::DeviceRadixSort::SortPairs(radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
			chunkIndexKeysArray, sortedChunkIndexKeysArray,
			chunkIndexValuesArray, sortedChunkIndexValuesArray, 
			chunkTotal));
	}

	void RayDecompressionWrapper(	
							// Input Arrays containing the Ray Chunk Bases and Sizes
							int* chunkBasesArray,
							int* chunkSizesArray,
							// Input Arrays containing the Ray Chunks
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray,
							// Total number of Ray Chunks
							int chunkTotal,
							// Auxiliary Array containing the Ray Chunk Arrays head flags 
							int* headFlagsArray, 
							// Auxiliary Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray,
							// Auxiliary Array containing the inclusive segmented scan result
							int* scanArray, 
							// Output Arrays containing the sorted Ray Indices
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray) {

		// Grid based on the Ray Chunk Count
		dim3 chunkBlock(1024);
		dim3 chunkGrid(chunkTotal/chunkBlock.x + 1);

		CreateChunkSkeleton<<<chunkBlock, chunkGrid>>>(
			chunkSizesArray, 
			sortedChunkIndexValuesArray,
			chunkTotal, 
			skeletonArray);

		// Prepare the Exclusive Scan
		if(scanTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::ExclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, skeletonArray, scanArray, chunkTotal));
			// Allocate temporary storage for exclusive prefix scan
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&scanTemporaryStorage, scanTemporaryStoreBytes));
		}

		// Update the Scan Array with each Chunks 
		Utility::checkCUDAError("cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::ExclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, skeletonArray, scanArray, chunkTotal));

		// Create the Chunk Bases
		CreateSortedRays<<<chunkBlock, chunkGrid>>>(
			chunkBasesArray, chunkSizesArray, 
			sortedChunkIndexKeysArray, sortedChunkIndexValuesArray,
			scanArray, 
			chunkTotal, 
			headFlagsArray, 
			skeletonArray, 
			sortedRayIndexKeysArray, sortedRayIndexValuesArray);
	}

	void HierarchyCreationWrapper(	
							// Input Arrays containing the Rays
							float3* rayArray, 
							// Input Arrays containing the trimmed Ray Indices
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Input Arrays containing the sorted Ray Indices
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Total number of Rays
							int rayTotal,
							// Auxiliary Array containing the Ray Chunk Arrays head flags 
							int* headFlagsArray, 
							// Auxiliary Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray,
							// Auxiliary Array containing the inclusive segmented scan result
							int* scanArray, 
							// Output Array containing the Ray Hierarchy
							float4* hierarchyArray) {

		int hierarchyNodeTotal = rayTotal/4 + (rayTotal % 4 != 0 ? 1 : 0);
								
		// Grid based on the Hierarchy Node Count
		dim3 baseLevelBlock(1024);
		dim3 baseLevelGrid(hierarchyNodeTotal/baseLevelBlock.x + 1);

		CreateHierarchyLevel1<<<baseLevelBlock, baseLevelGrid>>>(
			rayArray,
			trimmedRayIndexKeysArray, trimmedRayIndexValuesArray,
			sortedRayIndexKeysArray, sortedRayIndexValuesArray, 
			hierarchyNodeTotal, 
			hierarchyArray);

		cout << "Nodes : " << hierarchyNodeTotal << " Grid: " << baseLevelGrid.x << " Block: " << baseLevelBlock.x << endl;
		
		for(int hierarchyLevel=1; hierarchyLevel<HIERARCHY_MAXIMUM_DEPTH; hierarchyLevel++) {

			hierarchyNodeTotal = hierarchyNodeTotal/4 + (hierarchyNodeTotal % 4 != 0 ? 1 : 0);
			
			// Grid based on the Hierarchy Node Count
			dim3 nLevelBlock(1024);
			dim3 nLevelGrid(hierarchyNodeTotal/baseLevelBlock.x + 1);

			CreateHierarchyLevelN<<<nLevelBlock, nLevelGrid>>>(hierarchyArray, hierarchyLevel, hierarchyNodeTotal);
			
			//cout << "Nodes : " << hierarchyNodeTotal << " Grid: " << nLevelBlock.x << " Block: " << nLevelGrid.x << endl;
		}
	}

	void HierarchyTraversalWrapper(	
							// Input Array containing the Ray Hierarchy
							int* hierarchyArray,
							// Total number of Rays
							int rayTotal,
							// Auxiliary Array containing the Ray Chunk Arrays head flags 
							int* headFlagsArray, 
							// Auxiliary Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray,
							// Auxiliary Array containing the inclusive segmented scan result
							int* scanArray, 
							// Output Array containing the Triangle Hits
							int* hierarchyHitsArray) {
	}

	void IntersectionWrapper(	
							// Input Array containing the Triangle Hits
							int* hierarchyHitsArray,
							// Total number of Rays
							int rayTotal) {
	}

	void RayTraceWrapper(	unsigned int *pixelBufferObject,
							// Screen Dimensions
							int width, int height, 
							// Updated Triangle Position Array
							float4* trianglePositionsArray,
							// Updated Triangle Position Array
							float4* triangleNormalsArray,
							// Input Arrays containing the unsorted Ray Indices
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Input Array containing the unsorted Rays
							float3* rayArray,
							// Total Number of Triangles in the Scene
							int triangleTotal,
							// Total Number of Lights in the Scene
							int lightTotal,
							// Camera Definitions
							float3 cameraPosition) {

		// Ray-Casting
		dim3 rayCastingBlock(32,32);
		dim3 rayCastingGrid(width/rayCastingBlock.x + 1,height/rayCastingBlock.y + 1);

		RayTracePixel<<<rayCastingBlock, rayCastingGrid>>>(	pixelBufferObject,
															width, height,
															trianglePositionsArray, 
															triangleNormalsArray,
															rayIndexKeysArray,
															rayIndexValuesArray,
															rayArray,
															triangleTotal,
															lightTotal,
															depth, refractionIndex,
															cameraPosition);

		/*unsigned int bit_mask_1_4 = 15;
		unsigned int bit_mask_1_5 = 31;
		unsigned int bit_mask_1_9 = 511;

		unsigned int half_bit_mask_1_4 = 7;
		unsigned int half_bit_mask_1_5 = 15;
		unsigned int half_bit_mask_1_9 = 255;

		float bit_mask_1_4_f = 15.0f;
		float bit_mask_1_5_f = 31.0f;
		float bit_mask_1_9_f = 511.0f;

		float half_bit_mask_1_4_f = 7.0f;
		float half_bit_mask_1_5_f = 15.0f;
		float half_bit_mask_1_9_f = 255.0f;

		float3 origin = make_float3(5.0f, 5.0f, 5.0f);
		float3 direction = make_float3(0.333f, 0.333f, 0.333f);

		unsigned int index = 0;
			
		int azimuth = (int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);
		index = azimuth; 
		printf("Azimuth = %u (%u)\n", azimuth, index);

		int polar = (int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);
		index = (index << 9) | polar;
		printf("Polar = %u (%u)\n", polar, index);

		// Clamp the Origin to the 0-15 range
		int x = (int)clamp(origin.x + bit_mask_1_4 / 2, 0.0f, (float)bit_mask_1_4);
		index = (index << 4) | x;
		printf("Coordinate 1 = %u (%u)\n", x, index);
		int y = (int)clamp(origin.y + bit_mask_1_5 / 2, 0.0f, (float)bit_mask_1_5);
		index = (index << 5) | y;
		printf("Coordinate 2 = %u (%u)\n", y, index);
		int z = (int)clamp(origin.z + bit_mask_1_5 / 2, 0.0f, (float)bit_mask_1_5);
		index = (index << 5) | z;
		printf("Coordinate 3 = %u (%u)\n", z, index);
		
		printf("[R] Index = %u\n", index);
		printf("[R] Azimuth = %u (at %u)\n", (index & (bit_mask_1_9 << 23)) >> 23, bit_mask_1_9 << 23);		
		printf("[R] Polar = %u (at %u)\n", (index & (bit_mask_1_9 << 14)) >> 14, bit_mask_1_9 << 14);
		printf("[R] Coordinate 1 = %u (at %u)\n", (index & (bit_mask_1_4 << 10)) >> 10, bit_mask_1_4 << 10);
		printf("[R] Coordinate 2 = %u (at %u)\n", (index & (bit_mask_1_5 << 5)) >> 5, bit_mask_1_5 << 5);
		printf("[R] Coordinate 3 = %u (at %u)\n", (index & bit_mask_1_5), bit_mask_1_5);

		// Convert the Direction to Spherical Coordinates
		index = (int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);

		index = (index << 9) | (int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);

		// Clamp the Origin to the 0-15 range
		index = (index << 4) | (int)clamp(origin.x + half_bit_mask_1_4_f , 0.0f, bit_mask_1_4_f);
		index = (index << 5) | (int)clamp(origin.y + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);
		index = (index << 5) | (int)clamp(origin.z + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);
		
		printf("[R] Index = %u\n", index);
		printf("[R] Azimuth = %u (at %u)\n", (index & (bit_mask_1_9 << 23)) >> 23, bit_mask_1_9 << 23);		
		printf("[R] Polar = %u (at %u)\n", (index & (bit_mask_1_9 << 14)) >> 14, bit_mask_1_9 << 14);
		printf("[R] Coordinate 1 = %u (at %u)\n", (index & (bit_mask_1_4 << 10)) >> 10, bit_mask_1_4 << 10);
		printf("[R] Coordinate 2 = %u (at %u)\n", (index & (bit_mask_1_5 << 5)) >> 5, bit_mask_1_5 << 5);
		printf("[R] Coordinate 3 = %u (at %u)\n", (index & bit_mask_1_5), bit_mask_1_5);*/
	}

	// OpenGL Texture Binding Functions
	void bindDiffuseTextureArray(cudaArray *diffuseTextureArray) {
	
		diffuseTexture.normalized = false;					// access with normalized texture coordinates
		diffuseTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		diffuseTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		diffuseTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(diffuseTexture, diffuseTextureArray, channelDescriptor);
	}

	void bindSpecularTextureArray(cudaArray *specularTextureArray) {
	
		specularTexture.normalized = false;					// access with normalized texture coordinates
		specularTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		specularTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		specularTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(specularTexture, specularTextureArray, channelDescriptor);
	}

	void bindFragmentPositionArray(cudaArray *fragmentPositionArray) {
	
		fragmentPositionTexture.normalized = false;					// access with normalized texture coordinates
		fragmentPositionTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		fragmentPositionTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		fragmentPositionTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(fragmentPositionTexture, fragmentPositionArray, channelDescriptor);
	}

	void bindFragmentNormalArray(cudaArray *fragmentNormalArray) {
	
		fragmentNormalTexture.normalized = false;					// access with normalized texture coordinates
		fragmentNormalTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		fragmentNormalTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		fragmentNormalTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(fragmentNormalTexture, fragmentNormalArray, channelDescriptor);
	}

	// CUDA Triangle Texture Binding Functions
	void bindTrianglePositions(float *cudaDevicePointer, unsigned int triangleTotal) {

		trianglePositionsTexture.normalized = false;                      // access with normalized texture coordinates
		trianglePositionsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		trianglePositionsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, trianglePositionsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleNormals(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleNormalsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleNormalsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleNormalsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangleNormalsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleTextureCoordinatesTexture.normalized = false;                      // access with normalized texture coordinates
		triangleTextureCoordinatesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleTextureCoordinatesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float2) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float2>();
		cudaBindTexture(0, triangleTextureCoordinatesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleObjectIDs(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleMaterialIDsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleMaterialIDsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleMaterialIDsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(int1) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<int1>();
		cudaBindTexture(0, triangleObjectIDsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleMaterialIDs(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleMaterialIDsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleMaterialIDsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleMaterialIDsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(int1) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<int1>();
		cudaBindTexture(0, triangleMaterialIDsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	// CUDA Material Texture Binding Functions
	void bindMaterialDiffuseProperties(float *cudaDevicePointer, unsigned int materialTotal) {

		materialDiffusePropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		materialDiffusePropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		materialDiffusePropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * materialTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, materialDiffusePropertiesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindMaterialSpecularProperties(float *cudaDevicePointer, unsigned int materialTotal) {

		materialSpecularPropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		materialSpecularPropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		materialSpecularPropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * materialTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, materialSpecularPropertiesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	// CUDA Light Texture Binding Functions
	void bindLightPositions(float *cudaDevicePointer, unsigned int lightTotal) {

		lightPositionsTexture.normalized = false;                      // access with normalized texture coordinates
		lightPositionsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightPositionsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, lightPositionsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindLightColors(float *cudaDevicePointer, unsigned int lightTotal) {

		lightColorsTexture.normalized = false;                      // access with normalized texture coordinates
		lightColorsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightColorsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, lightColorsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindLightIntensities(float *cudaDevicePointer, unsigned int lightTotal) {

		lightIntensitiesTexture.normalized = false;                      // access with normalized texture coordinates
		lightIntensitiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightIntensitiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float2) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float2>();
		cudaBindTexture(0, lightIntensitiesTexture, cudaDevicePointer, channelDescriptor, size);
	}
}