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
static const float epsilon = 0.01f;

// Light Maximum Amount
static const int lightSourceMaximum = 10;
static const int raysPerPixel = 12;

// Ray indexing Constants
__constant__ __device__ static const unsigned int bit_mask_1_4 = 15;
__constant__ __device__ static const unsigned int bit_mask_1_5 = 31;
__constant__ __device__ static const unsigned int bit_mask_1_9 = 511;

__constant__ __device__ static const unsigned int half_bit_mask_1_4 = 7;
__constant__ __device__ static const unsigned int half_bit_mask_1_5 = 15;
__constant__ __device__ static const unsigned int half_bit_mask_1_9 = 255;

__constant__ __device__ static const float bit_mask_1_4_f = 15.0f;
__constant__ __device__ static const float bit_mask_1_5_f = 31.0f;
__constant__ __device__ static const float bit_mask_1_9_f = 511.0f;

__constant__ __device__ static const float half_bit_mask_1_4_f = 7.0f;
__constant__ __device__ static const float half_bit_mask_1_5_f = 15.0f;
__constant__ __device__ static const float half_bit_mask_1_9_f = 255.0f;

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

	// 32 bits
	//	- 14 bits for the origin (4 for x, 5 for y and 5 for z)
	//  - 18 bits for the direction (10 for longitude, 10 for latitude)
	int index = 0;

	// Convert the Direction to Spherical Coordinates
	index = (unsigned int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);
	index = (index << 9) | (unsigned int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);

	// Clamp the Origin to the 0-15 range
	index = (index << 4) | (unsigned int)clamp(origin.x + half_bit_mask_1_4_f , 0.0f, bit_mask_1_4_f);
	index = (index << 5) | (unsigned int)clamp(origin.y + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);
	index = (index << 5) | (unsigned int)clamp(origin.z + half_bit_mask_1_5_f, 0.0f, bit_mask_1_5_f);

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
__global__ void MultiplyVertex(
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

	// Vertices
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
	
	trianglePositionsArray[x] = make_float4(updatedVertex[0], updatedVertex[1], updatedVertex[2], matrixID);

	// Normals
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

	triangleNormalsArray[x] = make_float4(normalize(make_float3(updatedNormal[0], updatedNormal[1], updatedNormal[2])), 0.0f);
}

// Implementation of the Ray Creation and Indexing
__global__ void RayCreation(// Input Array containing the unsorted Rays
							float3* rayArray,
							// Screen Dimensions
							int windowWidth, int windowHeight,
							// Total number of Light Sources in the Scene
							int lightTotal,
							// Cameras Position in the Scene
							float3 cameraPosition,
							// Output Array containing the unsorted Ray Indices
							int2* rayIndicesArray) {

	// Ray Indexing		
	//
	// Use Directional Indexing first (24 bits)
	// Use Positional Indexing second (8 bits)
	//
	// Output
	//		Ray index Array	

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	int rayArrayBase = x * raysPerPixel + y * windowWidth * raysPerPixel;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	// Ray Origin Creation
	float3 rayOrigin = fragmentPosition;

	if(length(rayOrigin) != 0.0f) {
		
		// Ray Direction Creation
		float3 rayReflectionDirection = reflect(normalize(fragmentPosition-cameraPosition), normalize(fragmentNormal));
		float3 rayRefractionDirection = refract(normalize(fragmentPosition-cameraPosition), normalize(fragmentNormal), 1.0f / 1.52f);

		// Create the Reflection Ray and store its direction
		rayArray[rayArrayBase] = rayReflectionDirection;
		rayIndicesArray[rayArrayBase] = make_int2(rayToIndex(rayOrigin, rayReflectionDirection), rayArrayBase);
		rayArrayBase++;

		// Create the Refraction Ray and store its direction
		rayArray[rayArrayBase] = rayRefractionDirection;
		rayIndicesArray[rayArrayBase] = make_int2(rayToIndex(rayOrigin, rayRefractionDirection), rayArrayBase);
		rayArrayBase++;

		for(int l = 0; l < lightSourceMaximum; l++) {

			if(l < lightTotal) {

				// Fetch the Light Position
				float3 lightPosition = make_float3(tex1Dfetch(lightPositionsTexture, l));
				// Calculate the Light Direction
				float3 lightDirection = normalize(lightPosition - fragmentPosition);

				// Diffuse Factor
				float diffuseFactor = max(dot(lightDirection, fragmentNormal), 0.0f);
				clamp(diffuseFactor, 0.0f, 1.0f);
				
				if(diffuseFactor >
					0.0f) {
					
					// Create the Shadow Ray and store its direction
					rayArray[rayArrayBase] = lightDirection;
					rayIndicesArray[rayArrayBase] = make_int2(rayToIndex(lightPosition, lightDirection), rayArrayBase);
					rayArrayBase++;
				}
			}
			else {
				
				// Clean the Shadow Ray storage
				rayArray[rayArrayBase] = make_float3(0.0f);
				rayIndicesArray[rayArrayBase] = make_int2(0);
				rayArrayBase++;
			}
		}
	}
	else {
	
		// Clean the Reflection Ray storage
		rayArray[rayArrayBase] = make_float3(0.0f);
		rayIndicesArray[rayArrayBase] = make_int2(0);
		rayArrayBase++;

		// Clean the Refraction Ray storage
		rayArray[rayArrayBase] = make_float3(0.0f);
		rayIndicesArray[rayArrayBase] = make_int2(0);
		rayArrayBase++;

		// Clean the Shadow Ray storage
		rayArray[rayArrayBase] = make_float3(0.0f);
		rayIndicesArray[rayArrayBase] = make_int2(0);
		rayArrayBase++;
	}
}

// Implementation of the Ray Compression
__global__ void RayCompression(	
							// Input Array containing the unsorted Ray Indices
							int2* rayIndicesArray,
							// Auxiliary Array containing the head flags result
							int* headFlagsArray, 
							// Auxiliary Array containing the exclusing scan result
							int* scanArray, 
							// Output Array containing the unsorted Ray Chunks
							int2* chunkArray) {

	// Ray Compression - Compress Rays with the same index into chunks 
	//
	// Create the Head Flags Array (Initialized with 0)
	//		Head: (ray[i] != ray[i-1] => head[i] = 1 : head[i] = 0)
	//
	// Exclusive Scan on the Head Array (Initialized with 0)
	//		Scan: Sum of the Head Array
	//
	// Create the Chucks and Size Array (Initialized with 0 and 0)
	//		Base: (head[i] != head[i+1] => base[i] = i) 
	//		Size: (size[i] = base[i+1] - base[i])
	// 
	// Output 
	//		Base Array with the starting index of the chunk 
	//		Size Array with the size of the chunk
}

// Implementation of the Ray Sorting
__global__ void RaySorting(	
							// Input Array containing the unsorted Ray Chunks
							int2* chunkArray, 
							// Output Array containing the sorted Ray Chunks
							int2* sortedChunkArray) {

	// Ray Sorting - Radix Sort the Base Array and the Size Array
	//
	// Radix Sort on the Base Array
	//
	// Size Array doesn't have to be sorted, just needs to follow the sorting of the Base Array
}

// Implementation of the Ray Decompression
__global__ void RayDecompression(
							// Input Array containing the sorted Ray Chunks
							int2* sortedChunkArray, 
							// Auxiliary Array containing the Ray Chunk Arrays head flags 
							int* headFlagsArray, 
							// Auxiliary Array containing the Ray Chunk Arrays skeleton
							int* skeletonArray,
							// Auxiliary Array containing the inclusive segmented scan result
							int* scanArray, 
							// Output Array containing the sorted Ray Indices
							int2* sortedRayIndicesArray) {

	// Ray Decompression - Decompress Rays from the sorted chunks
	//
	// Exclusive Scan on the sorted Size Array
	//		Scan: Sum of the sorted Size Array
	//
	// Create the Skeleton and Head Flags Array (Initialized with 1 and 0)
	//		Skeleton: skeleton[i] = base[scan[i]]
	//		Head: head[i] = (skeleton[i] != 0)
	//
	// Inclusive Segmented Scan on the Skeleton and Head Arrays
	//
	// Output 
	//		Sorted ray index Array	
}

// Implementation of Whitteds Ray-Tracing Algorithm
__global__ void RayTracePixel(	unsigned int* pixelBufferObject,
								// Screen Dimensions
								const int width, 
								const int height,
								// Updated Triangle Position Array
								float4* trianglePositionsArray,
								// Updated Triangle Position Array
								float4* triangleNormalsArray,
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

	if(x >= width || y >= height)
		return;

	// Ray Creation
	float3 rayOrigin = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 rayDirection = reflect(normalize(rayOrigin-cameraPosition), normalize(make_float3(tex2D(fragmentNormalTexture, x,y))));

	if(length(rayOrigin) != 0.0f) {
			
		// Calculate the Final Color
		float3 finalColor = normalize(rayOrigin);
		//float3 finalColor = rayArray[x * raysPerPixel + y * width * raysPerPixel + 2];

		// Update the Pixel Buffer
		pixelBufferObject[y * width + x] = rgbToInt(finalColor.x * 255, finalColor.y * 255, finalColor.z * 255);
	}
	else {
	
		// Update the Pixel Buffer
		pixelBufferObject[y * width + x] = rgbToInt(0.0f, 0.0f, 0.0f);
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
		dim3 multiplicationGrid(triangleTotal*3/1024 + 1);
		
		// Model and Normal Matrix Multiplication
		MultiplyVertex<<<multiplicationBlock, multiplicationGrid>>>(modelMatricesArray, normalMatricesArray, trianglePositionsArray, triangleNormalsArray, triangleTotal * 3);
	}

	void RayCreationWrapper(// Input Array containing the unsorted Rays
							float3* rayArray,
							// Screen Dimensions
							int windowWidth, int windowHeight,
							// Total number of Light Sources in the Scene
							int lightTotal,
							// Cameras Position in the Scene
							float3 cameraPosition,
							// Output Array containing the unsorted Ray Indices
							int2* rayIndicesArray) {

		// Grid based on the Pixel Count
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1,windowHeight/block.y + 1);

		RayCreation<<<block, grid>>>(rayArray, windowWidth, windowHeight, lightTotal, cameraPosition, rayIndicesArray);
	}

	void RayCompressionWrapper(	// Input Array containing the unsorted Ray Indices
								int2* rayIndicesArray,
								// Auxiliary Array containing the head flags result
								int* headFlagsArray, 
								// Auxiliary Array containing the exclusing scan result
								int* scanArray, 
								// Output Array containing the unsorted Ray Chunks
								int2* chunkArray) {

		int rayTotal = 768*768*7;
		// Use Exclusing and Inclusive Sums to calculate Array Size and also to Truncate the Arrays

		// Grid based on the Ray Count
		dim3 block(1024);
		dim3 grid(rayTotal/1024 + 1);
		
		// Ray Compression
		RayCompression<<<block, grid>>>(rayIndicesArray, headFlagsArray, scanArray, chunkArray);
	}

	void RaySortingWrapper(	// Input Array containing the unsorted Ray Chunks
							int2* chunkArray, 
							// Output Array containing the sorted Ray Chunks
							int2* sortedChunkArray) {

	}

	void RayDecompressionWrapper(	// Input Array containing the sorted Ray Chunks
									int2* sortedChunkArray, 
									// Auxiliary Array containing the Ray Chunk Arrays head flags 
									int* headFlagsArray, 
									// Auxiliary Array containing the Ray Chunk Arrays skeleton
									int* skeletonArray,
									// Auxiliary Array containing the inclusive segmented scan result
									int* scanArray, 
									// Output Array containing the sorted Ray Indices
									int2* sortedRayIndicesArray) {

		int chunkTotal = 768*768*7;

		// Grid based on the Chunk Count
		dim3 block(1024);
		dim3 grid(chunkTotal/1024 + 1);
		
		// Ray Decompression
		RayDecompression<<<block, grid>>>(sortedChunkArray, headFlagsArray, skeletonArray, scanArray, sortedRayIndicesArray);
	}

	void RayTraceWrapper(	unsigned int *pixelBufferObject,
							// Screen Dimensions
							int width, int height, 			
							// Updated Normal Matrices Array
							float* modelMatricesArray,
							// Updated Normal Matrices Array
							float* normalMatricesArray,
							// Updated Triangle Position Array
							float4* trianglePositionsArray,
							// Updated Triangle Position Array
							float4* triangleNormalsArray,
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