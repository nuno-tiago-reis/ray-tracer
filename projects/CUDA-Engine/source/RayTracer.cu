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
#include <fstream>
#include <stdio.h>
#include <map>

// Utility Includes
#include "Utility.h"
#include "Constants.h"

// Test Includes
#include "TestManager.h"

// Temporary Storage
static void *scanTemporaryStorage = NULL;
static size_t scanTemporaryStoreBytes = 0;

static void *radixSortTemporaryStorage = NULL;
static size_t radixSortTemporaryStoreBytes = 0;

// Ray testing Constant
__constant__ __device__ static const float epsilon = 0.01f;

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

// CUDA Bounding Sphere Textures
texture<float4, 1, cudaReadModeElementType> boundingSpheresTexture;

// CUDA Light Textures
texture<float4, 1, cudaReadModeElementType> lightPositionsTexture;
texture<float4, 1, cudaReadModeElementType> lightColorsTexture;
texture<float2, 1, cudaReadModeElementType> lightIntensitiesTexture;

// Ray structure
struct Ray {

	float3 origin;
	float3 direction;

	__device__ Ray(const float3 &o,const float3 &d) {

		origin = o;
		direction = d;
	}
};

__device__ static inline unsigned int FloatFlip(unsigned int f) {

	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
}

__device__ static inline unsigned int IFloatFlip(unsigned int f) {

	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}

// Converts 8-bit integer to floating point rgb color
__device__ static inline float3 IntToRgb(int color) {

	float red	= color & 255;
	float green	= (color >> 8) & 255;
	float blue	= (color >> 16) & 255;

	return make_float3(red, green, blue);
}

// Converts floating point rgb color to 8-bit integer
__device__ static inline int RgbToInt(float red, float green, float blue) {

	red		= clamp(red,	0.0f, 255.0f);
	green	= clamp(green,	0.0f, 255.0f);
	blue	= clamp(blue,	0.0f, 255.0f);

	return (int(red)) | (int(green)<<8) | (int(blue)<<16); // notice switch red and blue to counter the GL_BGRA
}

// Converts a Direction Vector to Spherical Coordinates
__device__ static inline float2 CartesianToSpherical(float3 direction) {

	float azimuth = atan(direction.y / direction.x);
	float polar = acos(direction.z);

	return make_float2(azimuth,polar);
}

// Converts Spherical Coordinates to a Direction Vector
__device__ static inline float3 SphericalToCartesian(float2 spherical) {

	float x = cos(spherical.x) * sin(spherical.y);
	float y = sin(spherical.x) * sin(spherical.y);
	float z = cos(spherical.y);

	return make_float3(x,y,z);
}

// Compresses the Node ID and Triangle ID into a Hit Record
__device__ static inline unsigned int CreateHit(unsigned int nodeID, unsigned int triangleID) {
	
	return ((nodeID << 14) & 0xFFFFC000) + (triangleID & 0x00003FFF);
}

// Extracts the Node ID from the Hit Record (Last 18 Bits)
__device__ static inline unsigned int ExtractNodeID(unsigned int hit) {
	
	return (hit & 0xFFFFC000) >> 14;
}

// Extracts the Triangle ID from the Hit Record (First 14 Bits)
__device__ static inline unsigned int ExtractTriangleID(unsigned int hit) {

	return (hit & 0x00003FFF);
}

// Converts a ray to an Integer Hash Value
__device__ static inline unsigned int CreateShadowRayIndex(float3 origin, float3 direction, unsigned int lightIndex) {

	unsigned int index = 0;
	
	index = (lightIndex + 1) << 30;
	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index += clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)31) << 15;
	index += clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)31);

	index++;

	/*// Clamp the Origin to the 0-15 range
	index = (unsigned int)clamp(origin.z + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.y + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.x + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	
	// Convert the Direction to Spherical Coordinates
	index = (index << 4) | (unsigned int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);
	index = (index << 9) | (unsigned int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);*/

	return index;
}
	
__device__ static inline unsigned int CreateReflectionRayIndex(float3 origin, float3 direction) {

	unsigned int index = 0;
	
	/*// Clamp the Origin to the 0-15 range
	index = clamp((unsigned int)(origin.z + 64.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(origin.y + 64.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(origin.x + 64.0f), (unsigned int)0, (unsigned int)128);

	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index = (index << 7) | clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)30);
	index = (index << 5) | clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)30);*/

	float distance = length(origin);

	// Clamp the Origin to the 0-15 range
	//index = clamp((unsigned int)distance, (unsigned int)0, (unsigned int)128);
	//index = (index << 7) | clamp((unsigned int)((atan2(origin.x, origin.x) + HALF_PI) * 20.0f), (unsigned int)0, (unsigned int)128);
	index = clamp((unsigned int)((atan2(origin.x, origin.x) + HALF_PI) * 20.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(acos(origin.z/distance) * 20.0f), (unsigned int)0, (unsigned int)128);

	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index = (index << 7) | clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)30);
	index = (index << 5) | clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)30);

	index++;

	return index;
}

__device__ static inline unsigned int CreateRefractionRayIndex(float3 origin, float3 direction) {

	unsigned int index = 0;

	/*// Clamp the Origin to the 0-15 range
	index = clamp((unsigned int)(origin.z + 64.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(origin.y + 64.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(origin.x + 64.0f), (unsigned int)0, (unsigned int)128);

	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index = (index << 7) | clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)30);
	index = (index << 5) | clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)30);*/

	float distance = length(origin);

	// Clamp the Origin to the 0-15 range
	index = clamp((unsigned int)distance, (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)((atan2(origin.x, origin.x) + HALF_PI) * 20.0f), (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)(acos(origin.z/distance) * 20.0f), (unsigned int)0, (unsigned int)128);

	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index = (index << 7) | clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)30);
	index = (index << 5) | clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)30);

	index++;

	return index;
}

// Ray - Sphere Intersection Code
__device__ float RaySphereIntersection(const Ray &ray, const float3 sphereCenter, const float sphereRadius) {

	float3 sr = ray.origin - sphereCenter;

	float b = dot(sr, ray.direction);
	float c = dot(sr, sr) - (sphereRadius * sphereRadius);
	float d = b * b - c;

	float time;

	if(d > 0) {

		float e = sqrt(d);
		float t0 = -b-e;

		if(t0 < 0)
			time = -b+e;
		else
			time = min(-b-e,-b+e);

		return time;
	}

	return -1.0f;
}

// Ray - Triangle Intersection Code
__device__ static inline float RayTriangleIntersection(const Ray &ray, const float3 &vertex0, const float3 &edge1, const float3 &edge2) {  

	float3 tvec = ray.origin - vertex0;  
	float3 pvec = cross(ray.direction, edge2);  

	float  determinant  = dot(edge1, pvec);  
	determinant = __fdividef(1.0f, determinant);  
	
	// First Test
	float u = dot(tvec, pvec) * determinant;  
	if (u < 0.0f || u > 1.0f) 
		return -1.0f;
		//return length(tvec) * 1.5f;

	// Second Test
	float3 qvec = cross(tvec, edge1);  

	float v = dot(ray.direction, qvec) * determinant;  
	if (v < 0.0f || (u + v) > 1.0f)  
		return -1.0f;
		//return length(tvec) * 1.5f;

	return dot(edge2, qvec) * determinant;  
}  

// Ray - Node Intersection Code
__device__ static inline bool SphereNodeIntersection(const float4 &sphere, const float4 &cone, const float4 &triangle, const float &cosine, const float &tangent) {

	// intersection with hte reverse cone
	if(cone.w == HALF_PI)
		return true;

	float3 coneDirection = make_float3(cone);
	float3 sphereCenter = make_float3(sphere);
	float3 triangleCenter = make_float3(triangle);

	float3 sphereToTriangle = triangleCenter - sphereCenter;
	float3 sphereToTriangleProjection = dot(sphereToTriangle, coneDirection) * coneDirection;

	//if(dot is negative return false?
	return (dot(sphereToTriangle, coneDirection) * tangent + (sphere.w + triangle.w) / cosine) >= length(sphereToTriangle - sphereToTriangleProjection);
}

// Triangle Bounding Sphere Code
__device__ static inline float4 CreateTriangleBoundingSphere(const float3 &vertex0, const float3 &vertex1, const float3 &vertex2) {
	   
	float dotABAB = dot(vertex1 - vertex0, vertex1 - vertex0);
	float dotABAC = dot(vertex1 - vertex0, vertex2 - vertex0);
	float dotACAC = dot(vertex2 - vertex0, vertex2 - vertex0);

	float d = 2.0f * (dotABAB * dotACAC - dotABAC * dotABAC);

	float3 referencePoint = vertex0;

	float3 sphereCenter;
	float sphereRadius;

	if(abs(d) <= epsilon) {

		// a, b, and c lie on a line. 
		// Sphere center is the middle point of the largest side.
		// Sphere radius is the half the lenght of the largest side.

		float distanceAB = length(vertex0 - vertex1);
		float distanceBC = length(vertex1 - vertex2);
		float distanceCA = length(vertex2 - vertex0);

		if(distanceAB > distanceBC) {
		
			if(distanceAB > distanceCA) {

				sphereCenter = (vertex0 + vertex1) * 0.5f;
				sphereRadius = distanceAB * 0.5f;
			}
			else {

				sphereCenter = (vertex0 + vertex2) * 0.5f;
				sphereRadius = distanceCA * 0.5f;
			}
		}
		else {
			
			if(distanceBC > distanceCA)  {

				sphereCenter = (vertex1 + vertex2) * 0.5f;
				sphereRadius = distanceBC * 0.5f;
			}
			else {

				sphereCenter = (vertex0 + vertex2) * 0.5f;
				sphereRadius = distanceCA * 0.5f;
			}
		}
		
		return make_float4(sphereCenter, sphereRadius);
	} 
	else {

		float s = (dotABAB * dotACAC - dotACAC * dotABAC) / d;
		float t = (dotACAC * dotABAB - dotABAB * dotABAC) / d;

		// s controls height over AC, t over AB, (1-s-t) over BC
		if (s <= 0.0f) {
			sphereCenter = (vertex0 + vertex2) * 0.5f;
		} 
		else if (t <= 0.0f) {
			sphereCenter = (vertex0 + vertex1) * 0.5f;
		} 
		else if (s + t >= 1.0f) {
			sphereCenter = (vertex1 + vertex2) * 0.5f;
			referencePoint = vertex1;
		} 
		else 
			sphereCenter = vertex0 + s * (vertex1 - vertex0) + t * (vertex2 - vertex0);

		sphereRadius = length(sphereCenter - referencePoint);

		return make_float4(sphereCenter, sphereRadius);
	}
}

// Hierarchy Creation Code
__device__ static inline float4 CreateHierarchyCone0(const float4 &cone, const float4 &ray) {

	float3 coneDirection = make_float3(cone);
	float3 rayDirection = make_float3(ray);

	float spread = acos(dot(coneDirection, rayDirection));

	if(cone.w > spread)
		return cone;

	float3 q = normalize(dot(coneDirection, rayDirection) * coneDirection - rayDirection);
	float3 e = coneDirection * cos(cone.w) + q * sin(cone.w);
	
	float3 newConeDirection = normalize(rayDirection + e);
	float newConeSpread = acos(dot(newConeDirection, rayDirection));

	return make_float4(newConeDirection.x, newConeDirection.y, newConeDirection.z, newConeSpread);

	/*float3 coneDirection1 = make_float3(cone);
	float3 coneDirection2 = make_float3(ray);
	
	float spread = acos(dot(coneDirection1, coneDirection2)) * 0.5f;

	float3 coneDirection = normalize(coneDirection1 + coneDirection2);
	float coneSpread = clamp(spread + max(cone.w, ray.w), 0.0f, HALF_PI);

	return make_float4(coneDirection.x, coneDirection.y, coneDirection.z, coneSpread);*/
}

__device__ static inline float4 CreateHierarchyConeN(const float4 &cone1, const float4 &cone2) {

	float3 coneDirection1 = make_float3(cone1);
	float3 coneDirection2 = make_float3(cone2);
	
	float spread = acos(dot(coneDirection1, coneDirection2)) * 0.5f;

	/*if(cone1.w > spread + cone2.w)
		return cone1;

	if(cone2.w > spread + cone1.w)
		return cone2;*/
	
	float3 coneDirection = normalize(coneDirection1 + coneDirection2);
	float coneSpread = clamp(spread + max(cone1.w, cone2.w), 0.0f, HALF_PI);

	return make_float4(coneDirection.x, coneDirection.y, coneDirection.z, coneSpread); 
}

__device__ static inline float4 CreateHierarchySphere(const float4 &sphere1, const float4 &sphere2) {

	float3 sphereCenter1 = make_float3(sphere1);
	float3 sphereCenter2 = make_float3(sphere2);

	float3 sphereDirection = normalize(sphereCenter2 - sphereCenter1);
	float sphereDistance = length(sphereCenter2 - sphereCenter1);

	if(sphereDistance + sphere2.w <= sphere1.w)
		return sphere1;

	if(sphereDistance + sphere1.w <= sphere2.w)
		return sphere2;

	//float3 sphereCenter = sphereCenter1 + sphereDirection * sphereDistance * 0.5f;
	float3 sphereCenter = (sphereCenter1 + sphereCenter2) * 0.5f;
	float sphereRadius = sphereDistance * 0.5f + max(sphere1.w , sphere2.w);

	return make_float4(sphereCenter.x, sphereCenter.y, sphereCenter.z, sphereRadius);
}

__global__ void UpdateVertex(
							// Input Array containing the Updated Model Matrices.
							float* modelMatricesArray,
							// Input Array containing the Updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Vertex Total.
							const unsigned int vertexTotal,
							// Output Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Output Array containing the Updated Triangle Normals.
							float4* triangleNormalsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= vertexTotal)
		return;

	// Matrices ID
	unsigned int matrixID = tex1Dfetch(triangleObjectIDsTexture, x).x;

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

__global__ void UpdateBoundingSphere(
							// Input Array containing the updated Translation Matrices.
							float* translationMatricesArray,
							// Input Array containing the updated Scale Matrices.
							float* scaleMatricesArray,
							// Auxiliary Variable containing the Bounding Sphere Total.
							const unsigned int boundingSphereTotal,
							// Output Array containing the updated Bounding Spheres.
							float3* boundingSphereArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= boundingSphereTotal)
		return;

	// Matrices ID
	unsigned int matrixID = x;

	// Translation Matrix - Multiply the Bounding Sphere Center by it.
	float translationMatrix[16];

	for(int i=0; i<16; i++)
		translationMatrix[i] = translationMatricesArray[matrixID * 16 + i];
	
	// Scale Matrix - Multiply the Radius by it.
	float scaleMatrix[16];

	for(int i=0; i<16; i++)
		scaleMatrix[i] = scaleMatricesArray[matrixID * 16 + i];
	
	// Load the Bounding Sphere Center
	float4 center = tex1Dfetch(boundingSpheresTexture, x * 2);
	// Load the Bounding Sphere Radius and Bounds
	float4 radiusAndBounds = tex1Dfetch(boundingSpheresTexture, x * 2 + 1);

	float updatedCenter[4];

	for(int i=0; i<4; i++) {

		updatedCenter[i] = 0.0f;
		updatedCenter[i] += translationMatrix[i * 4 + 0] * center.x;
		updatedCenter[i] += translationMatrix[i * 4 + 1] * center.y;
		updatedCenter[i] += translationMatrix[i * 4 + 2] * center.z;
		updatedCenter[i] += translationMatrix[i * 4 + 3] * 1.0f;
	}

	float maximumScale = FLT_MIN;

	for(int i=0; i<3; i++)
		if(abs(scaleMatrix[i * 4 + i]) > maximumScale)
			maximumScale = abs(scaleMatrix[i * 4 + i]);

	float updatedRadius = radiusAndBounds.x * maximumScale;

	// Store the updated Bounding Sphere.
	boundingSphereArray[x * 2] = make_float3(updatedCenter[0], updatedCenter[1], updatedCenter[2]);
	boundingSphereArray[x * 2 + 1] = make_float3(updatedRadius, radiusAndBounds.y, radiusAndBounds.z);
}

__global__ void PrepareArray(	
							// Input Variable containing the Preparation Value
							const unsigned int value,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int arraySize,
							// Output Array to be prepared.
							unsigned int* preparedArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= arraySize)
		return;

	preparedArray[x] = value;
}

__global__ void PrepareArray2(	
							// Input Variable containing the Preparation Value
							const unsigned int value,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth,
							const unsigned int windowHeight,
							// Output Array to be prepared.
							unsigned int* preparedArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowHeight || y >= windowWidth)
		return;

	preparedArray[x + y * windowWidth] = value;
}

__global__ void TextureDebug(	
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	//float3 fragmentColor = make_float3(tex2D(diffuseTexture, x, y));

	//float3 fragmentColor = make_float3(tex2D(specularTexture, x, y));

	float3 fragmentColor = make_float3(tex2D(fragmentPositionTexture, x, y));
	fragmentColor = normalize(fragmentColor);

	//float3 fragmentColor = make_float3(tex2D(fragmentNormalTexture, x, y));
	fragmentColor += make_float3(1.0f);
	fragmentColor *= 0.5f;

	pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
}

__global__ void Debug(	
							// Input Array containing the Rays.
							float3* rayArray,
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayKeysArray, 
							unsigned int* rayValuesArray,
							// Input Array containing the updated Bounding Spheres.
							float3* boundingSphereArray,
							// Auxiliary Variable containing the Bounding Box Total.
							const unsigned int boundingSphereTotal,
							// Auxiliary Variable containing the Bounding Box Total.
							const unsigned int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraDirection,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraUp,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraRight,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	float3 fragmentColor = make_float3(tex2D(diffuseTexture, x,y));

	// Ray Creation
	float3 rayOrigin = cameraPosition;
	float3 rayDirection = normalize(cameraDirection + cameraRight * ((float)x / (float)windowWidth - 0.5f) + cameraUp * ((float)y / (float)windowHeight - 0.5f));

	float3 boundingSphereCenter;
	float3 boundingSphereBounds;

	// Bounding Sphere Loading 
	boundingSphereCenter = boundingSphereArray[((rayTotal) % boundingSphereTotal) * 2];
	boundingSphereBounds = boundingSphereArray[((rayTotal) % boundingSphereTotal) * 2 + 1];

	if(RaySphereIntersection(Ray(rayOrigin, rayDirection), make_float3(boundingSphereCenter.x, boundingSphereCenter.y, boundingSphereCenter.z), boundingSphereBounds.x * 10.0f) > 0.0f)
		fragmentColor += make_float3(1.0f, 0.5f, 0.0f);

	pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
}

__global__ void CreateShadowRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Light Total.
							const unsigned int lightTotal,
							// Output Array containing the unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray) {

	unsigned int threadID = blockIdx.x*blockDim.x + threadIdx.x;

	if(threadID >= windowWidth * windowHeight * lightTotal)
		return;

	unsigned int x = threadID % windowWidth;
	unsigned int y = threadID / (windowWidth * lightTotal);
	unsigned int l = (threadID / windowWidth) % lightTotal;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {

		// Calculate the Shadow Rays Origin and Direction
		float3 shadowRayOrigin = make_float3(tex1Dfetch(lightPositionsTexture, l));
		float3 shadowRayDirection = normalize(fragmentPosition - shadowRayOrigin);

		// Diffuse Factor (Negate the Normal because the Ray Origin is reversed)
		float diffuseFactor = max(dot(shadowRayDirection, -fragmentNormal), 0.0f);
		clamp(diffuseFactor, 0.0f, 1.0f);

		// Store the Shadow Rays its direction
		if(diffuseFactor > 0.0f) {
			
			// Store the Shadow Rays Origin
			rayArray[threadID * 2] = shadowRayOrigin;
			// Store the Shadow Rays Direction
			rayArray[threadID * 2 + 1] = shadowRayDirection;

			// Store the Shadow Rays Hash Key
			rayIndexKeysArray[threadID] = CreateShadowRayIndex(shadowRayOrigin, shadowRayDirection, l);
			// Store the Shadow Rays Index Value
			rayIndexValuesArray[threadID] = threadID;

			// Store the Shadow Rays Flag (Trimming)
			headFlagsArray[threadID] = 0;

			return;
		}
	}
			
	// Store the Shadow Rays Flag (Trimming)
	headFlagsArray[threadID] = 1;
}

__global__ void CreateReflectionRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Cameras World Space Position.
							float3 cameraPosition,
							// Input Array containing the Unsorted Rays
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	unsigned int rayIndex = x + y * windowWidth * LIGHT_SOURCE_MAXIMUM;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {
		
		// Calculate the Reflection Rays Position and Direction
		float3 reflectionRayOrigin = fragmentPosition;
		float3 reflectionRayDirection = reflect(normalize(reflectionRayOrigin-cameraPosition), normalize(fragmentNormal));
		
		// Store the Reflection Rays Origin
		rayArray[rayIndex * 2] = reflectionRayOrigin;
		// Store the Reflection Rays Direction
		rayArray[rayIndex * 2 + 1] = reflectionRayDirection;
		
		// Store the Reflection Rays Hash Key
		rayIndexKeysArray[rayIndex] = CreateReflectionRayIndex(reflectionRayOrigin, reflectionRayDirection);
		// Store the Reflection Rays Index Value
		rayIndexValuesArray[rayIndex] = rayIndex;
		
		// Store the Reflection Rays Flag (Trimming)
		headFlagsArray[rayIndex] = 0;

		return;
	}

	// Store the Reflection Rays Flag (Trimming)
	headFlagsArray[rayIndex] = 1;
}

__global__ void CreateRefractionRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Cameras World Space Position.
							float3 cameraPosition,
							// Input Array containing the Unsorted Rays
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	unsigned int rayIndex = x + y * windowWidth * LIGHT_SOURCE_MAXIMUM;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {
		
		// Calculate the Refraction Rays Position and Direction
		float3 refractionRayOrigin = fragmentPosition;
		float3 refractionRayDirection = refract(normalize(refractionRayOrigin-cameraPosition), normalize(fragmentNormal), 1.0f / 1.52f);
		
		// Store the Refraction Rays Origin
		rayArray[rayIndex * 2] = refractionRayOrigin;
		// Store the Refraction Rays Direction
		rayArray[rayIndex * 2 + 1] = refractionRayDirection;
		
		// Store the Refraction Rays Hash Key
		rayIndexKeysArray[rayIndex] = CreateRefractionRayIndex(refractionRayOrigin, refractionRayDirection);
		// Store the Refraction Rays Index Value
		rayIndexValuesArray[rayIndex] = rayIndex;
		
		// Store the Refraction Rays Flag (Trimming)
		headFlagsArray[rayIndex] = 0;

		return;
	}

	// Store the Refraction Rays Flag (Trimming)
	headFlagsArray[rayIndex] = 1;
}

__global__ void CreateTrimmedRays(	
							// Input Arrays containing the Untrimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray,
							// Auxiliary Variable containing the Screen Dimensions.
							const unsigned int screenDimensions,
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* inclusiveScanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= screenDimensions)
		return;

	// First Ray
	if(x == 0) {
		
		if(inclusiveScanArray[0] == 0) {

			trimmedRayIndexKeysArray[0] = rayIndexKeysArray[0];
			trimmedRayIndexValuesArray[0] = rayIndexValuesArray[0];
		}
	}
	// Remaining Rays
	else {
	
		// Current and Previous Offsets taken from the Inclusive Sum Array. 
		unsigned int currentOffset = inclusiveScanArray[x];
		unsigned int previousOffset = inclusiveScanArray[x - 1];

		// Equal Offsets means that the Ray should be shifted to the left.
		if(currentOffset == previousOffset) {

			trimmedRayIndexKeysArray[x - currentOffset] = rayIndexKeysArray[x];
			trimmedRayIndexValuesArray[x - currentOffset] = rayIndexValuesArray[x];
		}
	}
}
	
__global__ void CreateChunkFlags(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const int rayTotal,
							// Output Array containing the Chunk Head Flags.
							unsigned int* headFlagsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= rayTotal)
		return;

	// First Chunk
	if(x == 0) {

		headFlagsArray[x] = 1;
	}
	// Remaining Chunks
	else {
		
		// Current and Previous Hashes taken from the Trimmed Ray Keys Array
		unsigned int currentKey = trimmedRayIndexKeysArray[x];
		unsigned int previousKey = trimmedRayIndexKeysArray[x - 1];
		
		// Different Keys means that a new Chunk should be created.
		if(currentKey != previousKey)
			headFlagsArray[x] = 1;
		else
			headFlagsArray[x] = 0;
	}
}

__global__ void CreateChunkBases(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Array containing the Chunk Head Flags.
							unsigned int* headFlagsArray, 
							// Auxiliary Array containing the Exclusive Scan Output.
							unsigned int* scanArray, 
							// Output Array containing the Ray Chunk Bases.
							unsigned int* chunkBasesArray,
							// Output Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= rayTotal)
		return;
		
	// Head Flags containing 0 indicate there's no new Chunk to be created.
	if(headFlagsArray[x] == 0)
		return;

	// Store the Position of the new Chunk.
	unsigned int position = scanArray[x] - 1;

	// Store the Ray Index Base for the Chunk.
	chunkBasesArray[position] = x; 
	
	// Store the Ray Hash and the Chunk Position for the Chunk
	chunkIndexKeysArray[position] = trimmedRayIndexKeysArray[x];
	chunkIndexValuesArray[position] = position;
}

__global__ void CreateChunkSizes(
							// Input Array containing the Ray Chunk Bases.
							unsigned int* chunkBasesArray,
							// Auxiliary Variable containing the Chunk Total.
							const unsigned int chunkTotal,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Output Array containing the Ray Chunk Sizes.
							unsigned int* chunkSizesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	// Last Chunk
	if(x == chunkTotal - 1) {

		// Chunk Bases
		unsigned int currentBase = chunkBasesArray[x];
		unsigned int nextBase = rayTotal;
	
		// Store the Chunk Sizes based on the Chunk Base of the Current Chunk and the Ray Total.
		chunkSizesArray[x] = nextBase - currentBase;
	}
	// Remaining Chunks
	else {
		
		// Chunk Bases
		unsigned int currentBase = chunkBasesArray[x];
		unsigned int nextBase = chunkBasesArray[x+1];

		// Store the Chunk Sizes based on the Chunk Base of the Current and the Next Chunks.
		chunkSizesArray[x] = nextBase - currentBase;
	}
}

__global__ void CreateSortedRaySkeleton(
							// Input Array containing the Ray Chunk Sizes.
							unsigned int* chunkSizesArray,
							// Input Array containing the Ray Chunk Values [Values = Positions].
							unsigned int* sortedChunkValuesArray,
							// Auxiliary Variable containing the Chunk Total.
							const unsigned int chunkTotal,
							// Output Array containing the Sorted Ray Arrays Skeleton.
							unsigned int* skeletonArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;
	
	// Store the Sorted Ray Arrays Skeleton based on the Sorted Chunk Value (Position).
	skeletonArray[x] = chunkSizesArray[sortedChunkValuesArray[x]];
}

__global__ void CreateSortedRays(
							// Input Array containing the Ray Chunk Bases.
							unsigned int* chunkBasesArray,
							// Input Array containing the Ray Chunk Sizes.
							unsigned int* chunkSizesArray,
							// Input Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							unsigned int* sortedChunkKeysArray,
							unsigned int* sortedChunkValuesArray,
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Input Array containing the Exclusive Scan Output.
							unsigned int* scanArray, 
							// Input Array containing the Sorted Ray Arrays Skeleton.
							unsigned int* skeletonArray,
							// Auxiliary Variable containing the Chunk Total.
							const unsigned int chunkTotal,
							// Output Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	// Store the Chunk Key and Value.
	unsigned int chunkValue = sortedChunkValuesArray[x];
	
	// Store the Chunk Base and Size.
	unsigned int chunkBase = chunkBasesArray[chunkValue];
	unsigned int chunkSize = chunkSizesArray[chunkValue];

	// Store the Ray starting and final Positions.
	unsigned int startingPosition = scanArray[x];
	unsigned int finalPosition = startingPosition + chunkSize;

	// First Ray.
	sortedRayIndexKeysArray[startingPosition] = trimmedRayIndexKeysArray[chunkBase];
	sortedRayIndexValuesArray[startingPosition] = trimmedRayIndexValuesArray[chunkBase];

	// Remaining Rays.
	for(int i=startingPosition+1, j=1; i<finalPosition; i++, j++) {

		sortedRayIndexKeysArray[i] = trimmedRayIndexKeysArray[chunkBase + j];
		sortedRayIndexValuesArray[i] = trimmedRayIndexValuesArray[chunkBase + j];
	}
}

__global__ void CreateHierarchyLevel0(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray,
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Auxiliary Variable containing the Node Total.
							const unsigned int nodeTotal,
							// Auxiliary Variable containing the Initial Sphere Radius.
							const float initialRadius,
							// Auxiliary Variable containing the Initial Cone Spread.
							const float initialSpread,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	// Ray Origins are stored in the first offset
	float4 sphere = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2], initialRadius);
	// Ray Directions are stored in the second offset
	float4 cone = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2 + 1], initialSpread);
	
	for(unsigned int i=1; i<HIERARCHY_SUBDIVISION; i++) {

		if(rayTotal * 2 < (x * HIERARCHY_SUBDIVISION + i) * 2)
			break;

		// Ray Origins are stored in the first offset
		float4 rayOrigin = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION + i] * 2], initialRadius);
		// Ray Directions are stored in the second offset
		float4 rayDirection = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION + i] * 2 + 1], initialSpread);
		
		// Combine the Ray Sphere with the existing Node Sphere
		sphere = CreateHierarchySphere(sphere, rayOrigin);
		// Combine the Ray Cone with the existing Node Cone
		cone = CreateHierarchyCone0(cone, rayDirection);
	}

	// Store the Node Sphere
	hierarchyArray[x * 2] = sphere;
	// Store the Node Cone
	hierarchyArray[x * 2 + 1] = cone;
}

__global__ void CreateHierarchyLevelN(	
							// Input and Output Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Auxiliary Variable containing the Write Node Index.
							const unsigned int nodeWriteOffset,
							// Auxiliary Variable containing the Read Node Index.
							const unsigned int nodeReadOffset,
							// Auxiliary Variable containing the Node Total.
							const unsigned int nodeTotal) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	// Ray Origins are stored in the first offset
	float4 sphere = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION) * 2];
	// Ray Directions are stored in the second offset
	float4 cone = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION) * 2 + 1];
	
	for(unsigned int i=1; i<HIERARCHY_SUBDIVISION; i++) {

		if(nodeWriteOffset * 2 <= (nodeReadOffset + x * HIERARCHY_SUBDIVISION + i) * 2)
			break;
		
		// Ray Origins are stored in the first offset
		float4 currentSphere = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION + i) * 2];
		// Ray Directions are stored in the second offset
		float4 currentCone = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION + i) * 2 + 1];
		
		// Combine the new Node Sphere with the current Node Sphere
		sphere = CreateHierarchySphere(sphere, currentSphere);
		// Combine the new Node Cone with the current Node Cone
		cone = CreateHierarchyConeN(cone, currentCone);
	}
	
	// Store the Node Sphere
	hierarchyArray[(nodeWriteOffset + x) * 2] = sphere;
	// Store the Node Cone
	hierarchyArray[(nodeWriteOffset + x) * 2 + 1] = cone;
}

__global__ void CalculateBoundingSpheresIntersections(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the updated Bounding Spheres.
							float3* boundingSphereArray,
							// Auxiliary Variable containing the Bounding Sphere Total.
							const unsigned int boundingSphereTotal,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Output Array containing the Head Flags.
							unsigned int* headFlagsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / boundingSphereTotal;
	unsigned int boundingSphereID = x % boundingSphereTotal;

	if(nodeID >= nodeReadTotal || boundingSphereID >= boundingSphereTotal)
		return;

	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];

	// Sphere Center
	float3 center = boundingSphereArray[boundingSphereID * 2];
	// X = Sphere Radius Y = Objects Triangle Starting Index Z = Objects Triangle Final Index
	float3 radiusAndBounds = boundingSphereArray[boundingSphereID * 2 + 1];

	// Calculate the Intersection and store the result
	bool result = SphereNodeIntersection(sphere, cone, make_float4(center.x, center.y, center.z, radiusAndBounds.x), cos(cone.w), tan(cone.w));

	unsigned int lowerBound = triangleOffset;
	unsigned int upperBound = triangleOffset + triangleTotal;

	if(result == true) {

		// Interval completelly covered
		if(radiusAndBounds.y >= lowerBound && radiusAndBounds.y < upperBound && radiusAndBounds.z >= lowerBound && radiusAndBounds.z < upperBound) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal + (unsigned int)radiusAndBounds.y - lowerBound], 
			// Interval eg: (9009 - 8950) + 1 = 60
			(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1);

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Hits] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCompletelly Covered\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1, (unsigned int)radiusAndBounds.y - lowerBound, lowerBound, upperBound);*/
		}

		// True and Interval partially covered under
		else if(radiusAndBounds.y < lowerBound && radiusAndBounds.z >= lowerBound && radiusAndBounds.z < upperBound) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal], 
			// Interval minus the Offset eg: (9009 - 9000) + 1 = 10
			(unsigned int)radiusAndBounds.z - lowerBound + 1);

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Hits] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCovered Under\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - lowerBound + 1, 0, lowerBound, upperBound);*/
		}

		// True and Interval partially covered over
		else if(radiusAndBounds.y >= lowerBound && radiusAndBounds.y < upperBound && radiusAndBounds.z >= upperBound) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal + (unsigned int)radiusAndBounds.y - lowerBound], 
			// Interval minus the Overflow Interval eg: (9009 - 8950) + 1 - (9009 - (8000+1000) + 1) = 50
			triangleTotal);

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Hits] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCovered Over\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1 - ((unsigned int)radiusAndBounds.z - upperBound + 1), (unsigned int)radiusAndBounds.y - lowerBound, lowerBound, upperBound);*/
		}

		// Huge Mesh
		else if(radiusAndBounds.y < lowerBound && radiusAndBounds.z > upperBound) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal], 
			// Interval minus the Overflow Interval eg: (9009 - 8950) + 1 - (9009 - (8000+1000) + 1) = 50
			(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1 - ((unsigned int)radiusAndBounds.z - upperBound + 1));

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Hits] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCovered Over\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1 - ((unsigned int)radiusAndBounds.z - upperBound + 1), (unsigned int)radiusAndBounds.y - lowerBound, lowerBound, upperBound);*/
		}
	}
	else {

		// False and Interval completelly covered
		if(radiusAndBounds.y >= lowerBound && radiusAndBounds.y < upperBound && radiusAndBounds.z >= lowerBound && radiusAndBounds.z + 1 < upperBound) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal + (unsigned int)radiusAndBounds.z - lowerBound + 1], 
			// Interval eg: (9009 - 8950) + 1 = 60
			(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1);

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Miss] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCompletelly Covered\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1, (unsigned int)radiusAndBounds.z - lowerBound + 1, lowerBound, upperBound);*/
		}

		// False and Interval partially covered under
		else if(radiusAndBounds.y < triangleOffset && radiusAndBounds.z >= triangleOffset && (radiusAndBounds.z + 1) < (triangleOffset + triangleTotal)) {

			atomicAdd(&headFlagsArray[nodeID * triangleTotal + (unsigned int)radiusAndBounds.z - lowerBound + 1], 
			// Interval minus the Offset eg: (9009 - 9000) + 1 = 10
			(unsigned int)radiusAndBounds.z - lowerBound + 1);

			/*if(nodeID == 1 && triangleOffset == 10000)
			printf("[Miss] Node ID: %u\tBounding Box ID: %u\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCovered Under\n",  
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - lowerBound + 1, (unsigned int)radiusAndBounds.z - lowerBound + 1, lowerBound, upperBound);*/
		}
		else {

			/*printf("[Miss] Node ID: %u\tBounding Box ID: %u\t\tMinimum: %u\tMaximum: %u\tWritting: %u\tWritting to: %u\tLower Bound: %u\tUpper Bound: %u\tCovered Over\n", 
				nodeID, boundingSphereID, (unsigned int)radiusAndBounds.y, (unsigned int)radiusAndBounds.z, 
				(unsigned int)radiusAndBounds.z - (unsigned int)radiusAndBounds.y + 1 - ((unsigned int)radiusAndBounds.z - upperBound + 1), (unsigned int)radiusAndBounds.y - lowerBound, lowerBound, upperBound);*/
		}
	}
}

__global__ void CreateHierarchyBoundingSphereHits(
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Input Array containing the Inclusive Scan Output.
							unsigned int* scanArray,
							// Output Array containing the Head Flags.
							unsigned int* headFlagsArray,
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / triangleTotal;
	unsigned int triangleID = x % triangleTotal;

	if(nodeID >= nodeReadTotal || triangleID >= triangleTotal)
		return;

	int offset = scanArray[x];

	if(nodeID > 0)
		offset -= scanArray[nodeID * triangleTotal - 1];

	hierarchyHitsArray[x] = (offset > triangleID) ? CreateHit(nodeID, triangleID) : 0;
	headFlagsArray[x] = (offset > triangleID) ? 0 : 1;
}

__global__ void CreateCRSHHierarchyLevel0Hits(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Hit Total.
							const unsigned int hitTotal,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Output Array containing the Head Flags Output.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							unsigned int* trimmedHierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	unsigned int hit = trimmedHierarchyHitsArray[x];

	unsigned int nodeID = ExtractNodeID(hit);
	unsigned int triangleID = ExtractTriangleID(hit);

	// Load and the Triangle and Create the Triangle Bounding Sphere
	float4 triangle = CreateTriangleBoundingSphere(
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]));

	// Load the Hierarchy Node
	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];
	
	// Calculate the Intersection and store the result
	headFlagsArray[x] = (SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) ? 0 : 1;
	hierarchyHitsArray[x] = hit;
}

__global__ void CreateRAHHierarchyLevel0Hits(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Hit Total.
							const unsigned int hitTotal,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Output Array containing the Head Flags Output.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							unsigned int* trimmedHierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	unsigned int hit = CreateHit(x / triangleTotal, x % triangleTotal);

	unsigned int nodeID = ExtractNodeID(hit);
	unsigned int triangleID = ExtractTriangleID(hit);

	// Load and the Triangle and Create the Triangle Bounding Sphere
	float4 triangle = CreateTriangleBoundingSphere(
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]));

	// Load the Hierarchy Node
	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];
	
	// Calculate the Intersection and store the result
	headFlagsArray[x] = (SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) ? 0 : 1;
	hierarchyHitsArray[x] = hit;
}

__global__ void CreateHierarchyLevelNHits(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Hit Total.
							const unsigned int hitTotal,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Write Total.
							const unsigned int nodeWriteTotal,
							// Output Array containing the Inclusive Scan Output.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							unsigned int* trimmedHierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	unsigned int hit = trimmedHierarchyHitsArray[x];

	unsigned int nodeID = ExtractNodeID(hit);
	unsigned int triangleID = ExtractTriangleID(hit);

	// Load and the Triangle and Create the Triangle Bounding Sphere
	float4 triangle = CreateTriangleBoundingSphere(
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]),
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]));

	for(unsigned int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		if((nodeID * HIERARCHY_SUBDIVISION + i) < nodeWriteTotal) {

			// Load the Hierarchy Node
			float4 sphere = hierarchyArray[(nodeOffset + nodeID * HIERARCHY_SUBDIVISION + i) * 2];
			float4 cone = hierarchyArray[(nodeOffset + nodeID * HIERARCHY_SUBDIVISION + i) * 2 + 1];
	
			// Calculate the Intersection and store the result
			headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = (SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) ? 0 : 1;
			hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] =  CreateHit(nodeID * HIERARCHY_SUBDIVISION + i, triangleID);

			continue;
		}

		headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = 1;
		hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] = CreateHit(nodeID * HIERARCHY_SUBDIVISION + i, triangleID);
	}
}

__global__ void CreateTrimmedHierarchyHits(
							// Input Array containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray,
							// Input Array containing the Inclusive Scan Output.
							const unsigned int* scanArray, 
							// Auxiliary Variable containing the Hit Total.
							const unsigned int hitTotal,
							// Output Array containing the Trimmed Ray Hierarchy Hits.
							unsigned int* trimmedHierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// First Hit
	if(x == 0) {

		if(scanArray[0] == 0)
			trimmedHierarchyHitsArray[0] = hierarchyHitsArray[0];
	}
	// Remaining Hits
	else {
		
		// Equal Offsets means that the Ray should be shifted to the left.
		unsigned int currentOffset = scanArray[x];
		unsigned int previousOffset = scanArray[x - 1];
		
		// Equal Offsets means that the Hit should be shifted to the left.
		if(currentOffset == previousOffset)
			trimmedHierarchyHitsArray[x- currentOffset] = hierarchyHitsArray[x];
	}
}

__global__ void CalculateShadowRayIntersections(	
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
							unsigned int* shadowFlagsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Load the Hierarchy Hit
	unsigned int hit = hierarchyHitsArray[x];

	unsigned int nodeID = ExtractNodeID(hit);
	unsigned int triangleID = ExtractTriangleID(hit);

	// Load the Triangles Vertices and Edges
	float3 vertex0 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]);
	float3 edge1 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]) - vertex0;
	float3 edge2 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]) - vertex0;

	for(unsigned int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		// Check if the Extrapolated Ray exists.
		if(nodeID * HIERARCHY_SUBDIVISION + i >= rayTotal)
			return;

		// Fetch the Shadow Ray Index
		unsigned int rayIndex = sortedRayIndexValuesArray[nodeID * HIERARCHY_SUBDIVISION + i];

		// Fetch the Shadow Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		#ifdef SOFT_SHADOWS
			// Shadow Grid Axis
			float3 shadowAxis[2];

			// Check which is the component with the smallest coeficient
			float minimum = min(abs(rayDirection.x),max(abs(rayDirection.y),abs(rayDirection.z)));

			// Calculate the perpendicular plane base vectors
			if(abs(rayDirection.x) == minimum)
				shadowAxis[0] = cross(make_float3(1.0f,0.0f,0.0f), rayDirection);
			else if(abs(rayDirection.y) == minimum)
				shadowAxis[0] = cross(make_float3(0.0f,1.0f,0.0f), rayDirection);
			else if(abs(rayDirection.z) == minimum)
				shadowAxis[0] = cross(make_float3(0.0f,0.0f,1.0f), rayDirection);
		
			shadowAxis[1] = cross(rayDirection, shadowAxis[0]);
		#endif

		// Fetch the Fragment Position - Sent from the OpenGL Rasterizer
		float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, rayIndex % windowWidth, rayIndex / (windowWidth * LIGHT_SOURCE_MAXIMUM)));

		#ifdef SOFT_SHADOWS

			// Intersection Record
			unsigned int intersectionRecord = 0;

			// Compute the Shadow Grid
			for(int i=0, a=0; i<4; i++, a++) {

				// Ignore the Center Coordinate
				if(i==2)
					a++;

				for(int j=0, b=0; j<4; j++, b++) {

					// Ignore the Center Coordinate
					if(j==2)
						b++;

					// Calculate the Interpolated Shadow Ray Position
					float3 interpolatedPosition = rayOrigin + shadowAxis[0] * (float)(a - 2) * 0.125f + shadowAxis[1] * (float)(b - 2) * 0.125f;
					// Calculate the Interpolated Shadow Ray Direction
					float3 interpolatedDirection = normalize(fragmentPosition - interpolatedPosition);

					// Calculate the Interesection Time
					float intersectionDistance = RayTriangleIntersection(Ray(interpolatedPosition + interpolatedDirection * epsilon, interpolatedDirection), vertex0, edge1, edge2);

					// Calculate the Lights Distance to the Fragment
					if(intersectionDistance > epsilon && intersectionDistance < length(interpolatedPosition - fragmentPosition) - epsilon * 2.0f)
						intersectionRecord = intersectionRecord | (3 << (2 * (i + j * 4)));
				}
			}

			atomicOr(&shadowFlagsArray[rayIndex], intersectionRecord);
		#else

			// Calculate the Interesection Time
			float intersectionDistance = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, edge1, edge2);

			// Calculate the Lights Distance to the Fragment
			if(intersectionDistance > epsilon && intersectionDistance < length(rayOrigin - fragmentPosition) - epsilon * 2.0f)
				shadowFlagsArray[rayIndex] = UINT_MAX;
		#endif
	}
}

__global__ void ColorPrimaryShadowRay(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f, 0.0f, 0.0f);

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = make_float3(tex2D(fragmentNormalTexture, x,y));

	if(length(fragmentPosition) != 0.0f) {

		// Triangle Material Properties
		float4 fragmentDiffuseColor = tex2D(diffuseTexture, x,y);
		float4 fragmentSpecularColor = tex2D(specularTexture, x,y);

		for(unsigned int l = 0; l < lightTotal; l++) {

			// Load the Intersection Record
			unsigned int intersectionRecord = shadowFlagsArray[(x + y * windowWidth * LIGHT_SOURCE_MAXIMUM) + (windowWidth) * l];

			// Check if the Light is Blocked
			if(intersectionRecord != UINT_MAX) {

					// Light Direction
					float3 lightDirection = make_float3(tex1Dfetch(lightPositionsTexture, l)) - fragmentPosition;
					// Light Distance
					float lightDistance = length(lightDirection);
					// Normalize the Light Direction
					lightDirection = normalize(lightDirection);

					// Light Color
					float3 lightColor = make_float3(tex1Dfetch(lightColorsTexture, l));
					// Light Intensity (x = diffuse, y = specular)
					float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
					// Light Attenuation
					float attenuation = 1.0f / (0.75f + lightDistance * 0.0005f + lightDistance * lightDistance * 0.00005f);

					// Calculate the Diffuse Factor
					float diffuseFactor = clamp(max(dot(lightDirection, fragmentNormal), 0.0f), 0.0f, 1.0f);

					if(diffuseFactor > 0.0f) {

						// Blinn-Phong approximation Halfway Vector
						float3 halfwayVector = normalize(lightDirection - normalize(fragmentPosition - cameraPosition));
					
						// Calculate the Specular Factor
						float specularFactor = clamp(powf(max(dot(halfwayVector, fragmentNormal), 0.0f), fragmentSpecularColor.w), 0.0f, 1.0f);
					
						#ifdef SOFT_SHADOWS
							// Calculate the Shadow Factor
							float shadowFactor = 0.0f;
							for(int i=0; i<16; i++) {

								if((intersectionRecord & (0x00000003 << (i * 2))) == 0)
									if(i==5 || i==6 || i==9 || i==10)
										shadowFactor += 0.55f/16.0f;
									else if(i==0 || i==3 || i==12 || i==15)
										shadowFactor += 0.5375f/16.0f;
									else
										shadowFactor += 0.525f/16.0f;
							}
							// Diffuse Component
							fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation * log2(1.0f + shadowFactor * 2.0f);
							// Specular Component
							fragmentColor += make_float3(fragmentSpecularColor) * lightColor * specularFactor * lightIntensity.y * attenuation * log2(1.0f + shadowFactor * 2.0f);
						#else
							// Diffuse Component
							fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation;// * log2(1.0f + shadowFactor * 2.0f);
							// Specular Component
							fragmentColor += make_float3(fragmentSpecularColor) * lightColor * specularFactor * lightIntensity.y * attenuation;// * log2(1.0f + shadowFactor * 2.0f);
						#endif
					}
			}
		}
	}

	pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
}

__global__ void CalculateReflectionRayIntersections(
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
							unsigned int* intersectionTimeArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Load the Hierarchy Hit
	unsigned int hit = hierarchyHitsArray[x];

	unsigned int nodeID = ExtractNodeID(hit);
	unsigned int triangleID = ExtractTriangleID(hit);

	// Load the Triangles Vertices and Edges
	float3 vertex0 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]);
	float3 edge1 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]) - vertex0;
	float3 edge2 = make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]) - vertex0;

	for(unsigned int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		// Check if the Extrapolated Ray exists.
		if(nodeID * HIERARCHY_SUBDIVISION + i >= rayTotal)
			return;

		// Fetch the Ray Index
		unsigned int rayIndex = sortedRayIndexValuesArray[nodeID * HIERARCHY_SUBDIVISION + i];

		// Fetch the Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		float intersectionDistance = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, edge1, edge2);

		// Calculate the Intersection Time
		if(intersectionDistance > epsilon) {

			unsigned int newTime = ((unsigned int)(intersectionDistance * 10.0f + 1.0f) << 20) + (triangleOffset + triangleID);
			unsigned int oldTime = atomicMin((unsigned int*)&intersectionTimeArray[rayIndex], newTime);
		}
	}
}

__global__ void ColorReflectionRay(
							// Input Array containing the Unsorted Rays.
							float3* rayArray,
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
							// Auxiliary Array containing the Intersection Times.
							unsigned int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Reflection Ray Index
	unsigned int rayIndex = x + y * windowWidth * LIGHT_SOURCE_MAXIMUM;

	// Reflection Ray Intersection Triangle
	unsigned int intersectionRecord = (unsigned int)intersectionTimeArray[rayIndex];
	unsigned int intersectionTriangle = intersectionRecord & 0x000FFFFF;

	if(intersectionRecord != UINT_MAX) {

		// Load the Triangles Material Identifier
		int1 materialID = tex1Dfetch(triangleMaterialIDsTexture, intersectionTriangle * 3);

		// Store the Triangles Vertices and Edges
		float3 vertex0 = make_float3(trianglePositionsArray[intersectionTriangle * 3]);
		float3 vertex1 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 1]);
		float3 vertex2 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 2]);

		// Calculate the Reflection Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		// Calculate the Intersection Time
		float intersectionTime = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, vertex1 - vertex0, vertex2 - vertex0);

		// Calculate the Hit Point
		float3 fragmentPosition = rayOrigin + rayDirection * (epsilon + intersectionTime);

		// Normal calculation using Barycentric Interpolation
		float areaABC = length(cross(vertex1 - vertex0, vertex2 - vertex0));
		float areaPBC = length(cross(vertex1 - fragmentPosition, vertex2 - fragmentPosition));
		float areaPCA = length(cross(vertex0 - fragmentPosition, vertex2 - fragmentPosition));
		
		// Calculate the Hit Normal
		float3 fragmentNormal = 
			(areaPBC / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3]) + 
			(areaPCA / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 1]) + 
			(1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 2]);
		
		// Load the Triangles Material
		float4 fragmentDiffuseColor = tex1Dfetch(materialDiffusePropertiesTexture, materialID.x);
		float4 fragmentSpecularColor = tex1Dfetch(materialSpecularPropertiesTexture, materialID.x);

		for(unsigned int l = 0; l < lightTotal; l++) {

			// Light Direction
			float3 lightDirection = make_float3(tex1Dfetch(lightPositionsTexture, l)) - fragmentPosition;
			// Light Distance
			float lightDistance = length(lightDirection);
			// Normalize the Light Direction
			lightDirection = normalize(lightDirection);

			// Light Color
			float3 lightColor = make_float3(tex1Dfetch(lightColorsTexture, l));
			// Light Intensity (x = diffuse, y = specular)
			float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
			// Light Attenuation
			float attenuation = 1.0f / (0.75f + lightDistance * 0.0005f + lightDistance * lightDistance * 0.00005f);
			
			// Calculate the Diffuse Factor
			float diffuseFactor = clamp(max(dot(lightDirection, fragmentNormal), 0.0f), 0.0f, 1.0f);

			if(diffuseFactor > 0.0f) {

				// Blinn-Phong approximation Halfway Vector
				float3 halfwayVector = normalize(lightDirection - normalize(fragmentPosition - cameraPosition));
					
				// Calculate the Specular Factor
				float specularFactor = clamp(powf(max(dot(halfwayVector, fragmentNormal), 0.0f), fragmentSpecularColor.w), 0.0f, 1.0f);

				// Diffuse Component
				fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation * 0.75f;
				// Specular Component
				fragmentColor += make_float3(fragmentSpecularColor) * lightColor * specularFactor * lightIntensity.y * attenuation * 0.75f;
			}
		}

		pixelBufferObject[x + y * windowWidth] +=
			RgbToInt(fragmentColor.x * fragmentSpecularColor.w, fragmentColor.y * fragmentSpecularColor.w, fragmentColor.z * fragmentSpecularColor.w);
	}
}

__global__ void ColorAndCreateReflectionRay(
							// Input Array containing the Unsorted Rays.
							float3* rayArray,
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
							// Auxiliary Array containing the Intersection Times.
							unsigned int* intersectionTimeArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Reflection Ray Index
	unsigned int rayIndex = x + y * windowWidth * LIGHT_SOURCE_MAXIMUM;

	// Reflection Ray Intersection Triangle
	unsigned int intersectionRecord = (unsigned int)intersectionTimeArray[rayIndex];
	unsigned int intersectionTriangle = intersectionRecord & 0x000FFFFF;

	if(intersectionRecord != UINT_MAX) {

		// Load the Triangles Material Identifier
		int1 materialID = tex1Dfetch(triangleMaterialIDsTexture, intersectionTriangle * 3);

		// Store the Triangles Vertices and Edges
		float3 vertex0 = make_float3(trianglePositionsArray[intersectionTriangle * 3]);
		float3 vertex1 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 1]);
		float3 vertex2 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 2]);

		// Calculate the Reflection Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		// Calculate the Intersection Time
		float intersectionTime = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, vertex1 - vertex0, vertex2 - vertex0);

		// Calculate the Hit Point
		float3 fragmentPosition = rayOrigin + rayDirection * (epsilon + intersectionTime);

		// Normal calculation using Barycentric Interpolation
		float areaABC = length(cross(vertex1 - vertex0, vertex2 - vertex0));
		float areaPBC = length(cross(vertex1 - fragmentPosition, vertex2 - fragmentPosition));
		float areaPCA = length(cross(vertex0 - fragmentPosition, vertex2 - fragmentPosition));
		
		// Calculate the Hit Normal
		float3 fragmentNormal = 
			(areaPBC / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3]) + 
			(areaPCA / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 1]) + 
			(1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 2]);
		
		// Load the Triangles Material
		float4 fragmentDiffuseColor = tex1Dfetch(materialDiffusePropertiesTexture, materialID.x);
		float4 fragmentSpecularColor = tex1Dfetch(materialSpecularPropertiesTexture, materialID.x);

		for(unsigned int l = 0; l < lightTotal; l++) {

			// Light Direction
			float3 lightDirection = make_float3(tex1Dfetch(lightPositionsTexture, l)) - fragmentPosition;
			// Light Distance
			float lightDistance = length(lightDirection);
			// Normalize the Light Direction
			lightDirection = normalize(lightDirection);

			// Light Color
			float3 lightColor = make_float3(tex1Dfetch(lightColorsTexture, l));
			// Light Intensity (x = diffuse, y = specular)
			float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
			// Light Attenuation
			float attenuation = 1.0f / (0.75f + lightDistance * 0.0005f + lightDistance * lightDistance * 0.00005f);
			
			// Calculate the Diffuse Factor
			float diffuseFactor = clamp(max(dot(lightDirection, fragmentNormal), 0.0f), 0.0f, 1.0f);

			if(diffuseFactor > 0.0f) {

				// Blinn-Phong approximation Halfway Vector
				float3 halfwayVector = normalize(lightDirection - normalize(fragmentPosition - cameraPosition));
					
				// Calculate the Specular Factor
				float specularFactor = clamp(powf(max(dot(halfwayVector, fragmentNormal), 0.0f), fragmentSpecularColor.w), 0.0f, 1.0f);

				// Diffuse Component
				fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation * 0.75f;
				// Specular Component7
				fragmentColor += make_float3(fragmentSpecularColor) * lightColor * specularFactor * lightIntensity.y * attenuation * 0.75f;
			}
		}

		pixelBufferObject[x + y * windowWidth] +=
			RgbToInt(fragmentColor.x * fragmentSpecularColor.w, fragmentColor.y * fragmentSpecularColor.w, fragmentColor.z * fragmentSpecularColor.w);

		// Calculate the Reflection Rays Position and Direction
		float3 reflectionRayOrigin = fragmentPosition;
		float3 reflectionRayDirection = reflect(rayDirection, fragmentNormal);

		// Store the Reflection Rays Origin
		rayArray[rayIndex * 2] = reflectionRayOrigin;
		// Store the Reflection Rays Direction
		rayArray[rayIndex * 2 + 1] = reflectionRayDirection;

		// Store the Reflection Rays Hash Key
		rayIndexKeysArray[rayIndex] = CreateReflectionRayIndex(reflectionRayOrigin, reflectionRayDirection);
		// Store the Reflection Rays Index Value
		rayIndexValuesArray[rayIndex] = rayIndex;

		// Store the Reflection Rays Flag (Trimming)
		headFlagsArray[rayIndex] = 0;
	}
	else {

		// Store the Reflection Rays Flag (Trimming)
		headFlagsArray[rayIndex] = 1;
	}
}

__global__ void AntiAliasing(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Intput Array containing the Screen Buffer.
							unsigned int *primaryPixelBufferObject,
							// Output Array containing the Screen Buffer.
							unsigned int *secondaryPixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x+1 >= windowWidth || y+1 >= windowHeight || x < 1 || y < 1)
		return;

	float3 pixel = make_float3(0.0f);

	for(int i=-1; i<2; i++)
		for(int j=-1; j<2; j++)
			pixel += IntToRgb(primaryPixelBufferObject[x + (y + i) * windowWidth + i]) * 1.0f/9.0f;

	secondaryPixelBufferObject[x + y * windowWidth] = RgbToInt(pixel.x, pixel.y, pixel.z);
}

extern "C" {

	TestManager* testManager = TestManager::getInstance();

	void TriangleUpdateWrapper(
							// Input Array containing the updated Model Matrices.
							float* modelMatricesArray,
							// Input Array containing the updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Triangle Total.
							unsigned int triangleTotal,
							// Output Array containing the updated Triangle Positions.
							float4* trianglePositionsArray,
							// Output Array containing the updated Triangle Normals.
							float4* triangleNormalsArray) {
		
		unsigned int vertexTotal = triangleTotal * 3;

		// Grid based on the Triangle Count
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(vertexTotal / multiplicationBlock.x + 1);
		
		// Model and Normal Matrix Multiplication
		UpdateVertex<<<multiplicationGrid, multiplicationBlock>>>(
			modelMatricesArray, normalMatricesArray, 
			vertexTotal,
			trianglePositionsArray, triangleNormalsArray);
	}

	void BoundingSphereUpdateWrapper(
							// Input Array containing the updated Translation Matrices.
							float* translationMatricesArray,
							// Input Array containing the updated Scale Matrices.
							float* scaleMatricesArray,
							// Auxiliary Variable containing the Bounding Sphere Total.
							const unsigned int boundingSphereTotal,
							// Output Array containing the updated Bounding Spheres.
							float3* boundingSphereArray) {

		// Grid based on the Triangle Count
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(boundingSphereTotal / multiplicationBlock.x + 1);
		
		// Matrix Multiplication
		UpdateBoundingSphere<<<multiplicationGrid, multiplicationBlock>>>(
			translationMatricesArray, 
			scaleMatricesArray,
			boundingSphereTotal,
			boundingSphereArray);
	}

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
							unsigned int* sortedChunkIndexValuesArray) {

		// Number of Rays potentialy being cast per Frame
		unsigned int rayTotal = windowWidth * windowHeight * LIGHT_SOURCE_MAXIMUM;

		// Memory Allocated
		size_t allocated = 0;

		// Prepare the Scans by allocating temporary storage
		if(scanTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", 
				cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hierarchyHitMemoryTotal));

			// Allocate temporary storage for exclusive prefix scan
			Utility::checkCUDAError("cudaMalloc1()", cudaMalloc(&scanTemporaryStorage, scanTemporaryStoreBytes));
		}

		// Prepare the Radix Sort by allocating temporary storage
		if(radixSortTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceRadixSort::SortPairs()", 
				cub::DeviceRadixSort::SortPairs(
					radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
					chunkIndexKeysArray, sortedChunkIndexKeysArray,
					chunkIndexValuesArray, sortedChunkIndexValuesArray, 
					rayTotal));

			// Allocate the temporary storage
			Utility::checkCUDAError("cudaMalloc2()", cudaMalloc(&radixSortTemporaryStorage, radixSortTemporaryStoreBytes));
		}

		// Add the Scan Temporary Storage Bytes
		allocated += scanTemporaryStoreBytes;
		// Add the Radix Sort Temporary Storage Bytes
		allocated += radixSortTemporaryStoreBytes;

		size_t free, total;
		Utility::checkCUDAError("cudaGetMemInfo()", cudaMemGetInfo(&free, &total));

		printf("[Callback] Total Memory:\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
			total, total / 1024.0f, total / 1024.0f / 1024.0f, total / 1024.0f / 1024.0f / 1024.0f); 
		printf("[Callback] Free Memory:\t\t\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
			free, free / 1024.0f, free / 1024.0f / 1024.0f, free / 1024.0f / 1024.0f / 1024.0f);
		printf("[Callback] Allocated Memory:\t%010u B\t%011.03f KB\t%08.03f MB\t%05.03f GB\n", 
			allocated, allocated / 1024.0f, allocated / 1024.0f / 1024.0f, allocated / 1024.0f / 1024.0f / 1024.0f);

		cout << endl;
	}

	void ScreenCleaningWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y +1);

		TextureDebug << <grid, block >> >(windowWidth, windowHeight, pixelBufferObject);
	}

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
							// Auxiliary Variable containing the Bounding Box Total.
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
							unsigned int *pixelBufferObject) {

		// Grid based on the Screen Dimensions.
		//dim3 block2(32,32);
		//dim3 grid2(windowWidth/block2.x + 1, windowHeight/block2.y +1);

		//Debug2<<<grid2, block2>>>(windowWidth, windowHeight, pixelBufferObject);

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y +1);

		Debug<<<grid, block>>>(rayArray, sortedRayIndexKeysArray, sortedRayIndexValuesArray, boundingSphereArray, boundingSphereTotal, rayTotal, windowWidth, windowHeight, cameraPosition, cameraDirection, cameraUp, cameraRight, pixelBufferObject);
	}

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
							unsigned int* rayIndexValuesArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::rayCreationTimerID);
		#endif

			// Grid based on the Screen Dimensions.
			dim3 block(1024);
			dim3 grid((windowWidth * windowHeight * lightTotal) / block.x + 1);

			#ifdef BLOCK_GRID_DEBUG
				cout << "[ShadowRayCreationWrapper] Block = " << block.x  << " Threads " << "Grid = " << grid.x << " Blocks" << endl;
			#endif

			// Create the Shadow Rays
			CreateShadowRays<<<grid, block>>>(windowWidth, windowHeight, lightTotal, rayArray, headFlagsArray, rayIndexKeysArray, rayIndexValuesArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::rayCreationTimerID);
		#endif
	}

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
							unsigned int* rayIndexValuesArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::rayCreationTimerID);
		#endif

			// Grid based on the Screen Dimensions.
			dim3 block(32,32);
			dim3 grid(windowWidth/block.x + 1, windowHeight/block.y + 1);

			#ifdef BLOCK_GRID_DEBUG
				cout << "[ReflectionRayCreationWrapper] Block = " << block.x * block.y << " Threads " << "Grid = " << grid.x * grid.y << " Blocks" << endl;
			#endif

			// Create the Reflection Rays
			CreateReflectionRays<<<grid, block>>>(windowWidth, windowHeight, cameraPosition, rayArray, headFlagsArray, rayIndexKeysArray, rayIndexValuesArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::rayCreationTimerID);
		#endif
	}

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
							unsigned int* rayTotal) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::rayTrimmingTimerID);
		#endif

			// Maximum Number of Rays being cast per Frame
			unsigned int rayMaximum = windowWidth * windowHeight * lightTotal;

			// Calculate the Inclusive Scan using the Ray Head Flags.
			Utility::checkCUDAError("RayTrimmingWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayMaximum));

			// Grid based on the Ray Count
			dim3 block(1024);
			dim3 grid(rayMaximum/block.x + 1);

			#ifdef BLOCK_GRID_DEBUG
				cout << "[TrimRays] Block = " << block.x << " Threads " << "Grid = " << grid.x << " Blocks" << endl;
			#endif

			// Create the Trimmed Rays
			CreateTrimmedRays<<<grid, block>>>(rayIndexKeysArray, rayIndexValuesArray, rayMaximum, scanArray, trimmedRayIndexKeysArray, trimmedRayIndexValuesArray);

			// Check the Inclusive Scan Output (Last position gives us the number of Rays that weren't generated)
			Utility::checkCUDAError("RayTrimmingWrapper::cudaMemcpy()", cudaMemcpy(rayTotal, &scanArray[rayMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));

			// Calculate the Ray Total
			*rayTotal = rayMaximum - *rayTotal;

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::rayTrimmingTimerID);
		#endif
	}

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
							unsigned int* chunkTotal) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::rayCompressionTimerID);
		#endif

			// Grid based on the Ray Count
			dim3 rayBlock(1024);
			dim3 rayGrid(rayTotal/rayBlock.x + 1);
		
			#ifdef BLOCK_GRID_DEBUG
				cout << "[CreateChunkFlags] Block = " << rayBlock.x << " Threads " << "Grid = " << rayGrid.x << " Blocks" << endl;
			#endif

			// Create the Chunk Flags
			CreateChunkFlags<<<rayGrid, rayBlock>>>(trimmedRayIndexKeysArray, trimmedRayIndexValuesArray, rayTotal, headFlagsArray);

			// Calculate the Inclusive Scan using the Chunk Head Flags.
			Utility::checkCUDAError("RayCompressionWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));
		
			// Check the Inclusive Scan Output (Last position gives us the number of Chunks that were generated)
			Utility::checkCUDAError("RayCompressionWrapper::cudaMemcpy()", cudaMemcpy(chunkTotal, &scanArray[rayTotal-1], sizeof(int), cudaMemcpyDeviceToHost));

			// Create the Chunk Bases
			CreateChunkBases<<<rayGrid, rayBlock>>>(trimmedRayIndexKeysArray, trimmedRayIndexValuesArray, rayTotal, headFlagsArray, scanArray, chunkBasesArray, chunkIndexKeysArray, chunkIndexValuesArray);

			// Grid based on the Ray Chunk Count
			dim3 chunkBlock(1024);
			dim3 chunkGrid(*chunkTotal/chunkBlock.x + 1);
		
			#ifdef BLOCK_GRID_DEBUG
				cout << "[CreateChunkSizes] Block = " << chunkBlock.x << " Threads " << "Grid = " << chunkGrid.x << " Blocks" << endl;
			#endif

			// Create the Chunk Sizes
			CreateChunkSizes<<<chunkGrid, chunkBlock>>>(chunkBasesArray, *chunkTotal, rayTotal, chunkSizesArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::rayCompressionTimerID);
		#endif
	}

	void RaySortingWrapper(
							// Input Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray,
							// Auxiliary Variable containing the Number of Chunks.
							const unsigned int chunkTotal,
							// Output Arrays containing the Sorted Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* sortedChunkIndexKeysArray, 
							unsigned int* sortedChunkIndexValuesArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::raySortingTimerID);
		#endif

			// Sort the Chunks
			Utility::checkCUDAError("RaySortingWrapper::cub::DeviceRadixSort::SortPairs()", 
				cub::DeviceRadixSort::SortPairs(radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
				chunkIndexKeysArray, sortedChunkIndexKeysArray,
				chunkIndexValuesArray, sortedChunkIndexValuesArray, 
				chunkTotal));

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::raySortingTimerID);
		#endif
	}

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
							unsigned int* sortedRayIndexValuesArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::rayDecompressionTimerID);
		#endif

			// Grid based on the Ray Chunk Count
			dim3 chunkBlock(1024);
			dim3 chunkGrid(chunkTotal/chunkBlock.x + 1);
		
			#ifdef BLOCK_GRID_DEBUG
				cout << "[CreateChunkSkeleton] Block = " << chunkBlock.x << " Threads " << "Grid = " << chunkGrid.x << " Blocks" << endl;
			#endif

			// Create the Sorted Ray Skeleton
			CreateSortedRaySkeleton<<<chunkGrid, chunkBlock>>>(
				chunkSizesArray, 
				sortedChunkIndexValuesArray,
				chunkTotal, 
				skeletonArray);

			// Calculate the Exclusive Scan using the Sorted Ray Skeleton.
			Utility::checkCUDAError("RayDecompressionWrapper::cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::ExclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, skeletonArray, scanArray, chunkTotal));

			// Create the Sorted Rays
			CreateSortedRays<<<chunkGrid, chunkBlock>>>(
				chunkBasesArray, chunkSizesArray, 
				sortedChunkIndexKeysArray, sortedChunkIndexValuesArray,
				trimmedRayIndexKeysArray, trimmedRayIndexValuesArray,
				scanArray, 
				skeletonArray, 
				chunkTotal, 
				sortedRayIndexKeysArray, sortedRayIndexValuesArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::rayDecompressionTimerID);
		#endif
	}

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
							float4* hierarchyArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::hierarchyCreationTimerID);
		#endif

			unsigned int hierarchyNodeWriteOffset = 0;
			unsigned int hierarchyNodeReadOffset = 0;
			unsigned int hierarchyNodeTotal = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

			// Grid based on the Hierarchy Node Count
			dim3 baseLevelBlock(1024);
			dim3 baseLevelGrid(hierarchyNodeTotal/baseLevelBlock.x + 1);

			#ifdef BLOCK_GRID_DEBUG
				cout << "[CreateHierarchyLevel0] Block = " << baseLevelBlock.x << " Threads " << "Grid = " << baseLevelGrid.x << " Blocks" << endl;
			#endif

			// Create the First Level of the Ray Hierarchy.
			CreateHierarchyLevel0<<<baseLevelGrid, baseLevelBlock>>>(
				rayArray,
				sortedRayIndexKeysArray, sortedRayIndexValuesArray, 
				rayTotal,
				hierarchyNodeTotal, 
				initialRadius,
				initialSpread,
				hierarchyArray);
		
			// Create the Remaining Levels of the Ray Hierarchy.
			for(unsigned int hierarchyLevel=1; hierarchyLevel<HIERARCHY_MAXIMUM_DEPTH; hierarchyLevel++) {
			
				hierarchyNodeReadOffset = hierarchyNodeWriteOffset;
				hierarchyNodeWriteOffset += hierarchyNodeTotal;
				hierarchyNodeTotal = hierarchyNodeTotal / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

				// Grid based on the Hierarchy Node Count
				dim3 nLevelBlock(1024);
				dim3 nLevelGrid(hierarchyNodeTotal/nLevelBlock.x + 1);

				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevelN] Block = " << nLevelBlock.x << " Threads " << "Grid = " << nLevelGrid.x << " Blocks" << endl;
				#endif

				CreateHierarchyLevelN<<<nLevelGrid, nLevelBlock>>>(
					hierarchyArray, 
					hierarchyNodeWriteOffset, 
					hierarchyNodeReadOffset, 
					hierarchyNodeTotal);
			}

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::hierarchyCreationTimerID);
		#endif
	}

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
							unsigned int *hierarchyHitMemoryTotal) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::hierarchyTraversalTimerID);
		#endif

			// Create the Hierarchy Node Offses and Total
			unsigned int hierarchyNodeOffset = 0;
			unsigned int hierarchyNodeTotal = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

			for(unsigned int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

				hierarchyNodeOffset = hierarchyNodeTotal + hierarchyNodeOffset;
				hierarchyNodeTotal = hierarchyNodeTotal / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
			}

			// Create the Hit Maximum and Total
			unsigned int hitMaximum = hierarchyNodeTotal * triangleTotal;
			unsigned int hitTotal = 0;
			unsigned int missedHitTotal = 0;

			// Grid based on the Hit Maximum
			dim3 block(1024);
			dim3 grid(hitMaximum/block.x + 1);

			// If we're using CRSH
			if(algorithmID == 0) {
		
				#ifdef BLOCK_GRID_DEBUG 
					cout << "[PrepareArray] Grid = " << grid.x << endl;
				#endif

				// Clean the Head Flags Array
				PrepareArray<<<grid, block>>>(0, hitMaximum, headFlagsArray);

				// Grid based on the Hierarchy Node * Bounding Box Count
				dim3 boundingSphereBlock(1024);
				dim3 boundingSphereGrid((hierarchyNodeTotal * boundingSphereTotal)/boundingSphereBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevel0Hits] Block = " << boundingSphereBlock.x << " Threads " << "Grid = " << boundingSphereGrid.x << " Blocks" << endl;
				#endif

				CalculateBoundingSpheresIntersections<<<boundingSphereGrid, boundingSphereBlock>>>(
					hierarchyArray, 
					boundingSphereArray,
					boundingSphereTotal,
					triangleTotal,
					triangleOffset,
					hierarchyNodeOffset, 
					hierarchyNodeTotal, 
					headFlagsArray);

				// Create the Trim Scan Array
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

				// Grid based on the Hierarchy Hit Count
				dim3 hitMaximumBlock(1024);
				dim3 hitMaximumGrid(hitMaximum/hitMaximumBlock.x + 1);

				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevel0Hits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
				#endif

				CreateHierarchyBoundingSphereHits<<<hitMaximumGrid, hitMaximumBlock>>>(
					triangleTotal,
					triangleOffset,
					hierarchyNodeOffset, 
					hierarchyNodeTotal, 
					scanArray, 
					headFlagsArray,
					hierarchyHitsArray);

				// Create the Trim Scan Array
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateTrimmedHierarchyHits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
				#endif

				CreateTrimmedHierarchyHits<<<hitMaximumGrid, hitMaximumBlock>>>(
					hierarchyHitsArray,
					scanArray,
					hitMaximum,
					trimmedHierarchyHitsArray);

				// Check the Hit Total (last position of the scan array) 
				Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));

				// Calculate the Hit Total for this Level
				hitTotal = hitMaximum - missedHitTotal;
			}
			// If we're using RAH
			else {

				// Calculate the Hit Total for this Level
				hitTotal = hitMaximum;
			}

			// Calculate the Hit Total for this Level
			*hierarchyHitTotal = hitTotal;

			// Early Exit
			if(hitTotal == 0)
				return;

			// Clean the Head Flags Array
			PrepareArray<<<grid, block>>>(0, hitMaximum, headFlagsArray);

			// Grid based on the Hierarchy Hit Count
			dim3 hitTotalBlock(1024);
			dim3 hitTotalGrid(hitTotal/hitTotalBlock.x + 1);

			// If we're using CRSH
			if(algorithmID == 0) {

				CreateCRSHHierarchyLevel0Hits<<<hitTotalGrid, hitTotalBlock>>>(
					hierarchyArray,
					trianglePositionsArray,
					triangleTotal,
					triangleOffset,
					hitTotal,
					hierarchyNodeOffset,
					hierarchyNodeTotal,
					headFlagsArray,
					hierarchyHitsArray,
					trimmedHierarchyHitsArray);
			}
			// If we're using RAH
			else {

				CreateRAHHierarchyLevel0Hits<<<hitTotalGrid, hitTotalBlock>>>(
					hierarchyArray,
					trianglePositionsArray,
					triangleTotal,
					triangleOffset,
					hitTotal,
					hierarchyNodeOffset,
					hierarchyNodeTotal,
					headFlagsArray,
					hierarchyHitsArray,
					trimmedHierarchyHitsArray);
			}

			// Create the Trim Scan Array
			Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitTotal));

			#ifdef BLOCK_GRID_DEBUG
				cout << "[CreateTrimmedHierarchyHits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
			#endif

			CreateTrimmedHierarchyHits<<<hitTotalGrid, hitTotalBlock>>>(
				hierarchyHitsArray,
				scanArray,
				hitTotal,
				trimmedHierarchyHitsArray);

			// Check the Hit Total (last position of the scan array) 
			Utility::checkCUDAError("HierarchyTraversalWarmUpWrapper::cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitTotal - 1], sizeof(int), cudaMemcpyDeviceToHost));

			// Calculate the Hit Total for this Level
			*hierarchyHitTotal = hitTotal - missedHitTotal;

			#ifdef TRAVERSAL_DEBUG

				testManager->incrementAccumulatedMaximumHitTotal(hitTotal);
				testManager->incrementAccumulatedMissedHitTotal(missedHitTotal);
				testManager->incrementAccumulatedConnectedHitTotal(*hierarchyHitTotal);

				testManager->incrementFinalMaximumHitTotal(hitTotal, HIERARCHY_MAXIMUM_DEPTH-1);
				testManager->incrementFinalMissedHitTotal(missedHitTotal, HIERARCHY_MAXIMUM_DEPTH-1);
				testManager->incrementFinalConnectedHitTotal(*hierarchyHitTotal, HIERARCHY_MAXIMUM_DEPTH-1);
			#endif

		//#ifdef TIMER_DEBUG
		//	testManager->stopTimer(TestManager::hierarchyTraversalTimerID);
		//#endif
	}

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
							unsigned int* hierarchyHitMemoryTotal) {

		//#ifdef TIMER_DEBUG
		//	testManager->startTimer(TestManager::hierarchyTraversalTimerID);
		//#endif

			// Calculate the Nodes Offset and Total
			unsigned int hierarchyNodeOffset[HIERARCHY_MAXIMUM_DEPTH];
			unsigned int hierarchyNodeTotal[HIERARCHY_MAXIMUM_DEPTH];
		
			hierarchyNodeOffset[0] = 0;
			hierarchyNodeTotal[0] = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

			for(unsigned int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

				hierarchyNodeOffset[i] = hierarchyNodeTotal[i-1] + hierarchyNodeOffset[i-1];
				hierarchyNodeTotal[i] = hierarchyNodeTotal[i-1] / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal[i-1] % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
			}

			// Create the Hierarchy Hit Arrays
			for(int hierarchyLevel=HIERARCHY_MAXIMUM_DEPTH-2; hierarchyLevel>=0; hierarchyLevel--) {

				if((*hierarchyHitTotal) == 0)
					return;

				// Calculate the Hit Maximum for this Level
				unsigned int hitMaximum = (*hierarchyHitTotal) * HIERARCHY_SUBDIVISION;
				unsigned int hitTotal = *hierarchyHitTotal;

				//cout << "[Traversal Level  "<< hierarchyLevel << "] Memory Usage: " << (float)hitMaximum/(float)(*hierarchyHitMemoryTotal) << endl;

				// Grid based on the Hierarchy Hit Total
				dim3 hitTotalBlock(1024);
				dim3 hitTotalGrid(hitTotal/hitTotalBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevelNHits] Block = " << hitTotalBlock.x << " Threads " << "Grid = " << hitTotalGrid.x << " Blocks" << endl;
				#endif

				CreateHierarchyLevelNHits<<<hitTotalGrid, hitTotalBlock>>>(
					hierarchyArray,
					trianglePositionsArray,
					triangleTotal,
					triangleOffset,
					hitTotal,
					hierarchyNodeOffset[hierarchyLevel],
					hierarchyNodeTotal[hierarchyLevel],
					headFlagsArray,
					hierarchyHitsArray, trimmedHierarchyHitsArray);

				// Create the Trim Scan Array
				Utility::checkCUDAError("HierarchyTraversalWrapper::cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

				// Grid based on the Hierarchy Hit Count
				dim3 hitMaximumBlock(1024);
				dim3 hitMaximumGrid(hitMaximum / hitMaximumBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateTrimmedHierarchyHits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
				#endif

				CreateTrimmedHierarchyHits<<<hitMaximumGrid, hitMaximumBlock>>>(
					hierarchyHitsArray,
					scanArray,
					hitMaximum,
					trimmedHierarchyHitsArray);

				// Calculate the Hits Missed for this Level
				int missedHitTotal;
				// Check the Hit Total (last position of the scan array) 
				Utility::checkCUDAError("HierarchyTraversalWrapper::cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));

				// Calculate the Hit Total for this Level
				*hierarchyHitTotal = hitMaximum - missedHitTotal;

				#ifdef TRAVERSAL_DEBUG

					testManager->incrementAccumulatedMaximumHitTotal(hitMaximum);
					testManager->incrementAccumulatedMissedHitTotal(missedHitTotal);
					testManager->incrementAccumulatedConnectedHitTotal(*hierarchyHitTotal);

					testManager->incrementFinalMaximumHitTotal(hitMaximum, hierarchyLevel);
					testManager->incrementFinalMissedHitTotal(missedHitTotal, hierarchyLevel);
					testManager->incrementFinalConnectedHitTotal(*hierarchyHitTotal, hierarchyLevel);
				#endif
			}

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::hierarchyTraversalTimerID);
		#endif
	}

	void ShadowRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(1024);
		dim3 grid(windowWidth*windowHeight*lightTotal/ block.x + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[PrepareArray] Grid = " << grid.x << endl;
		#endif

		// Prepare the Array
		PrepareArray<<<grid, block>>>(0, windowWidth * windowHeight * lightTotal, shadowFlagsArray);
	}

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
							unsigned int* shadowFlagsArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::intersectionTimerID);
		#endif

			// Grid based on the Hierarchy Hit Count
			dim3 intersectionBlock(1024);
			dim3 intersectionGrid(hitTotal / intersectionBlock.x + 1);

			#ifdef BLOCK_GRID_DEBUG 
				cout << "[CalculateShadowRayIntersections] Grid = " << intersectionGrid.x << endl;
			#endif

			// Local Intersection
			CalculateShadowRayIntersections<<<intersectionGrid, intersectionBlock>>>(
				rayArray, 
				sortedRayIndexKeysArray, sortedRayIndexValuesArray,
				hierarchyHitsArray,
				trianglePositionsArray,
				hitTotal,
				rayTotal,
				triangleOffset,
				windowWidth, windowHeight,
				cameraPosition,
				shadowFlagsArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::intersectionTimerID);
		#endif
	}

	void ShadowRayColoringWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Input Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::shadingTimerID);
		#endif

			// Grid based on the Screen Dimensions.
			dim3 colouringBlock(32,32);
			dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

			#ifdef BLOCK_GRID_DEBUG 
				cout << "[ColorPrimaryShadowRay] Grid = " << colouringGrid.x << "," << colouringGrid.y << endl;
			#endif

			// Colour the Screen
			ColorPrimaryShadowRay<<<colouringGrid, colouringBlock>>>(windowWidth, windowHeight, lightTotal, cameraPosition, shadowFlagsArray, pixelBufferObject);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::shadingTimerID);
		#endif
	}

	void ReflectionRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* intersectionTimeArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(1024);
		dim3 grid(windowWidth*windowHeight*LIGHT_SOURCE_MAXIMUM/ block.x + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[PrepareArray] Grid = " << grid.x << endl;
		#endif

		// Prepare the Array
		PrepareArray<<<grid, block>>>(UINT_MAX, windowWidth * windowHeight *LIGHT_SOURCE_MAXIMUM, intersectionTimeArray);
	}

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
							unsigned int* intersectionTimeArray) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::intersectionTimerID);
		#endif

			// Grid based on the Hierarchy Hit Count
			dim3 intersectionBlock(1024);
			dim3 intersectionGrid(hitTotal / intersectionBlock.x + 1);

			#ifdef BLOCK_GRID_DEBUG 
				cout << "[CalculateReflectionRayIntersections] Grid = " << intersectionGrid.x << endl;
			#endif

			// Local Intersection
			CalculateReflectionRayIntersections<<<intersectionGrid, intersectionBlock>>>(
				rayArray,
				sortedRayIndexKeysArray, sortedRayIndexValuesArray,
				hierarchyHitsArray,
				trianglePositionsArray,
				hitTotal,
				rayTotal,
				triangleOffset,
				windowWidth, windowHeight,
				lightTotal,
				cameraPosition,
				intersectionTimeArray);

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::intersectionTimerID);
		#endif
	}

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
							unsigned int *pixelBufferObject) {

		#ifdef TIMER_DEBUG
			testManager->startTimer(TestManager::shadingTimerID);
		#endif

			if(createRays == false) {

				// Grid based on the Screen Dimensions.
				dim3 colouringBlock(32,32);
				dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

				// Colour the Screen
				ColorReflectionRay<<<colouringGrid, colouringBlock>>>(
					rayArray,
					trianglePositionsArray, triangleNormalsArray,
					windowWidth, windowHeight,
					lightTotal,
					cameraPosition,
					intersectionTimeArray,
					pixelBufferObject);
			}
			else {

				// Grid based on the Screen Dimensions.
				dim3 colouringBlock(32,32);
				dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

				// Colour the Screen
				ColorAndCreateReflectionRay<<<colouringGrid, colouringBlock>>>(
					rayArray,
					trianglePositionsArray, triangleNormalsArray,
					windowWidth, windowHeight,
					lightTotal,
					cameraPosition,
					intersectionTimeArray,
					headFlagsArray, 
					rayIndexKeysArray, 
					rayIndexValuesArray,
					pixelBufferObject);
			}

		#ifdef TIMER_DEBUG
			testManager->stopTimer(TestManager::shadingTimerID);
		#endif
	}

	void AntiAliasingWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Primary Screen Buffer.
							unsigned int *primaryPixelBufferObject,
							// Output Array containing the Secondary Screen Buffer.
							unsigned int *secondaryPixelBufferObject) {

		// Grid based on the Screen Dimensions.
		dim3 colouringBlock(32,32);
		dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

		// Colour the Screen
		AntiAliasing<<<colouringGrid, colouringBlock>>>(
			windowWidth, windowHeight,
			primaryPixelBufferObject,
			secondaryPixelBufferObject);
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

	// CUDA Bounxing Box Texture Binding Functions
	void bindBoundingSpheres(float *cudaDevicePointer, unsigned int boundingSphereTotal) {

		boundingSpheresTexture.normalized = false;                      // access with normalized texture coordinates
		boundingSpheresTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		boundingSpheresTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * boundingSphereTotal * 2;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, boundingSpheresTexture, cudaDevicePointer, channelDescriptor, size);
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