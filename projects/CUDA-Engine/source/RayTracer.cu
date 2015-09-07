// Debug Macros
//#define CUB_STDERR
//#define BLOCK_GRID_DEBUG

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

// Temporary Storage
static void *scanTemporaryStorage = NULL;
static size_t scanTemporaryStoreBytes = 0;

static void *radixSortTemporaryStorage = NULL;
static size_t radixSortTemporaryStoreBytes = 0;

// Ray testing Constant
__constant__ __device__ static const float epsilon = 0.01f;

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

__device__ static inline unsigned int FloatFlip(unsigned int f) {

	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
}

__device__ static inline unsigned int IFloatFlip(unsigned int f) {

	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}

// Converts 8-bit integer to floating point rgb color
__device__ static inline  float3 IntToRgb(int color) {

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

// Converts a ray to an Integer Hash Value
__device__ static inline int CreateShadowRayIndex(float3 origin, float3 direction) {

	unsigned int index = 0;

	// Convert the Direction to Spherical Coordinates (atan2 => [-HALF_PI, HALF_PI], acos => [0.0f, PI])
	index = clamp((unsigned int)((atan2(direction.x, direction.x) + HALF_PI) * 5.0f), (unsigned int)0, (unsigned int)31);
	index = (index << 16) | clamp((unsigned int)(acos(direction.z) * 5.0f), (unsigned int)0, (unsigned int)31);

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

__device__ static inline int CreateReflectionRayIndex(float3 origin, float3 direction) {

	int index = 0;
	
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

__device__ static inline int CreateRefractionRayIndex(float3 origin, float3 direction) {

	int index = 0;

	// Clamp the Origin to the 0-15 range
	index = (unsigned int)clamp(origin.z + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.y + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	index = (index << 4) | (unsigned int)clamp(origin.x + half_bit_mask_1_4_f, 0.0f, bit_mask_1_4_f);
	
	// Convert the Direction to Spherical Coordinates
	index = (index << 4) | (unsigned int)clamp((atan(direction.y / direction.x) + HALF_PI) * RADIANS_TO_DEGREES * 2.0f, 0.0f, 360.0f);
	index = (index << 9) | (unsigned int)clamp(acos(direction.z) * RADIANS_TO_DEGREES, 0.0f, 180.0f);

	index++;

	return index;
}

// Ray - Node Intersection Code
__device__ static inline bool SphereNodeIntersection(const float4 &sphere, const float4 &cone, const float4 &triangle, const float &cosine, const float &tangent) {

	if(cone.w == HALF_PI)
		return true;

	float3 coneDirection = make_float3(cone);
	float3 sphereCenter = make_float3(sphere);
	float3 triangleCenter = make_float3(triangle);

	float3 sphereToTriangle = triangleCenter - sphereCenter;
	float3 sphereToTriangleProjection = dot(sphereToTriangle, coneDirection) * coneDirection;

	return (length(sphereToTriangleProjection) * tangent + (sphere.w + triangle.w) / cosine) >= length((triangleCenter - sphereCenter) - sphereToTriangleProjection);
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

	// Second Test
	float3 qvec = cross(tvec, edge1);  

	float v = dot(ray.direction, qvec) * determinant;  
	if (v < 0.0f || (u + v) > 1.0f)  
		return -1.0f;  

	return dot(edge2, qvec) * determinant;  
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

	/*float3 coneDirection = make_float3(cone);
	float3 rayDirection = make_float3(ray);

	float spread = acos(dot(coneDirection, rayDirection));

	if(cone.w > spread)
		return cone;

	float3 q = normalize(dot(coneDirection, rayDirection) * coneDirection - rayDirection);
	float3 e = coneDirection * cos(cone.w) + q * sin(cone.w);
	
	float3 newConeDirection = normalize(rayDirection + e);
	float newConeSpread = acos(dot(newConeDirection, rayDirection));

	return make_float4(newConeDirection.x, newConeDirection.y, newConeDirection.z, newConeSpread);*/

	float3 coneDirection1 = make_float3(cone);
	float3 coneDirection2 = make_float3(ray);
	
	float spread = acos(dot(coneDirection1, coneDirection2)) * 0.5f;

	/*if(cone1.w > spread + cone2.w)
		return cone1;

	if(cone2.w > spread + cone1.w)
		return cone2;*/
	
	float3 coneDirection = normalize(coneDirection1 + coneDirection2);
	float coneSpread = clamp(spread + max(cone.w, ray.w), 0.0f, HALF_PI);

	return make_float4(coneDirection.x, coneDirection.y, coneDirection.z, coneSpread); 
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

	float3 sphereCenter = sphereCenter1 + sphereDirection * sphereDistance * 0.5f;
	float sphereRadius = sphereDistance * 0.5f + max(sphere1.w , sphere2.w);

	return make_float4(sphereCenter.x, sphereCenter.y, sphereCenter.z, sphereRadius);
}

__global__ void UpdateVertex(
							// Input Array containing the Updated Model Matrices.
							float* modelMatricesArray,
							// Input Array containing the Updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Vertex Total.
							const int vertexTotal,
							// Output Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Output Array containing the Updated Triangle Normals.
							float4* triangleNormalsArray) {

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

__global__ void PreparePixels(	
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const  int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	// Triangle Material Properties
	float4 fragmentDiffuseColor = tex2D(diffuseTexture, x,y);
	float4 fragmentSpecularColor = tex2D(specularTexture, x,y);

	// Primary Ray
	float3 rayOrigin = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 rayDirection = reflect(normalize(rayOrigin-cameraPosition), fragmentNormal);

	for(int l = 0; l < lightTotal; l++) {

		float3 lightPosition = make_float3(tex1Dfetch(lightPositionsTexture, l));

		// Light Direction and Distance
		float3 lightDirection = lightPosition - fragmentPosition;

		float lightDistance = length(lightDirection);
		lightDirection = normalize(lightDirection);

		// Diffuse Factor
		float diffuseFactor = max(dot(lightDirection, fragmentNormal), 0.0f);
		clamp(diffuseFactor, 0.0f, 1.0f);

		if(diffuseFactor > 0.0f) {

			// Blinn-Phong approximation Halfway Vector
			float3 halfwayVector = lightDirection - rayDirection;
			halfwayVector = normalize(halfwayVector);

			// Light Color
			float3 lightColor = make_float3(tex1Dfetch(lightColorsTexture, l));
			// Light Intensity (x = diffuse, y = specular)
			float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
			// Light Attenuation (x = constant, y = linear, z = exponential)
			float3 lightAttenuation = make_float3(0.0f, 0.0f, 0.0f);

			float attenuation = 1.0f / (1.0f + lightAttenuation.x + lightDistance * lightAttenuation.y + lightDistance * lightDistance * lightAttenuation.z);

			// Diffuse Component
			fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation;

			// Specular Factor
			float specularFactor = powf(max(dot(halfwayVector, fragmentNormal), 0.0f), fragmentSpecularColor.w);
			clamp(specularFactor, 0.0f, 1.0f);

			// Specular Component
			if(specularFactor > 0.0f)
				fragmentColor += make_float3(fragmentSpecularColor) * lightColor * specularFactor * lightIntensity.y * attenuation;
		}
	}
	
	pixelBufferObject[x + y * windowWidth] = RgbToInt(255.0f, 255.0f, 255.0f);
	//pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
}

__global__ void PrepareArray(	
							// Input Variable containing the Preparation Value
							const int value,
							// Auxiliary Variables containing the Screen Dimensions.
							const int arraySize,
							// Output Array to be prepared.
							int* preparedArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= arraySize)
		return;

	preparedArray[x] = value;
}

__global__ void Debug(	
							// Input Array containing the Rays.
							float3* rayArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const  int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	//float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	//float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	// Primary Ray
	//float3 rayOrigin = make_float3(tex2D(fragmentPositionTexture, x,y));
	//float3 rayDirection = reflect(normalize(rayOrigin-cameraPosition), fragmentNormal);

	// Primary Ray
	/*float3 rayOrigin = rayArray[(x + y * windowWidth) * 2];
	float3 rayDirection = rayArray[(x + y * windowWidth) * 2 + 1];

	float3 fragmentColor = normalize(rayOrigin);

	pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);*/
}

__global__ void CreateShadowRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Light Index.
							const int lightIndex,
							// Output Array containing the unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	int rayIndex = x + y * windowWidth;

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {

		// Calculate the Shadow Rays Origin and Direction
		float3 shadowRayOrigin = make_float3(tex1Dfetch(lightPositionsTexture, lightIndex));
		float3 shadowRayDirection = normalize(fragmentPosition - shadowRayOrigin);

		// Diffuse Factor (Negate the Normal because the Ray Origin is reversed)
		float diffuseFactor = max(dot(shadowRayDirection, -fragmentNormal), 0.0f);
		clamp(diffuseFactor, 0.0f, 1.0f);
				
		// Store the Shadow Rays its direction
		if(diffuseFactor > 0.0f) {
			
			// Store the Shadow Rays Origin
			rayArray[rayIndex * 2] = shadowRayOrigin;
			// Store the Shadow Rays Direction
			rayArray[rayIndex * 2 + 1] = shadowRayDirection;

			// Store the Shadow Rays Hash Key
			rayIndexKeysArray[rayIndex] = CreateShadowRayIndex(shadowRayOrigin, shadowRayDirection);
			// Store the Shadow Rays Index Value
			rayIndexValuesArray[rayIndex] = rayIndex;

			// Store the Shadow Rays Flag (Trimming)
			headFlagsArray[rayIndex] = 0;

			return;
		}
	}
			
	// Store the Shadow Rays Flag (Trimming)
	headFlagsArray[rayIndex] = 1;
}

__global__ void CreateReflectionRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Cameras World Space Position.
							float3 cameraPosition,
							// Input Array containing the Unsorted Rays
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	int rayIndex = x + y * windowWidth;

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
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Cameras World Space Position.
							float3 cameraPosition,
							// Input Array containing the Unsorted Rays
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= windowWidth || y >= windowHeight)
		return;

	int rayIndex = x + y * windowWidth;

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
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Auxiliary Variable containing the Screen Dimensions.
							const int screenDimensions,
							// Auxiliary Array containing the Inclusive Scan Output.
							int* inclusiveScanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray) {

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
		int currentOffset = inclusiveScanArray[x];
		int previousOffset = inclusiveScanArray[x - 1];

		// Equal Offsets means that the Ray should be shifted to the left.
		if(currentOffset == previousOffset) {

			trimmedRayIndexKeysArray[x - currentOffset] = rayIndexKeysArray[x];
			trimmedRayIndexValuesArray[x - currentOffset] = rayIndexValuesArray[x];
		}
	}
}
	
__global__ void CreateChunkFlags(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const int rayTotal,
							// Output Array containing the Chunk Head Flags.
							int* headFlagsArray) {

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
		int currentKey = trimmedRayIndexKeysArray[x];
		int previousKey = trimmedRayIndexKeysArray[x - 1];
		
		// Different Keys means that a new Chunk should be created.
		if(currentKey != previousKey)
			headFlagsArray[x] = 1;
		else
			headFlagsArray[x] = 0;
	}
}

__global__ void CreateChunkBases(	
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const int rayTotal,
							// Auxiliary Array containing the Chunk Head Flags.
							int* headFlagsArray, 
							// Auxiliary Array containing the Exclusive Scan Output.
							int* scanArray, 
							// Output Array containing the Ray Chunk Bases.
							int* chunkBasesArray,
							// Output Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= rayTotal)
		return;
		
	// Head Flags containing 0 indicate there's no new Chunk to be created.
	if(headFlagsArray[x] == 0)
		return;

	// Store the Position of the new Chunk.
	int position = scanArray[x] - 1;

	// Store the Ray Index Base for the Chunk.
	chunkBasesArray[position] = x; 
	
	// Store the Ray Hash and the Chunk Position for the Chunk
	chunkIndexKeysArray[position] = trimmedRayIndexKeysArray[x];
	chunkIndexValuesArray[position] = position;
}

__global__ void CreateChunkSizes(
							// Input Array containing the Ray Chunk Bases.
							int* chunkBasesArray,
							// Auxiliary Variable containing the Chunk Total.
							const int chunkTotal,
							// Auxiliary Variable containing the Ray Total.
							const int rayTotal,
							// Output Array containing the Ray Chunk Sizes.
							int* chunkSizesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	// Last Chunk
	if(x == chunkTotal - 1) {

		// Chunk Bases
		int currentBase = chunkBasesArray[x];
		int nextBase = rayTotal;
	
		// Store the Chunk Sizes based on the Chunk Base of the Current Chunk and the Ray Total.
		chunkSizesArray[x] = nextBase - currentBase;
	}
	// Remaining Chunks
	else {
		
		// Chunk Bases
		int currentBase = chunkBasesArray[x];
		int nextBase = chunkBasesArray[x+1];

		// Store the Chunk Sizes based on the Chunk Base of the Current and the Next Chunks.
		chunkSizesArray[x] = nextBase - currentBase;
	}
}

__global__ void CreateSortedRaySkeleton(
							// Input Array containing the Ray Chunk Sizes.
							int* chunkSizesArray,
							// Input Array containing the Ray Chunk Values [Values = Positions].
							int* sortedChunkValuesArray,
							// Auxiliary Variable containing the Chunk Total.
							const int chunkTotal,
							// Output Array containing the Sorted Ray Arrays Skeleton.
							int* skeletonArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;
	
	// Store the Sorted Ray Arrays Skeleton based on the Sorted Chunk Value (Position).
	skeletonArray[x] = chunkSizesArray[sortedChunkValuesArray[x]];
}

__global__ void CreateSortedRays(
							// Input Array containing the Ray Chunk Bases.
							int* chunkBasesArray,
							// Input Array containing the Ray Chunk Sizes.
							int* chunkSizesArray,
							// Input Arrays containing the Ray Chunks  [Keys = Hashes, Values = Indices]
							int* sortedChunkKeysArray,
							int* sortedChunkValuesArray,
							// Input Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Input Array containing the Exclusive Scan Output.
							int* scanArray, 
							// Input Array containing the Sorted Ray Arrays Skeleton.
							int* skeletonArray,
							// Auxiliary Variable containing the Chunk Total.
							const int chunkTotal,
							// Output Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= chunkTotal)
		return;

	// Store the Chunk Key and Value.
	int chunkValue = sortedChunkValuesArray[x];
	
	// Store the Chunk Base and Size.
	int chunkBase = chunkBasesArray[chunkValue];
	int chunkSize = chunkSizesArray[chunkValue];

	// Store the Ray starting and final Positions.
	int startingPosition = scanArray[x];
	int finalPosition = startingPosition + chunkSize;

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
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const int rayTotal,
							// Auxiliary Variable containing the Node Total.
							const int nodeTotal,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	// Ray Origins are stored in the first offset
	float4 sphere = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2], 0.0f);
	// Ray Directions are stored in the second offset
	float4 cone = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2 + 1], 0.0f);
	
	for(int i=1; i<HIERARCHY_SUBDIVISION; i++) {

		if(rayTotal * 2 < (x * HIERARCHY_SUBDIVISION + i) * 2)
			break;

		// Ray Origins are stored in the first offset
		float4 rayOrigin = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION + i] * 2], 0.0f);
		// Ray Directions are stored in the second offset
		float4 rayDirection = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION + i] * 2 + 1], 0.0f);
		
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
							const int nodeWriteOffset,
							// Auxiliary Variable containing the Read Node Index.
							const int nodeReadOffset,
							// Auxiliary Variable containing the Node Total.
							const int nodeTotal) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	// Ray Origins are stored in the first offset
	float4 sphere = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION) * 2];
	// Ray Directions are stored in the second offset
	float4 cone = hierarchyArray[(nodeReadOffset + x * HIERARCHY_SUBDIVISION) * 2 + 1];
	
	for(int i=1; i<HIERARCHY_SUBDIVISION; i++) {

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

__global__ void CreateHierarchyLevel0Hits(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Triangle Total.
							const int triangleTotal,
							// Auxiliary Variable containing the Node Offset.
							const int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const int nodeReadTotal,
							// Output Array containing the Inclusive Scan Output.
							int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							int2* trimmedHierarchyHitsArray) {

	/*unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeReadTotal)
		return;

	float4 sphere = hierarchyArray[(nodeOffset + x) * 2];
	float4 cone = hierarchyArray[(nodeOffset + x) * 2 + 1];

	float cosine = cos(cone.w);
	float tangent = tan(cone.w);

	float4 triangle;

	for(int i=0; i<triangleTotal; i++) {

		triangle = CreateTriangleBoundingSphere(
			make_float3(trianglePositionsArray[i*3]), 
			make_float3(trianglePositionsArray[i*3 + 1]), 
			make_float3(trianglePositionsArray[i*3 + 2]));
	
		// Calculate Intersection		
		if(SphereNodeIntersection(sphere, cone, triangle, cosine, tangent) == true) {
		
			headFlagsArray[x * triangleTotal + i] = 0;
			hierarchyHitsArray[x * triangleTotal + i] = make_int2(x,i);

			//headFlagsArray[i * nodeReadTotal + x] = 0;
			//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(x,i);
		}
		else {

			headFlagsArray[x * triangleTotal + i] = 1;
			hierarchyHitsArray[x * triangleTotal + i] = make_int2(0,0);
			
			//headFlagsArray[i * nodeReadTotal + x] = 1;
			//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(0,0);
		}
	}*/

	/*unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / triangleTotal;
	unsigned int triangleID = x % triangleTotal;

	if(nodeID >= nodeReadTotal || triangleID >= triangleTotal)
		return;

	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];

	float4 triangle = CreateTriangleBoundingSphere(
			make_float3(trianglePositionsArray[triangleID * 3]), 
			make_float3(trianglePositionsArray[triangleID * 3 + 1]), 
			make_float3(trianglePositionsArray[triangleID * 3 + 2]));
	
	// Calculate Intersection		
	if(SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) {
		
		headFlagsArray[x] = 0;
		hierarchyHitsArray[x] = make_int2(nodeID,triangleID);

		//headFlagsArray[i * nodeReadTotal + x] = 0;
		//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(x,i);
	}
	else {

		headFlagsArray[x] = 1;
		hierarchyHitsArray[x] = make_int2(0,0);
			
		//headFlagsArray[i * nodeReadTotal + x] = 1;
		//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(0,0);
	}*/

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / triangleTotal;
	unsigned int triangleID = x % triangleTotal;

	if(nodeID >= nodeReadTotal || triangleID >= triangleTotal)
		return;

	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];

	float4 triangle = CreateTriangleBoundingSphere(
			make_float3(trianglePositionsArray[triangleID * 3]), 
			make_float3(trianglePositionsArray[triangleID * 3 + 1]), 
			make_float3(trianglePositionsArray[triangleID * 3 + 2]));
	
	// Calculate Intersection		
	if(SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) {
		
		headFlagsArray[x] = 0;
		hierarchyHitsArray[x] = make_int2(nodeID,triangleID);

		//headFlagsArray[i * nodeReadTotal + x] = 0;
		//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(x,i);
	}
	else {

		headFlagsArray[x] = 1;
		hierarchyHitsArray[x] = make_int2(0,0);
			
		//headFlagsArray[i * nodeReadTotal + x] = 1;
		//hierarchyHitsArray[i * nodeReadTotal + x] = make_int2(0,0);
	}
}

__global__ void CreateHierarchyLevelNHits(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Auxiliary Variable containing the Triangle Total.
							const int triangleTotal,
							// Auxiliary Variable containing the Hit Total.
							const int hitTotal,
							// Auxiliary Variable containing the Node Offset.
							const int nodeOffset,
							// Auxiliary Variable containing the Node Write Total.
							const int nodeWriteTotal,
							// Output Array containing the Inclusive Scan Output.
							int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							int2* trimmedHierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	int2 hit = trimmedHierarchyHitsArray[x];

	float4 sphere;
	float4 cone;

	float4 triangle = CreateTriangleBoundingSphere(
		make_float3(trianglePositionsArray[hit.y * 3]), 
		make_float3(trianglePositionsArray[hit.y * 3 + 1]), 
		make_float3(trianglePositionsArray[hit.y * 3 + 2]));

	for(int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		if((hit.x * HIERARCHY_SUBDIVISION + i) < nodeWriteTotal) {

			sphere = hierarchyArray[(nodeOffset + hit.x * HIERARCHY_SUBDIVISION + i) * 2];
			cone = hierarchyArray[(nodeOffset + hit.x * HIERARCHY_SUBDIVISION + i) * 2 + 1];
	
			// Calculate Intersection		
			if(SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) {
		
				headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = 0;
				hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] = make_int2(hit.x * HIERARCHY_SUBDIVISION + i, hit.y);

				//headFlagsArray[hit.y * nodeWriteTotal + (hit.x * HIERARCHY_SUBDIVISION + i)] = 0;
				//hierarchyHitsArray[hit.y * nodeWriteTotal + (hit.x * HIERARCHY_SUBDIVISION + i)] = make_int2(hit.x * HIERARCHY_SUBDIVISION + i, hit.y);

				continue;
			}
		}

		headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = 1;
		hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] = make_int2(0, 0);
		
		//headFlagsArray[hit.y * nodeWriteTotal + (hit.x * HIERARCHY_SUBDIVISION + i)] = 1;
		//hierarchyHitsArray[hit.y * nodeWriteTotal + (hit.x * HIERARCHY_SUBDIVISION + i)] = make_int2(0,0);
	}
}

__global__ void CreateTrimmedHierarchyHits(	
							// Input Array containing the Ray Hierarchy Hits.
							int2* hierarchyHitsArray,
							// Input Array containing the Inclusive Scan Output.
							const int* scanArray, 
							// Auxiliary Variable containing the Hit Total.
							const int hitTotal,
							// Output Array containing the Trimmed Ray Hierarchy Hits.
							int2* trimmedHierarchyHitsArray) {

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
		int currentOffset = scanArray[x];
		int previousOffset = scanArray[x - 1];
		
		// Equal Offsets means that the Hit should be shifted to the left.
		if(currentOffset == previousOffset)
			trimmedHierarchyHitsArray[x- currentOffset] = hierarchyHitsArray[x];
	}
}

__global__ void CalculateShadowRayIntersections(	
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
							const int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							const int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Store the Hierarchy Hit
	int2 hierarchyHit = hierarchyHitsArray[x];

	// Store the Triangles Vertices and Edges
	float3 vertex0 = make_float3(trianglePositionsArray[hierarchyHit.y * 3]);
	float3 edge1 = make_float3(trianglePositionsArray[hierarchyHit.y * 3 + 1]) - vertex0;
	float3 edge2 = make_float3(trianglePositionsArray[hierarchyHit.y * 3 + 2]) - vertex0;

	for(int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		// Check if the Extrapolated Ray exists.
		if(hierarchyHit.x * HIERARCHY_SUBDIVISION + i >= rayTotal)
			return;

		// Fetch the Ray Index
		int rayIndex = sortedRayIndexValuesArray[hierarchyHit.x * HIERARCHY_SUBDIVISION + i];

		// Fetch the Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		float intersectionDistance = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, edge1, edge2);

		// Calculate the Interesection Time

			// Calculate the Lights Distance to the Fragment
			if(intersectionDistance > epsilon && intersectionDistance < length(rayOrigin - make_float3(tex2D(fragmentPositionTexture, rayIndex % windowWidth, rayIndex / windowWidth))) - epsilon * 2.0f)
				shadowFlagsArray[rayIndex] = INT_MAX;
	}
}

__global__ void ColorPrimaryShadowRay(	
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {

		// Triangle Material Properties
		float4 fragmentDiffuseColor = tex2D(diffuseTexture, x,y);
		float4 fragmentSpecularColor = tex2D(specularTexture, x,y);

		for(int l = 0; l < lightTotal; l++) {

			// Check if the Light is Blocked
			if(shadowFlagsArray[x + y * windowWidth] != INT_MAX) {

				// Light Direction and Distance
				float3 lightDirection = normalize(make_float3(tex1Dfetch(lightPositionsTexture, l)) - fragmentPosition);

				// Blinn-Phong approximation Halfway Vector
				float3 halfwayVector = normalize(lightDirection - normalize(fragmentPosition - cameraPosition));

				// Light Color
				float3 lightColor = make_float3(tex1Dfetch(lightColorsTexture, l));
				// Light Intensity (x = diffuse, y = specular)
				float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);

				// Diffuse Component
				fragmentColor += make_float3(fragmentDiffuseColor) * lightColor * clamp(max(dot(lightDirection, fragmentNormal), 0.0f), 0.0f, 1.0f) * lightIntensity.x;
				// Specular Component
				fragmentColor += make_float3(fragmentSpecularColor) * lightColor * clamp(powf(max(dot(halfwayVector, fragmentNormal), 0.0f), fragmentSpecularColor.w), 0.0f, 1.0f) * lightIntensity.y;
			}
		}
	
		pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);

		return;
	}

	pixelBufferObject[x + y * windowWidth] = RgbToInt(0.0f, 0.0f, 0.0f);
}

__global__ void CalculateReflectionRayIntersections(
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
							const int hitTotal,
							// Auxiliary Variable containing the Number of Rays.
							const int rayTotal,
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliary Array containing the Intersection Times.
							int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Store the Hierarchy Hit
	int2 hierarchyHit = hierarchyHitsArray[x];

	// Store the Triangles Vertices and Edges
	float3 vertex0 = make_float3(trianglePositionsArray[hierarchyHit.y * 3]);
	float3 edge1 = make_float3(trianglePositionsArray[hierarchyHit.y * 3 + 1]) - vertex0;
	float3 edge2 = make_float3(trianglePositionsArray[hierarchyHit.y * 3 + 2]) - vertex0;;

	for(int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		// Check if the Extrapolated Ray exists.
		if(hierarchyHit.x * HIERARCHY_SUBDIVISION + i >= rayTotal)
			return;

		// Fetch the Ray Index
		int rayIndex = sortedRayIndexValuesArray[hierarchyHit.x * HIERARCHY_SUBDIVISION + i];

		// Fetch the Ray
		float3 rayOrigin = rayArray[rayIndex * 2];
		float3 rayDirection = rayArray[rayIndex * 2 + 1];

		float intersectionDistance = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, edge1, edge2);

		// Calculate the Intersection Time
		if(intersectionDistance > epsilon) {

			unsigned int newTime = ((unsigned int)(intersectionDistance * 10.0f + 1.0f) << 20) + hierarchyHit.y;
			unsigned int oldTime = atomicMin((unsigned int*)&intersectionTimeArray[rayIndex], newTime);
		}
	}
}

__global__ void ColorReflectionRay(	
							// Input Array containing the Updated Triangle Positions.
							float4* trianglePositionsArray,
							// Input Array containing the Updated Triangle Normals.
							float4* triangleNormalsArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const int windowWidth, const int windowHeight,
							// Auxiliary Variable containing the Number of Lights.
							const int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliary Array containing the Intersection Times.
							int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	

	if(x >= windowWidth || y >= windowHeight)
		return;

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Reflection Ray Intersection Time
	unsigned int intersectionRecord = (unsigned int)intersectionTimeArray[x + y * windowWidth];
	unsigned int intersectionTriangle = intersectionRecord & 0x000FFFFF;

	if(intersectionRecord != UINT_MAX) {

		// Triangle Material Identifier
		int1 materialID = tex1Dfetch(triangleMaterialIDsTexture, intersectionTriangle * 3);

		// Fragment Position and Normal - Sent from the OpenGL Rasterizer
		float3 position = make_float3(tex2D(fragmentPositionTexture, x,y));
		float3 normal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

		// Calculate the Reflection Ray
		float3 rayDirection = reflect(normalize(position-cameraPosition), normal);

		// Store the Triangles Vertices and Edges
		float3 vertex0 = make_float3(trianglePositionsArray[intersectionTriangle * 3]);
		float3 vertex1 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 1]);
		float3 vertex2 = make_float3(trianglePositionsArray[intersectionTriangle * 3 + 2]);

		// Calculate the Intersection Time
		float intersectionTime = RayTriangleIntersection(Ray(position + rayDirection * epsilon, rayDirection), vertex0, vertex1 - vertex0, vertex2 - vertex0);

		// Calculate the Hit Point
		position = position + rayDirection * (epsilon + intersectionTime);

		// Normal calculation using Barycentric Interpolation
		float areaABC = length(cross(vertex1 - vertex0, vertex2 - vertex0));
		float areaPBC = length(cross(vertex1 - position, vertex2 - position));
		float areaPCA = length(cross(vertex0 - position, vertex2 - position));

		normal = 
			(areaPBC / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3]) + 
			(areaPCA / areaABC) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 1]) + 
			(1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(triangleNormalsArray[intersectionTriangle * 3 + 2]);

		for(int l = 0; l < lightTotal; l++) {

			// Light Direction
			float3 lightDirection = normalize(make_float3(tex1Dfetch(lightPositionsTexture, l)) - position);

			// Blinn-Phong approximation Halfway Vector
			float3 halfwayVector = normalize(lightDirection - rayDirection);

			// Light Color
			float3 lightColor =  make_float3(tex1Dfetch(lightColorsTexture, l));
			// Light Intensity (x = diffuse, y = specular)
			float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);

			// Diffuse Component
			fragmentColor += 
				make_float3(tex1Dfetch(materialDiffusePropertiesTexture, materialID.x)) * lightColor * 
				clamp(dot(lightDirection, normal), 0.0f, 1.0f) * lightIntensity.x;

			// Specular Component
			fragmentColor += 
				make_float3(tex1Dfetch(materialSpecularPropertiesTexture, materialID.x)) * lightColor * 
				clamp(powf(max(dot(halfwayVector, normal), 0.0f), tex1Dfetch(materialSpecularPropertiesTexture, materialID.x).w), 0.0f, 1.0f) * lightIntensity.y;
		}

		pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
	}
}

extern "C" {

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
							float4* triangleNormalsArray) {
		
		int vertexTotal = triangleTotal * 3;

		// Grid based on the Triangle Count
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(vertexTotal / multiplicationBlock.x + 1);
		
		// Model and Normal Matrix Multiplication
		UpdateVertex<<<multiplicationGrid, multiplicationBlock>>>(
			modelMatricesArray, normalMatricesArray, 
			vertexTotal,
			trianglePositionsArray, triangleNormalsArray);
	}

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
							int* sortedChunkIndexValuesArray) {

		// Number of Rays potentialy being cast per Frame
		int rayTotal = windowWidth * windowHeight;
		int nodeTotal = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

		// Prepare the Scans by allocating temporary storage
		if(scanTemporaryStorage == NULL) {

			// Check how much memory is necessary
			Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", 
				cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, nodeTotal * triangleTotal));
			// Allocate temporary storage for exclusive prefix scan
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&scanTemporaryStorage, scanTemporaryStoreBytes));
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
			Utility::checkCUDAError("cudaMalloc()", cudaMalloc(&radixSortTemporaryStorage, radixSortTemporaryStoreBytes));
		}
	}

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
							unsigned int *pixelBufferObject) {

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y + 1);

		// Prepare the Screen
		//PreparePixels<<<grid, block>>>(windowWidth, windowHeight, pixelBufferObject);

		//Debug<<<grid, block>>>(rayArray, windowWidth, windowHeight, lightTotal, cameraPosition, pixelBufferObject);
	}

	void ShadowRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, 
							int windowHeight,
							// Auxiliary Variables containing the Light Index.
							int lightIndex,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y + 1);

		#ifdef BLOCK_GRID_DEBUG
			cout << "[ShadowRayCreationWrapper] Block = " << block.x * block.y << " Threads " << "Grid = " << grid.x * grid.y << " Blocks" << endl;
		#endif

		// Create the Shadow Rays
		CreateShadowRays<<<grid, block>>>(windowWidth, windowHeight, lightIndex, rayArray, headFlagsArray, rayIndexKeysArray, rayIndexValuesArray);
	}

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
							int* rayIndexValuesArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y + 1);

		#ifdef BLOCK_GRID_DEBUG
			cout << "[ReflectionRayCreationWrapper] Block = " << block.x * block.y << " Threads " << "Grid = " << grid.x * grid.y << " Blocks" << endl;
		#endif

		// Create the Reflection Rays
		CreateReflectionRays<<<grid, block>>>(windowWidth, windowHeight, cameraPosition, rayArray, headFlagsArray, rayIndexKeysArray, rayIndexValuesArray);
	}

	void RayTrimmingWrapper(	
							// Input Arrays containing the Untrimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* rayIndexKeysArray, 
							int* rayIndexValuesArray,
							// Auxiliary Variables containing the Screen Dimensions.
							int windowWidth, 
							int windowHeight,
							// Auxiliary Array containing the Ray Head Flags.
							int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							int* scanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							int* trimmedRayIndexKeysArray, 
							int* trimmedRayIndexValuesArray,
							// Output Variable containing the Number of Rays.
							int* rayTotal) {
	
		// Maximum Number of Rays being cast per Frame
		int rayMaximum = windowWidth * windowHeight;

		// Calculate the Inclusive Scan using the Ray Head Flags.
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayMaximum));

		// Number of Pixels per Frame
		int screenDimensions = windowWidth * windowHeight;

		// Grid based on the Pixel Count
		dim3 block(1024);
		dim3 grid(screenDimensions/block.x + 1);	

		#ifdef BLOCK_GRID_DEBUG
			cout << "[TrimRays] Block = " << block.x << " Threads " << "Grid = " << grid.x << " Blocks" << endl;
		#endif

		// Create the Trimmed Rays
		CreateTrimmedRays<<<grid, block>>>(rayIndexKeysArray, rayIndexValuesArray, screenDimensions, scanArray, trimmedRayIndexKeysArray, trimmedRayIndexValuesArray);

		// Check the Inclusive Scan Output (Last position gives us the number of Rays that weren't generated)
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(rayTotal, &scanArray[rayMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));

		// Calculate the Ray Total
		*rayTotal = rayMaximum - *rayTotal;
	}

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
							int* chunkTotal) {

		// Grid based on the Ray Count
		dim3 rayBlock(1024);
		dim3 rayGrid(rayTotal/rayBlock.x + 1);
		
		#ifdef BLOCK_GRID_DEBUG
			cout << "[CreateChunkFlags] Block = " << rayBlock.x << " Threads " << "Grid = " << rayGrid.x << " Blocks" << endl;
		#endif

		// Create the Chunk Flags
		CreateChunkFlags<<<rayGrid, rayBlock>>>(trimmedRayIndexKeysArray, trimmedRayIndexValuesArray, rayTotal, headFlagsArray);
		
		// Calculate the Inclusive Scan using the Chunk Head Flags.
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayTotal));
		
		// Check the Inclusive Scan Output (Last position gives us the number of Chunks that were generated)
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(chunkTotal, &scanArray[rayTotal-1], sizeof(int), cudaMemcpyDeviceToHost));

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
	}

	void RaySortingWrapper(	
							// Input Arrays containing the Ray Chunks [Keys = Hashes, Values = Indices]
							int* chunkIndexKeysArray, 
							int* chunkIndexValuesArray,
							// Auxiliary Variable containing the Number of Chunks.
							int chunkTotal,
							// Output Arrays containing the Sorted Ray Chunks [Keys = Hashes, Values = Indices]
							int* sortedChunkIndexKeysArray, 
							int* sortedChunkIndexValuesArray) {
		
		// Sort the Chunks
		Utility::checkCUDAError("cub::DeviceRadixSort::SortPairs()", 
			cub::DeviceRadixSort::SortPairs(radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
			chunkIndexKeysArray, sortedChunkIndexKeysArray,
			chunkIndexValuesArray, sortedChunkIndexValuesArray, 
			chunkTotal));
	}

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
							int* sortedRayIndexValuesArray) {

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
		Utility::checkCUDAError("cub::DeviceScan::ExclusiveSum()", cub::DeviceScan::ExclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, skeletonArray, scanArray, chunkTotal));

		// Create the Sorted Rays
		CreateSortedRays<<<chunkGrid, chunkBlock>>>(
			chunkBasesArray, chunkSizesArray, 
			sortedChunkIndexKeysArray, sortedChunkIndexValuesArray,
			trimmedRayIndexKeysArray, trimmedRayIndexValuesArray,
			scanArray, 
			skeletonArray, 
			chunkTotal, 
			sortedRayIndexKeysArray, sortedRayIndexValuesArray);

	}

	void HierarchyCreationWrapper(	
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Input Arrays containing the Sorted Ray Indices [Keys = Hashes, Values = Indices]
							int* sortedRayIndexKeysArray, 
							int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							int rayTotal,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray) {
								
		int hierarchyNodeWriteOffset = 0;
		int hierarchyNodeReadOffset = 0;
		int hierarchyNodeTotal = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
								
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
			hierarchyArray);
		
		// Create the Remaining Levels of the Ray Hierarchy.
		for(int hierarchyLevel=1; hierarchyLevel<HIERARCHY_MAXIMUM_DEPTH; hierarchyLevel++) {
			
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
	}

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
							int* hierarchyHitTotal) {

		// Calculate the Nodes Offset and Total
		int hierarchyNodeOffset[HIERARCHY_MAXIMUM_DEPTH+1];
		int hierarchyNodeTotal[HIERARCHY_MAXIMUM_DEPTH+1];

		int hitTotal = 0;
		int hitMaximum = 0;
		
		hierarchyNodeOffset[0] = 0;
		hierarchyNodeTotal[0] = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

		for(int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

			hierarchyNodeOffset[i] = hierarchyNodeTotal[i-1] + hierarchyNodeOffset[i-1];
			hierarchyNodeTotal[i] = hierarchyNodeTotal[i-1] / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal[i-1] % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
		}

		cout << "::HierarchyTraversalWrapper::" << endl;

		// Create the Hierarchy Hit Arrays
		for(int hierarchyLevel=HIERARCHY_MAXIMUM_DEPTH-1; hierarchyLevel>=0; hierarchyLevel--) {

			// Calculate the Hierarchy Hits
			if(hierarchyLevel == HIERARCHY_MAXIMUM_DEPTH-1) {

				/*// Grid based on the Hierarchy Node Count
				dim3 baseLevelBlock(1024);
				dim3 baseLevelGrid(hierarchyNodeTotal[hierarchyLevel]/baseLevelBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevel0Hits] Block = " << baseLevelBlock.x << " Threads " << "Grid = " << baseLevelGrid.x << " Blocks" << endl;
				#endif

				CreateHierarchyLevel0Hits<<<baseLevelGrid, baseLevelBlock>>>(
					hierarchyArray, 
					trianglePositionsArray,
					triangleTotal,
					hierarchyNodeOffset[hierarchyLevel], 
					hierarchyNodeTotal[hierarchyLevel], 
					headFlagsArray, 
					hierarchyHitsArray, trimmedHierarchyHitsArray);*/

				;

				// Grid based on the Hierarchy Node Count
				dim3 baseLevelBlock(1024);
				dim3 baseLevelGrid((hierarchyNodeTotal[hierarchyLevel] * triangleTotal)/baseLevelBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevel0Hits] Block = " << baseLevelBlock.x << " Threads " << "Grid = " << baseLevelGrid.x << " Blocks" << endl;
				#endif

				CreateHierarchyLevel0Hits<<<baseLevelGrid, baseLevelBlock>>>(
					hierarchyArray, 
					trianglePositionsArray,
					triangleTotal,
					hierarchyNodeOffset[hierarchyLevel], 
					hierarchyNodeTotal[hierarchyLevel], 
					headFlagsArray, 
					hierarchyHitsArray, trimmedHierarchyHitsArray);
			
				// Calculate the Hit Maximum for this Level
				hitMaximum = hierarchyNodeTotal[hierarchyLevel] * triangleTotal;
			}
			else {

				// Grid based on the Hierarchy Node Count
				dim3 baseLevelBlock(1024);
				dim3 baseLevelGrid(hitTotal/baseLevelBlock.x + 1);
				
				#ifdef BLOCK_GRID_DEBUG
					cout << "[CreateHierarchyLevelNHits] Block = " << baseLevelBlock.x << " Threads " << "Grid = " << baseLevelGrid.x << " Blocks" << endl;
				#endif
				
				CreateHierarchyLevelNHits<<<baseLevelGrid, baseLevelBlock>>>(
					hierarchyArray, 
					trianglePositionsArray,
					triangleTotal, 
					hitTotal,
					hierarchyNodeOffset[hierarchyLevel], 
					hierarchyNodeTotal[hierarchyLevel], 
					headFlagsArray, 
					hierarchyHitsArray, trimmedHierarchyHitsArray);

				// Calculate the Hit Maximum for this Level
				hitMaximum = hitTotal * HIERARCHY_SUBDIVISION;
			}

			// Create the Trim Scan Array
			Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

			// Grid based on the Hierarchy Hit Count
			dim3 baseHitBlock(1024);
			dim3 baseHitGrid(hitMaximum / baseHitBlock.x + 1);
				
			#ifdef BLOCK_GRID_DEBUG
				cout << "[TrimHierarchyLevelNHits] Block = " << baseHitBlock.x << " Threads " << "Grid = " << baseHitGrid.x << " Blocks" << endl;
			#endif

			CreateTrimmedHierarchyHits<<<baseHitGrid, baseHitBlock>>>(
				hierarchyHitsArray,
				scanArray,
				hitMaximum,
				trimmedHierarchyHitsArray);

			// Calculate the Hits Missed for this Level
			int missedHitTotal;
			// Check the Hit Total (last position of the scan array) 
			Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));
			
			// Calculate the Hit Total for this Level
			hitTotal = hitMaximum - missedHitTotal;
			
			cout << "Hit Maximum = " << hitMaximum << endl;
			cout << "Missed Hit Total = " << missedHitTotal << endl;
			cout << "Connected Hit Total : " << hitTotal << endl;
			cout << "Node Total : " << hierarchyNodeTotal[hierarchyLevel] << " (Offset: " << hierarchyNodeOffset[hierarchyLevel] * 2 << ")" << endl;

			*hierarchyHitTotal = hitTotal;

			if(hitTotal == 0)
				return;
		}

		cout << "Ray Total: " << rayTotal << endl;
	}

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
							unsigned int *pixelBufferObject) {
								
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
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			shadowFlagsArray,
			pixelBufferObject);

		// Grid based on the Screen Dimensions.
		dim3 colouringBlock(32,32);
		dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);								

		// Colour the Screen
		ColorPrimaryShadowRay<<<colouringGrid, colouringBlock>>>(windowWidth, windowHeight, lightTotal, cameraPosition, shadowFlagsArray, pixelBufferObject);
	}

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
							unsigned int *pixelBufferObject) {
								

		// Grid based on the Screen Dimensions.
		dim3 block(1024);
		dim3 grid(windowWidth*windowHeight / block.x + 1);

		// Prepare the Array
		PrepareArray<<<grid, block>>>(UINT_MAX, windowWidth * windowHeight, intersectionTimeArray);

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
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			intersectionTimeArray,
			pixelBufferObject);

		// Grid based on the Screen Dimensions.
		dim3 colouringBlock(32,32);
		dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

		// Colour the Screen
		ColorReflectionRay<<<colouringGrid, colouringBlock>>>(
			trianglePositionsArray, triangleNormalsArray, 
			windowWidth, windowHeight, 
			lightTotal, 
			cameraPosition, 
			intersectionTimeArray, 
			pixelBufferObject);
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