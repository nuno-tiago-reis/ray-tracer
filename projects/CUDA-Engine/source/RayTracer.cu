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

// Intersection Constant
__constant__ __device__ static const float coneHeight = 65536.0f;

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

// CUDA Bounding Box Textures
texture<float4, 1, cudaReadModeElementType> boundingBoxesTexture;

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
__device__ static inline unsigned int CreateShadowRayIndex(float3 origin, float3 direction) {

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
	index = clamp((unsigned int)distance, (unsigned int)0, (unsigned int)128);
	index = (index << 7) | clamp((unsigned int)((atan2(origin.x, origin.x) + HALF_PI) * 20.0f), (unsigned int)0, (unsigned int)128);
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

// Axis Aligned Bounding Box - Node Intersection Code
__device__ static inline bool LineNodeIntersection(const float4 &sphere, const float4 &cone, const float3 &origin, const float3 &direction) {

	bool intersect(Cone cone, Vector3D dir, Vector3D P)

	// Beware, indigest formulaes !
	float tangent = tan(cone.w);

	// double sqTA = tan(cone.alpha) * tan(cone.alpha);
	float sqTA = tangent * tangent;

	// double A = dir.X * dir.X + dir.Y * dir.Y - dir.Z * dir.Z * sqTA;
	float a = direction.x * direction.x + direction.y * direction.y - direction.z * direction.z * sqTA;
	// double B = 2 * P.X * dir.X +2 * P.Y * dir.Y - 2 * (cone.H - P.Z) * dir.Z * sqTA;
	float b = origin.x * direction.x * 2.0f + origin.y * direction.y * 2.0f - (coneHeight - origin.z) * direction.z * 2.0f * sqTA;
	// double C = P.X * P.X + P.Y * P.Y - (cone.H - P.Z) * (cone.H - P.Z) * sqTA;
	float c = origin.x * origin.x + origin.y * origin.y - (coneHeight - origin.z) * (coneHeight - origin.z) * sqTA;

    // Now, we solve the polynom At² + Bt + C = 0
    //double delta = B * B - 4 * A * C;
	float delta = b * b - 4 * a * c;

	// No intersection between the cone and the line
	if(delta < 0.0f)
		return false; 

	// Solve for A
	if(a != 0) {

		// Check the two solutions (there might be only one, but that does not change a lot of things)

		//double t1 = (-B + sqrt(delta)) / (2 * A);
		float t1 = (-b + sqrt(delta)) / (2 * a);
		//double z1 = P.Z + t1 * dir.Z;
		float z1 = origin.z + t1 * direction.z;

		//bool t1_intersect = (t1 >= 0 && t1 <= 1 && z1 >= 0 && z1 <= cone.H);
		bool t1_intersect = (t1 >= 0.0f && t1 <= 1.0f && z1 >= 0.0f && z1 <= coneHeight);

		//double t2 = (-B - sqrt(delta)) / (2 * A);
		float t2 = (-b - sqrt(delta)) / (2 * a);
		//double z2 = P.Z + t2 * dir.Z;
		float z2 = origin.Z + t2 * direction.z;
		//bool t2_intersect = (t2 >= 0 && t2 <= 1 && z2 >= 0 && z2 <= cone.H);
		bool t2_intersect = (t2 >= 0.0f && t2 <= 1.0f && z2 >= 0.0f && z2 <= coneHeight);

		return (t1_intersect || t2_intersect);
	}

	// Solve for B
	if(b != 0.0f) {

		//double t = -C / B;
		double t = -c / b;

		//double z = P.Z + t * dir.Z;
		double z = origin.z + t * direction.z;

		return t >= 0.0f && t <= 1.0f && z >= 0.0f && z <= coneHeight;
	}
	
	// Solve for C
	return c == 0.0f;
}

__device__ static inline bool PlaneNodeIntersection(const float4 &sphere, const float4 &cone, const float3 &point, const float3 &u, const float3 &v) {

	bool intersection = LineNodeIntersection(sphere, cone, point, u) ||
						LineNodeIntersection(sphere, cone, point + v, u) ||
						LineNodeIntersection(sphere, cone, point, v) ||
						LineNodeIntersection(sphere, cone, point + u, v) ||;

	if(intersection == false) {

		// It is possible that either the part of the plan lies
		// entirely in the cone, or the inverse. We need to check.

		// Vector3D center = P + (u + v) / 2;
		float3 center = point + (u + v) / 2;

		// Is the face inside the cone (<=> center is inside the cone) ?
		if(center.z >= 0.0f && center.z <= coneHeight) {

			// double r = (H - center.Z) * tan(cone.alpha);
			double r = (coneHeight - center.Z) * tan(cone.W);

			if(center.x * center.x + center.y * center.y <= r)
				return true;
		}

		// Is the cone inside the face (this one is more tricky) ?
		// It can be resolved by finding whether the axis of the cone crosses the face.
		// First, find the plane coefficient (descartes equation)

		// Vector3D n = rect.u.crossProduct(rect.v);
		float3 normal = cross(u,v);
		
		// double d = -(rect.P.X * n.X + rect.P.Y * n.Y + rect.P.Z * n.Z);
		double d = -(point.x * normal.x + point.y * normal.y + point.z * normal.z);

	// Now, being in the face (ie, coordinates in (u, v) are between 0 and 1)
	// can be verified through scalar product
	if(normal.z != 0) {
		
		// Vector3D M(0, 0, -d/n.Z);
		// Vector3D MP = M - rect.P;
		float3 mp = make_float3(0.0f, 0.0f, -d/normal.z) - point;

		if(dot(mp, u) >= 0 || dot(mp, u) <= 1 || dot(mp, v) >= 0 || dot(mp, v) <= 1)
			return true;
	}

	return intersection;
}

__device__ static inline bool AxisAlignedBoundingBoxNodeIntersection(const float4 &sphere, const float4 &cone, const float4 &boundingBoxMaximum, const float4 &boundingBoxMinimum) {

	float3 x = make_float3(1.0f, 0.0f, 0.0f) * abs(boundingBoxMaximum.x - boundingBoxMinimum.x);
	float3 y = make_float3(0.0f, 1.0f, 0.0f) * abs(boundingBoxMaximum.y - boundingBoxMinimum.y);
	float3 z = make_float3(0.0f, 0.0f, 1.0f) * abs(boundingBoxMaximum.z - boundingBoxMinimum.z);

	return true;

	return	PlaneNodeIntersection(sphere, cone, boundingBoxMinimum, x, z) ||
			PlaneNodeIntersection(sphere, cone, boundingBoxMinimum, x, y) ||
			PlaneNodeIntersection(sphere, cone, boundingBoxMinimum, y, z) ||
			PlaneNodeIntersection(sphere, cone, boundingBoxMaximum,-x,-z) ||
			PlaneNodeIntersection(sphere, cone, boundingBoxMaximum,-x,-y) ||
			PlaneNodeIntersection(sphere, cone, boundingBoxMaximum,-y,-z);
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

__global__ void UpdateBoundingBox(
							// Input Array containing the updated Model Matrices.
							float* modelMatricesArray,
							// Auxiliary Variable containing the Vertex Total.
							unsigned int vertexTotal,
							// Output Array containing the updated Bounding Boxes.
							float4* boundingBoxArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= vertexTotal)
		return;

	// Matrices ID
	unsigned int matrixID = x / 2;

	// Model Matrix - Multiply each Vertex Position by it.
	float modelMatrix[16];

	for(int i=0; i<16; i++)
		modelMatrix[i] = modelMatricesArray[matrixID * 16 + i];
	
	float4 vertex = tex1Dfetch(boundingBoxesTexture, x);

	float updatedVertex[4];

	for(int i=0; i<3; i++) {

		updatedVertex[i] = 0.0f;
		updatedVertex[i] += modelMatrix[i * 4 + 0] * vertex.x;
		updatedVertex[i] += modelMatrix[i * 4 + 1] * vertex.y;
		updatedVertex[i] += modelMatrix[i * 4 + 2] * vertex.z;
	}
	
	// Store the updated Vertex Position.
	boundingBoxArray[x] = make_float4(updatedVertex[0], updatedVertex[1], updatedVertex[2], vertex.w);
}

__global__ void PreparePixels(	
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							const unsigned int lightTotal,
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

	for(unsigned int l = 0; l < lightTotal; l++) {

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

__global__ void Debug(	
							// Input Array containing the Rays.
							float3* rayArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							float3 cameraPosition,
							// Auxiliry Array containing the Head Flags.
							unsigned int *headFlagsArray,
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
	float3 rayOrigin = rayArray[(x + y * windowWidth) * 2];
	float3 rayDirection = rayArray[(x + y * windowWidth) * 2 + 1];

	float3 fragmentColor;

	if(headFlagsArray[x + y * windowWidth] == 0)
		fragmentColor = normalize(rayOrigin);
	else
		fragmentColor = make_float3(0.0f);

	pixelBufferObject[x + y * windowWidth] = RgbToInt(fragmentColor.x * 255.0f, fragmentColor.y * 255.0f, fragmentColor.z * 255.0f);
}

__global__ void CreateShadowRays(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variable containing the Light Index.
							const unsigned int lightIndex,
							// Output Array containing the unsorted Rays.
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

	unsigned int rayIndex = x + y * windowWidth;

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

	unsigned int rayIndex = x + y * windowWidth;

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

	unsigned int rayIndex = x + y * windowWidth;

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
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= nodeTotal)
		return;

	// Ray Origins are stored in the first offset
	float4 sphere = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2], 0.0f);
	// Ray Directions are stored in the second offset
	float4 cone = make_float4(rayArray[sortedRayIndexValuesArray[x * HIERARCHY_SUBDIVISION] * 2 + 1], 0.0f);
	
	for(unsigned int i=1; i<HIERARCHY_SUBDIVISION; i++) {

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

__global__ void CalculateBoundingBoxIntersections(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Output Array containing the updated Bounding Boxes.
							float4* boundingBoxArray,
							// Input Array containing the updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Bounding Box Total.
							const unsigned int boundingBoxTotal,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Node Offset.
							const unsigned int nodeOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Output Array containing the Inclusive Scan Output.
							unsigned int* headFlagsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / boundingBoxTotal;
	unsigned int boundingBoxID = x % boundingBoxTotal;

	if(nodeID >= nodeReadTotal || boundingBoxID >= boundingBoxTotal)
		return;

	float4 sphere = hierarchyArray[(nodeOffset + nodeID) * 2];
	float4 cone = hierarchyArray[(nodeOffset + nodeID) * 2 + 1];

	float4 boundingBoxMaximum = tex1Dfetch(boundingBoxesTexture, boundingBoxID * 2);
	float4 boundingBoxMinimum = tex1Dfetch(boundingBoxesTexture, boundingBoxID * 2 + 1);

	// Normal Matrix - Multiply each Vertex Position by it.
	float normalMatrix[16];

	for(int i=0; i<16; i++)
		normalMatrix[i] = normalMatricesArray[boundingBoxID * 16 + i];

	float updatedCone[3];

	for(int i=0; i<3; i++) {

		updatedCone[i] = 0.0f;
		updatedCone[i] += normalMatrix[i * 4 + 0] * cone.x;
		updatedCone[i] += normalMatrix[i * 4 + 1] * cone.y;
		updatedCone[i] += normalMatrix[i * 4 + 2] * cone.z;
	}

	// Calculate the Intersection and store the result
	headFlagsArray[nodeID * triangleTotal + (unsigned int)boundingBoxMaximum.w] = 
		((AxisAlignedBoundingBoxNodeIntersection(sphere, cone, boundingBoxMaximum, boundingBoxMinimum) == true) ? 
		((unsigned int)boundingBoxMinimum.w  - (unsigned int)boundingBoxMaximum.w + 1) : 0);

	headFlagsArray[nodeID * triangleTotal + (unsigned int)boundingBoxMaximum.w] = ((unsigned int)boundingBoxMinimum.w  - (unsigned int)boundingBoxMaximum.w + 1);
}

__global__ void CreateHierarchyLevel0Hits(
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Auxiliary Variable containing the Triangle Total.
							const unsigned int triangleTotal,
							// Auxiliary Variable containing the Triangle Offset.
							const unsigned int triangleOffset,
							// Auxiliary Variable containing the Node Read Total.
							const unsigned int nodeReadTotal,
							// Output Array containing the Inclusive Scan Output.
							unsigned int* scanArray,
							// Output Array containing the Head Flags Output.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Ray Hierarchy Hits.
							unsigned int* hierarchyHitsArray) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int nodeID = x / triangleTotal;
	unsigned int triangleID = x % triangleTotal;

	if(nodeID >= nodeReadTotal)
		return;
	
	// Load the Final Triangle ID
	unsigned int finalTriangleID = scanArray[x]; // - nodeID * triangleTotal;
	// Load the Final Triangle Offset
	unsigned int finalTriangleOffset = (nodeID > 0) ? scanArray[nodeID * triangleTotal - 1] : 0;

	hierarchyHitsArray[x] = ((nodeID << 16) & 0xFFFF0000) + ((triangleID - triangleOffset) & 0x0000FFFF);
	headFlagsArray[x] = (triangleID >= triangleOffset && triangleID < triangleTotal && triangleID < (finalTriangleID - finalTriangleOffset)) ? 0 : 1;
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

	unsigned int nodeID = (hit & 0xFFFF0000) >> 16;
	unsigned int triangleID = hit & 0x0000FFFF;

	float4 triangle = CreateTriangleBoundingSphere(
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3]), 
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 1]), 
		make_float3(trianglePositionsArray[(triangleOffset + triangleID) * 3 + 2]));

	for(unsigned int i=0; i<HIERARCHY_SUBDIVISION; i++) {

		if((nodeID * HIERARCHY_SUBDIVISION + i) < nodeWriteTotal) {

			float4 sphere = hierarchyArray[(nodeOffset + nodeID * HIERARCHY_SUBDIVISION + i) * 2];
			float4 cone = hierarchyArray[(nodeOffset + nodeID * HIERARCHY_SUBDIVISION + i) * 2 + 1];
	
			// Calculate the Intersection and store the result
			headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = (SphereNodeIntersection(sphere, cone, triangle, cos(cone.w), tan(cone.w)) == true) ? 0 : 1;
			hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] =  (((nodeID * HIERARCHY_SUBDIVISION + i) << 16) & 0xFFFF0000) + (triangleID & 0x0000FFFF);

			continue;
		}

		headFlagsArray[x * HIERARCHY_SUBDIVISION + i] = 1;
		hierarchyHitsArray[x * HIERARCHY_SUBDIVISION + i] = (((nodeID * HIERARCHY_SUBDIVISION + i) << 16) & 0xFFFF0000) + (triangleID & 0x0000FFFF);
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
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Load the Hierarchy Hit
	unsigned int hit = hierarchyHitsArray[x];

	unsigned int nodeID = (hit & 0xFFFF0000) >> 16;
	unsigned int triangleID = (hit & 0x0000FFFF);

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

		// Calculate the Interesection Time
		float intersectionDistance = RayTriangleIntersection(Ray(rayOrigin + rayDirection * epsilon, rayDirection), vertex0, edge1, edge2);

		// Calculate the Lights Distance to the Fragment
		if(intersectionDistance > epsilon && intersectionDistance < length(rayOrigin - make_float3(tex2D(fragmentPositionTexture, rayIndex % windowWidth, rayIndex / windowWidth))) - epsilon * 2.0f)
			shadowFlagsArray[rayIndex] = INT_MAX;
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
	float3 fragmentColor = make_float3(0.0f);

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = make_float3(tex2D(fragmentPositionTexture, x,y));
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	if(length(fragmentPosition) != 0.0f) {

		// Triangle Material Properties
		float4 fragmentDiffuseColor = tex2D(diffuseTexture, x,y);
		float4 fragmentSpecularColor = tex2D(specularTexture, x,y);

		for(unsigned int l = 0; l < lightTotal; l++) {

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
							unsigned int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= hitTotal)
		return;

	// Load the Hierarchy Hit
	unsigned int hit = hierarchyHitsArray[x];

	unsigned int nodeID = (hit & 0xFFFF0000) >> 16;
	unsigned int triangleID = (hit & 0x0000FFFF);

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

		for(unsigned int l = 0; l < lightTotal; l++) {

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

	
	void BoundingBoxUpdateWrapper(
							// Input Array containing the updated Model Matrices.
							float* modelMatricesArray,
							// Auxiliary Variable containing the Bounding Box Total.
							unsigned int boundingBoxTotal,
							// Output Array containing the updated Bounding Boxes.
							float4* boundingBoxArray) {

		unsigned int vertexTotal = boundingBoxTotal * 2;

		// Grid based on the Triangle Count
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(boundingBoxTotal / multiplicationBlock.x + 1);
		
		// Model Matrix Multiplication
		UpdateBoundingBox<<<multiplicationGrid, multiplicationBlock>>>(
			modelMatricesArray, 
			vertexTotal,
			boundingBoxArray);
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
		unsigned int rayTotal = windowWidth * windowHeight;

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

	void ScreenPreparationWrapper(
							// Input Array containing the Unsorted Rays.
							float3* rayArray, 
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Auxiliry Array containing the Head Flags.
							unsigned int *headFlagsArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

		// Grid based on the Screen Dimensions.
		dim3 block(32,32);
		dim3 grid(windowWidth/block.x + 1, windowHeight/block.y + 1);

		// Prepare the Screen
		//PreparePixels<<<grid, block>>>(windowWidth, windowHeight, pixelBufferObject);

		Debug<<<grid, block>>>(rayArray, windowWidth, windowHeight, lightTotal, cameraPosition, headFlagsArray, pixelBufferObject);
	}

	void ShadowRayCreationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Variables containing the Light Index.
							const unsigned int lightIndex,
							// Output Array containing the Unsorted Rays.
							float3* rayArray,
							// Output Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Output Arrays containing the Unsorted Ray Indices.
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray) {

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
							unsigned int* rayIndexKeysArray, 
							unsigned int* rayIndexValuesArray,
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Auxiliary Array containing the Ray Head Flags.
							unsigned int* headFlagsArray, 
							// Auxiliary Array containing the Inclusive Scan Output.
							unsigned int* scanArray, 
							// Output Arrays containing the Trimmed Ray Indices [Keys = Hashes, Values = Indices]
							unsigned int* trimmedRayIndexKeysArray, 
							unsigned int* trimmedRayIndexValuesArray,
							// Output Variable containing the Number of Rays.
							unsigned int* rayTotal) {
	
		// Maximum Number of Rays being cast per Frame
		unsigned int rayMaximum = windowWidth * windowHeight;

		// Calculate the Inclusive Scan using the Ray Head Flags.
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, rayMaximum));

		// Number of Pixels per Frame
		unsigned int screenDimensions = windowWidth * windowHeight;

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
							unsigned int* chunkIndexKeysArray, 
							unsigned int* chunkIndexValuesArray,
							// Auxiliary Variable containing the Number of Chunks.
							const unsigned int chunkTotal,
							// Output Arrays containing the Sorted Ray Chunks [Keys = Hashes, Values = Indices]
							unsigned int* sortedChunkIndexKeysArray, 
							unsigned int* sortedChunkIndexValuesArray) {
		
		// Sort the Chunks
		Utility::checkCUDAError("cub::DeviceRadixSort::SortPairs()", 
			cub::DeviceRadixSort::SortPairs(radixSortTemporaryStorage, radixSortTemporaryStoreBytes,
			chunkIndexKeysArray, sortedChunkIndexKeysArray,
			chunkIndexValuesArray, sortedChunkIndexValuesArray, 
			chunkTotal));
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
							unsigned int* sortedRayIndexKeysArray, 
							unsigned int* sortedRayIndexValuesArray,
							// Auxiliary Variable containing the Ray Total.
							const unsigned int rayTotal,
							// Output Array containing the Ray Hierarchy.
							float4* hierarchyArray) {
								
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
	}

	void HierarchyTraversalWarmUpWrapper(	
							// Input Array containing the Ray Hierarchy.
							float4* hierarchyArray,
							// Output Array containing the updated Bounding Boxes.
							float4* boundingBoxArray,
							// Input Array containing the updated Normal Matrices.
							float* normalMatricesArray,
							// Auxiliary Variable containing the Bounding Box Total.
							const unsigned int boundingBoxTotal,
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
	
		unsigned int hierarchyNodeOffset = 0;
		unsigned int hierarchyNodeTotal = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

		for(unsigned int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

			hierarchyNodeOffset = hierarchyNodeTotal + hierarchyNodeOffset;
			hierarchyNodeTotal = hierarchyNodeTotal / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
		}

		unsigned int hitMaximum = hierarchyNodeTotal * triangleTotal;

		// Grid based on the Hierarchy Hit Count
		dim3 hitMaximumBlock(1024);
		dim3 hitMaximumGrid(hitMaximum/hitMaximumBlock.x + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[PrepareArray] Grid = " << grid.x << endl;
		#endif

		// Prepare the Array
		PrepareArray<<<hitMaximumGrid, hitMaximumBlock>>>(0, hitMaximum, headFlagsArray);

		// Grid based on the Hierarchy Node * Bounding Box Count
		dim3 boundingBoxBlock(1024);
		dim3 boundingBoxGrid((hierarchyNodeTotal * boundingBoxTotal)/boundingBoxBlock.x + 1);
				
		#ifdef BLOCK_GRID_DEBUG
			cout << "[CreateHierarchyLevel0Hits] Block = " << boundingBoxBlock.x << " Threads " << "Grid = " << boundingBoxGrid.x << " Blocks" << endl;
		#endif

		CalculateBoundingBoxIntersections<<<boundingBoxGrid, boundingBoxBlock>>>(
			hierarchyArray, 
			boundingBoxArray,
			normalMatricesArray,
			boundingBoxTotal,
			triangleTotal,
			triangleOffset,
			hierarchyNodeOffset, 
			hierarchyNodeTotal, 
			headFlagsArray);

		/*unsigned int* duplicateHeadFlags = new unsigned int[hierarchyNodeTotal * triangleTotal];

		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&duplicateHeadFlags[0], headFlagsArray, hierarchyNodeTotal * triangleTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		for(int i=0; i<hierarchyNodeTotal; i++) {
		
			printf("Node [%d] :: ", i);
		
			for(int j=0; j<triangleTotal; j++) {

				printf("\t%u", duplicateHeadFlags[i * triangleTotal + j]);
			}
			printf("\n");
		}*/
		;

		// Create the Population Scan Array
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

		#ifdef BLOCK_GRID_DEBUG
			cout << "[CreateHierarchyLevel0Hits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
		#endif

		/*Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&duplicateHeadFlags[0], scanArray, hierarchyNodeTotal * triangleTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		for(int i=0; i<hierarchyNodeTotal; i++) {
		
			printf("Node [%d] :: ", i);
		
			for(int j=0; j<triangleTotal; j++) {

				printf("\t%u", duplicateHeadFlags[i * triangleTotal + j]);
			}
			printf("\n");
		}*/

		CreateHierarchyLevel0Hits<<<hitMaximumGrid ,hitMaximumBlock>>>(
			hierarchyArray,
			triangleTotal,
			triangleOffset,
			hierarchyNodeTotal,
			scanArray,
			headFlagsArray, 
			hierarchyHitsArray);

		/*unsigned int* duplicateHierarchyHits = new unsigned int[hierarchyNodeTotal * triangleTotal];

		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&duplicateHierarchyHits[0], hierarchyHitsArray, hierarchyNodeTotal * triangleTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		for(int i=0; i<hierarchyNodeTotal; i++) {
		
			printf("Node [%d] :: ", i);
		
			for(int j=0; j<triangleTotal; j++) {

				unsigned int hit = duplicateHierarchyHits[i * triangleTotal + j];

				unsigned int nodeID = (hit & 0xFFFF0000) >> 16;
				unsigned int triangleID = (hit & 0x0000FFFF);

				printf("\t%03u#%03u", nodeID, triangleID);
			}
			printf("\n");
		}
		exit(0);*/
		;

		// Create the Trim Scan Array
		Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

		#ifdef BLOCK_GRID_DEBUG
			cout << "[CreateTrimmedHierarchyHits] Block = " << hitMaximumBlock.x << " Threads " << "Grid = " << hitMaximumGrid.x << " Blocks" << endl;
		#endif

		CreateTrimmedHierarchyHits<<<hitMaximumGrid, hitMaximumBlock>>>(
			hierarchyHitsArray,
			scanArray,
			hitMaximum,
			trimmedHierarchyHitsArray);

		Utility::checkCUDAError("CreateTrimmedHierarchyHits::cudaDeviceSynchronize()", cudaDeviceSynchronize());
		Utility::checkCUDAError("CreateTrimmedHierarchyHits::cudaGetLastError()", cudaGetLastError());

		// Calculate the Hits Missed for this Level
		int missedHitTotal;
		// Check the Hit Total (last position of the scan array) 
		Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));

		// Calculate the Hit Total for this Level
		*hierarchyHitTotal = hitMaximum - missedHitTotal;

		/*cout << "["<< 0 << "]" << " Hit Maximum = " << hitMaximum << endl;
		cout << "["<< 0 << "]" << " Missed Hit Total = " << missedHitTotal << endl;
		cout << "["<< 0 << "]" << " Connected Hit Total : " << *hierarchyHitTotal << endl;

		exit(0);*/
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

		// Calculate the Nodes Offset and Total
		unsigned int hierarchyNodeOffset[HIERARCHY_MAXIMUM_DEPTH];
		unsigned int hierarchyNodeTotal[HIERARCHY_MAXIMUM_DEPTH];
		
		hierarchyNodeOffset[0] = 0;
		hierarchyNodeTotal[0] = rayTotal / HIERARCHY_SUBDIVISION + (rayTotal % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);

		for(unsigned int i=1; i<HIERARCHY_MAXIMUM_DEPTH; i++) {

			hierarchyNodeOffset[i] = hierarchyNodeTotal[i-1] + hierarchyNodeOffset[i-1];
			hierarchyNodeTotal[i] = hierarchyNodeTotal[i-1] / HIERARCHY_SUBDIVISION + (hierarchyNodeTotal[i-1] % HIERARCHY_SUBDIVISION != 0 ? 1 : 0);
		}

		#ifdef TRAVERSAL_DEBUG
			cout << "::HierarchyTraversalWrapper::" << endl;
		#endif

		// Create the Hierarchy Hit Arrays
		for(int hierarchyLevel=HIERARCHY_MAXIMUM_DEPTH-1; hierarchyLevel>=0; hierarchyLevel--) {

			if((*hierarchyHitTotal) == 0)
				return;

			// Calculate the Hit Maximum for this Level
			unsigned int hitMaximum = *hierarchyHitTotal * HIERARCHY_SUBDIVISION;
			unsigned int hitTotal = *hierarchyHitTotal;

			cout << "[Entry "<< hierarchyLevel << "] Hit Maximum = " << hitMaximum << endl;
			cout << "[Entry "<< hierarchyLevel << "] Hit Total = " << hitTotal << endl;

			#ifdef TRAVERSAL_DEBUG
				cout << "["<< hierarchyLevel << "]" << " Memory Usage: " << (float)hitMaximum/(float)(*hierarchyHitMemoryTotal) << endl;
			#endif

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
			
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaDeviceSynchronize()", cudaDeviceSynchronize());
			Utility::checkCUDAError("ShadowRayPreparationWrapper::cudaGetLastError()", cudaGetLastError());

			// Create the Trim Scan Array
			Utility::checkCUDAError("cub::DeviceScan::InclusiveSum()", cub::DeviceScan::InclusiveSum(scanTemporaryStorage, scanTemporaryStoreBytes, headFlagsArray, scanArray, hitMaximum));

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
			Utility::checkCUDAError("cudaMemcpy()", cudaMemcpy(&missedHitTotal, &scanArray[hitMaximum - 1], sizeof(int), cudaMemcpyDeviceToHost));
			
			// Calculate the Hit Total for this Level
			*hierarchyHitTotal = hitMaximum - missedHitTotal;

			cout << "[Exit "<< hierarchyLevel << "] Hit Maximum = " << hitMaximum << endl;
			cout << "[Exit "<< hierarchyLevel << "] Missed Hit Total = " << missedHitTotal << endl;
			cout << "[Exit "<< hierarchyLevel << "] Connected Hit Total : " << *hierarchyHitTotal << endl;
			
			#ifdef TRAVERSAL_DEBUG
				cout << "["<< hierarchyLevel << "]" << " Hit Maximum = " << hitMaximum << endl;
				cout << "["<< hierarchyLevel << "]" << " Missed Hit Total = " << missedHitTotal << endl;
				cout << "["<< hierarchyLevel << "]" << " Connected Hit Total : " << *hierarchyHitTotal << endl;
				cout << "["<< hierarchyLevel << "]" << " Node Total : " << hierarchyNodeTotal[hierarchyLevel] << " (Offset: " << hierarchyNodeOffset[hierarchyLevel] * 2 << ")" << endl;
			#endif
		}
		
		#ifdef TRAVERSAL_DEBUG
			cout << "Ray Total: " << rayTotal << endl;
		#endif
	}

	void ShadowRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(1024);
		dim3 grid(windowWidth*windowHeight / block.x + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[PrepareArray] Grid = " << grid.x << endl;
		#endif

		// Prepare the Array
		PrepareArray<<<grid, block>>>(0, windowWidth * windowHeight, shadowFlagsArray);
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
							// Auxiliary Variable containing the Number of Lights.
							const unsigned int lightTotal,
							// Auxiliary Variables containing the Camera Position.
							const float3 cameraPosition,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* shadowFlagsArray,
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
			triangleOffset,
			windowWidth, windowHeight,
			lightTotal,
			cameraPosition,
			shadowFlagsArray,
			pixelBufferObject);
	}

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
							unsigned int *pixelBufferObject) {

		// Grid based on the Screen Dimensions.
		dim3 colouringBlock(32,32);
		dim3 colouringGrid(windowWidth/colouringBlock.x + 1, windowHeight/colouringBlock.y + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[ColorPrimaryShadowRay] Grid = " << colouringGrid.x << "," << colouringGrid.y << endl;
		#endif

		// Colour the Screen
		ColorPrimaryShadowRay<<<colouringGrid, colouringBlock>>>(windowWidth, windowHeight, lightTotal, cameraPosition, shadowFlagsArray, pixelBufferObject);
	}

	void ReflectionRayPreparationWrapper(
							// Auxiliary Variables containing the Screen Dimensions.
							const unsigned int windowWidth, const unsigned int windowHeight,
							// Output Array containing the Shadow Ray Flags.
							unsigned int* intersectionTimeArray) {

		// Grid based on the Screen Dimensions.
		dim3 block(1024);
		dim3 grid(windowWidth*windowHeight / block.x + 1);

		#ifdef BLOCK_GRID_DEBUG 
			cout << "[PrepareArray] Grid = " << grid.x << endl;
		#endif

		// Prepare the Array
		PrepareArray<<<grid, block>>>(UINT_MAX, windowWidth * windowHeight, intersectionTimeArray);
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
							unsigned int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

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
			intersectionTimeArray,
			pixelBufferObject);
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
							// Auxiliary Array containing the Intersection Times.
							unsigned int* intersectionTimeArray,
							// Output Array containing the Screen Buffer.
							unsigned int *pixelBufferObject) {

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

	// CUDA Bounxing Box Texture Binding Functions
	void bindBoundingBoxes(float *cudaDevicePointer, unsigned int boundingBoxTotal) {

		materialDiffusePropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		materialDiffusePropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		materialDiffusePropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * boundingBoxTotal * 2;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, boundingBoxesTexture, cudaDevicePointer, channelDescriptor, size);
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