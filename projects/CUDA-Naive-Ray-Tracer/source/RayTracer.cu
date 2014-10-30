#include <stdio.h>

#include "cuda_runtime.h"

#include "math_functions.h"

#include "helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"

texture<float4, 1, cudaReadModeElementType> trianglePositionsTexture;	// the scene triangles store in a 1D float4 texture (they are stored as the 3 vertices)
texture<float4, 1, cudaReadModeElementType> triangleNormalsTexture;
texture<float4, 1, cudaReadModeElementType> triangleTangentsTexture;

texture<float2, 1, cudaReadModeElementType> triangleTextureCoordinatesTexture;

texture<float4, 1, cudaReadModeElementType> triangleDiffusePropertiesTexture;
texture<float4, 1, cudaReadModeElementType> triangleSpecularPropertiesTexture;

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> chessTexture;

const int initialDepth = 3;

const float initialRefractionIndex = 1.0f;

__device__ const int sphereTotal = 4;

__device__ const float sphereRadius = 5.0f;

// Hard coded for testing purposes
__device__ const float3 lightPosition =	{ 0.0f, 5.0f, 0.0f };
__device__ const float3 lightColor =	{ 1.0f, 1.0f, 1.0f };

// Hard coded for testing purposes
__device__ const float4 sphereDiffuses[] = {	{ 0.50754000f, 0.50754000f, 0.50754000f, 0.00f }, 
												{ 0.75164000f, 0.60648000f, 0.22648000f, 0.00f }, 
												{ 0.61424000f, 0.04136000f, 0.04136000f, 0.75f }, 
												{ 0.07568000f, 0.61424000f, 0.07568000f, 0.75f } };

__device__ const float4 sphereSpeculars[] = {	{ 0.50827300f, 0.50827300f, 0.50827300f, 102.0f }, 
												{ 0.62828100f, 0.55580200f, 0.36606500f, 102.0f }, 
												{ 0.72781100f, 0.62695900f, 0.62695900f, 152.0f }, 
												{ 0.63300000f, 0.72781100f, 0.63300000f, 152.0f } };

__device__ const float3 spherePositions[] = {	{  10.0f, -2.5f,  10.0f }, 
												{ -10.0f, -2.5f,  10.0f }, 
												{ -10.0f, -2.5f, -10.0f }, 
												{  10.0f, -2.5f, -10.0f } };

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

	int sphereIndex;
	int triangleIndex;

	__device__ HitRecord(const float3 &c) {

			time = UINT_MAX;

			color = c;

			point = make_float3(0,0,0);
			normal = make_float3(0,0,0);

			sphereIndex = -1;
			triangleIndex = -1; 
	}

	__device__ void resetTime() {
		
			time = UINT_MAX;

			point = make_float3(0,0,0);
			normal = make_float3(0,0,0);

			sphereIndex = -1;
			triangleIndex = -1;
	}
};

// Converts floating point rgb color to 8-bit integer
__device__ int rgbToInt(float red, float green, float blue) {

	red		= clamp(red,	0.0f, 255.0f);
	green	= clamp(green,	0.0f, 255.0f);
	blue	= clamp(blue,	0.0f, 255.0f);

	return (int(red)<<16) | (int(green)<<8) | int(blue); // notice switch red and blue to counter the GL_BGRA
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

// Casts a Ray and tests for intersections with the scenes geometry
__device__ float3 RayCast(	Ray ray,								
							// Triangle Dimensions
							const int triangleTotal,
							// Ray Bounce Depth
							const int depth,
							// Medium Refraction Index
							const float refractionIndex) {

	// Hit Record used to store Ray-Triangle Hit information - Initialized with Background Colour
	HitRecord hitRecord(make_float3(0.15f,0.15f,0.15f));
		
	// Search through the triangles and find the nearest hit point
	for(int i = 0; i < triangleTotal; i++) {

		float4 v0 = tex1Dfetch(trianglePositionsTexture, i * 3);
		float4 e1 = tex1Dfetch(trianglePositionsTexture, i * 3 + 1);
		e1 = e1 - v0;
		float4 e2 = tex1Dfetch(trianglePositionsTexture, i * 3 + 2);
		e2 = e2 - v0;

		float hitTime = RayTriangleIntersection(ray, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

		if(hitTime < hitRecord.time && hitTime > 0.001f) {

			hitRecord.time = hitTime; 
			hitRecord.sphereIndex = -1;
			hitRecord.triangleIndex = i;
		}
	}
	// Search through the spheres and find the nearest hit point
	for(int i = 0; i < sphereTotal; i++) {

		float hitTime = RaySphereIntersection(ray, spherePositions[i], sphereRadius);

		if(hitTime < hitRecord.time && hitTime > 0.001f) {

			hitRecord.time = hitTime; 
			hitRecord.sphereIndex = i;
			hitRecord.triangleIndex = -1;
		}
	}

	// If any Triangle was intersected
	if(hitRecord.triangleIndex >= 0 || hitRecord.sphereIndex >= 0) {

		// Needed for Ray Reflections
		float specularConstant;
		// Needed for Ray Refractions
		float refractionConstant;

		// Initialize the hit Color
		hitRecord.color = make_float3(0.0f, 0.0f, 0.0f);

		// Calculate the hit Triangle point
		hitRecord.point = ray.origin + ray.direction * hitRecord.time;

		// Calculate the hit Normal
		if(hitRecord.triangleIndex >= 0) {
			
			// Fetch the hit Triangles vertices
			float4 v0 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3);
			float4 v1 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3 + 1);
			float4 v2 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3 + 2);

			// Fetch the hit Triangles normals
			float4 n0 = tex1Dfetch(triangleNormalsTexture, hitRecord.triangleIndex * 3);
			float4 n1 = tex1Dfetch(triangleNormalsTexture, hitRecord.triangleIndex * 3 + 1);
			float4 n2 = tex1Dfetch(triangleNormalsTexture, hitRecord.triangleIndex * 3 + 2);

			// Normal calculation using Barycentric Interpolation
			float areaABC = length(cross(make_float3(v1) - make_float3(v0), make_float3(v2) - make_float3(v0)));
			float areaPBC = length(cross(make_float3(v1) - hitRecord.point, make_float3(v2) - hitRecord.point));
			float areaPCA = length(cross(make_float3(v0) - hitRecord.point, make_float3(v2) - hitRecord.point));

			hitRecord.normal = (areaPBC / areaABC) * make_float3(n0) + (areaPCA / areaABC) * make_float3(n1) + (1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(n2);
		}
		else { //if(hitRecord.sphereIndex >= 0) {
		
			hitRecord.normal = hitRecord.point - spherePositions[hitRecord.sphereIndex];

			if(length(ray.origin - spherePositions[hitRecord.sphereIndex]) < sphereRadius)
				hitRecord.normal = -hitRecord.normal;

			hitRecord.normal = normalize(hitRecord.normal);
		}

		// Blinn-Phong Shading - START
		float3 lightDirection = lightPosition - hitRecord.point;

		float lightDistance = length(lightDirection);
		lightDirection = normalize(lightDirection);

		// Diffuse Factor
		float diffuseFactor = max(dot(lightDirection, hitRecord.normal), 0.0f);
		clamp(diffuseFactor, 0.0f, 1.0f);

		if(diffuseFactor > 0.0f) {

			bool shadow = false;

			Ray shadowRay(hitRecord.point + lightDirection * 0.001, lightDirection);
			
			// Test Shadow Rays for each Triangle
			for(int i = 0; i < triangleTotal; i++) {

				float4 v0 = tex1Dfetch(trianglePositionsTexture, i * 3);
				float4 e1 = tex1Dfetch(trianglePositionsTexture, i * 3 + 1);
				e1 = e1 - v0;
				float4 e2 = tex1Dfetch(trianglePositionsTexture, i * 3 + 2);
				e2 = e2 - v0;

				float hitTime = RayTriangleIntersection(shadowRay, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

				if(hitTime > 0.001f) {

					shadow = true;
					break;
				}
			}

			// Test Shadow Rays for each Sphere
			for(int i = 0; i < sphereTotal; i++) {

				float hitTime = RaySphereIntersection(shadowRay, spherePositions[i], sphereRadius);

				if(hitTime > 0.001f) {

					shadow = true;
					break;
				}
			}

			// If there is no Triangle between the light source and the point hit
			if(shadow == false) {

				// Material Properties
				float4 diffuseColor;
				float4 specularColor;

				if(hitRecord.triangleIndex >= 0) {

					// Triangle Material Properties
					diffuseColor = tex1Dfetch(triangleDiffusePropertiesTexture, hitRecord.triangleIndex * 3);
					specularColor = tex1Dfetch(triangleSpecularPropertiesTexture, hitRecord.triangleIndex * 3);

					specularConstant = specularColor.w;
					refractionConstant = diffuseColor.w;
				}
				else {
				
					// Sphere Material Properties
					diffuseColor = sphereDiffuses[hitRecord.sphereIndex];
					specularColor = sphereSpeculars[hitRecord.sphereIndex];

					specularConstant = specularColor.w;
					refractionConstant = diffuseColor.w;
				}
				
				// Blinn-Phong approximation Halfway Vector
				float3 halfwayVector = lightDirection - ray.direction;
				halfwayVector = normalize(halfwayVector);

				// Light Attenuation
				float lightAttenuation = 16.0f / lightDistance;

				// Diffuse Component
				hitRecord.color += make_float3(diffuseColor) * lightColor * diffuseFactor * lightAttenuation;

				// Specular Factor
				float specularFactor = powf(max(dot(halfwayVector, hitRecord.normal), 0.0), specularConstant);
				clamp(specularFactor, 0.0f, 1.0f);

				// Specular Component
				if(specularFactor > 0.0f)
					hitRecord.color += make_float3(specularColor) * lightColor * specularFactor * lightAttenuation;
			}
		}
		// Blinn-Phong Shading - END

		// If max depth wasn't reached yet
		if(depth > 0)	{

			// If the Object Hit is reflective
			if(specularConstant > 0.0f) {

				// Calculate the Reflected Rays Direction
				float3 reflectedDirection = reflect(ray.direction, hitRecord.normal);

				// Cast the Reflected Ray
				//hitRecord.color += RayCast(Ray(hitRecord.point + reflectedDirection * 0.001f, reflectedDirection), triangleTotal, depth-1, refractionIndex) * 0.25;	//TODO
			}

			// If the Object Hit is translucid
			if(refractionConstant > 0.0f) {

				float newRefractionIndex;

				if(refractionIndex == 1.0f)
					newRefractionIndex = 0.75f;
				else
					newRefractionIndex = 1.0f;

				// Calculate the Refracted Rays Direction
				float3 refractedDirection = refract(ray.direction, hitRecord.normal, refractionIndex / newRefractionIndex);

				// Cast the Refracted Ray
				//hitRecord.color += RayCast(Ray(hitRecord.point + refractedDirection * 0.001f, refractedDirection), triangleTotal, depth-1, newRefractionIndex) * 0.75f * pow(0.95f,depth-3);
			}
		}
	}

	return hitRecord.color;
}

// Implementation of Whitteds Ray-Tracing Algorithm
__global__ void RayTracePixel(	unsigned int* pixelBufferObject,
								// Screen Dimensions
								const int width, 
								const int height, 
								// Triangle Dimensions
								const int triangleTotal,
								// Ray Bounce Depth
								const int depth,
								// Medium Refraction Index
								const float refractionIndex,
								// Camera Definitions
								const float3 cameraRight, 
								const float3 cameraUp, 
								const float3 cameraDirection,
								const float3 cameraPosition) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	/* Ray Creation */
	float3 rayOrigin = cameraPosition;
	float3 rayDirection = cameraDirection + cameraUp * (y / ((float)height) - 0.5f) + cameraRight * (x / ((float)width) - 0.5f);

	// Ray used to store Origin and Direction information
	Ray ray(rayOrigin, rayDirection);

	float3 pixelColor = RayCast(ray, triangleTotal, depth, refractionIndex);
	
	pixelBufferObject[y * width + x] = rgbToInt(pixelColor.x * 255, pixelColor.y * 255, pixelColor.z * 255);
}

extern "C" {

	void RayTraceWrapper(unsigned int *outputPixelBufferObject,
								int width, int height, 
								int triangleTotal,
								float3 cameraRight, float3 cameraUp, float3 cameraDirection,
								float3 cameraPosition) {

		/*outputPixelBufferObject[y * width + x] = rgbToInt(
			(float)((float)(width - x) / (float)width) * 255, 
			(float)((float)(height - y) / (float)height) * 255, 
			(float)((float)(width + height - x - y) / (float)(width + height)) * 255);*/

		dim3 block(16,16,1);
		dim3 grid(width/block.x,height/block.y, 1);

		RayTracePixel<<<grid, block>>>(	outputPixelBufferObject, 
										width, height,
										triangleTotal,
										initialDepth,
										initialRefractionIndex,
										cameraRight, cameraUp, cameraDirection,
										cameraPosition);
	}

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

	void bindTriangleTangents(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleTangentsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleTangentsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleTangentsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangleTangentsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleTextureCoordinates(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleTextureCoordinatesTexture.normalized = false;                      // access with normalized texture coordinates
		triangleTextureCoordinatesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleTextureCoordinatesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float2) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float2>();
		cudaBindTexture(0, triangleTextureCoordinatesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleDiffuseProperties(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleDiffusePropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		triangleDiffusePropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleDiffusePropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangleDiffusePropertiesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTriangleSpecularProperties(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleSpecularPropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		triangleSpecularPropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleSpecularPropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangleSpecularPropertiesTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindTextureArray(cudaArray *cudaArray) {

		chessTexture.normalized = true;                     // access with normalized texture coordinates
		chessTexture.filterMode = cudaFilterModeLinear;		// Point mode, so no 
		chessTexture.addressMode[0] = cudaAddressModeWrap;  // wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(chessTexture, cudaArray, channelDescriptor);
	}
}