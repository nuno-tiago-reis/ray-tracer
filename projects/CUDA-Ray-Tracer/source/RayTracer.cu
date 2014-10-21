#include <stdio.h>

#include "math_functions.h"

#include "helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"

texture<float4, 1, cudaReadModeElementType> trianglePositionsTexture;	// the scene triangles store in a 1D float4 texture (they are stored as the 3 vertices)
texture<float4, 1, cudaReadModeElementType> triangleNormalsTexture;
texture<float4, 1, cudaReadModeElementType> triangleTangentsTexture;

texture<float2, 1, cudaReadModeElementType> triangleTextureCoordinatesTexture;

texture<float4, 1, cudaReadModeElementType> triangleAmbientPropertiesTexture;
texture<float4, 1, cudaReadModeElementType> triangleDiffusePropertiesTexture;
texture<float4, 1, cudaReadModeElementType> triangleSpecularPropertiesTexture;
texture<float, 1, cudaReadModeElementType> triangleSpecularConstantsTexture;

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float red, float green, float blue) {

	red		= clamp(red,	0.0f, 255.0f);
	green	= clamp(green,	0.0f, 255.0f);
	blue	= clamp(blue,	0.0f, 255.0f);

	return (int(red)<<16) | (int(green)<<8) | int(blue); // notice switch red and blue to counter the GL_BGRA
}

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
	float3 normal;

	int triangleIndex;

	__device__ HitRecord() {

			time = UINT_MAX;

			color = make_float3(0,0,0);
			normal = make_float3(0,0,0);

			triangleIndex = -1; 
	}

	__device__ void resetTime() {
		
			time = UINT_MAX;

			triangleIndex = -1;
	}
};

/* Ray - BoundingBox Intersection Code */
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

/* Ray - Triangle Intersection Code */
__device__ float RayTriangleIntersection(const Ray &ray, const float3 &vertex0, const float3 &edge1, const float3 &edge2) {  

	float3 tvec = ray.origin - vertex0;  
	float3 pvec = cross(ray.direction, edge2);  

	float  determinant  = dot(edge1, pvec);  
	determinant = __fdividef(1.0f, determinant);  

	/* First Test */
	float u = dot(tvec, pvec) * determinant;  
	if (u < 0.0f || u > 1.0f)  
		return -1.0f;  

	/* Second Test */
	float3 qvec = cross(tvec, edge1);  

	float v = dot(ray.direction, qvec) * determinant;  
	if (v < 0.0f || (u + v) > 1.0f)  
		return -1.0f;  

	return dot(edge2, qvec) * determinant;  
}  

/* Ray - Sphere Intersection Code */
__device__ int RaySphereIntersection(const Ray &ray, const float3 sphereCenter, const float sphereRadius, float &time) {

	float3 sr = ray.origin - sphereCenter;

	float b = dot(sr, ray.direction);
	float c = dot(sr, sr) - (sphereRadius * sphereRadius);
	float d = b * b - c;

	if(d > 0) {

		float e = sqrt(d);
		float t0 = -b-e;

		if(t0 < 0)
			time = -b+e;
		else
			time = min(-b-e,-b+e);

		return 1;
	}

	return 0;
}

__global__ void raytrace( unsigned int *outputPixelBufferObject,
						   const int width, const int height,
						   const int triangleTotal,
						   const float3 cameraRight, const float3 cameraUp, const float3 cameraDirection, 
						   const float3 cameraPosition,
						   const float3 lightPosition,
						   const float3 lightColor) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int rayDepth = 0;

	float3 rayOrigin = cameraPosition;
	float3 rayDirection = cameraDirection + cameraUp * (y / ((float)height) - 0.5f) + cameraRight * (x / ((float)width) - 0.5f);
	rayDirection = normalize(rayDirection);

	/* Ray used for this pixel */
	Ray ray(rayOrigin, rayDirection);
	/* Hit Record used to store Ray-Triangle Hit information */
	HitRecord hitRecord;
	hitRecord.color = make_float3(0,0.5,0.5);

	while(rayDepth < 7) {
		
		// Search through the triangles and find the nearest hit point
		for(int i = 0; i < triangleTotal; i++) {

			//hitRecord.color += make_float3(0.1, 0.1, 0.1);

			float4 v0 = tex1Dfetch(trianglePositionsTexture, i*3);
			float4 e1 = tex1Dfetch(trianglePositionsTexture, i*3+1);		//should be v1
			e1 = e1 - v0;
			float4 e2 = tex1Dfetch(trianglePositionsTexture, i*3+2);		//should be v2
			e2 = e2 - v0;

			float hitTime = RayTriangleIntersection(ray, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

			if(hitTime < hitRecord.time && hitTime > 0.001) {

				hitRecord.time = hitTime; 
				hitRecord.triangleIndex = i;

				hitRecord.color =  make_float3(0.5, 0.5, 0.5);
			}
		}

		// If no Triangle and no Light were intersected
		if(hitRecord.time == UINT_MAX)	
			break;

		// If a Triangle was intersected
		if(hitRecord.triangleIndex >= 0) {
			
			// create the normal
			float4 v0 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3);
			float4 e1 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3 + 1);
			e1 = e1 - v0;
			float4 e2 = tex1Dfetch(trianglePositionsTexture, hitRecord.triangleIndex * 3 + 2);
			e2 = e2 - v0;

			hitRecord.normal = cross(make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));
			hitRecord.normal = normalize(hitRecord.normal);

			// Blinn-Phong Shading - START
			float3 hitPoint = ray.origin + ray.direction * hitRecord.time;

			float3 lightDirection = lightPosition - hitPoint;

			float lightDistance = length(lightDirection);
			lightDirection = normalize(lightDirection);

			/* Diffuse Factor */
			float diffuseFactor = max(dot(lightDirection, hitRecord.normal), 0.0);
			clamp(diffuseFactor, 0.0f, 1.0f);

			if(diffuseFactor > 0.0) {

				bool shadow = false;

				Ray shadowRay(hitPoint + lightDirection * 0.001, lightDirection);
			
				/* Test Shadow Rays for each Triangle */
				for(int i = 0; i < triangleTotal; i++) {

					float4 v0 = tex1Dfetch(trianglePositionsTexture, i*3);
					float4 e1 = tex1Dfetch(trianglePositionsTexture, i*3+1);		//should be v1
					e1 = normalize(e1 - v0);
					float4 e2 = tex1Dfetch(trianglePositionsTexture, i*3+2);		//should be v2
					e2 = normalize(e2 - v0);

					float hitTime = RayTriangleIntersection(shadowRay, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

					if(hitTime > 0.001) {

						shadow = true;
						break;
					}
				}

				/* If there is no Triangle between the light source and the point hit */
				if(shadow == false) {
				
					/* Blinn-Phong approximation Halfway Vector */
					float3 halfwayVector = lightDirection -ray.direction;
					halfwayVector = normalize(halfwayVector);

					/* Light Attenuation */
					float lightAttenuation = 16.0 / lightDistance;

					/* Diffuse Component */
					-hitRecord.color += make_float3(1.0,0.0,0.0) * lightColor * diffuseFactor * lightAttenuation * (4 - rayDepth) / 4;

					/* Specular Factor */
					float specularFactor = powf(max(dot(halfwayVector, hitRecord.normal), 0.0), 25.0f);
					clamp(specularFactor, 0.0f, 1.0f);

					/* Specular Component */
					if(specularFactor > 0.0)
						hitRecord.color += make_float3(0.75,0.75,0.75) * lightColor * specularFactor * lightAttenuation * (4 - rayDepth) / 4;
				}
			}
			// Blinn-Phong Shading - END

			/* If the Object Hit is reflective */
			if(true) { //TODO

				rayDepth++;

				hitRecord.resetTime();

				float3 reflectedDirection = reflect(ray.direction, hitRecord.normal);

				ray = Ray(hitPoint + reflectedDirection * 0.001, reflectedDirection);
			}
		}
	}

	int outputColor = rgbToInt(hitRecord.color.x * 255, hitRecord.color.y * 255, hitRecord.color.z * 255);

	outputPixelBufferObject[y * width + x] = outputColor;
}

extern "C" {

	void RayTraceImage(unsigned int *outputPixelBufferObject, 
								int width, int height, 
								int triangleTotal,
								float3 cameraRight, float3 cameraUp, float3 cameraDirection,
								float3 cameraPosition,
								float3 lightPosition,
								float3 lightColor)
	{

		/*size_t blockWidth = (size_t)ceilf( width / 16.0f );
		size_t blockHeight = (size_t)ceilf( height / 16.0f );

		dim3 block(blockWidth, blockHeight, 1);
		dim3 grid(16, 16, 1);*/

		
		dim3 block(8,8,1);
		dim3 grid(width/block.x,height/block.y, 1);

		raytrace<<<grid, block>>>(outputPixelBufferObject, width, height, triangleTotal, cameraRight, cameraUp, cameraDirection, cameraPosition, lightPosition, lightColor);
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

	void bindTriangleAmbientProperties(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleAmbientPropertiesTexture.normalized = false;                      // access with normalized texture coordinates
		triangleAmbientPropertiesTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleAmbientPropertiesTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangleAmbientPropertiesTexture, cudaDevicePointer, channelDescriptor, size);
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

	void bindTriangleSpecularConstants(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleSpecularConstantsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleSpecularConstantsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleSpecularConstantsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float>();
		cudaBindTexture(0, triangleSpecularConstantsTexture, cudaDevicePointer, channelDescriptor, size);
	}
}