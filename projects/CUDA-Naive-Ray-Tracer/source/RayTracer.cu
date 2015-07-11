#include <stdio.h>

#include "cuda_runtime.h"

#include "math_functions.h"

#include "helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"

// Triangle Textures
texture<float4, 1, cudaReadModeElementType> trianglePositionsTexture;
texture<float4, 1, cudaReadModeElementType> triangleNormalsTexture;
texture<float4, 1, cudaReadModeElementType> triangleTangentsTexture;

texture<float2, 1, cudaReadModeElementType> triangleTextureCoordinatesTexture;

texture<int1, 1, cudaReadModeElementType> triangleMaterialIDsTexture;

// Material Textures
texture<float4, 1, cudaReadModeElementType> materialDiffusePropertiesTexture;
texture<float4, 1, cudaReadModeElementType> materialSpecularPropertiesTexture;

// Light Textures
texture<float4, 1, cudaReadModeElementType> lightPositionsTexture;
texture<float4, 1, cudaReadModeElementType> lightDirectionsTexture;

texture<float1, 1, cudaReadModeElementType> lightCutOffsTexture;

texture<float4, 1, cudaReadModeElementType> lightColorsTexture;
texture<float2, 1, cudaReadModeElementType> lightIntensitiesTexture;
texture<float4, 1, cudaReadModeElementType> lightAttenuationsTexture;

// OpenGL Texture Textures
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> shadingTexture;

const int initialDepth = 3;

const float initialRefractionIndex = 1.0f;

// Hard coded for testing purposes
__device__ const int sphereTotal = 1;

__device__ const float sphereRadius = 1.25f;

__device__ const float4 sphereDiffuses[] = {	{ 0.50754000f, 0.50754000f, 0.50754000f, 0.00f }, 
												{ 0.75164000f, 0.60648000f, 0.22648000f, 0.00f }, 
												{ 0.61424000f, 0.04136000f, 0.04136000f, 0.00f }, 
												{ 0.07568000f, 0.61424000f, 0.07568000f, 0.00f },
												{ 0.80754000f, 0.80754000f, 0.80754000f, 1.00f } };

__device__ const float4 sphereSpeculars[] = {	{ 0.50827300f, 0.50827300f, 0.50827300f, 155.0f }, 
												{ 0.62828100f, 0.55580200f, 0.36606500f, 155.0f }, 
												{ 0.72781100f, 0.62695900f, 0.62695900f, 155.0f }, 
												{ 0.63300000f, 0.72781100f, 0.63300000f, 155.0f },
												{ 0.50827300f, 0.50827300f, 0.50827300f,   0.0f } };

__device__ const float3 spherePositions[] = {	{   0.0f,  0.5f,   0.0f }, 
												{ -10.0f, -2.5f,  10.0f }, 
												{ -10.0f, -2.5f, -10.0f }, 
												{  10.0f, -2.5f, -10.0f },
												{   0.0f, -2.5f,  0.0f }};

// Ray testing Constant
__device__ const float epsilon = 0.01f;

// Shadow Grid Dimensions and pre-calculated Values
__device__ const int shadowGridWidth = 3;
__device__ const int shadowGridHeight = 3;

__device__ const int shadowGridHalfWidth = 1;
__device__ const int shadowGridHalfHeight = 1;

//__device__ const int shadowGridDimension = 25;
__device__ const float shadowGridDimensionInverse = 1.0f/9.0f;

__device__ const float shadowCellSize = 0.20f;

// Anti-Aliasing Constants
__device__ const int antiAliasingGridWidth = 2;
__device__ const int antiAliasingGridHeight = 2;

__device__ const int antiAliasingGridHalfWidth = 1;
__device__ const int antiAliasingGridHalfHeight = 1;

__device__ const float antiAliasingGridDimensionInverse = 1.0f/5.0f; //account for the center of the pixel too

__device__ const float antiAliasingCellSize = 0.50f;

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

// Converts 8-bit integer to floating point rgb color
__device__ float3 intToRgb(int color) {

	float red	= color & 255;
	float green	= (color >> 8) & 255;
	float blue	= (color >> 16) & 255;

	return make_float3(red, green, blue);
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

__device__ float RaySphereIntersection2(const Ray &ray, const float3 sphereCenter, const float sphereRadius) {

	float3 distance = sphereCenter - ray.origin;

	float d = pow(sphereCenter.x - ray.origin.x,2) + pow(sphereCenter.y - ray.origin.y,2) + pow(sphereCenter.z - ray.origin.z,2);
	
	if(d == pow(sphereRadius,2))
		return -1.0f;
	
	float B = dot(distance, ray.direction);

	if(d > pow(sphereRadius,2) && B < 0.0f)
		return -1.0f;

	float C = d - pow(sphereRadius,2);

	float R = pow(B,2) - C;

	if(R < 0.0f)
		return -1.0f;

	float Ti = 0.0f;

	if(d > pow(sphereRadius,2))
		Ti = B - sqrt(R);
	else if(d < pow(sphereRadius,2))
		Ti = B + sqrt(R);

	return Ti;
}

// Casts a Ray and tests for intersections with the scenes geometry
__device__ float3 RayCast(	Ray ray, 
							// Total Number of Triangles in the Scene
							const int triangleTotal,
							// Total Number of Lights in the Scene
							const int lightTotal,
							// Ray Bounce Depth
							const int depth,
							// Medium Refraction Index
							const float refractionIndex) {

	// Hit Record used to store Ray-Triangle Hit information - Initialized with Background Colour
	HitRecord hitRecord(make_float3(0.15f,0.15f,0.15f));

	if(depth == 2)
		return ray.direction;
		
	// Search through the triangles and find the nearest hit point
	for(int i = 0; i < triangleTotal; i++) {

		float4 v0 = tex1Dfetch(trianglePositionsTexture, i * 3);
		float4 e1 = tex1Dfetch(trianglePositionsTexture, i * 3 + 1);
		e1 = e1 - v0;
		float4 e2 = tex1Dfetch(trianglePositionsTexture, i * 3 + 2);
		e2 = e2 - v0;

		float hitTime = RayTriangleIntersection(ray, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

		if(hitTime < hitRecord.time && hitTime > epsilon) {

			hitRecord.time = hitTime; 
			hitRecord.sphereIndex = -1;
			hitRecord.triangleIndex = i;
		}
	}

	// Search through the spheres and find the nearest hit point
	for(int i = 0; i < sphereTotal; i++) {

		float hitTime = RaySphereIntersection(ray, spherePositions[i], sphereRadius);

		if(hitTime < hitRecord.time && hitTime > epsilon) {

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

		float areaABC;
		float areaPBC;
		float areaPCA;

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
			areaABC = length(cross(make_float3(v1) - make_float3(v0), make_float3(v2) - make_float3(v0)));
			areaPBC = length(cross(make_float3(v1) - hitRecord.point, make_float3(v2) - hitRecord.point));
			areaPCA = length(cross(make_float3(v0) - hitRecord.point, make_float3(v2) - hitRecord.point));

			hitRecord.normal = (areaPBC / areaABC) * make_float3(n0) + (areaPCA / areaABC) * make_float3(n1) + (1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(n2);
		}
		else { //if(hitRecord.sphereIndex >= 0) {
		
			hitRecord.normal = hitRecord.point - spherePositions[hitRecord.sphereIndex];

			if(length(ray.origin - spherePositions[hitRecord.sphereIndex]) < sphereRadius)
				hitRecord.normal = -hitRecord.normal;

			hitRecord.normal = normalize(hitRecord.normal);
		}

		// Blinn-Phong Shading (Soft Shadows - START)

		// Material Properties
		float4 diffuseColor;
		float4 specularColor;

		if(hitRecord.triangleIndex >= 0) {

			// Triangle Material Properties
			int1 materialID = tex1Dfetch(triangleMaterialIDsTexture, hitRecord.triangleIndex * 3);

			diffuseColor = tex1Dfetch(materialDiffusePropertiesTexture, materialID.x);
			specularColor = tex1Dfetch(materialSpecularPropertiesTexture, materialID.x);

			specularConstant = specularColor.w;
			refractionConstant = diffuseColor.w;

			//If using Textures

			/*float2 uv0 = tex1Dfetch(triangleTextureCoordinatesTexture, hitRecord.triangleIndex * 3);
			float2 uv1 = tex1Dfetch(triangleTextureCoordinatesTexture, hitRecord.triangleIndex * 3 + 1);
			float2 uv2 = tex1Dfetch(triangleTextureCoordinatesTexture, hitRecord.triangleIndex * 3 + 2);

			float2 uv = (areaPBC / areaABC) * uv0 + (areaPCA / areaABC) * uv1 + (1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * uv2;

			uchar4 textureColor = tex2D(shadingTexture, uv.x, uv.y);

			diffuseColor = make_float4((float)textureColor.x / 255.0f, (float)textureColor.y / 255.0f, (float)textureColor.z / 255.0f, 1.0f);
			specularColor = make_float4((float)textureColor.x / 255.0f, (float)textureColor.y / 255.0f, (float)textureColor.z / 255.0f, 1.0f);

			specularConstant = 150.0f;
			refractionConstant = 0.0f;*/
		}
		else { //if(hitRecord.sphereIndex >= 0) {
				
			// Sphere Material Properties
			diffuseColor = sphereDiffuses[hitRecord.sphereIndex];
			specularColor = sphereSpeculars[hitRecord.sphereIndex];

			specularConstant = specularColor.w;
			refractionConstant = diffuseColor.w;
		}

		for(int l = 0; l < lightTotal; l++) {

			float3 lightPosition = make_float3(tex1Dfetch(lightPositionsTexture, l));

			// Light Direction and Distance
			float3 lightDirection = lightPosition - hitRecord.point;

			float lightDistance = length(lightDirection);
			lightDirection = normalize(lightDirection);

			// Light Direction perpendicular plane base vectors 
			float3 lightPlaneAxisA;
			float3 lightPlaneAxisB;
			float3 w;

			// Check which is the component with the smallest coeficient
			float m = min(abs(lightDirection.x),max(abs(lightDirection.y),abs(lightDirection.z)));

			if(abs(lightDirection.x) == m) {

				w = make_float3(1.0f,0.0f,0.0f);
			}
			else if(abs(lightDirection.y) == m) {

				w = make_float3(0.0f,1.0f,0.0f);
			}
			else { //if(abs(lightDirection.z) == m) {

				w = make_float3(0.0f,0.0f,1.0f);
			}

			// Calculate the perpendicular plane base vectors
			lightPlaneAxisA = cross(w, lightDirection);
			lightPlaneAxisB = cross(lightDirection,lightPlaneAxisA);

			// Shadow Grid for Soft Shadows
			for(int i=0; i<shadowGridWidth; i++) {
				for(int j=0; j<shadowGridHeight; j++) {

					float3 interpolatedPosition = lightPosition + lightPlaneAxisA * (i-shadowGridHalfWidth) * shadowCellSize + lightPlaneAxisB * (j-shadowGridHalfHeight) * shadowCellSize;

					float3 interpolatedDirection = interpolatedPosition - hitRecord.point;
					interpolatedDirection = normalize(interpolatedDirection);

					// Diffuse Factor
					float diffuseFactor = max(dot(interpolatedDirection, hitRecord.normal), 0.0f);
					clamp(diffuseFactor, 0.0f, 1.0f);

					if(diffuseFactor > 0.0f) {

						bool shadow = false;

						Ray shadowRay(hitRecord.point + interpolatedDirection * epsilon, interpolatedDirection);
			
						// Test Shadow Rays for each Triangle
						for(int k = 0; k < triangleTotal; k++) {

							float4 v0 = tex1Dfetch(trianglePositionsTexture, k * 3);
							float4 e1 = tex1Dfetch(trianglePositionsTexture, k * 3 + 1);
							e1 = e1 - v0;
							float4 e2 = tex1Dfetch(trianglePositionsTexture, k * 3 + 2);
							e2 = e2 - v0;

							float hitTime = RayTriangleIntersection(shadowRay, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

							if(hitTime > epsilon) {

								shadow = true;
								break;
							}
						}

						// Test Shadow Rays for each Sphere
						for(int k = 0; k < sphereTotal; k++) {

							float hitTime = RaySphereIntersection(shadowRay, spherePositions[k], sphereRadius);

							if(hitTime > epsilon) {

								shadow = true;
								break;
							}
						}

						if(shadow == false) {

							// Blinn-Phong approximation Halfway Vector
							float3 halfwayVector = interpolatedDirection - ray.direction;
							halfwayVector = normalize(halfwayVector);

							// Light Color
							float3 lightColor =  make_float3(tex1Dfetch(lightColorsTexture, l));
							// Light Intensity (x = diffuse, y = specular)
							float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
							// Light Attenuation (x = constant, y = linear, z = exponential)
							float3 lightAttenuation = make_float3(tex1Dfetch(lightAttenuationsTexture, l));

							float attenuation = 1.0f / (lightAttenuation.x + lightDistance * lightAttenuation.y + lightDistance * lightDistance * lightAttenuation.z);

							// Diffuse Component
							hitRecord.color += make_float3(diffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation * shadowGridDimensionInverse;

							// Specular Factor
							float specularFactor = powf(max(dot(halfwayVector, hitRecord.normal), 0.0f), specularConstant);
							clamp(specularFactor, 0.0f, 1.0f);

							// Specular Component
							if(specularFactor > 0.0f)
								hitRecord.color += make_float3(specularColor) * lightColor * specularFactor * lightIntensity.y * attenuation * shadowGridDimensionInverse;
						}
					}
				}
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
				//hitRecord.color += RayCast(Ray(hitRecord.point + reflectedDirection * epsilon, reflectedDirection), triangleTotal, lightTotal, depth-1, refractionIndex) * 0.50f;
				hitRecord.color = RayCast(Ray(hitRecord.point + reflectedDirection * epsilon, reflectedDirection), triangleTotal, lightTotal, depth-1, refractionIndex) * 0.50f;
			}

			// If the Object Hit is translucid
			if(refractionConstant > 0.0f) {

				/*float newRefractionIndex;

				if(refractionIndex == 1.0f)
					newRefractionIndex = 1.50f;
				else
					newRefractionIndex = 1.0f;

				// Calculate the Refracted Rays Direction
				float3 refractedDirection = refract(ray.direction, hitRecord.normal, refractionIndex / newRefractionIndex);

				// Cast the Refracted Ray
				if(length(refractedDirection) > 0.0f && hitRecord.sphereIndex > 0)
					hitRecord.color += RayCast(Ray(hitRecord.point + refractedDirection * epsilon, refractedDirection), triangleTotal, lightTotal, depth-1, newRefractionIndex) * 0.50f;*/
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
								// Total Number of Triangles in the Scene
								const int triangleTotal,
								// Total Number of Lights in the Scene
								const int lightTotal,
								// Ray Bounce Depth
								const int depth,
								// Medium Refraction Index
								const float refractionIndex,
								// Camera Definitions
								const float3 cameraPosition, 
								const float3 cameraUp, const float3 cameraRight, const float3 cameraDirection) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float3 pixelColor = make_float3(0.0f);

	//Anti-Aliasing - 4x Super Sampling
	/*for(int i=0; i<antiAliasingGridWidth; i++) {

		for(int j=0; j<antiAliasingGridHeight; j++) {

			// Ray Creation
			float3 rayOrigin = cameraPosition;
			float3 rayDirection = cameraDirection + 
				cameraRight * (((float)x + (i * 2 - antiAliasingGridHalfWidth * antiAliasingCellSize)) / (float)width - 0.5f) + 
				cameraUp * (((float)y + (j * 2 - antiAliasingGridHalfHeight * antiAliasingCellSize)) / (float)height - 0.5f);

			// Ray used to store Origin and Direction information
			Ray ray(rayOrigin, rayDirection);

			pixelColor += RayCast(ray, triangleTotal, lightTotal, depth, refractionIndex) * antiAliasingGridDimensionInverse;
		}
	}*/

	// Ray Creation
	float3 rayOrigin = cameraPosition;
	float3 rayDirection = cameraDirection + cameraRight * ((float)x / (float)width - 0.5f) + cameraUp * ((float)y / (float)height - 0.5f);

	// Ray used to store Origin and Direction information
	Ray ray(rayOrigin, rayDirection);

	//pixelColor += RayCast(ray, triangleTotal, lightTotal, depth, refractionIndex) * antiAliasingGridDimensionInverse;
	pixelColor = RayCast(ray, triangleTotal, lightTotal, depth, refractionIndex);

	pixelBufferObject[y * width + x] = rgbToInt(pixelColor.x * 255, pixelColor.y * 255, pixelColor.z * 255);
}

extern "C" {

	void RayTraceWrapper(unsigned int *outputPixelBufferObject,
								int width, int height, 
								int triangleTotal,
								int lightTotal,
								float3 cameraPosition,
								float3 cameraUp, float3 cameraRight, float3 cameraDirection
								) {

		dim3 block(8,8,1);
		dim3 grid(width/block.x,height/block.y, 1);

		RayTracePixel<<<grid, block>>>(	outputPixelBufferObject, 
										width, height,
										triangleTotal,
										lightTotal,
										initialDepth,
										initialRefractionIndex,
										cameraPosition,
										cameraUp, cameraRight, cameraDirection);
	}

	// Triangle Texture Binding Functions
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

	void bindTriangleMaterialIDs(float *cudaDevicePointer, unsigned int triangleTotal) {

		triangleMaterialIDsTexture.normalized = false;                      // access with normalized texture coordinates
		triangleMaterialIDsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangleMaterialIDsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(int1) * triangleTotal * 3;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<int1>();
		cudaBindTexture(0, triangleMaterialIDsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	// Material Texture Binding Functions
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

	// Light Texture Binding Functions
	void bindLightPositions(float *cudaDevicePointer, unsigned int lightTotal) {

		lightPositionsTexture.normalized = false;                      // access with normalized texture coordinates
		lightPositionsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightPositionsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, lightPositionsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindLightDirections(float *cudaDevicePointer, unsigned int lightTotal) {

		lightDirectionsTexture.normalized = false;                      // access with normalized texture coordinates
		lightDirectionsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightDirectionsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, lightDirectionsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	void bindLightCutOffs(float *cudaDevicePointer, unsigned int lightTotal) {

		lightCutOffsTexture.normalized = false;                      // access with normalized texture coordinates
		lightCutOffsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightCutOffsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float1) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float1>();
		cudaBindTexture(0, lightCutOffsTexture, cudaDevicePointer, channelDescriptor, size);
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

	void bindLightAttenuations(float *cudaDevicePointer, unsigned int lightTotal) {

		lightAttenuationsTexture.normalized = false;                      // access with normalized texture coordinates
		lightAttenuationsTexture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		lightAttenuationsTexture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * lightTotal;

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, lightAttenuationsTexture, cudaDevicePointer, channelDescriptor, size);
	}

	// OpenGL Textutures Texture Binding Functions
	void bindTextureArray(cudaArray *cudaArray) {

		shadingTexture.normalized = true;						// access with normalized texture coordinates
		shadingTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		shadingTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		shadingTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<uchar4>();
		cudaBindTextureToArray(shadingTexture, cudaArray, channelDescriptor);
	}
}