#include <stdio.h>

#include "cuda_runtime.h"

#include "math_functions.h"

#include "helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"

#include "Utility.h"

// Ray initial depth 
const int initialDepth = 0;
// Ray initial refraction index
const float initialRefractionIndex = 1.0f;

// Ray testing Constant
__device__ const float epsilon = 0.01f;

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

texture<int1, 1, cudaReadModeElementType> triangleObjectIDsTexture;
texture<int1, 1, cudaReadModeElementType> triangleMaterialIDsTexture;

texture<float2, 1, cudaReadModeElementType> triangleModelMatrixTexture;

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


// Implementation of Matrix Multiplication
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

// Casts a Ray and tests for intersections with the scenes geometry
__device__ float3 PrimaryRayCast(
							Ray ray,
							// Updated Triangle Position Array
							float4* trianglePositionsArray,
							// Updated Triangle Position Array
							float4* triangleNormalsArray,
							// Total Number of Triangles in the Scene
							const int triangleTotal,
							// Total Number of Lights in the Scene
							const int lightTotal) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		

	// Fragment Color
	float3 fragmentColor = make_float3(0.0f);

	// Fragment Position and Normal - Sent from the OpenGL Rasterizer
	float3 fragmentPosition = ray.origin;
	float3 fragmentNormal = normalize(make_float3(tex2D(fragmentNormalTexture, x,y)));

	// Triangle Material Properties
	float4 fragmentDiffuseColor = tex2D(diffuseTexture, x,y);
	float4 fragmentSpecularColor = tex2D(specularTexture, x,y);

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

			bool shadow = false;

			Ray shadowRay(fragmentPosition + lightDirection * epsilon, lightDirection);
			
			// Test Shadow Rays for each Triangle
			for(int k = 0; k < triangleTotal; k++) {

				float4 v0 = trianglePositionsArray[k * 3];
				float4 e1 = trianglePositionsArray[k * 3 + 1];
				e1 = e1 - v0;
				float4 e2 = trianglePositionsArray[k * 3 + 2];
				e2 = e2 - v0;

				float hitTime = RayTriangleIntersection(shadowRay, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

				if(hitTime > epsilon) {

					shadow = true;
					break;
				}
			}

			if(shadow == false) {

				// Blinn-Phong approximation Halfway Vector
				float3 halfwayVector = lightDirection - ray.direction;
				halfwayVector = normalize(halfwayVector);

				// Light Color
				float3 lightColor =  make_float3(tex1Dfetch(lightColorsTexture, l));
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
	}

	return fragmentColor;
}

// Casts a Ray and tests for intersections with the scenes geometry
__device__ float3 SecondaryRayCast(	
							Ray ray, 
							// Updated Triangle Position Array
							float4* trianglePositionsArray,
							// Updated Triangle Position Array
							float4* triangleNormalsArray,
							// Total Number of Triangles in the Scene
							const int triangleTotal,
							// Total Number of Lights in the Scene
							const int lightTotal,
							// Ray Bounce Depth
							const int depth,
							// Medium Refraction Index
							const float refractionIndex) {

	// Hit Record used to store Ray-Triangle Hit information - Initialized with Background Colour
	HitRecord hitRecord(make_float3(0.0f,0.0f,0.0f));

	// Search through the triangles and find the nearest hit point
	for(int i = 0; i < triangleTotal; i++) {

		float4 v0 = trianglePositionsArray[i * 3];
		float4 e1 = trianglePositionsArray[i * 3 + 1];
		e1 = e1 - v0;
		float4 e2 = trianglePositionsArray[i * 3 + 2];
		e2 = e2 - v0;

		float hitTime = RayTriangleIntersection(ray, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

		if(hitTime < hitRecord.time && hitTime > epsilon) {

			hitRecord.time = hitTime;
			hitRecord.triangleIndex = i;
		}
	}

	// If any Triangle was intersected
	if(hitRecord.triangleIndex >= 0) {

		// Illumination

		// Initialize the hit Color
		hitRecord.color = make_float3(0.0f, 0.0f, 0.0f);
		// Calculate the hit Triangle point
		hitRecord.point = ray.origin + ray.direction * hitRecord.time;
			
		// Fetch the hit Triangles vertices
		float4 v0 = trianglePositionsArray[hitRecord.triangleIndex * 3];
		float4 v1 = trianglePositionsArray[hitRecord.triangleIndex * 3 + 1];
		float4 v2 = trianglePositionsArray[hitRecord.triangleIndex * 3 + 2];

		// Fetch the hit Triangles normals
		float4 n0 = triangleNormalsArray[hitRecord.triangleIndex * 3];
		float4 n1 = triangleNormalsArray[hitRecord.triangleIndex * 3 + 1];
		float4 n2 = triangleNormalsArray[hitRecord.triangleIndex * 3 + 2];

		// Normal calculation using Barycentric Interpolation
		float areaABC = length(cross(make_float3(v1) - make_float3(v0), make_float3(v2) - make_float3(v0)));
		float areaPBC = length(cross(make_float3(v1) - hitRecord.point, make_float3(v2) - hitRecord.point));
		float areaPCA = length(cross(make_float3(v0) - hitRecord.point, make_float3(v2) - hitRecord.point));

		hitRecord.normal = (areaPBC / areaABC) * make_float3(n0) + (areaPCA / areaABC) * make_float3(n1) + (1.0f - (areaPBC / areaABC) - (areaPCA / areaABC)) * make_float3(n2);

		// Triangle Material Identifier
		int1 materialID = tex1Dfetch(triangleMaterialIDsTexture, hitRecord.triangleIndex * 3);

		// Triangle Material Properties
		float4 diffuseColor = tex1Dfetch(materialDiffusePropertiesTexture, materialID.x);
		float4 specularColor = tex1Dfetch(materialSpecularPropertiesTexture, materialID.x);

		float specularConstant = specularColor.w;
		float refractionConstant = diffuseColor.w;

		for(int l = 0; l < lightTotal; l++) {

			float3 lightPosition = make_float3(tex1Dfetch(lightPositionsTexture, l));

			// Light Direction and Distance
			float3 lightDirection = lightPosition - hitRecord.point;

			float lightDistance = length(lightDirection);
			lightDirection = normalize(lightDirection);

			// Diffuse Factor
			float diffuseFactor = max(dot(lightDirection, hitRecord.normal), 0.0f);
			clamp(diffuseFactor, 0.0f, 1.0f);

			if(diffuseFactor > 0.0f) {

				bool shadow = false;

				Ray shadowRay(hitRecord.point + lightDirection * epsilon, lightDirection);
			
				// Test Shadow Rays for each Triangle
				for(int k = 0; k < triangleTotal; k++) {

					float4 v0 = trianglePositionsArray[k * 3];
					float4 e1 = trianglePositionsArray[k * 3 + 1];
					e1 = e1 - v0;
					float4 e2 = trianglePositionsArray[k * 3 + 2];
					e2 = e2 - v0;

					float hitTime = RayTriangleIntersection(shadowRay, make_float3(v0.x,v0.y,v0.z), make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

					if(hitTime < hitRecord.time && hitTime > epsilon) {

						shadow = true;
						break;
					}
				}

				if(shadow == false) {

					// Blinn-Phong approximation Halfway Vector
					float3 halfwayVector = lightDirection - ray.direction;
					halfwayVector = normalize(halfwayVector);

					// Light Color
					float3 lightColor =  make_float3(tex1Dfetch(lightColorsTexture, l));
					// Light Intensity (x = diffuse, y = specular)
					float2 lightIntensity = tex1Dfetch(lightIntensitiesTexture, l);
					// Light Attenuation (x = constant, y = linear, z = exponential)
					float3 lightAttenuation = make_float3(0.0f, 0.0f, 0.0f);

					float attenuation = 1.0f / (1.0f + lightAttenuation.x + lightDistance * lightAttenuation.y + lightDistance * lightDistance * lightAttenuation.z);

					// Diffuse Component
					hitRecord.color += make_float3(diffuseColor) * lightColor * diffuseFactor * lightIntensity.x * attenuation;

					// Specular Factor
					float specularFactor = powf(max(dot(halfwayVector, hitRecord.normal), 0.0f), specularConstant);
					clamp(specularFactor, 0.0f, 1.0f);

					// Specular Component
					if(specularFactor > 0.0f)
						hitRecord.color += make_float3(specularColor) * lightColor * specularFactor * lightIntensity.y * attenuation;
				}
			}
		}

		// Ray Bounces
		if(depth > 0) {

			// If the Object Hit is reflective
			if(specularConstant > 0.0f) {

				// Calculate the Reflected Rays Direction
				float3 reflectedDirection = reflect(ray.direction, hitRecord.normal);

				// Cast the Reflected Ray
				hitRecord.color = 
					SecondaryRayCast(
						Ray(hitRecord.point + reflectedDirection * epsilon, reflectedDirection), 
						trianglePositionsArray,
						triangleNormalsArray, 
						triangleTotal, lightTotal, depth-1, refractionIndex) * 0.50f;
			}

			// If the Object Hit is translucid
			/*if(refractionConstant > 0.0f) {

				float newRefractionIndex;

				if(refractionIndex == 1.0f)
					newRefractionIndex = 1.50f;
				else
					newRefractionIndex = 1.0f;

				// Calculate the Refracted Rays Direction
				float3 refractedDirection = refract(ray.direction, hitRecord.normal, refractionIndex / newRefractionIndex);

				// Cast the Refracted Ray
				if(length(refractedDirection) > 0.0f && hitRecord.sphereIndex > 0)
					hitRecord.color += SecondaryRayCast(Ray(hitRecord.point + refractedDirection * epsilon, refractedDirection), triangleTotal, lightTotal, depth-1, newRefractionIndex) * 0.50f;
			}*/
		}
	}

	return hitRecord.color;
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

		Ray ray(rayOrigin + rayDirection * epsilon, rayDirection);

		// Calculate the +Primary Ray Shadows
		float3 primaryColor = PrimaryRayCast(ray, trianglePositionsArray, triangleNormalsArray,  triangleTotal, lightTotal);
		// Calculate the Primary Ray Bounces
		float3 secondaryColor = SecondaryRayCast(ray, trianglePositionsArray, triangleNormalsArray,  triangleTotal, lightTotal, depth, refractionIndex);
			
		// Calculate the Final Color
		float3 finalColor = primaryColor + secondaryColor;
		
		// Update the Pixel Buffer
		pixelBufferObject[y * width + x] = rgbToInt(finalColor.x * 255, finalColor.y * 255, finalColor.z * 255);
	}
	else {
	
		// Update the Pixel Buffer
		pixelBufferObject[y * width + x] = rgbToInt(0.0f, 0.0f, 0.0f);
	}
}

extern "C" {

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
							// Total Number of Triangles in the Scene
							int triangleTotal,
							// Total Number of Lights in the Scene
							int lightTotal,
							// Camera Definitions
							float3 cameraPosition) {

		// Model-Matrix Multiplication
		dim3 multiplicationBlock(1024);
		dim3 multiplicationGrid(triangleTotal*3/1024 + 1);

		MultiplyVertex<<<multiplicationBlock, multiplicationGrid>>>(modelMatricesArray, normalMatricesArray, trianglePositionsArray, triangleNormalsArray, triangleTotal * 3);

		// Ray-Casting
		dim3 rayCastingBlock(32,32);
		dim3 rayCastingGrid(width/rayCastingBlock.x + 1,height/rayCastingBlock.y + 1);

		RayTracePixel<<<rayCastingBlock, rayCastingGrid>>>(	pixelBufferObject,
															width, height,
															trianglePositionsArray, 
															triangleNormalsArray,
															triangleTotal,
															lightTotal,
															initialDepth, initialRefractionIndex,
															cameraPosition);
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