#include <stdio.h>

#include "cuda_runtime.h"

#include "math_functions.h"

#include "helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"

// Ray initial depth 
const int initialDepth = 3;
// Ray initial refraction index
const float initialRefractionIndex = 1.0f;

// OpenGL Rendering Texture
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> renderTexture;

// OpenGL Ray Origin, Reflection and Refraction Textures
texture<float4, cudaTextureType2D, cudaReadModeElementType> rayOriginTexture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> rayReflectionTexture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> rayRefractionTexture;

// CUDA Triangle Textures
texture<float4, 1, cudaReadModeElementType> trianglePositionsTexture;
texture<float4, 1, cudaReadModeElementType> triangleNormalsTexture;
texture<float4, 1, cudaReadModeElementType> triangleTangentsTexture;
texture<float2, 1, cudaReadModeElementType> triangleTextureCoordinatesTexture;

texture<int1, 1, cudaReadModeElementType> triangleMaterialIDsTexture;

// CUDA Material Textures
texture<float4, 1, cudaReadModeElementType> materialDiffusePropertiesTexture;
texture<float4, 1, cudaReadModeElementType> materialSpecularPropertiesTexture;

// CUDA Light Textures
texture<float4, 1, cudaReadModeElementType> lightPositionsTexture;
texture<float4, 1, cudaReadModeElementType> lightColorsTexture;
texture<float2, 1, cudaReadModeElementType> lightIntensitiesTexture;

// Ray testing Constant
__device__ const float epsilon = 0.01f;

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

	return (int(red)<<16) | (int(green)<<8) | int(blue); // notice switch red and blue to counter the GL_BGRA
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

	int3 pixelColor = make_int3(0);

	pixelColor.x = 255 - tex2D(renderTexture, x, y).x;
	pixelColor.y = 255 - tex2D(renderTexture, x, y).y;
	pixelColor.z = 255 - tex2D(renderTexture, x, y).z;

	int rgb = pixelColor.x;
	rgb = (rgb << 8) + pixelColor.y;
	rgb = (rgb << 8) + pixelColor.z;

	pixelBufferObject[y * width + x] = rgb;

	float4 color = tex2D(rayReflectionTexture, x, y);
	pixelBufferObject[y * width + x] = rgbToInt((color.x + 1.0f)* 128.0f, (color.y + 1.0f) * 128.0f, (color.z + 1.0f) * 128.0f);
	
	pixelBufferObject[y * width + x] = rgb;
}

extern "C" {

	void RayTraceWrapper(	unsigned int *pixelBufferObject,
								int width, int height, 
								int triangleTotal,
								int lightTotal,
								float3 cameraPosition,
								float3 cameraUp, float3 cameraRight, float3 cameraDirection
								) {

		dim3 block(8,8,1);
		dim3 grid(width/block.x,height/block.y, 1);

		RayTracePixel<<<grid, block>>>(	pixelBufferObject,
										width, height,
										triangleTotal,
										lightTotal,
										initialDepth,
										initialRefractionIndex,
										cameraPosition,
										cameraUp, cameraRight, cameraDirection);
	}

	// OpenGL Texture Binding Functions
	void bindRenderTextureArray(cudaArray *renderArray) {
	
		renderTexture.normalized = false;					// access with normalized texture coordinates
		renderTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		renderTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		renderTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<uchar4>();
		cudaBindTextureToArray(renderTexture, renderArray, channelDescriptor);
	}

	// OpenGL Texture Binding Functions
	void bindRayOriginTextureArray(cudaArray *rayOriginArray) {
	
		rayOriginTexture.normalized = false;					// access with normalized texture coordinates
		rayOriginTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		rayOriginTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		rayOriginTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(rayOriginTexture, rayOriginArray, channelDescriptor);
	}

	void bindRayReflectionTextureArray(cudaArray *rayReflectionArray) {
	
		rayReflectionTexture.normalized = false;					// access with normalized texture coordinates
		rayReflectionTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		rayReflectionTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		rayReflectionTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(rayReflectionTexture, rayReflectionArray, channelDescriptor);
	}

	void bindRayRefractionTextureArray(cudaArray *rayRefractionArray) {
	
		rayRefractionTexture.normalized = false;					// access with normalized texture coordinates
		rayRefractionTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		rayRefractionTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		rayRefractionTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
		cudaBindTextureToArray(rayRefractionTexture, rayRefractionArray, channelDescriptor);
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