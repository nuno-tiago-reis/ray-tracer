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
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> frameBufferTexture;

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

	pixelColor.x = 255 - tex2D(frameBufferTexture, x, y).x;
	pixelColor.y = 255 - tex2D(frameBufferTexture, x, y).y;
	pixelColor.z = 255 - tex2D(frameBufferTexture, x, y).z;

	int rgb = pixelColor.x;
	rgb = (rgb << 8) + pixelColor.y;
	rgb = (rgb << 8) + pixelColor.z;

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


	// OpenGL FrameBuffer Texture Binding Functions
	void bindFrameTextureArray(cudaArray *frameTextureArray) {

		frameBufferTexture.normalized = false;						// access with normalized texture coordinates
		frameBufferTexture.filterMode = cudaFilterModePoint;		// Point mode, so no 
		frameBufferTexture.addressMode[0] = cudaAddressModeWrap;	// wrap texture coordinates
		frameBufferTexture.addressMode[1] = cudaAddressModeWrap;	// wrap texture coordinates

		cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<uchar4>();
		cudaBindTextureToArray(frameBufferTexture, frameTextureArray, channelDescriptor);
	}
}