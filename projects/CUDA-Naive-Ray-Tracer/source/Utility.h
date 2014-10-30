#ifndef UTILITY_H
#define UTILITY_H

#include "GL/glew.h"
#include "GL/glut.h"

#include "cuda_runtime.h"

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class Utility {

	private:

		Utility();

		~Utility();

	public:

		static void checkOpenGLError(char* call) {

			GLenum errorCode;

			while ((errorCode = glGetError()) != GL_NO_ERROR) {

				fprintf(stderr, "[OpenGL Error] %s failed: Code %d - %s.\n", call, errorCode, gluErrorString(errorCode));
				fflush(stderr);

				exit(1);
			}
		}

		static void checkCUDAError(char* call, cudaError cudaResult) {
	
			if(cudaResult != cudaSuccess) {

				fprintf(stderr, "[CUDA Error] %s failed: %s - %s.\n", call, cudaGetErrorName(cudaResult), cudaGetErrorString(cudaResult));
				fflush(stderr);

				exit(1);
			}
		}
};

#endif