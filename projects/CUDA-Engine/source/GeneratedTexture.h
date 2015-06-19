#ifndef GENERATED_TEXTURE_H
#define GENERATED_TEXTURE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Perlin Noise Generator */
#include "PerlinNoise.h"

/* Generic Texture */
#include "Texture.h"

/* Wood Shader */
#include "WoodShader.h"

/* Generated Texture Definitions */
#define H 512
#define W 512
#define D 4
#define S H*W*D

class GeneratedTexture : public Texture {

	private:

		/* Noise Generation Parameters */
		GLfloat noiseAlpha;
		GLfloat noiseBeta;

		GLint noiseOctaves;

		GLfloat noiseScale;

		/* Noise Texture */
		GLfloat noiseTexture[S];

	public:

		GeneratedTexture(string name, GLenum textureFormat, GLfloat noiseAlpha, GLfloat noiseBeta, GLint noiseOctaves, GLfloat noiseScale, string uniform);
		~GeneratedTexture();

		/* Loading Methods */
		void createTexture();

		void loadUniforms(GLuint programID, GLuint textureID);

		/* Bind & Unbind to OpenGL Methods */
		void bind(GLuint textureID);

		/* Getters */
		GLfloat getNoiseAlpha();

		GLfloat getNoiseBeta();

		GLint getNoiseOctaves();

		GLfloat getNoiseScale();

		/* Setters */
		void setNoiseAlpha(GLfloat noiseAlpha);

		void setNoiseBeta(GLfloat noiseBeta);

		void setNoiseOctaves(GLint noiseOctaves);

		void setNoiseScale(GLfloat noiseScale);
};

#endif
