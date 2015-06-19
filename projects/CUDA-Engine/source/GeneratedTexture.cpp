#include "GeneratedTexture.h"

GeneratedTexture::GeneratedTexture(string name, GLenum textureFormat, GLfloat noiseAlpha, GLfloat noiseBeta, GLint noiseOctaves, GLfloat noiseScale, string uniform) 
	: Texture(name, textureFormat, uniform) {

	this->noiseAlpha = noiseAlpha;
	this->noiseBeta = noiseBeta;
	this->noiseOctaves = noiseOctaves;

	this->noiseScale = noiseScale;
}

GeneratedTexture::~GeneratedTexture() {
}

void GeneratedTexture::createTexture() {

	/* Initialize the Perlin Noise Generator */
	PerlinNoise* perlinNoise = PerlinNoise::getInstance();
	perlinNoise->init();

	for(int i=0;i<W;i++) {

		for(int j=0;j<H;j++) {

			for(int k=0;k<D;k++) {

				GLfloat fi=(1.0f*i)/W,fj=(1.0f*j)/H, fk=(1.0f*k)/D;
				GLfloat n=perlinNoise->PerlinNoise3D(fi,fj,fk,this->noiseAlpha,this->noiseBeta,this->noiseOctaves);
				n=n*0.5f+0.5f;

				this->noiseTexture[k*W*H+j*W+i]=n;
			}
		}
	}

	glGenTextures(1, &this->handler);

	glBindTexture(this->format, this->handler);

	glTexParameteri(this->format, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(this->format, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(this->format, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glTexParameteri(this->format, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

	glTexImage3D(this->format, 0, GL_R32F, H, W, D, 0, GL_RED, GL_FLOAT, this->noiseTexture);

	glBindTexture(this->format, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->filename + "\" loading failed.");

	// Register the Textures with CUDA.
	Utility::checkCUDAError("cudaGraphicsGLRegisterImage()", cudaGraphicsGLRegisterImage(&cudaGraphicsResourceReference, this->handler, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

	/* Destroy Perlin Noise Generator */
	PerlinNoise::destroyInstance();
}

void GeneratedTexture::loadUniforms(GLuint programID, GLuint textureID) {

	glProgramUniform1i(programID, glGetUniformLocation(programID, NOISE_TEXTURE_UNIFORM), textureID);
	Utility::checkOpenGLError("ERROR: Uniform Location \"" NOISE_TEXTURE_UNIFORM "\" error.");

	glProgramUniform1f(programID,glGetUniformLocation(programID,NOISE_SCALE_UNIFORM), this->noiseScale);
	Utility::checkOpenGLError("ERROR: Uniform Location \"" NOISE_SCALE_UNIFORM "\" error.");
}

void GeneratedTexture::bind(GLuint textureID) {

	glActiveTexture(textureID);
	glBindTexture(this->format, this->handler);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->filename + "\" binding failed.");
}

GLfloat GeneratedTexture::getNoiseAlpha() {

	return this->noiseAlpha;
}

GLfloat GeneratedTexture::getNoiseBeta() {

	return this->noiseBeta;
}

GLint GeneratedTexture::getNoiseOctaves() {

	return this->noiseOctaves;
}

GLfloat GeneratedTexture::getNoiseScale() {

	return this->noiseScale;
}

void GeneratedTexture::setNoiseAlpha(GLfloat noiseAlpha) {

	this->noiseAlpha = noiseAlpha;
}

void GeneratedTexture::setNoiseBeta(GLfloat noiseBeta) {

	this->noiseBeta = noiseBeta;
}

void GeneratedTexture::setNoiseOctaves(GLint noiseOctaves) {

	this->noiseOctaves = noiseOctaves;
}

void GeneratedTexture::setNoiseScale(GLfloat noiseScale) {

	this->noiseScale = noiseScale;
}