#include "WoodShader.h"

WoodShader::WoodShader(string name) : 
	ShaderProgram(name,WOOD_TEXTURE_VERTEX_SHADER_FILE,WOOD_TEXTURE_FRAGMENT_SHADER_FILE) {
}

WoodShader::~WoodShader(){
}

void WoodShader::bindAttributes() {

	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + this->name + "\" attribute binding error.");
}

void WoodShader::bindUniforms() {

	/* Binds Projection, View Matrices and Light Sources */
	ShaderProgram::bindUniforms();

	this->uniformLocationMap[NOISE_TEXTURE_UNIFORM] = glGetUniformLocation(_programID, NOISE_TEXTURE_UNIFORM);
	this->uniformLocationMap[NOISE_SCALE_UNIFORM] = glGetUniformLocation(_programID, NOISE_SCALE_UNIFORM);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" uniform binding error.");
}