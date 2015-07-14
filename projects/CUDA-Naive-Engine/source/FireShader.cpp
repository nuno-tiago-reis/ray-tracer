#include "FireShader.h"

FireShader::FireShader(string name) : 
	ShaderProgram(name,FIRE_VERTEX_SHADER_FILE,FIRE_FRAGMENT_SHADER_FILE) {
}

FireShader::~FireShader() {
}

void FireShader::bindAttributes() {

	/* Binds Generic Vertex Attributes (Position, Normal and Material) */
	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + name + "\" attribute binding error.");
}

void FireShader::bindUniforms() {

	/* Binds Projection, View Matrices and Light Sources */
	ShaderProgram::bindUniforms();
}