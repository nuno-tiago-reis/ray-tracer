#include "SphereMappingShader.h"

SphereMappingShader::SphereMappingShader(string name) :
	ShaderProgram(name, SPHERE_MAPPING_VERTEX_SHADER_FILE, SPHERE_MAPPING_FRAGMENT_SHADER_FILE) {
}

SphereMappingShader::~SphereMappingShader() {
}

void SphereMappingShader::bindAttributes() {

	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + this->name + "\" attribute binding error.");
}

void SphereMappingShader::bindUniforms() {

	/* Binds Projection, View Matrices and Light Sources */
	ShaderProgram::bindUniforms();

	this->uniformLocationMap[SPHERE_MAPPING_UNIFORM] = glGetUniformLocation(_programID, SPHERE_MAPPING_UNIFORM);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" uniform binding error.");
}