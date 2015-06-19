#include "CubeMappingShader.h"

CubeMappingShader::CubeMappingShader(string name) :
	ShaderProgram(name, CUBE_MAPPING_VERTEX_SHADER_FILE, CUBE_MAPPING_FRAGMENT_SHADER_FILE) {
}

CubeMappingShader::~CubeMappingShader() {
}

void CubeMappingShader::bindAttributes() {

	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + this->name + "\" attribute binding error.");
}

void CubeMappingShader::bindUniforms() {

	/* Binds Projection, View Matrices and Light Sources */
	ShaderProgram::bindUniforms();

	this->uniformLocationMap[CUBE_MAPPING_UNIFORM] = glGetUniformLocation(_programID, CUBE_MAPPING_UNIFORM);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" uniform binding error.");
}