#include "BumpMappingShader.h"

BumpMappingShader::BumpMappingShader(string name) :
	ShaderProgram(name, BUMP_MAPPING_VERTEX_SHADER_FILE, BUMP_MAPPING_FRAGMENT_SHADER_FILE) {
}

BumpMappingShader::~BumpMappingShader() {
}

void BumpMappingShader::bindAttributes() {

	/* Binds Generic Vertex Attributes (Position, TextureUV, Normal, Tangent and Material) */
	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, TEXTURE_UV, TEXTURE_UV_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);
	glBindAttribLocation(_programID, TANGENT, TANGENT_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + this->name + "\" attribute binding error.");
}

void BumpMappingShader::bindUniforms() {

	/* Binds Projection, View Matrices and Light Sources */
	ShaderProgram::bindUniforms();
	
	this->uniformLocationMap[DIFFUSE_TEXTURE_UNIFORM] = glGetUniformLocation(_programID, DIFFUSE_TEXTURE_UNIFORM);
	this->uniformLocationMap[NORMAL_TEXTURE_UNIFORM] = glGetUniformLocation(_programID, NORMAL_TEXTURE_UNIFORM);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" uniform binding error.");
}