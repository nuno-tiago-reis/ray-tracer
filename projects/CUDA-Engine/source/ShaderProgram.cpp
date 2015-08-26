#include "ShaderProgram.h"

map<string,GLuint> ShaderProgram::uniformBufferIndexMap;

bool ShaderProgram::bufferCreated=false;

ShaderProgram::ShaderProgram(string name, string vertexShaderFile, string fragmentShaderFile) {

	this->name = name;

	ifstream fileStream;
	stringstream buffer;
	
	/* Read Vertex Shader File */
	string vertexShader;

	fileStream.open(vertexShaderFile,fstream::in);
	buffer << fileStream.rdbuf();
	vertexShader.append(buffer.str());

	/* Close the File Stream */
	fileStream.close();

	/* Clear the Buffer */
	buffer = stringstream();

	/* Read Fragment Shader File */
	string fragmentShader;

	fileStream.open(fragmentShaderFile,fstream::in);
	buffer << fileStream.rdbuf();
	fragmentShader.append(buffer.str());

	/* Close the File Stream */
	fileStream.close();

	this->vertexShader = _strdup(vertexShader.c_str());
	this->fragmentShader = _strdup(fragmentShader.c_str());

	this->vertexShaderFile = vertexShaderFile;
	this->fragmentShaderFile = fragmentShaderFile;

	_programID = UINT_MAX;

	this->uniformBlockBindingPointMap[LIGHT_SOURCES_UNIFORM] = LIGHT_SOURCES_BINDING_POINT;
	this->uniformBlockBindingPointMap[MATRICES_UNIFORM] = MATRICES_BINDING_POINT;

	this->vertexShaderID = UINT_MAX;
	this->fragmentShaderID = UINT_MAX;
}

ShaderProgram::~ShaderProgram() {

	destroyShaderProgram();

	delete this->vertexShader;
	delete this->fragmentShader;
}

void ShaderProgram::createShaderProgram() {

	/* Create Vertex Shader */
	this->vertexShaderID = glCreateShader(GL_VERTEX_SHADER);

	/* Compile Vertex Shader */
	glShaderSource(this->vertexShaderID, 1, &this->vertexShader, 0);
	glCompileShader(this->vertexShaderID);

	/* Check if Vertex Shader Compiled */
	GLint shaderCompiled;
	glGetShaderiv(this->vertexShaderID, GL_COMPILE_STATUS, &shaderCompiled);

	if(shaderCompiled == GL_FALSE)
		Utility::checkShaderCompileError(this->vertexShaderID,this->vertexShader,this->vertexShaderFile.c_str());

	/* Create Fragment Shader */
	this->fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	/* Compile Fragment Shader */
	glShaderSource(this->fragmentShaderID, 1, &this->fragmentShader, 0);
	glCompileShader(this->fragmentShaderID);

	/* Check if Fragment Shader Compiled */
	glGetShaderiv(this->fragmentShaderID, GL_COMPILE_STATUS, &shaderCompiled);

	if(shaderCompiled == GL_FALSE)
		Utility::checkShaderCompileError(this->fragmentShaderID,this->fragmentShader,this->fragmentShaderFile.c_str());

	/* Create Shader Program */
	_programID = glCreateProgram();

	/* Attach Vertex & Fragment Shader */
	glAttachShader(_programID, this->vertexShaderID);
	glAttachShader(_programID, this->fragmentShaderID);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" creation error.");
}

void ShaderProgram::linkShaderProgram() {

	/* Link Program Shader */
	glLinkProgram(_programID);

	/* Check if Program Linked Correctly */
	GLint shaderLinked;
	glGetProgramiv(_programID, GL_LINK_STATUS, &shaderLinked);

	if(shaderLinked == GL_FALSE)
		Utility::checkShaderLinkageError(_programID,this->vertexShader,this->vertexShaderFile.c_str(),this->fragmentShader,this->fragmentShaderFile.c_str());

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" linking error.");
}

void ShaderProgram::bindAttributes() {

	glBindAttribLocation(_programID, POSITION, POSITION_ATTRIBUTE);

	glBindAttribLocation(_programID, NORMAL, NORMAL_ATTRIBUTE);
	glBindAttribLocation(_programID, TANGENT, TANGENT_ATTRIBUTE);

	glBindAttribLocation(_programID, TEXTURE_UV, TEXTURE_UV_ATTRIBUTE);

	glBindAttribLocation(_programID, AMBIENT, AMBIENT_ATTRIBUTE);
	glBindAttribLocation(_programID, DIFFUSE, DIFFUSE_ATTRIBUTE);
	glBindAttribLocation(_programID, SPECULAR, SPECULAR_ATTRIBUTE);
	glBindAttribLocation(_programID, SHININESS, SHININESS_ATTRIBUTE);

	Utility::checkOpenGLError("ERROR: Shader program \"" + this->name + "\" attribute binding error.");
}

void ShaderProgram::bindUniforms() {

	/* Initialize Uniforms */
	this->uniformLocationMap["ModelMatrix"] = glGetUniformLocation(_programID, "ModelMatrix");

	/* Lights Block */
	this->uniformBlockIndexMap[LIGHT_SOURCES_UNIFORM] = glGetUniformBlockIndex(_programID, LIGHT_SOURCES_UNIFORM);

	glUniformBlockBinding(_programID, this->uniformBlockIndexMap[LIGHT_SOURCES_UNIFORM], this->uniformBlockBindingPointMap[LIGHT_SOURCES_UNIFORM]);

	if(!bufferCreated) {

		unsigned char buffer[(sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * LIGHT_SOURCE_LIMIT];
		memset(buffer,0,(sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * LIGHT_SOURCE_LIMIT);

		GLuint uniformBufferIndex;
	
		glGenBuffers(1, &uniformBufferIndex);

		this->uniformBufferIndexMap[LIGHT_SOURCES_UNIFORM] = uniformBufferIndex;

		glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBufferIndexMap[LIGHT_SOURCES_UNIFORM]);

			glBufferData(GL_UNIFORM_BUFFER, (sizeof(GLfloat) * LIGHT_SOURCE_STRUCT_SIZE + sizeof(GLint)) * LIGHT_SOURCE_LIMIT, buffer, GL_DYNAMIC_DRAW);

			glBindBufferBase(GL_UNIFORM_BUFFER, this->uniformBlockBindingPointMap[LIGHT_SOURCES_UNIFORM], this->uniformBufferIndexMap[LIGHT_SOURCES_UNIFORM]);

		glBindBuffer(GL_UNIFORM_BUFFER,0);
	}

	/* Shared Matrices Block */
	this->uniformBlockIndexMap[MATRICES_UNIFORM] = glGetUniformBlockIndex(_programID, MATRICES_UNIFORM);

	glUniformBlockBinding(_programID, this->uniformBlockIndexMap[MATRICES_UNIFORM], this->uniformBlockBindingPointMap[MATRICES_UNIFORM]);

	if(!bufferCreated) {
	
		GLuint uniformBufferIndex;

		glGenBuffers(1, &uniformBufferIndex);

		this->uniformBufferIndexMap[MATRICES_UNIFORM] = uniformBufferIndex;

		glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBufferIndexMap[MATRICES_UNIFORM]);

		glBufferData(GL_UNIFORM_BUFFER, sizeof(GLfloat) * MATRICES_STRUCT_SIZE, 0, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, this->uniformBlockBindingPointMap[MATRICES_UNIFORM], this->uniformBufferIndexMap[MATRICES_UNIFORM]);

		glBindBuffer(GL_UNIFORM_BUFFER,0);
	}

	bufferCreated=true;

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" uniform binding error.");
}

void ShaderProgram::destroyShaderProgram() {
	
	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" destruction error 11.");
	glUseProgram(0);

	//cout << "[" << this->name << "]" << " ProgramID = " << _programID << endl;
	//cout << "[" << this->name << "]" << " VertexProgramID = " << this->vertexShaderID << endl;
	//cout << "[" << this->name << "]" << " FragmentProgramID = " << this->fragmentShaderID << endl;

	glDetachShader(_programID, this->vertexShaderID);
	glDetachShader(_programID, this->fragmentShaderID);

	glDeleteShader(this->fragmentShaderID);
	glDeleteShader(this->vertexShaderID);

	glDeleteProgram(_programID);

	Utility::checkOpenGLError("ERROR: Shader Program \"" + this->name + "\" destruction error 22.");
}

string ShaderProgram::getName() {

	return this->name;
}

string ShaderProgram::getVertexShaderFile() {

	return this->vertexShaderFile;
}

string ShaderProgram::getFragmentShaderFile() {

	return this->fragmentShaderFile;
}

const char* ShaderProgram::getVertexShader() {

	return this->vertexShader;
}

const char* ShaderProgram::getFragmentShader() {

	return this->fragmentShader;
}

GLuint ShaderProgram::getProgramID() {

	return _programID;
}

GLuint ShaderProgram::getUniformLocation(string uniformName) {

	return this->uniformLocationMap[uniformName];
}

GLuint ShaderProgram::getUniformBlockIndex(string uniformBlockName) {

	return this->uniformBlockIndexMap[uniformBlockName];
}

GLuint ShaderProgram::getUniformBlockBindingPoint(string uniformBlockName) {

	return this->uniformBlockBindingPointMap[uniformBlockName];
}

GLuint ShaderProgram::getUniformBufferIndex(string uniformBlockName) {

	return this->uniformBufferIndexMap[uniformBlockName];
}

GLuint ShaderProgram::getVertexShaderID() {

	return this->vertexShaderID;
}

GLuint ShaderProgram::getFragmentShaderID() {

	return this->fragmentShaderID;
}

void ShaderProgram::setName(string name) {

	this->name = name;
}

void ShaderProgram::setVertexShaderFile(string vertexShaderFile) {

	this->vertexShaderFile = vertexShaderFile;
}

void ShaderProgram::setFragmentShaderFile(string fragmentShaderFile) {

	this->fragmentShaderFile = fragmentShaderFile;
}

void ShaderProgram::setVertexShader(const char* vertexShader) {

	this->vertexShader = _strdup(vertexShader);
}

void ShaderProgram::setFragmentShader(const char* fragmentShader) {

	this->fragmentShader = _strdup(fragmentShader);
}

void ShaderProgram::setProgramID(GLuint programID) {

	_programID = programID;
}

void ShaderProgram::setUniformLocation(string uniformName, GLuint uniformLocation) {

	this->uniformLocationMap[uniformName] = uniformLocation;
}

void ShaderProgram::setUniformBlockIndex(string uniformName, GLuint uniformBlockIndex) {

	this->uniformBlockIndexMap[uniformName] = uniformBlockIndex;
}

void ShaderProgram::setUniformBlockBindingPoint(string uniformBlockName, GLuint uniformBlockBindingPoint) {

	this->uniformBlockBindingPointMap[uniformBlockName] = uniformBlockBindingPoint;
}

void ShaderProgram::setUniformBufferIndex(string uniformBlockName,  GLuint uniformBufferIndex) {

	this->uniformBufferIndexMap[uniformBlockName] = uniformBufferIndex;
}

void ShaderProgram::setVertexShaderID(GLuint vertexShaderID) {

	this->vertexShaderID = vertexShaderID;
}

void ShaderProgram::setFragmentShaderID(GLuint fragmentShaderID) {

	this->fragmentShaderID = fragmentShaderID;
}

void ShaderProgram::dump() {

	cout << "<Shader Dump>" << endl;

	/* Shader Program File Paths */
	cout << "<Shader Vertex File Path> = " << this->vertexShaderFile << " ;" << endl;
	cout << "<Shader Fragment File Path> = " << this->fragmentShaderFile << " ;" << endl;

	/* Shader Program Code */
	//cout << "<Shader Vertex Code> = \n" << this->vertexShader << " ;" << endl;
	//cout << "<Shader Fragment Code> = \n" << this->fragmentShader << " ;" << endl;

	/* Shader Program ID */
	cout << "<Shader Program ID> = " << _programID << " ;" << endl;

	/* Shader Uniform Attributes */
	cout << "<Shader Uniform Locations> = " << endl;
	map<string,GLint>::const_iterator uniformLocationIterator;
	for(uniformLocationIterator = this->uniformLocationMap.begin(); uniformLocationIterator != this->uniformLocationMap.end(); uniformLocationIterator++)
		cout << "\t" << uniformLocationIterator->first << " = " << uniformLocationIterator->second << endl;

	cout << "<Shader Uniform Block Indexes> = " << endl;
	map<string,GLuint>::const_iterator uniformBlockIndexIterator;
	for(uniformBlockIndexIterator = this->uniformBlockIndexMap.begin(); uniformBlockIndexIterator != this->uniformBlockIndexMap.end(); uniformBlockIndexIterator++)
		cout << "\t" << uniformBlockIndexIterator->first << " = " << uniformBlockIndexIterator->second << endl;

	/* Shader Vertex & Fragment Program IDs */
	cout << "<Shader Vertex Program ID> = " << this->vertexShaderID << " ;" << endl;
	cout << "<Shader Fragment Program ID> = " << this->fragmentShaderID << " ;" << endl;
}