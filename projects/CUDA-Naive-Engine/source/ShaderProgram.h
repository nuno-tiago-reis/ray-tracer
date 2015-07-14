#ifndef SHADER_H
#define SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"
/* OpenGL Error Check */
#include "Utility.h"

/* C++ Includes */
#include <fstream>
#include <climits>
#include <map>

/* Model Matrix Uniforms */
#define MODEL_MATRIX_UNIFORM "ModelMatrix"

/* Shared Matrices Uniforms */
#define MATRICES_UNIFORM "SharedMatrices"
#define MATRICES_BINDING_POINT 0
#define MATRICES_STRUCT_SIZE 32

/* Light Source Uniforms */
#define LIGHT_SOURCES_UNIFORM "SharedLightSources"
#define LIGHT_SOURCES_BINDING_POINT 1
#define LIGHT_SOURCE_STRUCT_SIZE 19
#define LIGHT_SOURCE_LIMIT 10

/* Position Attribute */
#define POSITION 0
#define POSITION_ATTRIBUTE "VertexPosition"

/* Normal Attributes */
#define NORMAL 1
#define NORMAL_ATTRIBUTE "VertexNormal"
#define TANGENT 2
#define TANGENT_ATTRIBUTE "VertexTangent"

/* Texture UV Attribute */
#define TEXTURE_UV 3
#define TEXTURE_UV_ATTRIBUTE "VertexTextureUV"

/* Color Attributes */
#define AMBIENT 4
#define AMBIENT_ATTRIBUTE "VertexAmbient"
#define DIFFUSE 5
#define DIFFUSE_ATTRIBUTE "VertexDiffuse"
#define SPECULAR 6
#define SPECULAR_ATTRIBUTE "VertexSpecular"
#define SHININESS 7
#define SHININESS_ATTRIBUTE "VertexShininess"

using namespace std;

class ShaderProgram {

	protected:

		/* Shader Identifier */
		string name;

		/* Shader Program File Paths */
		string vertexShaderFile;
		string fragmentShaderFile;

		/* Shader Program Code */
		const char* vertexShader;
		const char* fragmentShader;

		/* Shader Program ID */
		GLuint _programID;

		/* Shader Vertex & Fragment Program IDs */
		GLuint vertexShaderID;
		GLuint fragmentShaderID;

		/* Shader Uniform Attributes */
		map<string,GLint> uniformLocationMap;

		map<string,GLuint> uniformBlockIndexMap;
		map<string,GLuint> uniformBlockBindingPointMap;

		static bool bufferCreated;
		static map<string,GLuint> uniformBufferIndexMap;

	public:

		/* Constructors & Destructors */
		ShaderProgram(string name, string vertexShaderFile, string fragmentShaderFile);
		~ShaderProgram();

		/* GPU Creation & Destruction Methods */
		void createShaderProgram();

		void linkShaderProgram();

		virtual void bindAttributes();
		virtual void bindUniforms();

		void destroyShaderProgram();

		/* Getters */
		string getName();

		string getVertexShaderFile();
		string getFragmentShaderFile();

		const char* getVertexShader();
		const char* getFragmentShader();

		GLuint getProgramID();

		GLuint getUniformLocation(string uniformName);

		GLuint getUniformBlockIndex(string uniformBlockName);
		GLuint getUniformBlockBindingPoint(string uniformBlockName);
		GLuint getUniformBufferIndex(string uniformBlockName);

		GLuint getVertexShaderID();
		GLuint getFragmentShaderID();

		/* Setters */
		void setName(string name);

		void setVertexShaderFile(string vertexShaderFile);
		void setFragmentShaderFile(string fragmentShaderFile);

		void setVertexShader(const char* vertexShader);
		void setFragmentShader(const char* fragmentShader);

		void setProgramID(GLuint programID);

		void setUniformLocation(string uniformName, GLuint uniformLocation);

		void setUniformBlockIndex(string uniformBlockName, GLuint uniformBlockIndex);
		void setUniformBlockBindingPoint(string uniformBlockName, GLuint uniformBlockBindingPoint);
		void setUniformBufferIndex(string uniformBlockName, GLuint uniformBufferIndex);

		void setVertexShaderID(GLuint vertexShaderID);
		void setFragmentShaderID(GLuint fragmentShaderID);

		/* Debug Methods */
		void dump();
};

#endif