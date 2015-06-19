#ifndef MESH_H
#define MESH_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/freeglut.h"
/* OpenGL Error check */
#include "Utility.h"

/* C++ Includes */
#include <string>

/* Generic Shader Program */
#include "ShaderProgram.h"

/* Mesh Reader */
#include "OBJ_Reader.h"

typedef struct {

	GLfloat position[4];
	
	GLfloat normal[4];
	GLfloat tangent[4];
	GLfloat textureUV[2];

	GLfloat ambient[4];
	GLfloat diffuse[4];
	GLfloat specular[4];
	GLfloat specularConstant;

} Vertex;

using namespace std;

class Mesh {

	protected:

		/* Meshs Name */
		string name;

		/* Meshes Vertex Attributes */
		GLint vertexCount;
		Vertex* vertices;

		/* Meshes OpenGL IDs */
		GLuint arrayObjectID;
		GLuint bufferObjectID;

	public:
		
		/* Constructors & Destructors */
		Mesh(string name, string meshFilename, string materialFilename);
		~Mesh();

		/* GPU Creation & Destruction Methods */
		void createMesh();
		void destroyMesh();

		/* Getters */
		string getName();

		GLint getVertexCount();
		Vertex* getVertices();

		GLuint getArrayObjectID();
		GLuint getBufferObjectID();

		/* Setters */
		void setName(string name);

		void setVertexCount(GLint vertexCount);
		void setVertices(Vertex* vertices, GLint vertexCount);

		void setArrayObjectID(GLuint arrayObjectID);
		void setBufferObjectID(GLuint bufferObjectID);

		/* Debug Methods */
		void dump();
};

#endif