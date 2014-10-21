#ifndef MESH_H
#define MESH_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL Includes */
#include "GL/glew.h"
#include "GL/freeglut.h"

/* C++ Includes */
#include <string>

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

		Vertex getVertex(GLint vertexID);

		/* Setters */
		void setName(string name);

		void setVertexCount(GLint vertexCount);
		void setVertices(Vertex* vertices, GLint vertexCount);

		/* Debug Methods */
		void dump();
};

#endif