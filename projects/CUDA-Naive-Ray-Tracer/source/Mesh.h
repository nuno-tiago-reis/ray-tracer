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

	float position[4];
	
	float normal[4];
	float tangent[4];
	float textureUV[2];

	float ambient[4];
	float diffuse[4];
	float specular[4];
	float specularConstant;

} Vertex;

using namespace std;

class Mesh {

	protected:

		/* Meshs Name */
		string name;

		/* Meshes Vertex Attributes */
		int vertexCount;
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

		int getVertexCount();
		Vertex* getVertices();

		Vertex getVertex(int vertexID);

		/* Setters */
		void setName(string name);

		void setVertexCount(int vertexCount);
		void setVertices(Vertex* vertices, int vertexCount);

		/* Debug Methods */
		void dump();
};

#endif