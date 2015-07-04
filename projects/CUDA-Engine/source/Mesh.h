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

/* Vertex */
#include "Vertex.h"
/* Material */
#include "Material.h"

/* Mesh Reader */
#include "OBJ_Reader.h"

using namespace std;

class Material;

class Mesh {

	protected:

		/* Meshs Name */
		string name;

		/* Meshes Vertex Attributes */
		map<int, Vertex*> vertexMap;

	public:
		
		/* Constructors & Destructors */
		Mesh(string name, string meshFilename);
		~Mesh();

		/* Getters */
		string getName();

		/* Setters */
		void setName(string name);

		/* Vertex Map Methods */
		int getVertexCount();

		void addVertex(Vertex* vertex);
		void removeVertex(int vertexID);

		map<int, Vertex*> getVertexMap();

		/* Debug Methods */
		void dump();
};

#endif