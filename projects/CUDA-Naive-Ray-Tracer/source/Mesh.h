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

/* Custom Includes */
#include "Vertex.h"
#include "Material.h"

/* Mesh Reader */
#include "OBJ_Reader.h"

using namespace std;

class Mesh {

	protected:

		/* Meshs Name */
		string name;

		/* Meshes Vertex Map */
		map<int, Vertex*> vertexMap;
		/* Meshes Material Map */
		map<int, Material*> materialMap;

	public:
		
		/* Constructors & Destructors */
		Mesh(string name, string meshFilename, string materialFilename);
		~Mesh();

		/* Getters */
		string getName();

		/* Setters */
		void setName(string name);

		/* Vertex Map Manipulation Methods */
		void addVertex(Vertex* vertex);
		void removeVertex(int vertexID);

		Vertex* getVertex(int vertexID);

		map<int,Vertex*> getVertexMap();

		/* Material Map Manipulation Methods */
		void addMaterial(Material* material);
		void removeMaterial(int materialID);

		Material* getMaterial(int materialID);

		map<int,Material*> getMaterialMap();

		/* Debug Methods */
		void dump();
};

#endif