#ifndef OBJ_READER_H
#define OBJ_READER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* C++ Includes */
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <map>

/* Mesh */
#include "Mesh.h"

/* Constants */
#define LOCATION "models/"

typedef struct {

	float x;
	float y;
	float z;

} Coordinate3D;

typedef struct {

	float u;
	float v;

} Coordinate2D;

using namespace std;

class Mesh;

class OBJ_Reader {

	private:

		/* Singleton Instance */
		static OBJ_Reader *instance;

		OBJ_Reader();
		~OBJ_Reader();

	public:

		/* Singleton Methods */
		static OBJ_Reader* getInstance();
		static void destroyInstance();

		void loadMesh(string meshFilename, string materialFilename, Mesh* mesh);
};

#endif