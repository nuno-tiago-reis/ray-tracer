#ifndef OBJ_READER_H
#define OBJ_READER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/freeglut.h"

/* C++ Includes */
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <map>

/* Mesh */
#include "Mesh.h"

/* Vector Implementation */
#include "Vector.h"

/* Constants */
#define LOCATION "models/"

#define X 0
#define Y 1
#define Z 2

typedef struct {

	float ambient[3];
	float diffuse[3];
	float specular[3];

	float specularConstant;

} MaterialStruct;

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