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

/* OpenGL Error check */
#include "Utility.h"

/* C++ Includes */
#include <fstream>
#include <sstream>

#include <string>
#include <vector>

/* Mesh */
#include "Mesh.h"
/* Material */
#include "Material.h"

/* Constants */
#define LOCATION "models/"

#define X 0
#define Y 1
#define Z 2

typedef struct {

	GLfloat x;
	GLfloat y;
	GLfloat z;

} Coordinate3D;

typedef struct {

	GLfloat u;
	GLfloat v;

} Coordinate2D;

using namespace std;

class Mesh;
class Material;

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

		void loadMesh(string meshFilename, Mesh* mesh);
		void loadMaterial(string materialFilename, Material* material);
};

#endif