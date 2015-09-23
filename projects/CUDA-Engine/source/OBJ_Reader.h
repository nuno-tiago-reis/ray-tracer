#ifndef OBJ_READER_H
#define OBJ_READER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

// OpenGL definitions
#include "GL/glew.h"
#include "GL/freeglut.h"

// OpenGL Error check
#include "Utility.h"

// C++ Includes
#include <fstream>
#include <sstream>

#include <string>
#include <vector>

// Mesh 
#include "Mesh.h"
// Material
#include "Material.h"
// Bounding Sphere
#include "BoundingSphere.h"

// Constants
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

		// Singleton Instance
		static OBJ_Reader *instance;

		// Mesh Line Number
		int meshLineNumber;
		// Mesh Read Indicator
		bool meshEndOfFile;

		// Mesh Offset Vertex
		int offsetVertex;
		// Mesh Offset Normal
		int offsetNormal;
		// Mesh Offset Texture Coordinate
		int offsetTextureCoordinate;

		// Mesh Name
		string meshName;
		// Mesh File Name
		string meshFilename;
		// Mesh File Stream
		ifstream meshFileStream;

		// Material Name
		string materialName;

		OBJ_Reader();
		~OBJ_Reader();

	public:

		// Singleton Methods
		static OBJ_Reader* getInstance();
		static void destroyInstance();

		// Loading Methods
		void loadMesh(string meshFilename, Mesh* mesh);
		void loadMaterial(string materialFilename, Material* material);

		// Checking Methods
		bool canReadMesh(string meshFilename);
		bool canReadMaterial(string materialFilename);
};

#endif