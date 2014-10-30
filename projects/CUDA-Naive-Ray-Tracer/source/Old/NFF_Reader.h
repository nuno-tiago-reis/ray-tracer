#ifndef NFF_READER_H
#define NFF_READER_H

#include "GL/glew.h"
#include "GL/freeglut.h"

#include <fstream>
#include <sstream>
#include <iostream>

#include <string>
#include <vector>

#include "Scene.h"

#include "NFF_Plane.h"
#include "NFF_Sphere.h"
#include "NFF_Polygon.h"
#include "NFF_BoundingBox.h"

#include "Structures.h"

#define LOCATION "models/"

using namespace std;

class NFF_Reader {

	private:

		/* Singleton Instance */
		static NFF_Reader *instance;

		/* Constructors and Destructors */
		NFF_Reader();
		~NFF_Reader();

	public:
		
		/* Singleton Methods */
		static NFF_Reader* getInstance();
		static void destroyInstance();

		/* NFF Parser */
		void parseNFF(string modelFileName, Scene *scene);
};

#endif