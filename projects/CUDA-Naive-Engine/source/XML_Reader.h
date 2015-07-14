#ifndef XML_READER_H
#define XML_READER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/freeglut.h"
#include "tinyxml2.h"

/* OpenGL Error check */
#include "Utility.h"

/* C++ Includes */
#include <string>

/* Transform */
#include "Transform.h"

/* Constants */
#define TRANSFORMATION_FILE "models/transformation - Child Nodes.xml"
#define TRANSFORMATION_OUTPUT_FILE "models/transformationOutput.xml"

using namespace tinyxml2;

class Transform;

class XML_Reader {

	private:

		/* Singleton Instance */
		static XML_Reader *instance;

		/* TinyXML2 Document */
		XMLDocument* _transformationDocument;

		XML_Reader();
		~XML_Reader();

	public:

		/* Singleton Methods */
		static XML_Reader* getInstance();
		static void destroyInstance();

		void openTransformFile(string filename);

		void loadTransform(Transform* transform);
		void saveTransform(Transform* transform);
};

#endif