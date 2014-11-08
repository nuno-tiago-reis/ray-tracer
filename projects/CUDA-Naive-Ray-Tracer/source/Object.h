#ifndef OBJECT_H
#define OBJECT_H

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

/* Math Library */
#include "Matrix.h"

/* Object Components */
#include "Mesh.h"
#include "Transform.h"

using namespace std;

class Object {

	protected:

		/* Object Identifier */
		string name;

		/* Object Components: Mesh, Material and Transform */
		Mesh* mesh;
		Transform* transform;

	public:

		/* Constructors & Destructors */
		Object(string name);
		~Object();

		/* Scene Methods */
		virtual void update();
		virtual void update(float elapsedTime);

		/* Getters */
		string getName();
		string getParentName();

		Mesh* getMesh();
		Transform* getTransform();

		/* Setters */
		void setName(string name);
		void setParentName(string parentName);

		void setMesh(Mesh* mesh);
		void setTransform(Transform* transform);

		/* Debug Methods */
		void dump();
};

#endif