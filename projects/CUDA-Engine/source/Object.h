#ifndef OBJECT_H
#define OBJECT_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"

/* C++ Includes */
#include <string>
#include <map>

/* Math Library */
#include "Matrix.h"

/* Object Components */
#include "Mesh.h"
#include "Material.h"
#include "Transform.h"

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

class Object {

	protected:

		/* Object Identifier */
		string name;
		string parentName;

		/* Object Components: Mesh, Material and Transform */
		Mesh* mesh;
		Material* material;
		Transform* transform;

	public:

		/* Constructors & Destructors */
		Object(string name);
		~Object();

		/* Scene Methods */
		virtual void draw();

		virtual void update();
		virtual void update(GLfloat elapsedTime);

		/* Ray Cast Intersection Method */
		GLfloat isIntersecting(Vector origin, Vector direction);

		/* Getters */
		string getName();
		string getParentName();

		Mesh* getMesh();
		Material* getMaterial();
		Transform* getTransform();

		/* Setters */
		void setName(string name);
		void setParentName(string parentName);

		void setMesh(Mesh* mesh);
		void setMaterial(Material* material);
		void setTransform(Transform* transform);

		/* Debug Methods */
		void dump();
};

#endif