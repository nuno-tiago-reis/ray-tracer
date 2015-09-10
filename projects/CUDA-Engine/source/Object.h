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
#include "BoundingBox.h"

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

class Object {

	protected:

		// Object Identifiers
		int id;
		string name;
		string parentName;

		// Object Components: Mesh, Material and Transform
		Mesh* mesh;
		Material* material;
		Transform* transform;

		// Objects Mesh OpenGL IDs
		GLuint arrayObjectID;
		GLuint bufferObjectID;

	public:

		// Constructors & Destructors
		Object(string name);
		~Object();

		// GPU Creation & Destruction Methods
		void createMesh();
		void destroyMesh();

		// Scene Methods
		virtual void draw();

		virtual void update();
		virtual void update(GLfloat elapsedTime);

		// Ray Cast Intersection Method
		GLfloat isIntersecting(Vector origin, Vector direction);

		// Getters
		int getID();
		string getName();
		string getParentName();

		Mesh* getMesh();
		Material* getMaterial();
		Transform* getTransform();

		GLuint getArrayObjectID();
		GLuint getBufferObjectID();

		// Setters
		void setID(int id);
		void setName(string name);
		void setParentName(string parentName);

		void setMesh(Mesh* mesh);
		void setMaterial(Material* material);
		void setTransform(Transform* transform);

		void setArrayObjectID(GLuint arrayObjectID);
		void setBufferObjectID(GLuint bufferObjectID);

		// Debug Methods
		void dump();
};

#endif