#ifndef VERTEX_H
#define VERTEX_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Math Library */
#include "Vector.h"

using namespace std;

typedef struct {

	GLfloat position[4];
	
	GLfloat normal[4];
	GLfloat tangent[4];
	GLfloat textureUV[2];

	GLfloat ambient[4];
	GLfloat diffuse[4];
	GLfloat specular[4];
	GLfloat specularConstant;

} VertexStructure;

class Vertex {

	protected:

		/* Vertex Identifier */
		int id;

		/* Vertex Position */
		Vector position;

		/* Vertex Normal and Tangent - Used in Normal Mapping */
		Vector normal;
		Vector tangent;

		/* Vertex Texture Coordinates - Used in Texture Mapping */
		Vector textureCoordinates;

	public:

		/* Constructors & Destructors */
		Vertex(int id);
		~Vertex();
		
		/* GPU Methods */
		VertexStructure getVertexStructure();

		/* Getters */
		int getID();

		Vector getPosition();

		Vector getNormal();
		Vector getTangent();

		Vector getTextureCoordinates();

		/* Setters */
		void setID(int id);

		void setPosition(Vector position);

		void setNormal(Vector normal);
		void setTangent(Vector tangent);

		void setTextureCoordinates(Vector textureCoordinates);

		/* Debug Methods */
		void dump();
};

#endif