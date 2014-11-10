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

		/* Vertex Material Identifier */
		int materialID;

	public:

		/* Constructors & Destructors */
		Vertex(int id);
		~Vertex();

		/* Getters */
		int getID();

		Vector getPosition();

		Vector getNormal();
		Vector getTangent();

		Vector getTextureCoordinates();

		int getMaterialID();

		/* Setters */
		void setID(int id);

		void setPosition(Vector position);

		void setNormal(Vector normal);
		void setTangent(Vector tangent);

		void setTextureCoordinates(Vector textureCoordinates);

		void setMaterialID(int materialID);

		/* Debug Methods */
		void dump();
};

#endif