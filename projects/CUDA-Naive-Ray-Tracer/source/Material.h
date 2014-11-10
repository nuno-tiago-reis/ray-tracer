#ifndef MATERIAL_H
#define MATERIAL_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Math Library */
#include "Vector.h"

using namespace std;

class Material {

	protected:

		/* Material Identifier */
		int id;
		/* Material Name */
		string name;

		/* Material Ambient, Diffuse and Specular Properties */
		Vector ambient;
		Vector diffuse;
		Vector specular;

		/* Material Specular Constant */
		float specularConstant;

	public:

		/* Constructors & Destructors */
		Material(int id, string name);
		~Material();

		/* Getters */
		int getID();
		string getName();

		Vector getAmbient();
		Vector getDiffuse();
		Vector getSpecular();

		float getSpecularConstant();

		/* Setters */
		void setID(int id);
		void setName(string name);

		void setAmbient(Vector ambient);
		void setDiffuse(Vector diffuse);
		void setSpecular(Vector specular);

		void setSpecularConstant(float specularConstant);

		/* Debug Methods */
		void dump();
};

#endif