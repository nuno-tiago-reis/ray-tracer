#ifndef GRID_H
#define GRID_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

#include "Vector.h"
#include "Math.h"
#include <map>

#include "Voxel.h"

#include "BoundingBox.h"

class Grid {

	private:

		/* Scene Bounding Box */
		BoundingBox * boundingBox;

		/* Scene Object Number */
		int objectNumber;

		/* Grid Dimensions */
		GLfloat wx;
		GLfloat wy;
		GLfloat wz;

		/* Grid Voxel Number */
		GLint nx;
		GLint ny;
		GLint nz;

		/* Grid Map */
		map<GLint,Voxel*> voxelMap;

	public:

		/* Constructors & Destructors */
		Grid();
		~Grid();

		void initialize();

		Object* traverse(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		/* Voxel Map Operations */
		void addVoxel(Voxel* voxel);
		void removeVoxel(int index);

		Voxel* getVoxel(int index);

		/* Getters */
		BoundingBox* getBoundingBox();

		int getObjectNumber();

		GLfloat getWx();
		GLfloat getWy();
		GLfloat getWz();

		GLint getNx();
		GLint getNy();
		GLint getNz();

		/* Setters */
		void setBoundingBox(BoundingBox * boundingBox);

		void setObjectNumber(int ObjectNumber);

		void setWx(GLfloat wx);
		void setWy(GLfloat wy);
		void setWz(GLfloat wz);

		void setNx(GLint nx);
		void setNy(GLint nx);
		void setNz(GLint nx);

		void dump();
};

#endif