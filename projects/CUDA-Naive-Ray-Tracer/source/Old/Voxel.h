#ifndef VOXEL_H
#define VOXEL_H

#include <map>

#include "Object.h"
#include "NFF_Plane.h"

class Voxel {

	private:

		/* Voxel Index inside the Grid */
		int indexX;
		int indexY;
		int indexZ;

		/* Map of Objects contained in the Voxel */
		map<GLint,Object*> objectMap;

	public:

		/* Constructors & Destructors */
		Voxel(int indexX, int indexY, int indexZ);
		~Voxel();

		Object* intersect(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit, GLfloat *distance);

		/* Map Operations */
		void addObject(Object* object);
		void removeObject(int identifier);

		map<GLint, Object*> getObjectMap();

		/* Getters */
		int getIndexX();
		int getIndexY();
		int getIndexZ();

		/* Setters */
		void setIndexX(int indexX);
		void setIndexY(int indexY);
		void setIndexZ(int indexZ);

		void dump();
};

#endif