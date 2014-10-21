#ifndef NFF_POLYGON_H
#define	NFF_POLYGON_H

#include "Object.h"

#include <map>

using namespace std;

class NFF_Polygon : public Object {

	private:

		map<int,Vector> vertexMap;

	public:

		NFF_Polygon(int identifier);
		~NFF_Polygon();

		void addVertex(Vector vertex);
		void removeVertex(int index);

		Vector getVertex(int index);
		map<int,Vector> getVertexMap();

		bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		void createBoundingBox();

		void dump();
};

#endif