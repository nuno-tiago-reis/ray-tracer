#ifndef SCENE_H
#define SCENE_H

#include <stdio.h>
#include <map>

#include "Vector.h"
#include "Math.h"

#include "Camera.h"

#include "Light.h"
#include "Object.h"

#include "Grid.h"

//#define EPSILON 0.002f

#define GRID true
#define GRID_N 2.0f

#define MAX_DEPTH 6

#define SHADOW_N 5
#define SHADOW_N2 (float)(SHADOW_N*SHADOW_N)

#define EPSILON 0.0001f

#define CONSTANT_ATTENUATION 1.00f
#define LINEAR_ATTENUATION 0.0025f
#define QUADRATIC_ATTENUATION 0.0000025f

class Scene {

	private:

		/* Scene containing the Grid */
		Grid* grid;

		/* Scene Camera */
		Camera* camera;

		/* Scene Light Map */
		map<GLint,Light*> lightMap;
		/* Scene Object Map */
		map<GLint,Object*> objectMap;

		/* Scene Background Color */
		Vector backgroundColor;

	public:

		/* Constructors & Destructors */
		Scene();
		~Scene();

		Vector rayTracing(Vector rayOrigin, Vector rayDirection, int depth, float ior);

		Vector rayTracingNaive(Vector rayOrigin, Vector rayDirection, int depth, float ior);
		Vector rayTracingGrid(Vector rayOrigin, Vector rayDirection, int depth, float ior);

		void initializeGrid();

		Object * traverseGrid(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit);

		void addObject(Object* object);
		void removeObject(int identifier);

		void addLight(Light* light);
		void removeLight(int identifier);	

		/* Getters */
		Grid* getGrid();

		Camera* getCamera();

		Vector getBackgroundColor();

		/* Setters */
		void setGrid(Grid* grid);

		void setCamera(Camera* camera);

		void setBackgroundColor(Vector color);

		void dump();
};

#endif