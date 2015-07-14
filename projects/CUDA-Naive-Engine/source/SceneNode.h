#ifndef SCENE_NODE_H
#define SCENE_NODE_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Object */
#include "Object.h"

/* Matrix Stack */
#include "MatrixStack.h"

class SceneNode {

	private:

		/* Scene Node Identifier */
		string name;

		/* Scene Node Graphic Object (may be NULL) */
		Object* object;

		/* Scene Node Map */
		map<string,SceneNode*> nodeMap;

	public:
		
		/* Constructors & Destructors */
		SceneNode(string name);
		~SceneNode();

		/* Scene Graph Methods */
		void draw();

		void update();
		void update(GLfloat elapsedTime);

		/* Object List Manipulation Methods */
		void addChildNode(SceneNode* sceneNode);

		bool removeChildNode(string childNodeName);
		bool removeChildNode(string childNodeName, int depth);

		SceneNode* getChildNode(string childNodeName);
		SceneNode* getChildNode(string childNodeName, int depth);

		/* Getters */
		string getName();

		Object* getObject();

		/* Setters */
		void setName(string name);

		void setObject(Object* object);

		/* Debug Methods */
		void dump();
};

#endif