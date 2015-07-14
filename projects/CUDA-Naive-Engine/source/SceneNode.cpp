#include "SceneNode.h"

SceneNode::SceneNode(string name) {

	this->name = name;

	this->object = NULL;
}

SceneNode::~SceneNode() {

	delete this->object;

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++)
		delete sceneNodeIterator->second;
}

void SceneNode::draw() {

	if(this->object != NULL)
		this->object->draw();

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->draw();
}

void SceneNode::update() {

	MatrixStack* matrixStack = MatrixStack::getInstance();

	Transform* objectTransform = this->object->getTransform();

	/* Update Node Matrix with Matrix Stack */
	objectTransform->setModelMatrix(matrixStack->peek());
	
	this->object->update();

	/* Push Model Matrix to the Matrix Stack */
	matrixStack->push(objectTransform->getModelMatrix());

	/* Update Child Nodes */
	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->update();

	/* Pop Model Matrix to the Matrix Stack */
	matrixStack->pop();
}

void SceneNode::update(GLfloat elapsedTime) {

	MatrixStack* matrixStack = MatrixStack::getInstance();

	Transform* objectTransform = this->object->getTransform();

	/* Update Node Matrix with Matrix Stack */
	objectTransform->setModelMatrix(matrixStack->peek());
	
	this->object->update(elapsedTime);

	/* Push Model Matrix to the Matrix Stack */
	matrixStack->push(objectTransform->getModelMatrix());

		/* Update Child Nodes */
		for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++)
			sceneNodeIterator->second->update(elapsedTime);

	/* Pop Model Matrix to the Matrix Stack */
	matrixStack->pop();
}

void SceneNode::addChildNode(SceneNode* sceneNode) {

	sceneNode->getObject()->setParentName(this->object->getName().c_str());

	this->nodeMap[sceneNode->getObject()->getName()] = sceneNode;
}

bool SceneNode::removeChildNode(string childNodeName) {

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++) {

		Object* graphicObject = sceneNodeIterator->second->getObject();

		if(graphicObject->getName().compare(childNodeName) == 0) {

			graphicObject->setParentName("None");

			this->nodeMap.erase(childNodeName);

			return true;
		}

		if(sceneNodeIterator->second->removeChildNode(childNodeName) == true) {

			return true;
		}
	}

	return false;
}

bool SceneNode::removeChildNode(string childNodeName, int depth) {

	if(depth == 0)
		return false;

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++) {

		Object* graphicObject = sceneNodeIterator->second->getObject();

		if(graphicObject->getName().compare(childNodeName) == 0) {

			graphicObject->setParentName("None");

			this->nodeMap.erase(childNodeName);

			return true;
		}

		if(sceneNodeIterator->second->removeChildNode(childNodeName,depth - 1) == true) {

			return true;
		}
	}

	return false;
}

SceneNode* SceneNode::getChildNode(string childNodeName) {

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++) {

		Object* graphicObject = sceneNodeIterator->second->getObject();

		if(graphicObject->getName().compare(childNodeName) == 0)
			return sceneNodeIterator->second;

		SceneNode* sceneNode = sceneNodeIterator->second->getChildNode(childNodeName);

		if(sceneNode != NULL)
			return sceneNode;
	}

	return NULL;
}

SceneNode* SceneNode::getChildNode(string childNodeName, int depth) {

	if(depth == 0)
		return NULL;

	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++) {

		Object* graphicObject = sceneNodeIterator->second->getObject();

		if(graphicObject->getName().compare(childNodeName) == 0)
			return sceneNodeIterator->second;

		SceneNode* sceneNode = sceneNodeIterator->second->getChildNode(childNodeName,depth - 1);

		if(sceneNode != NULL)
			return sceneNode;
	}

	return NULL;
}

string SceneNode::getName() {

	return this->name;
}

Object* SceneNode::getObject() {

	return this->object;
}	

void SceneNode::setName(string name) {

	this->name = name;
}

void SceneNode::setObject(Object* object) {

	this->object = object;
}

void SceneNode::dump() {

	cout << "<SceneNode Dump> = " << this->name << endl;

	/* Scene Node Map */
	cout << "<SceneNode Node List> = " << endl;
	for(map<string,SceneNode*>::const_iterator sceneNodeIterator = this->nodeMap.begin(); sceneNodeIterator != this->nodeMap.end(); sceneNodeIterator++)
		sceneNodeIterator->second->dump();
}