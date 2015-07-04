#ifndef SCENE_MANAGER_H
#define SCENE_MANAGER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* FMODEX Sound Library */
#include "fmod.h"
#include "fmod.hpp"
#include "fmod_errors.h"

#define MAX_SOUND_CHANNELS 32

/* OpenGL definitions */
#include "GL/glew.h"
#include "GL/glut.h"
/* OpenGL Error Check */
#include "Utility.h"

/* C++ Includes */
#include <map>
#include <iterator>

/* Matrix - Ray Cast */
#include "Matrix.h"

/* Light - Light Map */
#include "Light.h"
/* Camera - Camera Map */
#include "Camera.h"
/* Shader - Shader Map */
#include "ShaderProgram.h"
/* Sound - Sound Map */
#include "Sound.h"

/* SceneNode - SceneNode Map*/
#include "SceneNode.h"

/* Object - Object Map*/
#include "Object.h"

/* User Interaction Handlers */
#include "KeyboardHandler.h"
#include "MouseHandler.h"

/* Engine Names */
#include "Names.h"

class SceneManager {

	private:

		/* Singleton Instance */
		static SceneManager *instance;

		int objectID;

		int rotationAxis;
		int currentObject;

		/* FMOD Sound System */
		FMOD::System* fmodSystem;
		FMOD::Channel* channel[MAX_SOUND_CHANNELS];
		
		/* Active Camera */
		Camera* _activeCamera;

		/* Sound Map */
		map<string,Sound*> soundMap;
		/* Light Map  */
		map<string,Light*> lightMap;
		/* Camera Map  */
		map<string,Camera*> cameraMap;

		/* Object Map  */
		map<string,Object*> objectMap;
		/* Scene Graph Map  */
		map<string,SceneNode*> sceneNodeMap;
		/* Shader Program Map  */
		map<string,ShaderProgram*> shaderProgramMap;

		/* Constructors & Destructors */
		SceneManager();
		~SceneManager();

	public:

		/* Singleton Methods */
		static SceneManager* getInstance();
		static void destroyInstance();

		/* Scene Methods */
		void init();

		void loadUniforms();

		void draw();

		void update(GLfloat elapsedTime);

		void reshape(GLint width, GLint height);

		/* Scene Interaction Methods */
		void readKeyboard(GLfloat elapsedTime);
		void readMouse(GLfloat elapsedTime);

		void rayCast(GLint* mousePosition, GLfloat elapsedTime);

		/* Object ID */
		int getObjectID();

		/* Getters */
		Camera* getActiveCamera();

		/* Setters */
		void setActiveCamera(Camera* camera);

		/* Sound Map Manipulation Methods */
		void addSound(Sound* sound);
		void removeSound(string soundName);

		Sound* getSound(string soundName);

		/* Light Map Manipulation Methods */
		void addLight(Light* light);
		void removeLight(string lightName);

		Light* getLight(string lightName);

		/* Camera Map Manipulation Methods */
		void addCamera(Camera* camera);
		void removeCamera(string cameraName);

		Camera* getCamera(string cameraName);

		/* Shader Map Manipulation Methods */
		void addShaderProgram(ShaderProgram* shaderProgram);
		void removeShaderProgram(string shaderProgramName);

		ShaderProgram* getShaderProgram(string shaderProgramName);

		/* Object Map Manipulation Methods */
		void addObject(Object* graphicObject);
		void removeObject(string graphicObjectName);

		Object* getObject(string graphicObjectName);

		/* Scene Node Map Manipulation Methods */
		void addSceneNode(SceneNode* sceneNode);
		void removeSceneNode(string sceneNodeName);

		SceneNode* getSceneNode(string sceneNodeName);

		/* Debug Methods */
		void dump();
};

#endif