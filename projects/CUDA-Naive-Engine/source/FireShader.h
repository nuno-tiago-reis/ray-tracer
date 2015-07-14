#ifndef FIRE_SHADER_H
#define FIRE_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Blinn-phong Shader definitions */
#define FIRE_SHADER "Fire"

#define FIRE_VERTEX_SHADER_FILE		"shaders/fire_vertex_shader.glsl"
#define FIRE_FRAGMENT_SHADER_FILE	"shaders/fire_fragment_shader.glsl"

class FireShader : public ShaderProgram {

	protected:

	public:

		/* Constructors & Destructors */
		FireShader(string name);
		~FireShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif