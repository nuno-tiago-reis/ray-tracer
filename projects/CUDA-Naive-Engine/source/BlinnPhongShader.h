#ifndef BLINN_PHONG_SHADER_H
#define BLINN_PHONG_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Blinn-phong Shader definitions */
#define BLINN_PHONG_SHADER "Blinn Phong"

#define BLINN_PHONG_VERTEX_SHADER_FILE		"shaders/blinn_phong_vertex_shader.glsl"
#define BLINN_PHONG_FRAGMENT_SHADER_FILE	"shaders/blinn_phong_fragment_shader.glsl"

class BlinnPhongShader : public ShaderProgram {

	protected:

	public:

		/* Constructors & Destructors */
		BlinnPhongShader(string name);
		~BlinnPhongShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif