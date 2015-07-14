#ifndef REAL_WOOD_SHADER_H
#define REAL_WOOD_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Wood Shader definitions */
#define WOOD_SHADER "Realistic Wood"

#define WOOD_TEXTURE_VERTEX_SHADER_FILE		"shaders/real_wood_vertex_shader.glsl"
#define WOOD_TEXTURE_FRAGMENT_SHADER_FILE	"shaders/real_wood_fragment_shader.glsl"

#define NOISE_TEXTURE_UNIFORM	"Noise"
#define NOISE_SCALE_UNIFORM		"NoiseScale"

class WoodShader : public ShaderProgram {

	protected:

	public:

		WoodShader(string name);
		~WoodShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif