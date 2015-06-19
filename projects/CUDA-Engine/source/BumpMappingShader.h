#ifndef BUMP_MAP_SHADER_H
#define BUMP_MAP_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Bump Mapping Shader definitions */
#define BUMP_MAPPING_SHADER "Bump Mapping"

#define BUMP_MAPPING_VERTEX_SHADER_FILE		"shaders/bump_map_vertex_shader.glsl"
#define BUMP_MAPPING_FRAGMENT_SHADER_FILE	"shaders/bump_map_fragment_shader.glsl"

#define DIFFUSE_TEXTURE_UNIFORM "DiffuseTexture"
#define NORMAL_TEXTURE_UNIFORM	"NormalTexture"

class BumpMappingShader : public ShaderProgram {

	protected:

	public:

		/* Constructors & Destructors */
		BumpMappingShader(string name);
		~BumpMappingShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif