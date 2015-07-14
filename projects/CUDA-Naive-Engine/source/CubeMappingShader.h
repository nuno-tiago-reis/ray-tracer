#ifndef CUBE_MAP_SHADER_H
#define CUBE_MAP_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Cube Mapping Shader definitions */
#define CUBE_MAPPING_SHADER "Cube Mapping"

#define CUBE_MAPPING_VERTEX_SHADER_FILE		"shaders/cube_map_vertex_shader.glsl"
#define CUBE_MAPPING_FRAGMENT_SHADER_FILE	"shaders/cube_map_fragment_shader.glsl"

#define CUBE_MAPPING_UNIFORM "CubeMap"

class CubeMappingShader : public ShaderProgram {

	protected:

	public:

		/* Constructors & Destructors */
		CubeMappingShader(string name);
		~CubeMappingShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif