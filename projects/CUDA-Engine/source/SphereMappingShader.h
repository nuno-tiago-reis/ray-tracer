#ifndef SPHERE_MAP_SHADER_H
#define SPHERE_MAP_SHADER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* Generic Shader */
#include "ShaderProgram.h"

using namespace std;

/* Sphere Mapping Shader definitions */
#define SPHERE_MAPPING_SHADER "Sphere Mapping"

#define SPHERE_MAPPING_VERTEX_SHADER_FILE	"shaders/sphere_map_vertex_shader.glsl"
#define SPHERE_MAPPING_FRAGMENT_SHADER_FILE	"shaders/sphere_map_fragment_shader.glsl"

#define SPHERE_MAPPING_UNIFORM "SphereMap"

class SphereMappingShader : public ShaderProgram {

	protected:

	public:

		/* Constructors & Destructors */
		SphereMappingShader(string name);
		~SphereMappingShader();

		/* GPU Creation & Destruction Methods */
		void bindAttributes();
		void bindUniforms();

		/* Debug Methods */
		void dump();
};

#endif