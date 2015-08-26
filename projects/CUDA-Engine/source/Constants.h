// Degree & Radian Conversion Constants
#define PI 3.1415927f
#define HALF_PI PI/2.0f

#define DEGREES_TO_RADIANS PI/180.0f
#define RADIANS_TO_DEGREES 180.0f/PI

// Viewport Size Constants
#define WIDTH 128
#define HEIGHT 128

// Ray Casting Constants
#define LIGHT_SOURCE_MAXIMUM 1
#define RAYS_PER_PIXEL_MAXIMUM 3

// Chunk Division Constants
#define CHUNK_DIVISION 10

// Hierarchy Depth Constants
#define HIERARCHY_MAXIMUM_DEPTH 4
#define HIERARCHY_SUBDIVISION 4

#define HIERARCHY_HIT_SUBDIVISION 100

	/*float3 sphereCenter1 = make_float3(-6.522256f, -7.25f, -1.542503f);
	float3 sphereCenter2 = make_float3(-6.522256f, -7.25f, -2.570542f);
	float3 sphereCenter3 = make_float3(-6.522256f, -7.25f, -5.654659f);
	float3 sphereCenter4 = make_float3(-6.522256f, -7.25f, -6.682692f);

	float3 sphereDirection1 = make_float3(-0.693206f, 0.71871f, -0.054057f);
	float3 sphereDirection2 = make_float3(-0.691413f, 0.716851f, -0.089851f);
	float3 sphereDirection3 = make_float3(-0.680941f, 0.705993f, -0.194661f);
	float3 sphereDirection4 = make_float3(-0.675881f, 0.700747f, -0.228341f);
	
	// First Time
	float3 sphereDirection12 = normalize(sphereCenter1 - sphereCenter2);
	float sphereDistance12 = length(sphereCenter1 - sphereCenter2) * 0.5f;
	
	float3 sphereCenter12 = sphereCenter1 - sphereDirection12 * sphereDistance12;
	float sphereRadius12 = sphereDistance12 + max(0.0f, 0.0f);

	printf("Direction 12\t %2.5f %2.5f %2.5f\t Distance = %2.5f\n", sphereDirection12.x, sphereDirection12.y, sphereDirection12.z, sphereDistance12);
	printf("Sphere 12\t %2.5f\t%2.5f\t%2.5f\t%2.5f\n\n", sphereCenter12.x, sphereCenter12.y, sphereCenter12.z, sphereRadius12);
	
	// Second Time
	float3 sphereDirection123 = normalize(sphereCenter12 - sphereCenter3);
	float sphereDistance123 = length(sphereCenter12 - sphereCenter3) * 0.5f;
	
	float3 sphereCenter123 = sphereCenter12 - sphereDirection123 * sphereDistance123;
	float sphereRadius123 = sphereDistance123 + max(sphereRadius12 , 0.0f);

	printf("Direction 123\t %2.5f %2.5f %2.5f\t Distance = %2.5f\n", sphereDirection123.x, sphereDirection123.y, sphereDirection123.z, sphereDistance123);
	printf("Sphere 123\t %2.5f\t%2.5f\t%2.5f\t%2.5f\n\n", sphereCenter123.x, sphereCenter123.y, sphereCenter123.z, sphereRadius123);

	// Third Time
	float3 sphereDirection1234 = normalize(sphereCenter123 - sphereCenter4);
	float sphereDistance1234 = length(sphereCenter123 - sphereCenter4) * 0.5f;

	float3 sphereCenter1234 = sphereCenter123 - sphereDirection1234 * sphereDistance1234;
	float sphereRadius1234 = sphereDistance1234 + max(sphereRadius123 , 0.0f);
	
	printf("Direction 1234\t %2.5f %2.5f %2.5f\t Distance = %2.5f\n", sphereDirection1234.x, sphereDirection1234.y, sphereDirection1234.z, sphereDistance1234);
	printf("Sphere 1234\t %2.5f\t%2.5f\t%2.5f\t%2.5f\n\n", sphereCenter1234.x, sphereCenter1234.y, sphereCenter1234.z, sphereRadius1234);*/

	/*printf("Ray Total %d\n", rayTotal);
	printf("Chunk Total %d\n", chunkTotal);
	
	printf("Trimmed Ray Counter %d\n", trimmedRayCounter);
	printf("Sorted Ray Counter \t%d\n", sortedRayCounter);
	printf("Chunk Counter %d\n", chunkCounter);
	printf("Node Counter %d\n", nodeCounter);

	exit(0);*/
