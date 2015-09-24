// Degree & Radian Conversion Constants
#define PI 3.1415927f
#define HALF_PI PI/2.0f

#define DEGREES_TO_RADIANS PI/180.0f
#define RADIANS_TO_DEGREES 180.0f/PI

// Viewport Size Constants
#define WIDTH 512
#define HEIGHT 512

// Ray-Tracing Depth Constants
#define DEPTH 3

// Ray Tracing Light Constants
#define LIGHT_SOURCE_MAXIMUM 1

// Ray Tracing Shadow Constants
#define SHADOW_RAY_RADIUS 0.0f //0.25f
#define SHADOW_RAY_DIVISION 16
#define SHADOW_RAY_SPREAD 0.0f //PI / 256.0f

// Ray-Tracing Memory Constants
#define HIERARCHY_TRIANGLE_MAXIMUM 5000
#define HIERARCHY_TRIANGLE_ALLOCATION_MAXIMUM 10000

// Ray-Tracing Hierarchy Depth Constants
#define HIERARCHY_MAXIMUM_DEPTH 3
#define HIERARCHY_SUBDIVISION 8

// Mode Constants
//#define ANTI_ALIASING
//#define SOFT_SHADOWS
#define IMPROVED_ALGORITHM

// Main Debug Constants
//#define SYNCHRONIZE_DEBUG
//#define BOUNDING_SPHERE_DEBUG
//#define TRIANGLE_DIVISION_DEBUG

// Kernel Debug Constants 
//#define CUB_STDERR
//#define BLOCK_GRID_DEBUG
//#define TRAVERSAL_DEBUG