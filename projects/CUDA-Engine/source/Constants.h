// Degree & Radian Conversion Constants
#define PI 3.1415927f
#define HALF_PI PI/2.0f

#define DEGREES_TO_RADIANS PI/180.0f
#define RADIANS_TO_DEGREES 180.0f/PI

// Viewport Size Constants
#define WIDTH 512
#define HEIGHT 512

// Ray-Tracing Depth Constants
#define DEPTH 2

// Ray Tracing Light Constants
#define LIGHT_SOURCE_MAXIMUM 4

// Ray Tracing Shadow Constants
#define SHADOW_RAY_RADIUS 0.5f
#define SHADOW_RAY_SPREAD PI / 128.0f

// Ray-Tracing Memory Constants
#define HIERARCHY_TRIANGLE_MAXIMUM 10000
#define HIERARCHY_TRIANGLE_ALLOCATION_MAXIMUM 5000

// Ray-Tracing Hierarchy Depth Constants
#define HIERARCHY_MAXIMUM_DEPTH 2
#define HIERARCHY_SUBDIVISION 8