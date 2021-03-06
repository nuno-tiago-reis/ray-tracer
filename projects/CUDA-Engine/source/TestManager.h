#ifndef TEST_MANAGER_H
#define TEST_MANAGER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

// CUDA definitions
#include <cuda_runtime.h>

// Utility
#include "Constants.h"

// CUB Timer
struct GpuTimer {

	cudaEvent_t start;
	cudaEvent_t stop;

	float elapsedTime;

	GpuTimer() {

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {

		cudaEventRecord(start, 0);
	}

	void Stop() {

		cudaEventRecord(stop, 0);

		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
	}

	float ElapsedMillis() {

		return elapsedTime;
	}
};

// C++ Includes
#include <map>
#include <fstream>

// Utility Includes
#include "Utility.h"

using namespace std;

class TestManager {

	public:

		// Ray Creation Timer ID
		static int rayCreationTimerID;
		// Ray Trimming Timer ID
		static int rayTrimmingTimerID;
		// Ray Compression Timer ID
		static int rayCompressionTimerID;
		// Ray Sorting Timer ID
		static int raySortingTimerID;
		// Ray Decompression Timer ID
		static int rayDecompressionTimerID;
		
		// Hierarchy Creation Timer ID
		static int hierarchyCreationTimerID;
		// Hierarchy Traversal Timer ID
		static int hierarchyTraversalTimerID;

		// Intersection Timer ID
		static int intersectionTimerID;
		// Shading Timer ID
		static int shadingTimerID;

		// Timer Count
		static int timerCount;

	private:

		// Singleton Instance
		static TestManager *instance;

		// Debug Timer Map
		map<int, GpuTimer*> timerMap;

		// Accumulated Maximum Hit Total
		int accumulatedMaximumHitTotal;
		// Accumulated Missed Hit Total
		int accumulatedMissedHitTotal;
		// Accumulated Connected Hit Total
		int accumulatedConnectedHitTotal;

		// Final Maximum Hit Total
		int finalMaximumHitTotal[HIERARCHY_MAXIMUM_DEPTH];
		// Final Missed Hit Total
		int finalMissedHitTotal[HIERARCHY_MAXIMUM_DEPTH];
		// Final Connected Hit Total
		int finalConnectedHitTotal[HIERARCHY_MAXIMUM_DEPTH];

		// Hierarchy Traversal Elapsed Times
		float hierarchyTraversalElapsedTimes;
		// Intersection Elapsed Times
		float intersectionElapsedTimes;

		// Constructors & Destructors - Private due to Singleton
		TestManager();
		~TestManager();

	public:

		// Singleton Methods
		static TestManager* getInstance();
		static void destroyInstance();

		// Methods
		void initialize();

		// Timers
		void startTimer(int timerID);
		void stopTimer(int timerID);

		// Incrementers
		void incrementAccumulatedMaximumHitTotal(int maximumHitTotal);
		void incrementAccumulatedMissedHitTotal(int missedHitTotal);
		void incrementAccumulatedConnectedHitTotal(int connectedHitTotal);

		void incrementFinalMaximumHitTotal(int maximumHitTotal, int hierarchyLevel);
		void incrementFinalMissedHitTotal(int missedHitTotal, int hierarchyLevel);
		void incrementFinalConnectedHitTotal(int connectedHitTotal, int hierarchyLevel);

		// Getters
		GpuTimer* getTimer(int timerID);

		int getAccumulatedMaximumHitTotal();
		int getAccumulatedMissedHitTotal();
		int getAccumulatedConnectedHitTotal();

		int getFinalMaximumHitTotal(int hierarchyLevel);
		int getFinalMissedHitTotal(int hierarchyLevel);
		int getFinalConnectedHitTotal(int hierarchyLevel);

		// Setters
		void setTimer(GpuTimer* timer, int timerID);

		void setAccumulatedMaximumHitTotal(int accumulatedMaximumHitTotal);
		void setAccumulatedMissedHitTotal(int accumulatedMissedHitTotal);
		void setAccumulatedConnectedHitTotal(int accumulatedConnectedHitTotal);

		void setFinalMaximumHitTotal(int finalMaximumHitTotal, int hierarchyLevel);
		void setFinalMissedHitTotal(int finalMissedHitTotal, int hierarchyLevel);
		void setFinalConnectedHitTotal(int finalConnectedHitTotal, int hierarchyLevel);

		// Dump
		void dump(int algorithmID, int sceneID, int iterationID, int rayTotal, int triangleTotal);
};

#endif