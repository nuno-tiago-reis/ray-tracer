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
	}

	float ElapsedMillis() {

		float elapsed;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		return elapsed;
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
		int finalMaximumHitTotal;
		// Final Missed Hit Total
		int finalMissedHitTotal;
		// Final Connected Hit Total
		int finalConnectedHitTotal;

		/* Constructors & Destructors - Private due to Singleton */
		TestManager();
		~TestManager();

	public:

		// Singleton Methods
		static TestManager* getInstance();
		static void destroyInstance();

		// Timers
		void startTimer(int timerID);
		void stopTimer(int timerID);

		// Incrementers
		void incrementAccumulatedMaximumHitTotal(int maximumHitTotal);
		void incrementAccumulatedMissedHitTotal(int missedHitTotal);
		void incrementAccumulatedConnectedHitTotal(int connectedHitTotal);

		void incrementFinalMaximumHitTotal(int maximumHitTotal);
		void incrementFinalMissedHitTotal(int missedHitTotal);
		void incrementFinalConnectedHitTotal(int connectedHitTotal);

		// Getters
		GpuTimer* getTimer(int timerID);

		int getAccumulatedMaximumHitTotal();
		int getAccumulatedMissedHitTotal();
		int getAccumulatedConnectedHitTotal();

		int getFinalMaximumHitTotal();
		int getFinalMissedHitTotal();
		int getFinalConnectedHitTotal();

		// Setters
		void setTimer(GpuTimer* timer, int timerID);

		void setAccumulatedMaximumHitTotal(int accumulatedMaximumHitTotal);
		void setAccumulatedMissedHitTotal(int accumulatedMissedHitTotal);
		void setAccumulatedConnectedHitTotal(int accumulatedConnectedHitTotal);

		void setFinalMaximumHitTotal(int finalMaximumHitTotal);
		void setFinalMissedHitTotal(int finalMissedHitTotal);
		void setFinalConnectedHitTotal(int finalConnectedHitTotal);

		// Dump
		void dump(int algorithmID, int sceneID, int iterationID, int rayTotal, int triangleTotal);
};

#endif