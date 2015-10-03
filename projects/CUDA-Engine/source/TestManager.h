#ifndef TEST_MANAGER_H
#define TEST_MANAGER_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

// CUDA definitions
#include <cuda_runtime.h>

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

		// Maximum Hit Total
		int maximumHitTotal;
		// Missed Hit Total
		int missedHitTotal;
		// Connected Hit Total
		int connectedHitTotal;

		// Current Maximum Hit Total
		int currentMaximumHitTotal;
		// Current Missed Hit Total
		int currentMissedHitTotal;
		// Current Connected Hit Total
		int currentConnectedHitTotal;

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
		void incrementMaximumHitTotal(int maximumHitTotal);
		void incrementMissedHitTotal(int missedHitTotal);
		void incrementConnectedHitTotal(int connectedHitTotal);

		// Getters
		GpuTimer* getTimer(int timerID);

		int getMaximumHitTotal();
		int getMissedHitTotal();
		int getConnectedHitTotal();

		int getCurrentMaximumHitTotal();
		int getCurrentMissedHitTotal();
		int getCurrentConnectedHitTotal();

		// Setters
		void setTimer(GpuTimer* timer, int timerID);

		void setMaximumHitTotal(int maximumHitTotal);
		void setMissedHitTotal(int missedHitTotal);
		void setConnectedHitTotal(int connectedHitTotal);

		void setCurrentMaximumHitTotal(int currentMaximumHitTotal);
		void setCurrentMissedHitTotal(int currentMissedHitTotal);
		void setCurrentConnectedHitTotal(int currentConnectedHitTotal);

		// Dump
		void dump(int algorithmID, int sceneID, int iterationID, int rayTotal);
};

#endif