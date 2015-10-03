#include "TestManager.h"

// Singleton Instance
TestManager* TestManager::instance = NULL;

// Ray Creation Timer ID
int TestManager::rayCreationTimerID = 0;
// Ray Trimming Timer ID
int TestManager::rayTrimmingTimerID = 1;

// Ray Compression Timer ID
int TestManager::rayCompressionTimerID = 2;
// Ray Sorting Timer ID
int TestManager::raySortingTimerID = 3;
// Ray Decompression Timer ID
int TestManager::rayDecompressionTimerID = 4;

// Hierarchy Creation Timer ID
int TestManager::hierarchyCreationTimerID = 5;
// Hierarchy Traversal Timer ID
int TestManager::hierarchyTraversalTimerID = 6;

// Intersection Timer ID
int TestManager::intersectionTimerID = 7;
// Shading Timer ID
int TestManager::shadingTimerID = 8;

TestManager::TestManager() {

	// Initialize the Maximum Hit Total
	this->maximumHitTotal = 0;
	// Initialize the Missed Hit Total
	this->missedHitTotal = 0;
	// Initialize the Connected Hit Total
	this->connectedHitTotal = 0;

	// Initialize the Current Maximum Hit Total
	this->currentMaximumHitTotal = 0;
	// Initialize the Current Missed Hit Total
	this->currentMissedHitTotal = 0;
	// Initialize the Current Connected Hit Total
	this->currentConnectedHitTotal = 0;
}

TestManager::~TestManager() {

}

TestManager* TestManager::getInstance() {

	if(instance == NULL)
		instance = new TestManager();

	return instance;
}

void TestManager::destroyInstance() {

	delete instance;

	instance = NULL;
}

void TestManager::startTimer(int timerID) {

	// Delete the Timer if it exists
	if(timerMap.find(timerID) != timerMap.end())
		delete timerMap[timerID];

	// Create the Timer
	timerMap[timerID] = new GpuTimer();

	// Make sure there isn't a Kernel Running
	Utility::checkCUDAError("cudaDeviceSynchronize()", cudaDeviceSynchronize());

	this->timerMap[timerID]->Start();

	Utility::checkCUDAError("TestManager::GpuTimer::Start()", cudaDeviceSynchronize());
	Utility::checkCUDAError("TestManager::GpuTimer::Start()", cudaGetLastError());
}

void TestManager::stopTimer(int timerID) {

	// Make sure there isn't a Kernel Running
	Utility::checkCUDAError("cudaDeviceSynchronize()", cudaDeviceSynchronize());

	this->timerMap[timerID]->Stop();

	Utility::checkCUDAError("TestManager::GpuTimer::Stop()", cudaDeviceSynchronize());
	Utility::checkCUDAError("TestManager::GpuTimer::Stop()", cudaGetLastError());
}

void TestManager::incrementMaximumHitTotal(int maximumHitTotal) {

	this->maximumHitTotal += maximumHitTotal;

	this->currentMaximumHitTotal = maximumHitTotal;
}

void TestManager::incrementMissedHitTotal(int missedHitTotal) {

	this->missedHitTotal += missedHitTotal;

	this->currentMissedHitTotal = missedHitTotal;
}

void TestManager::incrementConnectedHitTotal(int connectedHitTotal) {

	this->connectedHitTotal += connectedHitTotal;

	this->currentConnectedHitTotal = connectedHitTotal;
}

GpuTimer* TestManager::getTimer(int timerID) {

	return this->timerMap[timerID];
}

int TestManager::getMaximumHitTotal() {

	return this->maximumHitTotal;
}

int TestManager::getMissedHitTotal() {

	return this->missedHitTotal;
}

int TestManager::getConnectedHitTotal() {

	return this->connectedHitTotal;
}

int TestManager::getCurrentMaximumHitTotal() {

	return this->currentMaximumHitTotal;
}

int TestManager::getCurrentMissedHitTotal() {

	return this->currentMissedHitTotal;
}

int TestManager::getCurrentConnectedHitTotal() {

	return this->currentConnectedHitTotal;
}

void TestManager::setTimer(GpuTimer* timer, int timerID) {

	this->timerMap[timerID] = timer;
}

void TestManager::setMaximumHitTotal(int maximumHitTotal) {

	this->maximumHitTotal = maximumHitTotal;
}

void TestManager::setMissedHitTotal(int missedHitTotal) {

	this->missedHitTotal = missedHitTotal;
}

void TestManager::setConnectedHitTotal(int connectedHitTotal) {

	this->connectedHitTotal = connectedHitTotal;
}

void TestManager::setCurrentMaximumHitTotal(int currentMaximumHitTotal) {

	this->currentMaximumHitTotal = currentMaximumHitTotal;
}

void TestManager::setCurrentMissedHitTotal(int currentMissedHitTotal) {

	this->currentMissedHitTotal = currentMissedHitTotal;
}

void TestManager::setCurrentConnectedHitTotal(int currentConnectedHitTotal) {

	this->currentConnectedHitTotal = currentConnectedHitTotal;
}

void TestManager::dump(int algorithmID, int sceneID, int iterationID, int rayTotal) {

	ostringstream filename;
	
	// Append the Algorithm Name
	if(algorithmID == 0)
		filename << "tests/test-crsh-";
	else
		filename << "tests/test-rah-";
	
	// Append the Scene Name
	if(sceneID == 0)
		filename << "office.txt";
	else if(sceneID == 1)
		filename << "cornell.txt";
	else if(sceneID == 2)
		filename << "sponza.txt";

	ofstream filestream;
	filestream.open(filename.str(), ofstream::out | ofstream::app);

	// Initial Line
	if(iterationID == 0)
		filestream << "Shadow Ray Iteration" << endl;
	else
		filestream << "Reflection Ray Iteration " << iterationID << endl << endl;

	// Intersection Results
	filestream << "[Accumulated] Maximum Hit Total:\t" << this->maximumHitTotal << endl;
	filestream << "[Accumulated] Missed Hit Total:\t" << this->missedHitTotal << endl;
	filestream << "[Accumulated] Connected Hit Total:\t" << this->connectedHitTotal << endl;

	// Current Intersection Results
	filestream << "[Current] Maximum Hit Total:\t" << this->currentMaximumHitTotal << endl;
	filestream << "[Current] Missed Hit Total:\t" << this->currentMissedHitTotal << endl;
	filestream << "[Current] Connected Hit Total:\t" << this->currentConnectedHitTotal << endl;

	// Timer Results
	filestream << "[Timer] Ray Creation:\t" << this->timerMap[rayCreationTimerID]->ElapsedMillis() << endl; 
	filestream << "[Timer] Ray Trimming:\t" << this->timerMap[rayTrimmingTimerID]->ElapsedMillis() << endl;

	// If we're using CRSH Algorithm
	if(algorithmID == 0) {

		filestream << "[Timer] Ray Compression:\t" << this->timerMap[rayCompressionTimerID]->ElapsedMillis() << endl;
		filestream << "[Timer] Ray Sorting:\t" << this->timerMap[raySortingTimerID]->ElapsedMillis() << endl;
		filestream << "[Timer] Ray Decompression:\t" << this->timerMap[rayDecompressionTimerID]->ElapsedMillis() << endl;
	}

	filestream << "[Timer] Hierarchy Creation:\t" << this->timerMap[hierarchyCreationTimerID]->ElapsedMillis() << endl;
	filestream << "[Timer] Hierarchy Traversal:\t" << this->timerMap[hierarchyTraversalTimerID]->ElapsedMillis() << endl;

	filestream << "[Timer] Intersection:\t" << this->timerMap[intersectionTimerID]->ElapsedMillis() << endl;
	filestream << "[Timer] Shading:\t" << this->timerMap[shadingTimerID]->ElapsedMillis() << endl;

	filestream << endl;

	filestream.close();
}