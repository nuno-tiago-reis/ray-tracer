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

	// Initialize the Accumulated Maximum Hit Total
	this->accumulatedMaximumHitTotal = 0;
	// Initialize the Accumulated Missed Hit Total
	this->accumulatedMissedHitTotal = 0;
	// Initialize the Accumulated Connected Hit Total
	this->accumulatedConnectedHitTotal = 0;

	// Initialize the Final Maximum Hit Total
	this->finalMaximumHitTotal = 0;
	// Initialize the Final Missed Hit Total
	this->finalMissedHitTotal = 0;
	// Initialize the Final Connected Hit Total
	this->finalConnectedHitTotal = 0;
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

void TestManager::incrementAccumulatedMaximumHitTotal(int maximumHitTotal) {

	this->accumulatedMaximumHitTotal += maximumHitTotal;

	/*ofstream filestream;
	filestream.open("tests/output.txt", ofstream::out | ofstream::app);

	filestream << "[Adding] Maximum Hits:\t" << maximumHitTotal << endl;

	filestream.close();*/
}

void TestManager::incrementAccumulatedMissedHitTotal(int missedHitTotal) {
	
	this->accumulatedMissedHitTotal += missedHitTotal;

	/*ofstream filestream;
	filestream.open("tests/output.txt", ofstream::out | ofstream::app);

	filestream << "[Adding] Missed Hits:\t" << missedHitTotal << endl;

	filestream.close();*/
}

void TestManager::incrementAccumulatedConnectedHitTotal(int connectedHitTotal) {
	
	this->accumulatedConnectedHitTotal += connectedHitTotal;

	/*ofstream filestream;
	filestream.open("tests/output.txt", ofstream::out | ofstream::app);

	filestream << "[Adding] Connected Hits:\t" << connectedHitTotal << endl;

	filestream.close();*/
}

void TestManager::incrementFinalMaximumHitTotal(int maximumHitTotal) {

	this->finalMaximumHitTotal += maximumHitTotal;
}

void TestManager::incrementFinalMissedHitTotal(int missedHitTotal) {
	
	this->finalMissedHitTotal += missedHitTotal;
}

void TestManager::incrementFinalConnectedHitTotal(int connectedHitTotal) {
	
	this->finalConnectedHitTotal += connectedHitTotal;
}

GpuTimer* TestManager::getTimer(int timerID) {

	return this->timerMap[timerID];
}

int TestManager::getAccumulatedMaximumHitTotal() {

	return this->accumulatedMaximumHitTotal;
}

int TestManager::getAccumulatedMissedHitTotal() {

	return this->accumulatedMissedHitTotal;
}

int TestManager::getAccumulatedConnectedHitTotal() {

	return this->accumulatedConnectedHitTotal;
}

int TestManager::getFinalMaximumHitTotal() {

	return this->finalMaximumHitTotal;
}

int TestManager::getFinalMissedHitTotal() {

	return this->finalMissedHitTotal;
}

int TestManager::getFinalConnectedHitTotal() {

	return this->finalConnectedHitTotal;
}

void TestManager::setTimer(GpuTimer* timer, int timerID) {

	this->timerMap[timerID] = timer;
}

void TestManager::setAccumulatedMaximumHitTotal(int accumulatedMaximumHitTotal) {

	this->accumulatedMaximumHitTotal = accumulatedMaximumHitTotal;
}

void TestManager::setAccumulatedMissedHitTotal(int accumulatedMissedHitTotal) {

	this->accumulatedMissedHitTotal = accumulatedMissedHitTotal;
}

void TestManager::setAccumulatedConnectedHitTotal(int accumulatedConnectedHitTotal) {

	this->accumulatedConnectedHitTotal = accumulatedConnectedHitTotal;
}

void TestManager::setFinalMaximumHitTotal(int finalMaximumHitTotal) {

	this->finalMaximumHitTotal = finalMaximumHitTotal;
}

void TestManager::setFinalMissedHitTotal(int finalMissedHitTotal) {

	this->finalMissedHitTotal = finalMissedHitTotal;
}

void TestManager::setFinalConnectedHitTotal(int finalConnectedHitTotal) {

	this->finalConnectedHitTotal = finalConnectedHitTotal;
}

void TestManager::dump(int algorithmID, int sceneID, int iterationID, int rayTotal, int triangleTotal) {

	ostringstream filename;
	
	// Append the Algorithm Name
	if(algorithmID == 0)
		filename << "tests/test-crsh" << "-d" << HIERARCHY_MAXIMUM_DEPTH << "-n" << HIERARCHY_SUBDIVISION;
	else
		filename << "tests/test-rah" << "-d" << HIERARCHY_MAXIMUM_DEPTH << "-n" << HIERARCHY_SUBDIVISION;
	
	// Append the Scene Name
	if(sceneID == 0)
		filename << "-office.txt";
	else if(sceneID == 1)
		filename << "-cornell.txt";
	else if(sceneID == 2)
		filename << "-sponza.txt";

	ofstream filestream;
	filestream.open(filename.str(), ofstream::out | ofstream::app);

	// Initial Line
	if(iterationID == 0)
		filestream << "Shadow Ray Iteration" << endl;
	else
		filestream << "Reflection Ray Iteration " << iterationID << endl << endl;

	// Test Setup
	filestream << "Ray Total:\t" << rayTotal << endl;
	filestream << "TriangleTotal:\t" << triangleTotal << endl;

	// Intersection Results
	filestream << "[Accumulated] Maximum Hit Total:\t" << this->accumulatedMaximumHitTotal << endl;
	filestream << "[Accumulated] Missed Hit Total:\t" << this->accumulatedMissedHitTotal << endl;
	filestream << "[Accumulated] Connected Hit Total:\t" << this->accumulatedConnectedHitTotal << endl;

	// Final Intersection Results
	filestream << "[Final] Maximum Hit Total:\t" << this->finalMaximumHitTotal << endl;
	filestream << "[Final] Missed Hit Total:\t" << this->finalMissedHitTotal << endl;
	filestream << "[Final] Connected Hit Total:\t" << this->finalConnectedHitTotal << endl;

	// Algorithm Intersection Results
	filestream << "[Algorithm] Brute Force Total:\t" << rayTotal * triangleTotal << endl;
	filestream << "[Algorithm] Algorithm Total:\t" << this->accumulatedMaximumHitTotal + this->finalConnectedHitTotal * HIERARCHY_SUBDIVISION << endl;

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