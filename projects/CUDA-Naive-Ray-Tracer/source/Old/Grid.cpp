#include "Grid.h"

Grid::Grid() {

}

Grid::~Grid() {
}

void Grid::initialize() {

	for(int z = 0 ; z < nz ; z++)
		for(int y = 0 ; y < ny ; y++)
			for(int x = 0 ; x < nx ; x++)
				voxelMap[x + y * nx + z * nx * ny] = new Voxel(x,y,z);
}

Object* Grid::traverse(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) {

	Vector entryPoint;
	Vector exitPoint;

	/* If the Grid BoundingBox wasn't hit */
	if(boundingBox->rayIntersection(rayOrigin, rayDirection, &entryPoint, &exitPoint, NULL) == false)
		return NULL;

	Vector maximum = boundingBox->getMaximum();
	Vector minimum = boundingBox->getMinimum();

	/* Voxel Indices */
	int ix, iy, iz;

	/* Calculate starting Voxel */
	if(	rayOrigin[VX] > minimum[VX] && rayOrigin[VX] < maximum[VX] &&
		rayOrigin[VY] > minimum[VY] && rayOrigin[VY] < maximum[VY] &&
		rayOrigin[VZ] > minimum[VZ] && rayOrigin[VZ] < maximum[VZ])
		entryPoint = rayOrigin;

	ix = Math::clamp((entryPoint[VX] - minimum[VX]) * nx / wx, 0.0f, (GLfloat)nx - 1.0f);
	iy = Math::clamp((entryPoint[VY] - minimum[VY]) * ny / wy, 0.0f, (GLfloat)ny - 1.0f);
	iz = Math::clamp((entryPoint[VZ] - minimum[VZ]) * nz / wz, 0.0f, (GLfloat)nz - 1.0f);	

	/* Voxel Dimensions */
	GLfloat dtx = (wx / nx) / fabs(rayDirection[VX]);
	GLfloat dty = (wy / ny) / fabs(rayDirection[VY]);
	GLfloat dtz = (wz / nz) / fabs(rayDirection[VZ]);

	/*cout << "dtx = " << dtx;
	cout << " dty = " << dty;
	cout << " dtz = " << dtz << endl;*/

	/* X Axis Step and Limit */
	GLfloat txNext = minimum[VX] + (ix + 1.0f) *  (wx / nx);
	GLint ixStep = 1;
	GLint ixStop = nx;
	
	if(rayDirection[VX] < 0.0f) {

		txNext = minimum[VX] + ix *  (wx / nx);
	
		ixStep = -1;
		ixStop = -1;
	}

	/* Y Axis Step and Limit */
	GLfloat tyNext = minimum[VY] + (iy + 1.0f) * (wy / ny);
	GLint iyStep = 1;
	GLint iyStop = ny;
	
	if(rayDirection[VY] < 0.0f) {

		tyNext = minimum[VY] + iy * (wy / ny);

		iyStep = -1;
		iyStop = -1;
	}

	/* Z Axis Step and Limit */
	GLfloat tzNext = minimum[VZ] + (iz + 1.0f) * (wz / nz);
	GLint izStep = 1;
	GLint izStop = nz;
	
	if(rayDirection[VZ] < 0.0f) {

		tzNext = minimum[VZ] + iz * (wz / nz);

		izStep = -1;
		izStop = -1;
	}

	txNext = fabs(txNext - entryPoint[VX]) / fabs(rayDirection[VX]);
	tyNext = fabs(tyNext - entryPoint[VY]) / fabs(rayDirection[VY]);
	tzNext = fabs(tzNext - entryPoint[VZ]) / fabs(rayDirection[VZ]);

	//cout << "txNext = " << txNext << " tyNext = " << tyNext << " tzNext = " << tzNext << endl;

	while(true) {

		if(ix == ixStop || iy == iyStop || iz == izStop)
			return NULL;

		Voxel *voxel = voxelMap[ix + iy * nx + iz * nx * ny];

		Vector point;
		Vector normal;

		if(txNext < tyNext && txNext < tzNext) {

			GLfloat tx = FLT_MAX;
			Object* objectHit = voxel->intersect(entryPoint,rayDirection,&point,&normal,&tx);

			if(objectHit != NULL && tx < txNext) {

				if(pointHit != NULL && normalHit != NULL) {
				
					*pointHit = point;
					*normalHit = normal;
				}

				return objectHit;
			}

			txNext += dtx;
			ix += ixStep;
		}
		else {
		
			if(tyNext < tzNext) {

				GLfloat ty = FLT_MAX;
				Object* objectHit = voxel->intersect(entryPoint,rayDirection,&point,&normal,&ty);

				if(objectHit != NULL && ty < tyNext) {

					if(pointHit != NULL && normalHit != NULL) {
				
						*pointHit = point;
						*normalHit = normal;
					}

					return objectHit;
				}

				tyNext += dty;
				iy += iyStep;
			}
			else {

				GLfloat tz = FLT_MAX;
				Object* objectHit = voxel->intersect(entryPoint,rayDirection,&point,&normal,&tz);

				if(objectHit != NULL && tz < tzNext) {

					if(pointHit != NULL && normalHit != NULL) {
				
						*pointHit = point;
						*normalHit = normal;
					}

					return objectHit;
				}

				tzNext += dtz;
				iz += izStep;
			}
		}
	}
}

/* Voxel Map Operations */
void Grid::addVoxel(Voxel* voxel) {

	voxelMap[voxel->getIndexX() + voxel->getIndexY() * nx + voxel->getIndexZ() * nx * ny] = voxel;
}

void Grid::removeVoxel(int index) {

	voxelMap.erase(index);
}

Voxel* Grid::getVoxel(int index) {

	return voxelMap[index];
}

/* Getters */
BoundingBox* Grid::getBoundingBox() {
	return boundingBox;
}

int Grid::getObjectNumber() {
	return objectNumber;
}

GLfloat Grid::getWx() {
	return wx;
}

GLfloat Grid::getWy() {
	return wy;
}

GLfloat Grid::getWz() {
	return wz;
}

GLint Grid::getNx() {
	return nx;
}

GLint Grid::getNy() {
	return ny;
}

GLint Grid::getNz() {
	return nz;
}

/* Setters */
void Grid::setBoundingBox(BoundingBox *boundingBox) {
	this->boundingBox = boundingBox;
}

void Grid::setObjectNumber(int objectNumber) {
	this->objectNumber = objectNumber;
}

void Grid::setWx(GLfloat wx) {
	this->wx = wx;
}

void Grid::setWy(GLfloat wy) {
	this->wy = wy;
}

void Grid::setWz(GLfloat wz) {
	this->wz = wz;
}

void Grid::setNx(GLint nx) {
	this->nx = nx;
}

void Grid::setNy(GLint ny) {
	this->ny = ny;
}

void Grid::setNz(GLint nz) {
	this->nz = nz;
}

void Grid::dump() {

	cout << endl << "Debugging Grid" << endl;

	cout << "[Grid] w.x = " << wx << endl;
	cout << "[Grid] w.y = " << wy << endl;
	cout << "[Grid] w.z = " << wz << endl;

	cout << "[Grid] n.x = " << nx << endl;
	cout << "[Grid] n.y = " << ny << endl;
	cout << "[Grid] n.z = " << nz << endl;

	cout << "[Grid] Contains " << voxelMap.size() << " Voxels." << endl << endl;

	for(map<GLint, Voxel*>::const_iterator voxelIterator = voxelMap.begin(); voxelIterator != voxelMap.end(); voxelIterator++)
		voxelIterator->second->dump();
}