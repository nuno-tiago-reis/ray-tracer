#include "Scene.h"

Scene::Scene() {

	grid = new Grid();
	camera = new Camera();
}

Scene::~Scene() {
}

Vector Scene::rayTracing(Vector rayOrigin, Vector rayDirection, int depth, float ior) {

	if(GRID)
		return rayTracingGrid(rayOrigin, rayDirection, depth, ior);
	else
		return rayTracingNaive(rayOrigin, rayDirection, depth, ior);
}

Vector Scene::rayTracingNaive(Vector rayOrigin, Vector rayDirection, int depth, float ior) {

	Vector pointHit;
	Vector normalHit;
	Object* objectHit = NULL;

	float minimumDistance = FLT_MAX;

	/* Test Ray intersection wih the Scenes Objects */
	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Vector point;
		Vector normal;
		Object* object = objectIterator->second;

		if(object->rayIntersection(rayOrigin, rayDirection, &point, &normal) == true) {

			float distance = (point - rayOrigin).length();

			if(minimumDistance > distance) {

				pointHit = point;
				normalHit = normal;
				objectHit = object;

				minimumDistance = distance;
			}
		}
	}

	/* If no Object was hit */
	if(objectHit == NULL)
		return backgroundColor;

	Vector color;

	/* Tests Lights intersection with the Scenes Objects */
	for(map<GLint, Light*>::const_iterator lightIterator = lightMap.begin(); lightIterator != lightMap.end(); lightIterator++) {

		Light* light = lightIterator->second;
		
		/* Light Attributes */
		Vector lightPosition = light->getPosition();
		Vector lightDirection = lightPosition - pointHit;
		lightDirection.normalize();

		/* Light Direction perpendicular plane base vectors */
		Vector lightPlaneAxisA;
		Vector lightPlaneAxisB;
		Vector w;

		/* Check which is the component with the smallest coeficient */
		GLfloat m = min(abs(lightDirection[VX]),max(abs(lightDirection[VY]),abs(lightDirection[VZ])));

		if(abs(lightDirection[VX]) == m) {

			w = Vector(1.0f,0.0f,0.0f,1.0f);
		}
		else if(abs(lightDirection[VY]) == m) {

			w = Vector(0.0f,1.0f,0.0f,1.0f);
		}
		else if(abs(lightDirection[VZ]) == m) {

			w = Vector(0.0f,0.0f,1.0f,1.0f);
		}

		/* Calculate the perpendicular plane base vectors */
		lightPlaneAxisA = Vector::crossProduct(w, lightDirection);
		lightPlaneAxisB = Vector::crossProduct(lightDirection,lightPlaneAxisA);

		/* Calculate Shadows for each of the interpolated positions along the perpendicular plane */
		for(int i=0;i<SHADOW_N;i++) {

			for(int j=0;j<SHADOW_N;j++) {

				Vector interpolatedPosition = lightPosition - lightPlaneAxisA*0.25f - lightPlaneAxisB*0.25f;
				interpolatedPosition += lightPlaneAxisA*(i*0.5f/SHADOW_N) + lightPlaneAxisB*(j*0.5f/SHADOW_N);

				Vector interpolatedDirection = interpolatedPosition - pointHit;
				interpolatedDirection.normalize();

				/* Calculate Pixel Color */
				GLfloat diffuseFactor = max(Vector::dotProduct(interpolatedDirection,normalHit),0.0f);
		
				if(diffuseFactor > 0.0f) {

					/* Test for Shadow Feelers */
					bool blockedLight = false;

					for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

						Object* object = objectIterator->second;

						if(object->rayIntersection(pointHit + interpolatedDirection * EPSILON, interpolatedDirection, NULL, NULL) == true) {

							blockedLight = true;
							break;
						}
					}
			
					if(blockedLight == false) {

						Vector vertexToEye = rayOrigin - pointHit;
						vertexToEye.normalize();

						/* Blinn-Phong approximation Halfway Vector */
						Vector halfVector = interpolatedDirection + vertexToEye;
						halfVector.normalize();

						/* Light Attenuation */
						GLfloat lightDistance =  (lightPosition - pointHit).length();
						GLfloat lightIntensity = 1.0f /
							(CONSTANT_ATTENUATION + lightDistance * LINEAR_ATTENUATION + pow(lightDistance,2.0f) * QUADRATIC_ATTENUATION);

						/* Diffuse Component */
						color += objectHit->getColor() * objectHit->getDiffuseIntensity() * diffuseFactor* lightIntensity * (1.0f/SHADOW_N2);

						/* Specular Component */
						GLfloat specularAngle = max(Vector::dotProduct(halfVector,normalHit),0.0f);
						GLfloat specularFactor = pow(specularAngle,objectHit->getShininess());
						if(specularFactor > 0.0f)
							color += objectHit->getColor() * objectHit->getSpecularIntensity() * specularFactor * lightIntensity * (1.0f/SHADOW_N2);
					}
				}
			}
		}
	}

	/* If maximum depth was reached */
	if(depth-1 == 0)
		return color;

	/* If the Object Hit is reflective */
	if(objectHit->getShininess() > 0.0f) {

		Vector reflectionDirection = Vector::reflect(rayDirection,normalHit);
		Vector reflectionColor = rayTracing(pointHit + reflectionDirection * EPSILON, reflectionDirection,depth-1, ior);

		color += reflectionColor * objectHit->getSpecularIntensity();
	}
	
	/* If the Object Hit is translucid */
	if(objectHit->getTransmittance() > 0.0f) {

		GLfloat newRefractionIndex;

		if(ior == 1.0f)
			newRefractionIndex = objectHit->getRefractionIndex();
		else
			newRefractionIndex = 1.0f;

		Vector refractionDirection = Vector::refract(rayDirection,normalHit, ior / newRefractionIndex);
		Vector refractionColor = rayTracing(pointHit + refractionDirection * EPSILON, refractionDirection,depth-1, newRefractionIndex);

		color += refractionColor * objectHit->getTransmittance() * pow(0.95f,depth-MAX_DEPTH+1);
	}

	return color;
}

Vector Scene::rayTracingGrid(Vector rayOrigin, Vector rayDirection, int depth, float ior) {

	/* Hit Object Properties */
	Vector pointHit;
	Vector normalHit;
	Object* objectHit = NULL;

	/* Test Ray intersection wih the Scenes Objects */
	objectHit = grid->traverse(rayOrigin,rayDirection,&pointHit,&normalHit);

	/* If no Object was hit */
	if(objectHit == NULL)
		return backgroundColor;

	Vector color;

	/* Tests Lights intersection with the Scenes Objects */
	for(map<GLint, Light*>::const_iterator lightIterator = lightMap.begin(); lightIterator != lightMap.end(); lightIterator++) {

		Light* light = lightIterator->second;
		
		/* Light Attributes */
		Vector lightPosition = light->getPosition();
		Vector lightDirection = lightPosition - pointHit;
		lightDirection.normalize();

		/* Light Direction perpendicular plane base vectors */
		Vector lightPlaneAxisA;
		Vector lightPlaneAxisB;
		Vector w;

		/* Check which is the component with the smallest coeficient */
		GLfloat m = min(abs(lightDirection[VX]),max(abs(lightDirection[VY]),abs(lightDirection[VZ])));

		if(abs(lightDirection[VX]) == m) {

			w = Vector(1.0f,0.0f,0.0f,1.0f);
		}
		else if(abs(lightDirection[VY]) == m) {

			w = Vector(0.0f,1.0f,0.0f,1.0f);
		}
		else if(abs(lightDirection[VZ]) == m) {

			w = Vector(0.0f,0.0f,1.0f,1.0f);
		}

		/* Calculate the perpendicular plane base vectors */
		lightPlaneAxisA = Vector::crossProduct(w, lightDirection);
		lightPlaneAxisB = Vector::crossProduct(lightDirection,lightPlaneAxisA);

		/* Calculate Shadows for each of the interpolated positions along the perpendicular plane */
		for(int i=0;i<SHADOW_N;i++) {

			for(int j=0;j<SHADOW_N;j++) {

				Vector interpolatedPosition = lightPosition - lightPlaneAxisA*0.25f - lightPlaneAxisB*0.25f;
				interpolatedPosition += lightPlaneAxisA*(i*0.5f/SHADOW_N) + lightPlaneAxisB*(j*0.5f/SHADOW_N);

				Vector interpolatedDirection = interpolatedPosition - pointHit;
				interpolatedDirection.normalize();

				/* Calculate Pixel Color */
				GLfloat diffuseFactor = max(Vector::dotProduct(interpolatedDirection,normalHit),0.0f);
		
				if(diffuseFactor > 0.0f) {

					/* Test for Shadow Feelers */
					bool blockedLight = false;

					if(grid->traverse(pointHit + interpolatedDirection * EPSILON, interpolatedDirection, NULL, NULL) != NULL)
						blockedLight = true;
			
					if(blockedLight == false) {

						Vector vertexToEye = rayOrigin - pointHit;
						vertexToEye.normalize();

						/* Blinn-Phong approximation Halfway Vector */
						Vector halfVector = interpolatedDirection + vertexToEye;
						halfVector.normalize();

						/* Light Attenuation */
						GLfloat lightDistance =  (lightPosition - pointHit).length();
						GLfloat lightIntensity = 1.0f /
							(CONSTANT_ATTENUATION + lightDistance * LINEAR_ATTENUATION + pow(lightDistance,2.0f) * QUADRATIC_ATTENUATION);

						/* Diffuse Component */
						color += objectHit->getColor() * objectHit->getDiffuseIntensity() * diffuseFactor* lightIntensity * (1.0f/SHADOW_N2);

						/* Specular Component */
						GLfloat specularAngle = max(Vector::dotProduct(halfVector,normalHit),0.0f);
						GLfloat specularFactor = pow(specularAngle,objectHit->getShininess());
						if(specularFactor > 0.0f)
							color += objectHit->getColor() * objectHit->getSpecularIntensity() * specularFactor * lightIntensity * (1.0f/SHADOW_N2);
					}
				}
			}
		}
	}

	/* If maximum depth was reached */
	if(depth-1 == 0)
		return color;

	/* If the Object Hit is reflective */
	if(objectHit->getShininess() > 0.0f) {

		Vector reflectionDirection = Vector::reflect(rayDirection,normalHit);
		Vector reflectionColor = rayTracing(pointHit + reflectionDirection * EPSILON, reflectionDirection,depth-1, ior);

		color += reflectionColor * objectHit->getSpecularIntensity();
	}
	
	/* If the Object Hit is translucid */
	if(objectHit->getTransmittance() > 0.0f) {

		GLfloat newRefractionIndex;

		if(ior == 1.0f)
			newRefractionIndex = objectHit->getRefractionIndex();
		else
			newRefractionIndex = 1.0f;

		Vector refractionDirection = Vector::refract(rayDirection,normalHit, ior / newRefractionIndex);
		Vector refractionColor = rayTracing(pointHit + refractionDirection * EPSILON, refractionDirection,depth-1, newRefractionIndex);

		color += refractionColor * objectHit->getTransmittance() * pow(0.95f,depth-MAX_DEPTH+1);
	}

	return color;
}

void Scene::initializeGrid() {

	if(GRID == false)
		return;

	Vector maximum = Vector(FLT_MIN);
	Vector minimum = Vector(FLT_MAX);

	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {
		
		Object* object = objectIterator->second;

		if(dynamic_cast<NFF_Plane*>(object))
			continue;

		object->createBoundingBox();

		BoundingBox* objectBoundingBox = object->getBoundingBox();

		/* Bounding Box Maximum */
		Vector objectMaximum = objectBoundingBox->getMaximum();

		if(maximum[VX] < objectMaximum[VX])
			maximum[VX] = objectMaximum[VX];

		if(maximum[VY] < objectMaximum[VY])
			maximum[VY] = objectMaximum[VY];

		if(maximum[VZ] < objectMaximum[VZ])
			maximum[VZ] = objectMaximum[VZ];

		/* Bounding Box Minimum */
		Vector objectMinimum = objectBoundingBox->getMinimum();
		
		if(minimum[VX] > objectMinimum[VX])
			minimum[VX] = objectMinimum[VX];

		if(minimum[VY] > objectMinimum[VY])
			minimum[VY] = objectMinimum[VY];

		if(minimum[VZ] > objectMinimum[VZ])
			minimum[VZ] = objectMinimum[VZ];
	}

	maximum += Vector(0.05f);
	minimum -= Vector(0.05f);

	/* Scene Bounding Box */
	BoundingBox *boundingBox = new BoundingBox();
	boundingBox->setMaximum(maximum);
	boundingBox->setMinimum(minimum);

	grid->setBoundingBox(boundingBox);

	/* Scene Object Number */
	GLint objectNumber = objectMap.size();

	grid->setObjectNumber(objectNumber);

	/* Grid dimensions */
	GLfloat wx = fabs(maximum[VX] - minimum[VX]);
	GLfloat wy = fabs(maximum[VY] - minimum[VY]);
	GLfloat wz = fabs(maximum[VZ] - minimum[VZ]);
	
	grid->setWx(wx);
	grid->setWy(wy);
	grid->setWz(wz);

	GLfloat s = pow((wx * wy * wz) / (GLfloat)objectNumber,1.0f/3.0f);

	/* Grid Voxel Number */
	GLint nx = (int)(GRID_N * wx / s) + 1;
	GLint ny = (int)(GRID_N * wy / s) + 1;
	GLint nz = (int)(GRID_N * wz / s) + 1;

	grid->setNx(nx);
	grid->setNy(ny);
	grid->setNz(nz);

	/* Create the Voxel Grid */
	grid->initialize();

	/* Fill each Voxel with the Scene Objects */
	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {
		
		Object* object = objectIterator->second;

		if(dynamic_cast<NFF_Plane*>(object))
			continue;

		BoundingBox * objectBoundingBox = object->getBoundingBox();

		Vector objectMaximum = objectBoundingBox->getMaximum();
		Vector objectMinimum = objectBoundingBox->getMinimum();

		int xmin = Math::clamp(fabs(objectMinimum[VX] - minimum[VX]) * nx/wx, 0.0f , (GLfloat)nx - 1.0f);
		int ymin = Math::clamp(fabs(objectMinimum[VY] - minimum[VY]) * ny/wy, 0.0f , (GLfloat)ny - 1.0f);
		int zmin = Math::clamp(fabs(objectMinimum[VZ] - minimum[VZ]) * nz/wz, 0.0f , (GLfloat)nz - 1.0f);

		int xmax = Math::clamp(fabs(objectMaximum[VX] - minimum[VX]) * nx/wx, 0.0f , (GLfloat)nx - 1.0f);
		int ymax = Math::clamp(fabs(objectMaximum[VY] - minimum[VY]) * ny/wy, 0.0f , (GLfloat)ny - 1.0f);
		int zmax = Math::clamp(fabs(objectMaximum[VZ] - minimum[VZ]) * nz/wz, 0.0f , (GLfloat)nz - 1.0f);

		for(int z = zmin; z <= zmax; z++)
			for(int y = ymin; y <= ymax; y++)
				for(int x = xmin; x <= xmax; x++)
					grid->getVoxel(x + y * nx + z * nx * ny)->addObject(object);

		/*for(int z = 0; z < nz ; z++)
			for(int y = 0; y < ny ; y++)
				for(int x = 0; x < nx ; x++)
					grid->getVoxel(x + y * nx + z * nx * ny)->addObject(object);*/
	}

	grid->getBoundingBox()->dump();
}

/* Getters and Setters */
void Scene::addObject(Object* object) {

	objectMap[object->getIdentifier()] = object;
}

void Scene::removeObject(int identifier) {

	objectMap.erase(identifier);
}

void Scene::addLight(Light* light) {

	lightMap[light->getIdentifier()] = light;
}

void Scene::removeLight(int identifier) {

	lightMap.erase(identifier);
}

/* Getters */
Grid* Scene::getGrid() {

	return grid;
}

Camera* Scene::getCamera() {

	return camera;
}

Vector Scene::getBackgroundColor() {

	return backgroundColor;
}

/* Setters */
void Scene::setGrid(Grid* grid) {

	this->grid = grid;
}

void Scene::setCamera(Camera* camera) {

	this->camera = camera;
}

void Scene::setBackgroundColor(Vector color) {

	backgroundColor = color;
}

void Scene::dump() {

	cout << "Debugging Scene" << endl;
 
	cout << "Number Objects " << objectMap.size() << endl;

	cout << "Number Lights " << lightMap.size() << endl;

	cout << "OBJECTS:::::" << endl;

	for(map<GLint, Object*>::const_iterator objectIterator = objectMap.begin(); objectIterator != objectMap.end(); objectIterator++) {

		Object* object = objectIterator->second;

		//cout << "ID = " << object->getIdentifier() << endl;

	}
}