#include "NFF_Reader.h"

NFF_Reader* NFF_Reader::instance = NULL;

NFF_Reader::NFF_Reader() {
}

NFF_Reader::~NFF_Reader() {
}

NFF_Reader* NFF_Reader::getInstance() {

	if(instance == NULL)
		instance = new NFF_Reader();

	return instance;
}

void NFF_Reader::destroyInstance() {

	delete instance;

	instance = NULL;
}


void NFF_Reader::parseNFF(string modelFileName, Scene  *scene) {

	cout << "NFF_Reader::loadModel(" << modelFileName << ");" << endl;

	string line;

	/* Reading NFF file - First Pass */
	int numberLights = 0;
	int numberObjects = 0;
	int numberMaterials = 0;

	ifstream modelFile(LOCATION + modelFileName);

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;

		if(start == "l")
			numberLights++;
		else if(start == "f")
			numberMaterials++;
		else if(start == "pl" || start == "s" || start == "p")
			numberObjects++;
	}

	modelFile.close();

	/* Reading NFF file - Second Pass */
	modelFile.open(LOCATION + modelFileName);

	/* Storage Strucure */
	MaterialProperty *materialProperties = new MaterialProperty[numberMaterials]; //f

	/* Index Trackers */
	int currentLight = 0;
	int currentObject = 0;
	int currentMaterial = 0;

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;

		/* Background */
		if(start == "b") {

			GLfloat r,g,b;
			iss >> r >> g >> b;

			scene->setBackgroundColor(Vector(r,g,b,1.0f));
		}
		/* Camera Eye */
		else if(start == "from") {
			
			GLfloat x,y,z;
			iss >> x >> y >> z;

			scene->getCamera()->setEye(Vector(x,y,z,1.0f));
		}
		/* Camera Target */
		else if(start == "at") {

			GLfloat x,y,z;
			iss >> x >> y >> z;

			scene->getCamera()->setTarget(Vector(x,y,z,1.0f));
		}
		/* Camera Up Vector */
		else if(start == "up") {

			GLfloat x,y,z;
			iss >> x >> y >> z;

			scene->getCamera()->setUp(Vector(x,y,z,1.0f));
		}
		/* Camera Field of View */
		else if(start == "angle") {

			GLfloat angle;
			iss >> angle;

			scene->getCamera()->setFieldOfView(angle);
		}
		/* Camera Hither Plane */
		else if(start == "hither") {

			GLfloat hither;
			iss >> hither;

			scene->getCamera()->setNear(hither);
		}
		/* Viewport Resolution */
		else if(start == "resolution") {

			GLint width, height;
			iss >> width >> height;

			scene->getCamera()->setWidth(width);
			scene->getCamera()->setHeight(height);
		}
		/* Light Source */
		else if(start == "l") {

			GLfloat x,y,z;
			GLfloat r,g,b;
			iss >> x >> y >> z >> r >> g >> b;

			Light* light = new Light(currentLight);

			light->setPosition(Vector(x,y,z,1.0f));
			light->setColor(Vector(r,g,b,1.0f));

			scene->addLight(light);
		
			currentLight++;
		}
		/* Object Material */
		else if(start == "f") {

			GLfloat r,g,b;
			GLfloat Kd, Ks;
			GLfloat shine,transmittance,indexOfRefraction;
			iss >> r >> g >> b >> Kd >> Ks >> shine >> transmittance >> indexOfRefraction;

			materialProperties[currentMaterial].r = r;
			materialProperties[currentMaterial].g = g;
			materialProperties[currentMaterial].b = b;
			materialProperties[currentMaterial].Kd = Kd;
			materialProperties[currentMaterial].Ks = Ks;
			materialProperties[currentMaterial].shine = shine;
			materialProperties[currentMaterial].transmittance = transmittance;
			materialProperties[currentMaterial].indexOfRefraction = indexOfRefraction;

			currentMaterial++;
		}
		/* Plane Object */
		else if(start == "pl") {

			GLfloat x1, y1, z1, x2, y2, z2, x3, y3, z3;
			iss >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> x3 >> y3 >> z3;

			NFF_Plane* plane = new NFF_Plane(currentObject);
			
			plane->setVertex(Vector(x1,y1,z1,1.0f),0);
			plane->setVertex(Vector(x2,y2,z2,1.0f),1);
			plane->setVertex(Vector(x3,y3,z3,1.0f),2);

			plane->setColor(Vector(	materialProperties[currentMaterial-1].r,
									materialProperties[currentMaterial-1].g,
									materialProperties[currentMaterial-1].b,
									1.0f));

			plane->setDiffuseIntensity(materialProperties[currentMaterial-1].Kd);
			plane->setSpecularIntensity(materialProperties[currentMaterial-1].Ks);

			plane->setShininess(materialProperties[currentMaterial-1].shine);
			plane->setTransmittance(materialProperties[currentMaterial-1].transmittance);
			plane->setRefractionIndex(materialProperties[currentMaterial-1].indexOfRefraction);

			scene->addObject(plane);

			currentObject++;

		}
		/* Sphere Object */
		else if(start == "s") {

			GLfloat x,y,z,r;
			iss >> x >> y >> z >> r;

			NFF_Sphere* sphere = new NFF_Sphere(currentObject);

			sphere->setPosition(Vector(x,y,z,1.0f));
			sphere->setRadius(r);

			sphere->setColor(Vector(materialProperties[currentMaterial-1].r,
									materialProperties[currentMaterial-1].g,
									materialProperties[currentMaterial-1].b,
									1.0f));

			sphere->setDiffuseIntensity(materialProperties[currentMaterial-1].Kd);
			sphere->setSpecularIntensity(materialProperties[currentMaterial-1].Ks);

			sphere->setShininess(materialProperties[currentMaterial-1].shine);
			sphere->setTransmittance(materialProperties[currentMaterial-1].transmittance);
			sphere->setRefractionIndex(materialProperties[currentMaterial-1].indexOfRefraction);

			scene->addObject(sphere);

			currentObject++;
		}
		/* Polygon Object */
		else if(start == "p") {

			int numberVertices;
			iss >> numberVertices;

			NFF_Polygon* polygon = new NFF_Polygon(currentObject);

			for(int i=0;i<numberVertices;i++) {

				getline(modelFile, line);
				iss = istringstream(line);

				float x,y,z;
				iss >> x >> y >> z;

				polygon->addVertex(Vector(x,y,z,1.0f));
				
				iss.clear();
			}

			polygon->setColor(Vector(	materialProperties[currentMaterial-1].r,
										materialProperties[currentMaterial-1].g,
										materialProperties[currentMaterial-1].b,
										1.0f));

			polygon->setDiffuseIntensity(materialProperties[currentMaterial-1].Kd);
			polygon->setSpecularIntensity(materialProperties[currentMaterial-1].Ks);

			polygon->setShininess(materialProperties[currentMaterial-1].shine);
			polygon->setTransmittance(materialProperties[currentMaterial-1].transmittance);
			polygon->setRefractionIndex(materialProperties[currentMaterial-1].indexOfRefraction);

			scene->addObject(polygon);

			currentObject++;
		}
		/* AxisAligned Bounding Box Object */
		else if(start == "aabb") {
		
			GLfloat xMax, yMax, zMax;
			GLfloat xMin, yMin, zMin;

			iss >> xMin >> yMin >> zMin;
			iss >> xMax >> yMax >> zMax;

			NFF_BoundingBox* boundingBox = new NFF_BoundingBox(currentObject);

			boundingBox->setMaximum(Vector(xMax,yMax,zMax,1.0f));
			boundingBox->setMinimum(Vector(xMin,yMin,zMin,1.0f));

			boundingBox->setColor(Vector(	materialProperties[currentMaterial-1].r,
														materialProperties[currentMaterial-1].g,
														materialProperties[currentMaterial-1].b,
														1.0f));

			boundingBox->setDiffuseIntensity(materialProperties[currentMaterial-1].Kd);
			boundingBox->setSpecularIntensity(materialProperties[currentMaterial-1].Ks);

			boundingBox->setShininess(materialProperties[currentMaterial-1].shine);
			boundingBox->setTransmittance(materialProperties[currentMaterial-1].transmittance);
			boundingBox->setRefractionIndex(materialProperties[currentMaterial-1].indexOfRefraction);

			scene->addObject(boundingBox);

			currentObject++;
		}

		iss.clear();
	}

	scene->dump();

	modelFile.close();
}