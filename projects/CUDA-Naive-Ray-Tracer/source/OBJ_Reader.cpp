#include "OBJ_Reader.h"

OBJ_Reader* OBJ_Reader::instance = NULL;

OBJ_Reader::OBJ_Reader() {
}

OBJ_Reader::~OBJ_Reader() {
}

OBJ_Reader* OBJ_Reader::getInstance() {

	if(instance == NULL)
		instance = new OBJ_Reader();

	return instance;
}

void OBJ_Reader::destroyInstance() {

	delete instance;

	instance = NULL;
}

vector<int> &split(const string &s, char delim, vector<int> &elems);
vector<int> split(const string &s, char delim);

void OBJ_Reader::loadMesh(string meshFilename, string materialFilename, Mesh* mesh) {

	cout << "[Initialization] LoadMesh(" << meshFilename << "," << materialFilename << "," << mesh->getName() << ")" << endl;

	/* Holds the current line being read */
	string line;

	/* Map holding all the declared materials in the .mtl file */
	map<string, Material*> materialMap;	

	/* ID of the Material currently being read */
	int currentMaterialID = 0;
	/* Name of the Material currently being read */
	string currentMaterialName = string("Uninitialized");

	/* Reading the Materials .mtl */
	ifstream materialFile(LOCATION + materialFilename);

	while(getline(materialFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;
		
		/* Reading a new Material */
		if(start == "newmtl") {

			iss >> currentMaterialName;

			materialMap[currentMaterialName] = new Material(currentMaterialID++, currentMaterialName);
		}
		/* Reading Ambient Component */
		else if(start == "Ka") {

			float x,y,z;
			iss >> x >> y >> z;

			materialMap[currentMaterialName]->setAmbient(Vector(x, y, z, 1.0f));
		}
		/* Reading Diffuse Component */
		else if(start == "Kd") {

			float x,y,z;
			iss >> x >> y >> z;

			materialMap[currentMaterialName]->setDiffuse(Vector(x, y, z, 1.0f));
		}
		/* Reading Specular Component */
		else if(start == "Ks") {

			float x,y,z;
			iss >> x >> y >> z;

			materialMap[currentMaterialName]->setSpecular(Vector(x, y, z, 1.0f));
		}
		/* Reading Specular Constant */
		else if(start == "Ns") {

			float s;
			iss >> s;

			materialMap[currentMaterialName]->setSpecularConstant(s);
		}
	}

	materialFile.close();

	/* If no Material was read, add a default one */
	if(currentMaterialName == string("Uninitialized")) {

		currentMaterialName = string("Default Material");
	
		materialMap[currentMaterialName] = new Material(currentMaterialID++, currentMaterialName);

		materialMap[currentMaterialName]->setAmbient(Vector(0.75f, 0.75f, 0.75f, 1.0f));
		materialMap[currentMaterialName]->setDiffuse(Vector(0.75f, 0.75f, 0.75f, 1.0f));
		materialMap[currentMaterialName]->setSpecular(Vector(0.75f, 0.75f, 0.75f, 1.0f));
		materialMap[currentMaterialName]->setSpecularConstant(100.0f);
	}

	/* Reading the Model .obj - First pass */
	int faceNumber = 0;
	int positionNumber = 0;
	int normalNumber = 0;
	int textureCoordinateNumber = 0;

	ifstream modelFile(LOCATION + meshFilename);

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;
		
		/* Add a Vertex */
		if(start == "v")
			positionNumber++;
		/* Add a Vertex Normal */
		else if(start == "vn")
			normalNumber++;
		/* Add a Vertex Texture UV */
		else if(start == "vt")
			textureCoordinateNumber++;
		/* Add a Face (Triangle) */
		else if(start == "f")
			faceNumber++;
	}

	modelFile.close();

	/* Reading the Model .obj - Second pass */
	modelFile.open(LOCATION + meshFilename);

	/* Storage Structures */
	Coordinate3D *positionArray = new Coordinate3D[positionNumber];
	Coordinate3D *normalArray = new Coordinate3D[normalNumber];
	Coordinate2D *textureCoordinateArray = new Coordinate2D[textureCoordinateNumber];

	/* Auxiliary Array to calculate Tangents */
	int *bufferVerticesID = new int[faceNumber * 3];

	/* Auxiliary Arrays to accumulate Tangents */
	Vector *sTangentArray = new Vector[positionNumber];
	Vector *tTangentArray = new Vector[positionNumber];

	/* Map holding all the declared vertices in the .obj file */
	map<unsigned int, Vertex*> vertexMap;

	/* Index Trackers */
	int currentFace = 0;
	int currentPosition = 0;
	int currentNormal = 0;
	int currentTextureCoordinate = 0;

	string activeMaterialName = string("Uninitialized");

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;

		/* Change Active Material */
		if(start == "usemtl") {

			iss >> activeMaterialName;
		}
		/* Add a Vertex */
		else if(start == "v") {

			float x,y,z;
			iss >> x >> y >> z;

			positionArray[currentPosition].x = x;
			positionArray[currentPosition].y = y;
			positionArray[currentPosition].z = z;			

			currentPosition++;
		}
		/* Add a Vertex Normal */
		else if(start == "vn") {

			float x,y,z;
			iss >> x >> y >> z;

			normalArray[currentNormal].x = x;
			normalArray[currentNormal].y = y;
			normalArray[currentNormal].z = z;			

			currentNormal++;
		} 
		/* Add a Vertex Texture UV */
		else if(start == "vt") {

			float u,v;
			iss >> u >> v;

			textureCoordinateArray[currentTextureCoordinate].u = u;
			textureCoordinateArray[currentTextureCoordinate].v = v;

			currentTextureCoordinate++;
		}
		/* Add a Face (Triangle) */
		else if(start == "f") {

			string faceVertex[3];
			iss >> faceVertex[0] >> faceVertex[1] >> faceVertex[2];

			for(int i=0; i<3; i++) {

				vector<int> index = split(faceVertex[i], '/');

				/* IDs of the Vertex Components */
				int positionID = index[0]-1;
				int normalID = index[2]-1;
				int textureCoordinatesID = index[1]-1;

				Vertex* vertex = new Vertex(currentFace * 3 + i);

				/* Vertex ID */
				bufferVerticesID[currentFace * 3 + i] = positionID;

				/* Vertex Position */
				vertex->setPosition(Vector(positionArray[positionID].x, positionArray[positionID].y, positionArray[positionID].z, 1.0f));
			
				/* Vertex Texture Coordinates */
				if(index.size() >= 2)
					vertex->setTextureCoordinates(Vector(textureCoordinateArray[textureCoordinatesID].u, textureCoordinateArray[textureCoordinatesID].v, 0.0f, 0.0f));
				else
					vertex->setTextureCoordinates(Vector(0.0f));

				/* Vertex Normal */
				if(index.size() >= 3)
					vertex->setNormal(Vector(normalArray[normalID].x, normalArray[normalID].y, normalArray[normalID].z, 0.0f));
				else
					vertex->setNormal(Vector(0.0f)); 				

				/* Vertex Material */
				if(activeMaterialName != string("Uninitialized") && materialMap.find(activeMaterialName) != materialMap.end())	
					vertex->setMaterialID(materialMap[activeMaterialName]->getID());			
				else
					vertex->setMaterialID(materialMap.begin()->second->getID());

				vertexMap[vertex->getID()] = vertex;
			}

			/* Load the Vertices */
			Vertex* vertex0 = vertexMap[currentFace * 3];
			Vertex* vertex1 = vertexMap[currentFace * 3 + 1];
			Vertex* vertex2 = vertexMap[currentFace * 3 + 2];

			/* Create the Vertex-based Edges */
			Vector positionEdge01 = vertex1->getPosition() - vertex0->getPosition();
			Vector positionEdge02 = vertex2->getPosition() - vertex0->getPosition();

			/* Create the UV-based Edges */
			Vector textureCoordinatesEdge01 = vertex1->getTextureCoordinates() - vertex0->getTextureCoordinates();
			Vector textureCoordinatesEdge02 = vertex2->getTextureCoordinates() - vertex0->getTextureCoordinates();

			float r = 1.0f / (textureCoordinatesEdge01[VX] * textureCoordinatesEdge02[VY] - textureCoordinatesEdge02[VX] * textureCoordinatesEdge01[VY]);

			Vector s(
				(textureCoordinatesEdge02[VY] * positionEdge01[VX] - textureCoordinatesEdge01[VY] * positionEdge02[VX]) * r, 
				(textureCoordinatesEdge02[VY] * positionEdge01[VY] - textureCoordinatesEdge01[VY] * positionEdge02[VY]) * r,
				(textureCoordinatesEdge02[VY] * positionEdge01[VZ] - textureCoordinatesEdge01[VY] * positionEdge02[VZ]) * r, 0.0f);

			Vector t(
				(textureCoordinatesEdge01[VX] * positionEdge02[VX] - textureCoordinatesEdge02[VX] * positionEdge01[VX]) * r, 
				(textureCoordinatesEdge01[VX] * positionEdge02[VY] - textureCoordinatesEdge02[VX] * positionEdge01[VY]) * r,
				(textureCoordinatesEdge01[VX] * positionEdge02[VZ] - textureCoordinatesEdge02[VX] * positionEdge01[VZ]) * r, 0.0f);

			/* Acumulate the new Tangents */
			sTangentArray[bufferVerticesID[currentFace * 3]] += s;
			tTangentArray[bufferVerticesID[currentFace * 3]] += t;

			sTangentArray[bufferVerticesID[currentFace * 3 + 1]] += s;
			tTangentArray[bufferVerticesID[currentFace * 3 + 1]] += t;

			sTangentArray[bufferVerticesID[currentFace * 3 + 2]] += s;
			tTangentArray[bufferVerticesID[currentFace * 3 + 2]] += t;

			currentFace++;
		}

		iss.clear();
	}

	/* Average the Tangents */
	for(map<unsigned int,Vertex*>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++) {

		Vertex* vertex =  vertexIterator->second;
	
		Vector normal = vertex->getNormal();
		Vector t1 = sTangentArray[bufferVerticesID[vertex->getID()]];
		Vector t2 = tTangentArray[bufferVerticesID[vertex->getID()]];
        
		// Gram-Schmidt orthogonalize
		Vector tangent = (t1 - normal * Vector::dotProduct(normal, t1));
		tangent.normalize();

		// Calculate handedness
		tangent[3] = (Vector::dotProduct(Vector::crossProduct(normal, tangent), t2) < 0.0f) ? -1.0f : 1.0f;
	
		/* Vertex Tangent */
		vertex->setTangent(tangent);

		if(Vector::dotProduct(normal,tangent) > Vector::threshold)
			cerr << "Failed calculating Tangent." << endl;
	}

	/* Add the Vertices to the Mesh */
	for(map<unsigned int,Vertex*>::const_iterator vertexIterator = vertexMap.begin(); vertexIterator != vertexMap.end(); vertexIterator++)
		mesh->addVertex(vertexIterator->second);

	/* Add the Materials to the Mesh */
	for(map<string,Material*>::const_iterator materialIterator = materialMap.begin(); materialIterator != materialMap.end(); materialIterator++)
		mesh->addMaterial(materialIterator->second);

	/* Cleanup */
	delete[] positionArray;
	delete[] normalArray;	
	delete[] textureCoordinateArray;

	delete[] sTangentArray;
	delete[] tTangentArray;

	delete[] bufferVerticesID;

	cout << "[Initialization] LoadMesh(" << meshFilename << "," << materialFilename << "," << mesh->getName() << ") Successfull!" << endl;

}

vector<int> &split(const string &s, char delim, vector<int> &elems) {

    std::stringstream ss(s);
    string item;

    while (std::getline(ss, item, delim)) {

		int i = atoi(item.c_str());
		elems.push_back(i);
    }

    return elems;
}

vector<int> split(const string &s, char delim) {

    std::vector<int> elems;
    split(s, delim, elems);

    return elems;
}