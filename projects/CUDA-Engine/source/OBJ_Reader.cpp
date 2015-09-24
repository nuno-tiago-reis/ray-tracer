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

void OBJ_Reader::loadMesh(string meshFilename, Mesh* mesh) {

	cout << "[Initialization] LoadMesh(" << meshFilename << ");" << endl;

	// Reading the Model .obj - First pass
	int faceNumber = 0;
	int vertexNumber = 0;
	int normalNumber = 0;
	int textureCoordinateNumber = 0;

	// Check if we're opening a new Mesh
	if(this->meshFilename.compare(meshFilename) != 0) {

		// Store the new Filename
		this->meshFilename = meshFilename;

		// Reset the Mesh Name
		this->meshName = string("Uninitialized");
		// Reset the Mesh Line
		this->meshLineNumber = 0;
		// Reset the Mesh End of File Indicator
		this->meshEndOfFile = false;

		// Reset the Material Name
		this->materialName = string("Uninitialized");

		// Reset the Vertex offset
		this->offsetVertex = 0;
		// Reset the Normal offset
		this->offsetNormal = 0;
		// Reset the Texture UV offset
		this->offsetTextureCoordinate = 0;
	}

	// Open the new Stream
	this->meshFileStream.open(string(LOCATION).append(this->meshFilename), ifstream::in);

	// Line
	string currentMeshLine;

	// Mesh Name
	string currentMeshName = string("Uninitialized");
	// Line Counter
	int currentMeshLineNumber = 0;

	while(true) {

		currentMeshLineNumber++;

		// Check if we're past the previous mesh
		if(currentMeshLineNumber < this->meshLineNumber) {

			meshFileStream.ignore(numeric_limits<streamsize>::max(), meshFileStream.widen('\n'));

			continue;
		}

		// Open the Line and exit if EOF
		if(!getline(meshFileStream, currentMeshLine))
			break;

		// Create the Line Stream
		istringstream iss(currentMeshLine);

		string meshLine0;
		iss >> meshLine0;

		// Check if there is an object
		if(meshLine0.compare("o") == 0) {

			if(currentMeshName.compare(string("Uninitialized")) != 0)
				break;

			iss >> currentMeshName;
		}

		// Add a Vertex
		if(meshLine0.compare("v") == 0)
			vertexNumber++;
		// Add a Vertex Normal
		else if(meshLine0.compare("vn") == 0)
			normalNumber++;
		// Add a Vertex Texture UV
		else if(meshLine0.compare("vt") == 0)
			textureCoordinateNumber++;
		// Add a Face (Triangle)
		else if(meshLine0.compare("f") == 0)
			faceNumber++;

		// Check if there is a material
		if(meshLine0.compare("usemtl") == 0)
			iss >> this->materialName;

		iss.clear();
	}

	// Close the Model File
	this->meshFileStream.close();

	// Mesh is Over
	if(currentMeshName.compare("Uninitialized") == 0 && this->meshLineNumber != 0) {
	
		this->meshEndOfFile = true;

		return;
	}

	// Reading the Model .obj - Second pass
	this->meshFileStream.open(LOCATION + this->meshFilename, ifstream::in);

	// Store the Mesh Name
	this->meshName = currentMeshName;

	// Storage Structures
	Coordinate3D *vertexArray = new Coordinate3D[vertexNumber];
	Coordinate3D *normalArray = new Coordinate3D[normalNumber];
	Coordinate2D *textureCoordinateArray = new Coordinate2D[textureCoordinateNumber];

	// Calculated after parsing
	Vector *sTangentArray = new Vector[vertexNumber];
	Vector *tTangentArray = new Vector[vertexNumber];

	for(int i=0; i<vertexNumber; i++) {
	
		sTangentArray[i] = Vector(0.0f);
		tTangentArray[i] = Vector(0.0f);
	}

	// Final GPU-ready Structure
	VertexStructure *bufferVertices = new VertexStructure[faceNumber * 3];
	int *bufferVerticesID = new int[faceNumber * 3];

	// Index Trackers
	int currentFace = 0;
	int currentVertex = 0;
	int currentNormal = 0;
	int currentTextureCoordinate = 0;

	// Mesh Name
	currentMeshName = string("Uninitialized");
	// Line Counter
	currentMeshLineNumber = 0;

	while(true) {

		currentMeshLineNumber++;

		// Check if we're past the previous mesh
		if(currentMeshLineNumber < this->meshLineNumber) {

			meshFileStream.ignore(numeric_limits<streamsize>::max(), meshFileStream.widen('\n'));

			continue;
		}

		// Open the Line and exit if EOF
		if(!getline(meshFileStream, currentMeshLine))
			break;

		// Create the Line Stream
		istringstream iss(currentMeshLine);

		string meshLine0;
		iss >> meshLine0;

		// Check if there is an object
		if(meshLine0.compare("o") == 0) {

			if(currentMeshName.compare("Uninitialized") != 0)
				break;

			iss >> currentMeshName;
		}

		// Add a Vertex
		if(meshLine0.compare("v") == 0) {

			float x,y,z;
			iss >> x >> y >> z;

			vertexArray[currentVertex].x = x;
			vertexArray[currentVertex].y = y;
			vertexArray[currentVertex].z = z;

			currentVertex++;
		}
		// Add a Vertex Normal
		else if(meshLine0.compare("vn") == 0) {

			float x,y,z;
			iss >> x >> y >> z;

			normalArray[currentNormal].x = x;
			normalArray[currentNormal].y = y;
			normalArray[currentNormal].z = z;

			currentNormal++;
		} 
		// Add a Vertex Texture UV
		else if(meshLine0.compare("vt") == 0) {

			float u,v;
			iss >> u >> v;

			textureCoordinateArray[currentTextureCoordinate].u = u;
			textureCoordinateArray[currentTextureCoordinate].v = v;

			currentTextureCoordinate++;
		}
		// Add a Face (Triangle)
		else if(meshLine0.compare("f") == 0) {

			string faceVertex[3];
			iss >> faceVertex[0] >> faceVertex[1] >> faceVertex[2];

			for(int i=0; i<3; i++) {

				vector<int> index = split(faceVertex[i], '/');

				// Vertex ID
				bufferVerticesID[currentFace * 3 + i] = index[0]-1-this->offsetVertex;

				// Vertex Position
				bufferVertices[currentFace * 3 + i].position[0] = vertexArray[index[0]-1-this->offsetVertex].x;
				bufferVertices[currentFace * 3 + i].position[1] = vertexArray[index[0]-1-this->offsetVertex].y;
				bufferVertices[currentFace * 3 + i].position[2] = vertexArray[index[0]-1-this->offsetVertex].z;
				bufferVertices[currentFace * 3 + i].position[3] = 1.0f;
			
				// Vertex Texture Coordinates
				if(index.size() >= 2) {

					bufferVertices[currentFace * 3 + i].textureUV[0] = textureCoordinateArray[index[1]-1-this->offsetTextureCoordinate].u;
					bufferVertices[currentFace * 3 + i].textureUV[1] = textureCoordinateArray[index[1]-1-this->offsetTextureCoordinate].v;
				} 
				else {

					bufferVertices[currentFace * 3 + i].textureUV[0] = 0.0f;
					bufferVertices[currentFace * 3 + i].textureUV[1] = 0.0f;
				}

				// Vertex Normals
				if(index.size() >= 3) {

					bufferVertices[currentFace * 3 + i].normal[0] = normalArray[index[2]-1-this->offsetNormal].x;
					bufferVertices[currentFace * 3 + i].normal[1] = normalArray[index[2]-1-this->offsetNormal].y;
					bufferVertices[currentFace * 3 + i].normal[2] = normalArray[index[2]-1-this->offsetNormal].z;
					bufferVertices[currentFace * 3 + i].normal[3] = 0.0f;
				}
				else {

					bufferVertices[currentFace * 3 + i].normal[0] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[1] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[2] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[3] = 0.0f;
				}
			}

			// Create the Vertex-based Edges
			Coordinate3D xyz1;
			xyz1.x = bufferVertices[currentFace * 3 + 1].position[0] - bufferVertices[currentFace * 3].position[0];
			xyz1.y = bufferVertices[currentFace * 3 + 1].position[1] - bufferVertices[currentFace * 3].position[1];
			xyz1.z = bufferVertices[currentFace * 3 + 1].position[2] - bufferVertices[currentFace * 3].position[2];

			Coordinate3D xyz2;
			xyz2.x = bufferVertices[currentFace * 3 + 2].position[0] - bufferVertices[currentFace * 3].position[0];
			xyz2.y = bufferVertices[currentFace * 3 + 2].position[1] - bufferVertices[currentFace * 3].position[1];
			xyz2.z = bufferVertices[currentFace * 3 + 2].position[2] - bufferVertices[currentFace * 3].position[2];

			// Create the UV-based Edges
			Coordinate2D uv1;
			uv1.u = bufferVertices[currentFace * 3 + 1].textureUV[0] - bufferVertices[currentFace * 3].textureUV[0];
			uv1.v = bufferVertices[currentFace * 3 + 1].textureUV[1] - bufferVertices[currentFace * 3].textureUV[1];

			Coordinate2D uv2;
			uv2.u = bufferVertices[currentFace * 3 + 2].textureUV[0] - bufferVertices[currentFace * 3].textureUV[0];
			uv2.v = bufferVertices[currentFace * 3 + 2].textureUV[1] - bufferVertices[currentFace * 3].textureUV[1];

			float r = 1.0f / (uv1.u * uv2.v - uv2.u * uv1.v);

			Vector s((uv2.v * xyz1.x - uv1.v * xyz2.x) * r, (uv2.v * xyz1.y - uv1.v * xyz2.y) * r,(uv2.v * xyz1.z - uv1.v * xyz2.z) * r, 0.0f);
			Vector t((uv1.u * xyz2.x - uv2.u * xyz1.x) * r, (uv1.u * xyz2.y - uv2.u * xyz1.y) * r,(uv1.u * xyz2.z - uv2.u * xyz1.z) * r, 0.0f);

			// Acumulate the new Tangents
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

	// Check if the Model File is Over
	if(this->meshFileStream.eof() == true)
		this->meshEndOfFile = true;

	// Close the Model File
	this->meshFileStream.close();

	// Increment the Line Number for the next object
	this->meshLineNumber = currentMeshLineNumber;

	// Increment the Vertex offset for the next object
	this->offsetVertex += vertexNumber;
	// Increment the Normal offset for the next object
	this->offsetNormal += normalNumber;
	// Increment the Texture UV offset for the next object
	this->offsetTextureCoordinate += textureCoordinateNumber;

	// Average the Tangents
	for(int i=0; i<faceNumber * 3; i++) {

		Vector n = Vector(bufferVertices[i].normal);
		Vector t1 = sTangentArray[bufferVerticesID[i]];
		Vector t2 = tTangentArray[bufferVerticesID[i]];

		// Gram-Schmidt orthogonalize
		Vector tangent = (t1 - n * Vector::dotProduct(n, t1));
		tangent.normalize();

		// Calculate handedness
		tangent[3] = (Vector::dotProduct(Vector::crossProduct(n, tangent), t2) < 0.0f) ? -1.0f : 1.0f;
	
		for(int j=0; j<4; j++)
			bufferVertices[i].tangent[j] = tangent[j];

		//if(Vector::dotProduct(n,tangent) > Vector::threshold)
			//cerr << "[Initialization] Tangent calculation failed." << endl;

		// Create the Vertex
		Vertex* vertex = new Vertex(i);

		vertex->setPosition(Vector(bufferVertices[i].position));
		vertex->setNormal(Vector(bufferVertices[i].normal));
		vertex->setTangent(Vector(bufferVertices[i].tangent));
		vertex->setTextureCoordinates(Vector(bufferVertices[i].textureUV));

		mesh->addVertex(vertex);
	}

	// Create the Bounding Sphere
	BoundingSphere* boundingSphere = new BoundingSphere();
	// Initialize the Bounding Sphere
	boundingSphere->calculateMiniball(mesh);

	// Set the Bounding Sphere
	mesh->setBoundingSphere(boundingSphere);
	// Set the Name
	mesh->setName(currentMeshName);

	// Cleanup
	delete[] vertexArray;
	delete[] normalArray;
	delete[] textureCoordinateArray;

	delete[] sTangentArray;
	delete[] tTangentArray;
	
	delete[] bufferVertices;
	delete[] bufferVerticesID;
}

void OBJ_Reader::loadMaterial(string materialFilename, Material* material) {

	cout << "[Initialization] LoadMaterial(" << materialFilename << ");" << endl;

	// Load the Default Values
	material->setAmbient(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setDiffuse(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setSpecular(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setSpecularConstant(100.0f);

	// Open the Material File Stream
	ifstream materialFileStream(LOCATION + materialFilename, ifstream::in);

	// Line
	string currentMaterialLine;

	// Material
	string currentMaterialName = string("Uninitialized");

	while(true) {

		// Open the Line and exit if EOF
		if(!getline(materialFileStream, currentMaterialLine))
			break;

		// Create the Line Stream
		istringstream iss(currentMaterialLine);

		string materialLine0;
		iss >> materialLine0;

		// Check if there is a material
		if(materialLine0.compare("newmtl") == 0) {

			iss >> currentMaterialName;
		}
		
		if(this->materialName.compare(currentMaterialName) == 0 || this->materialName.compare("Uninitialized") == 0) {

			// Reading Ambient Component
			if(materialLine0.compare("Ka") == 0) {

				float ax, ay, az;
				iss >> ax >> ay >> az;

				material->setAmbient(Vector(ax, ay, az, 1.0f));
			}
			// Reading Diffuse Component
			else if(materialLine0.compare("Kd") == 0) {

				float dx, dy, dz;
				iss >> dx >> dy >> dz;

				material->setDiffuse(Vector(dx, dy, dz, 1.0f));
			}
			// Reading Specular Component
			else if(materialLine0.compare("Ks") == 0) {

				float sx, sy, sz;
				iss >> sx >> sy >> sz;

				material->setSpecular(Vector(sx, sy, sz, 1.0f));
			}
			// Reading Specular Constant
			else if(materialLine0.compare("Ns") == 0) {

				float specularConstnat;
				iss >> specularConstnat;

				material->setSpecularConstant(specularConstnat);
			}
		}
	}

	// Set the Name
	material->setName(this->materialName);

	// Close the Material File Stream
	materialFileStream.close();
}

bool OBJ_Reader::canReadMesh(string meshFilename) {

	if(this->meshFilename.compare(meshFilename) == 0)
		return !meshEndOfFile;

	return true;
}

bool OBJ_Reader::canReadMaterial(string materialFilename) {

	return true;
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