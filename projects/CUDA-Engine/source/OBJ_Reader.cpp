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

	string line;

	// Reading the Model .obj - First pass
	int faceNumber = 0;
	int vertexNumber = 0;
	int normalNumber = 0;
	int textureCoordinateNumber = 0;

	// Open the Model File
	ifstream modelFile(LOCATION + meshFilename);

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;
		
		/* Add a Vertex */
		if(start == "v")
			vertexNumber++;
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

	// Close the Model File
	modelFile.close();

	// Reading the Model .obj - Second pass
	modelFile.open(LOCATION + meshFilename);

	// Storage Structures
	Coordinate3D *vertexArray = new Coordinate3D[vertexNumber];
	Coordinate3D *normalArray = new Coordinate3D[normalNumber];
	Coordinate2D *textureCoordinateArray = new Coordinate2D[textureCoordinateNumber];

	// Calculated after parsing
	Vector *sTangentArray = new Vector[vertexNumber];
	Vector *tTangentArray = new Vector[vertexNumber];

	// Final GPU-ready Structure
	VertexStructure *bufferVertices = new VertexStructure[faceNumber * 3];
	GLint *bufferVerticesID = new GLint[faceNumber * 3];

	// Bounding Box Variables
	float xMaximum = FLT_MIN, yMaximum = FLT_MIN, zMaximum = FLT_MIN;
	float xMinimum = FLT_MAX, yMinimum = FLT_MAX, zMinimum = FLT_MAX;

	// Index Trackers
	int currentFace = 0;
	int currentVertex = 0;
	int currentNormal = 0;
	int currentTextureCoordinate = 0;

	while(getline(modelFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;

		// Add a Vertex
		if(start == "v") {

			float x,y,z;
			iss >> x >> y >> z;

			vertexArray[currentVertex].x = x;
			vertexArray[currentVertex].y = y;
			vertexArray[currentVertex].z = z;

			if(x > xMaximum)
				xMaximum = x;
			if(y > yMaximum)
				yMaximum = y;
			if(z > zMaximum)
				zMaximum = z;

			if(x < xMinimum)
				xMinimum = x;
			if(y < yMinimum)
				yMinimum = y;
			if(z < zMinimum)
				zMinimum = z;

			currentVertex++;
		}
		// Add a Vertex Normal
		else if(start == "vn") {

			float x,y,z;
			iss >> x >> y >> z;

			normalArray[currentNormal].x = x;
			normalArray[currentNormal].y = y;
			normalArray[currentNormal].z = z;

			currentNormal++;
		} 
		// Add a Vertex Texture UV
		else if(start == "vt") {

			float u,v;
			iss >> u >> v;

			textureCoordinateArray[currentTextureCoordinate].u = u;
			textureCoordinateArray[currentTextureCoordinate].v = v;

			currentTextureCoordinate++;
		}
		// Add a Face (Triangle)
		else if(start == "f") {

			string faceVertex[3];
			iss >> faceVertex[0] >> faceVertex[1] >> faceVertex[2];

			for(int i=0; i<3; i++) {

				vector<int> index = split(faceVertex[i], '/');

				// Vertex ID
				bufferVerticesID[currentFace * 3 + i] = index[0]-1;

				// Vertex Position
				bufferVertices[currentFace * 3 + i].position[0] = vertexArray[index[0]-1].x;
				bufferVertices[currentFace * 3 + i].position[1] = vertexArray[index[0]-1].y;
				bufferVertices[currentFace * 3 + i].position[2] = vertexArray[index[0]-1].z;
				bufferVertices[currentFace * 3 + i].position[3] = 1.0f;
			
				// Vertex Texture Coordinates
				if(index.size() >= 2) {

					bufferVertices[currentFace * 3 + i].textureUV[0] = textureCoordinateArray[index[1]-1].u;
					bufferVertices[currentFace * 3 + i].textureUV[1] = textureCoordinateArray[index[1]-1].v;
				} 
				else {

					bufferVertices[currentFace * 3 + i].textureUV[0] = 0.0f;
					bufferVertices[currentFace * 3 + i].textureUV[1] = 0.0f;
				}

				// Vertex Normals
				if(index.size() >= 3) {

					bufferVertices[currentFace * 3 + i].normal[0] = normalArray[index[2]-1].x;
					bufferVertices[currentFace * 3 + i].normal[1] = normalArray[index[2]-1].y;
					bufferVertices[currentFace * 3 + i].normal[2] = normalArray[index[2]-1].z;
					bufferVertices[currentFace * 3 + i].normal[3] = 0.0f;
				}
				else {

					bufferVertices[currentFace * 3 + i].normal[0] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[1] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[2] = 0.0f;
					bufferVertices[currentFace * 3 + i].normal[3] = 0.0f;
				}
			}

			/* Create the Vertex-based Edges */
			Coordinate3D xyz1;
			xyz1.x = bufferVertices[currentFace * 3 + 1].position[0] - bufferVertices[currentFace * 3].position[0];
			xyz1.y = bufferVertices[currentFace * 3 + 1].position[1] - bufferVertices[currentFace * 3].position[1];
			xyz1.z = bufferVertices[currentFace * 3 + 1].position[2] - bufferVertices[currentFace * 3].position[2];

			Coordinate3D xyz2;
			xyz2.x = bufferVertices[currentFace * 3 + 2].position[0] - bufferVertices[currentFace * 3].position[0];
			xyz2.y = bufferVertices[currentFace * 3 + 2].position[1] - bufferVertices[currentFace * 3].position[1];
			xyz2.z = bufferVertices[currentFace * 3 + 2].position[2] - bufferVertices[currentFace * 3].position[2];

			/* Create the UV-based Edges */
			Coordinate2D uv1;
			uv1.u = bufferVertices[currentFace * 3 + 1].textureUV[0] - bufferVertices[currentFace * 3].textureUV[0];
			uv1.v = bufferVertices[currentFace * 3 + 1].textureUV[1] - bufferVertices[currentFace * 3].textureUV[1];

			Coordinate2D uv2;
			uv2.u = bufferVertices[currentFace * 3 + 2].textureUV[0] - bufferVertices[currentFace * 3].textureUV[0];
			uv2.v = bufferVertices[currentFace * 3 + 2].textureUV[1] - bufferVertices[currentFace * 3].textureUV[1];

			float r = 1.0f / (uv1.u * uv2.v - uv2.u * uv1.v);

			Vector s((uv2.v * xyz1.x - uv1.v * xyz2.x) * r, (uv2.v * xyz1.y - uv1.v * xyz2.y) * r,(uv2.v * xyz1.z - uv1.v * xyz2.z) * r, 0.0f);
			Vector t((uv1.u * xyz2.x - uv2.u * xyz1.x) * r, (uv1.u * xyz2.y - uv2.u * xyz1.y) * r,(uv1.u * xyz2.z - uv2.u * xyz1.z) * r, 0.0f);

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

		if(Vector::dotProduct(n,tangent) > Vector::threshold)
			cerr << "[Initialization] Tangent calculation failed." << endl;

		// Create the Vertex
		Vertex* vertex = new Vertex(i);

		vertex->setPosition(Vector(bufferVertices[i].position));
		vertex->setNormal(Vector(bufferVertices[i].normal));
		vertex->setTangent(Vector(bufferVertices[i].tangent));
		vertex->setTextureCoordinates(Vector(bufferVertices[i].textureUV));

		mesh->addVertex(vertex);
	}

	/* Create the Bounding Box */
	BoundingBox* boundingBox = new BoundingBox();

	boundingBox->setMaximum(Vector(xMaximum, yMaximum, zMaximum, 1.0f));
	boundingBox->setMinimum(Vector(xMinimum, yMinimum, zMinimum, 1.0f));
	
	mesh->setBoundingBox(boundingBox);

	/* Cleanup */
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

	string line;

	// Load the Default Values
	material->setAmbient(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setDiffuse(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setSpecular(Vector(0.75f, 0.75f, 0.75f, 1.0f));
	material->setSpecularConstant(100.0f);

	// Open the Material File
	ifstream materialFile(LOCATION + materialFilename);

	while(getline(materialFile, line)) {

		istringstream iss(line);

		string start;
		iss >> start;

		// Reading Ambient Component
		if(start == "Ka") {

			float x,y,z;
			iss >> x >> y >> z;

			material->setAmbient(Vector(x, y, z, 1.0f));
		}
		// Reading Diffuse Component
		else if(start == "Kd") {

			float x,y,z;
			iss >> x >> y >> z;

			material->setDiffuse(Vector(x, y, z, 1.0f));
		}
		// Reading Specular Component
		else if(start == "Ks") {

			float x,y,z;
			iss >> x >> y >> z;

			material->setSpecular(Vector(x, y, z, 1.0f));
		}
		// Reading Specular Constant
		else if(start == "Ns") {

			float s;
			iss >> s;

			material->setSpecularConstant(s);
		}
	}

	// Close the Material File
	materialFile.close();
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