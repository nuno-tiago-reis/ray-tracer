#include "XML_Reader.h"

XML_Reader* XML_Reader::instance = NULL;

XML_Reader::XML_Reader() {

	_transformationDocument = new XMLDocument();
}

XML_Reader::~XML_Reader() {

	_transformationDocument->Clear();
}

XML_Reader* XML_Reader::getInstance() {

	if(instance == NULL)
		instance = new XML_Reader();

	return instance;
}

void XML_Reader::destroyInstance() {

	delete instance;

	instance = NULL;
}

void XML_Reader::openTransformFile(string filename) {

	_transformationDocument->LoadFile(filename.c_str());

	if(_transformationDocument->ErrorID() != XML_NO_ERROR) {
	
		cout << "XML File " << filename.c_str() << " parsing error." << endl << endl;

		cout << _transformationDocument->GetErrorStr1() << endl;
		cout << _transformationDocument->GetErrorStr2() << endl;
	}
}

void XML_Reader::loadTransform(Transform* transform) {

	cout << "XML_Reader::loadTransform(" << transform->getName() << ");" << endl;

	for(XMLNode* modelXML = _transformationDocument->FirstChildElement()->FirstChildElement("model"); modelXML != NULL; modelXML = modelXML->NextSibling()) {

		if(transform->getName().compare(modelXML->ToElement()->Attribute("identifier")) != 0)
			continue;

		/*XMLNode* boundingVolumeXML = modelXML->FirstChildElement("boundingVolume");

		string boundingVolume(boundingVolumeXML->ToElement()->Attribute("type"));
		
		if(boundingVolume.compare(BOUNDING_BOX) == 0) {

			GLfloat height = boundingVolumeXML->ToElement()->FloatAttribute("height");
			GLfloat width = boundingVolumeXML->ToElement()->FloatAttribute("width");
			GLfloat depth = boundingVolumeXML->ToElement()->FloatAttribute("depth");
		
			BoundingBox* boundingBox = new BoundingBox(height,width,depth);
			graphicObject->setBoundingVolume(boundingBox);
		}
		else if(boundingVolume.compare(BOUNDING_SPHERE) == 0) {

			GLfloat radius = boundingVolumeXML->ToElement()->FloatAttribute("radius");
		
			BoundingSphere* boundingSphere = new BoundingSphere(radius);
			graphicObject->setBoundingVolume(boundingSphere);
		}
		else if(boundingVolume.compare("None") == 0) {
		}*/
		
		XMLNode* initialTransformationXML = modelXML->FirstChildElement("initial");

		Vector position;

		position[0] = initialTransformationXML->ToElement()->FloatAttribute("px");
		position[1] = initialTransformationXML->ToElement()->FloatAttribute("py");
		position[2] = initialTransformationXML->ToElement()->FloatAttribute("pz");

		transform->setPosition(position);

		Vector scale;
		
		scale[0] = initialTransformationXML->ToElement()->FloatAttribute("sx");
		scale[1] = initialTransformationXML->ToElement()->FloatAttribute("sy");
		scale[2] = initialTransformationXML->ToElement()->FloatAttribute("sz");

		transform->setScale(scale);

		Vector rotation;

		rotation[0] = initialTransformationXML->ToElement()->FloatAttribute("rx");
		rotation[1] = initialTransformationXML->ToElement()->FloatAttribute("ry");
		rotation[2] = initialTransformationXML->ToElement()->FloatAttribute("rz");

		transform->setRotation(rotation);

		break;
	}
}

void XML_Reader::saveTransform(Transform* transform) {

	XMLDocument* document = new XMLDocument();

	document->LoadFile(TRANSFORMATION_OUTPUT_FILE);

	XMLElement* model = document->NewElement("model");

	for(XMLNode* modelIterator = document->FirstChildElement()->FirstChildElement("model"); modelIterator != NULL; modelIterator = modelIterator->NextSibling())
		if(transform->getName().compare(modelIterator->ToElement()->Attribute("identifier")) == 0) {

			document->FirstChildElement("geometry")->DeleteChild(modelIterator); 
			break;
		}

	model->SetAttribute("identifier",transform->getName().c_str());

		XMLElement* initial = document->NewElement("initial");

		Vector position = transform->getPosition();

		initial->SetAttribute("px",position[0]);
		initial->SetAttribute("py",position[1]);
		initial->SetAttribute("pz",position[2]);

		Vector scale = transform->getScale();

		initial->SetAttribute("sx",scale[0]);
		initial->SetAttribute("sy",scale[1]);
		initial->SetAttribute("sz",scale[2]);

		Vector rotation = transform->getRotation();

		initial->SetAttribute("rx",rotation[0]);
		initial->SetAttribute("ry",rotation[1]);
		initial->SetAttribute("rz",rotation[2]);

	model->InsertFirstChild(initial);

	document->FirstChildElement("geometry")->InsertEndChild(model);
	document->SaveFile(TRANSFORMATION_OUTPUT_FILE);
}