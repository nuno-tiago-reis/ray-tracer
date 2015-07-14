#include "CubeTexture.h"

CubeTexture::CubeTexture(
	string name,
	string positiveXFileName, string negativeXFileName, string positiveYFileName,
	string negativeYFileName, string positiveZFileName,string negativeZFileName,
	string uniform) : Texture(name, GL_TEXTURE_CUBE_MAP, uniform) {

	this->textureFilenameMap[POSITIVE_X] = positiveXFileName;
	this->textureFilenameMap[NEGATIVE_X] = negativeXFileName;

	this->textureFilenameMap[POSITIVE_Y] = positiveYFileName;
	this->textureFilenameMap[NEGATIVE_Y] = negativeYFileName;

	this->textureFilenameMap[POSITIVE_Z] = positiveZFileName;
	this->textureFilenameMap[NEGATIVE_Z] = negativeZFileName;
}

CubeTexture::~CubeTexture() {
}

void CubeTexture::createTexture() {

	/* Load the Texture */
	this->handler = SOIL_load_OGL_cubemap(
		this->textureFilenameMap[POSITIVE_X].c_str(),this->textureFilenameMap[NEGATIVE_X].c_str(),
		this->textureFilenameMap[POSITIVE_Y].c_str(), this->textureFilenameMap[NEGATIVE_Y].c_str(),
		this->textureFilenameMap[POSITIVE_Z].c_str(), this->textureFilenameMap[NEGATIVE_Z].c_str(),
		SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS);

	/* Check for an error during the load process */
	if(this->handler == 0)
		cout << "SOIL loading error (\"" << 
		this->textureFilenameMap[POSITIVE_X].c_str() << this->textureFilenameMap[NEGATIVE_X].c_str() << 
		this->textureFilenameMap[POSITIVE_Y].c_str() << this->textureFilenameMap[NEGATIVE_Y].c_str() <<
		this->textureFilenameMap[POSITIVE_Z].c_str() << this->textureFilenameMap[NEGATIVE_Z].c_str() <<
		"\": " << SOIL_last_result() << endl;

	Utility::checkOpenGLError("ERROR: Texture CubeMap loading failed.");
}

void CubeTexture::bind(GLuint textureID) {

	glActiveTexture(textureID);

    glBindTexture(this->format, this->handler);

	glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->filename + "\" binding failed.");
}

void CubeTexture::unbind(GLuint textureID) {

	glBindTexture(this->format, 0);

	Utility::checkOpenGLError("ERROR: Texture \"" + this->filename + "\" unbinding failed.");
}