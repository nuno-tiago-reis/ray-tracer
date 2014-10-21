#ifndef OBJECT_H
#define	OBJECT_H

#include <iostream>

#include "Vector.h"

#include "BoundingBox.h"

class Object {

	protected:

		GLint identifier;

		/* Object Bounding Box */
		BoundingBox *boundingBox;

		/* Object Attributes */
		Vector color;
				
		GLfloat diffuseIntensity;
		GLfloat specularIntensity;

		GLfloat shininess;

		GLfloat transmittance;

		GLfloat refractionIndex;

	public:

		/* Constructors & Destructors */
		Object(int identifier);
		~Object();

		virtual bool rayIntersection(Vector rayOrigin, Vector rayDirection, Vector *pointHit, Vector *normalHit) = 0;

		virtual void createBoundingBox() = 0;

		/* Getters & Setters */
		GLint getIdentifier();

		BoundingBox* getBoundingBox();

		Vector getColor();

		GLfloat getDiffuseIntensity();
		GLfloat getSpecularIntensity();
		GLfloat getShininess();

		GLfloat getTransmittance();
		GLfloat getRefractionIndex();

		/* Setters */
		void setIdentifier(GLint identifier);

		void setBoundingBox(BoundingBox *boundingBox);

		void setColor(Vector color);

		void setDiffuseIntensity(GLfloat diffuseIntensity);
		void setSpecularIntensity(GLfloat specularIntensity);
		void setShininess(GLfloat shininess);

		void setTransmittance(GLfloat transmittance);
		void setRefractionIndex(GLfloat refractionIndex);

		virtual void dump();
};

#endif