#ifndef LIGHT_H
#define	LIGHT_H

#include "Vector.h"

class Light {

	protected:

		/* Light Identifier (0-9) */
		GLint identifier;

		/* Light Attributes */
		Vector position;
		Vector color;

	public:

		/* Constructors & Destructors */
		Light(GLint identifier);
		~Light();

		/* Getters & Setters */

		GLint getIdentifier();

		Vector getPosition();
		Vector getColor();

		/* Setters */
		void setIdentifier(GLint identifier);

		void setPosition(Vector position);
		void setColor(Vector color);

		void dump();
};

#endif