#ifndef MATRIX_STACK_H
#define MATRIX_STACK_H

#ifdef MEMORY_LEAK
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>
#endif

/* C++ Includes */
#include <list>

/* Math Library */
#include "Matrix.h"

using namespace std;

class MatrixStack {

	private:

		/* Singleton Instance */
		static MatrixStack *instance;

		/* Matrix Stack */
		list<Matrix> stack;

		/* Constructors & Destructors */
		MatrixStack();
		~MatrixStack();

	public:

		/* Singleton Methods */
		static MatrixStack* getInstance();
		static void destroyInstance();

		/* Basic Stack Operations */
		void push(Matrix matrix);

		Matrix pop();
		Matrix pop(int count);

		Matrix peek();

		void clean();

		/* Getters & Setters */
		int getMatrixCount();

		/* Debug */
		void dump();
};

#endif