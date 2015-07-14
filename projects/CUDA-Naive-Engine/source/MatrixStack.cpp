#include "MatrixStack.h"

MatrixStack* MatrixStack::instance = NULL;

MatrixStack::MatrixStack() {
}

MatrixStack::~MatrixStack() {

	this->stack.clear();
}

MatrixStack* MatrixStack::getInstance() {

	if(instance == NULL)
		instance = new MatrixStack();

	return instance;
}

void MatrixStack::destroyInstance() {

	delete instance;

	instance = NULL;
}

void MatrixStack::push(Matrix matrix) {

	this->stack.push_back(matrix);
}

Matrix MatrixStack::pop() {

	if(this->stack.empty() == true)
		return Matrix();

	Matrix matrix = this->stack.back();

	this->stack.pop_back();

	return matrix;
}

Matrix MatrixStack::pop(int count) {

	Matrix matrix;

	while(count-- > 0 && this->stack.empty() == false) {

		matrix = this->stack.back();

		 this->stack.pop_back();
	}

	return matrix;
}

Matrix MatrixStack::peek() {

	if(this->stack.empty() == true)
		return Matrix();

	return this->stack.back();
}

void MatrixStack::clean() {

	while(this->stack.empty() == false)
		this->stack.pop_back();
}

int MatrixStack::getMatrixCount() {

	return this->stack.size();
}

void MatrixStack::dump() {

	cout << "<MatrixStack Dump>" << endl;

	/* MatrixStack Size */
	cout << "<MatrixStack Size> = " << this->stack.size() << " ;" << endl;

	/* Matrix Stack */
	cout << "<MatrixStack Matrix Stack> = " << endl;
	for(list<Matrix>::iterator matrixIterator = this->stack.begin(); matrixIterator != this->stack.end(); matrixIterator++)
		matrixIterator->dump();
}