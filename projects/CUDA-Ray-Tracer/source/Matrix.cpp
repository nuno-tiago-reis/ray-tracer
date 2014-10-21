#include "Matrix.h"

const GLfloat Matrix::threshold = (GLfloat)1.0e-5;

Matrix::Matrix() {

	loadIdentity();
}

Matrix::Matrix(Quaternion quaternion) {
	
	quaternion.normalize();

	GLfloat xt = quaternion[QX] * quaternion[QT];
	GLfloat xx = quaternion[QX] * quaternion[QX];
	GLfloat xy = quaternion[QX] * quaternion[QY];
	GLfloat xz = quaternion[QX] * quaternion[QZ];
	
	GLfloat yt = quaternion[QY] * quaternion[QT];
	GLfloat yy = quaternion[QY] * quaternion[QY];
	GLfloat yz = quaternion[QY] * quaternion[QZ];

	GLfloat zt = quaternion[QZ] * quaternion[QT];
	GLfloat zz = quaternion[QZ] * quaternion[QZ];

	this->matrix[0][0] = 1.0f - 2.0f * (yy + zz);
	this->matrix[0][1] = 2.0f * (xy + zt);
	this->matrix[0][2] = 2.0f * (xz - yt);
	this->matrix[0][3] = 0.0f;

	this->matrix[1][0] = 2.0f * (xy - zt);
	this->matrix[1][1] = 1.0f - 2.0f * (xx + zz);
	this->matrix[1][2] = 2.0f * (yz + xt);
	this->matrix[1][3] = 0.0f;

	this->matrix[2][0] = 2.0f * (xz + yt);
	this->matrix[2][1] = 2.0f * (yz - xt);
	this->matrix[2][2] = 1.0f - 2.0f * (xx + yy);
	this->matrix[2][3] = 0.0f;

	this->matrix[3][0] = 0.0f;
	this->matrix[3][1] = 0.0f;
	this->matrix[3][2] = 0.0f;
	this->matrix[3][3] = 1.0f;

	clean();
}

Matrix::Matrix(GLfloat initialValue) {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			this->matrix[i][j] = initialValue;
}

Matrix::Matrix(const GLfloat initialValue[16]) {

	for(int i=0; i<16; i++)
		this->matrix[i/4][i%4] = initialValue[i];
}


Matrix::Matrix(const GLfloat initialValue[4][4]) {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			this->matrix[i][j] = initialValue[i][j];
}

Matrix::~Matrix() {
}

void Matrix::loadIdentity() {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			if(i==j)
				this->matrix[i][j] = 1;
			else 
				this->matrix[i][j] = 0;
}

void Matrix::clean() {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			if(fabs(this->matrix[i][j]) < threshold) 
				this->matrix[i][j] = 0.0f;
}

void Matrix::scale(Vector scaleVector) {

	scale(scaleVector[VX], scaleVector[VY], scaleVector[VZ]);
}

void Matrix::scale(GLfloat xScale, GLfloat yScale, GLfloat zScale) {

	Matrix scaleMatrix;

	scaleMatrix[S_X] = xScale;
	scaleMatrix[S_Y] = yScale;
	scaleMatrix[S_Z] = zScale;

	(*this) *= scaleMatrix;
}

void Matrix::translate(Vector translationVector) {

	translate(translationVector[VX], translationVector[VY], translationVector[VZ]);
}

void Matrix::translate(GLfloat xCoordinate, GLfloat yCoordinate, GLfloat zCoordinate) {

	Matrix translationMatrix;

	translationMatrix[T_X] = xCoordinate;
	translationMatrix[T_Y] = yCoordinate;
	translationMatrix[T_Z] = zCoordinate;

	(*this) *= translationMatrix;
}

void Matrix::rotate(GLfloat angle, GLfloat xRotation, GLfloat yRotation, GLfloat zRotation) {

	Matrix rotationMatrix;
	Matrix xRotationMatrix;
	Matrix yRotationMatrix;
	Matrix zRotationMatrix;

	GLfloat cos_angle = cos(angle * (GLfloat)DEGREES_TO_RADIANS);
	GLfloat sin_angle = sin(angle * (GLfloat)DEGREES_TO_RADIANS);

	if(xRotation > 0) {

		xRotationMatrix[R1_X] = cos_angle;
		xRotationMatrix[R2_X] =-sin_angle;
		xRotationMatrix[R3_X] = sin_angle;
		xRotationMatrix[R4_X] = cos_angle;

		rotationMatrix *= xRotationMatrix;
	}

	if(yRotation > 0) {

		yRotationMatrix[R1_Y] = cos_angle;
		yRotationMatrix[R2_Y] = sin_angle;
		yRotationMatrix[R3_Y] =-sin_angle;
		yRotationMatrix[R4_Y] = cos_angle;

		rotationMatrix *= yRotationMatrix;
	}

	if(zRotation > 0) {

		zRotationMatrix[R1_Z] = cos_angle;
		zRotationMatrix[R2_Z] =-sin_angle;
		zRotationMatrix[R3_Z] = sin_angle;
		zRotationMatrix[R4_Z] = cos_angle;

		rotationMatrix *= zRotationMatrix;
	}

	rotationMatrix.transpose();

	(*this)*=rotationMatrix;
}

void Matrix::quaternionRotate(Quaternion quaternion) {

	Matrix rotationMatrix(quaternion);

	rotationMatrix.transpose();

	(*this) *= rotationMatrix;
}

void Matrix::transpose() {

	Matrix temporary;

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			temporary.setValue(i,j,this->matrix[j][i]);

	swap(*this,temporary);
}

void Matrix::invert() {

	GLfloat determinant;
	GLfloat inverseMatrix[16];

	inverseMatrix[0] =	(*this)[5]  * (*this)[10] * (*this)[15] - 
						(*this)[5]  * (*this)[11] * (*this)[14] - 
						(*this)[9]  * (*this)[6]  * (*this)[15] + 
						(*this)[9]  * (*this)[7]  * (*this)[14] +
						(*this)[13] * (*this)[6]  * (*this)[11] - 
						(*this)[13] * (*this)[7]  * (*this)[10];

	inverseMatrix[4] = -(*this)[4]  * (*this)[10] * (*this)[15] + 
						(*this)[4]  * (*this)[11] * (*this)[14] + 
						(*this)[8]  * (*this)[6]  * (*this)[15] - 
						(*this)[8]  * (*this)[7]  * (*this)[14] - 
						(*this)[12] * (*this)[6]  * (*this)[11] + 
						(*this)[12] * (*this)[7]  * (*this)[10];

	inverseMatrix[8] =	(*this)[4]  * (*this)[9] * (*this)[15] - 
						(*this)[4]  * (*this)[11] * (*this)[13] - 
						(*this)[8]  * (*this)[5] * (*this)[15] + 
						(*this)[8]  * (*this)[7] * (*this)[13] + 
						(*this)[12] * (*this)[5] * (*this)[11] - 
						(*this)[12] * (*this)[7] * (*this)[9];

	inverseMatrix[12] = -(*this)[4]  * (*this)[9] * (*this)[14] + 
						 (*this)[4]  * (*this)[10] * (*this)[13] +
						 (*this)[8]  * (*this)[5] * (*this)[14] - 
						 (*this)[8]  * (*this)[6] * (*this)[13] - 
						 (*this)[12] * (*this)[5] * (*this)[10] + 
						 (*this)[12] * (*this)[6] * (*this)[9];

	inverseMatrix[1] =  -(*this)[1]  * (*this)[10] * (*this)[15] + 
						 (*this)[1]  * (*this)[11] * (*this)[14] + 
						 (*this)[9]  * (*this)[2] * (*this)[15] - 
						 (*this)[9]  * (*this)[3] * (*this)[14] - 
						 (*this)[13] * (*this)[2] * (*this)[11] + 
						 (*this)[13] * (*this)[3] * (*this)[10];

	inverseMatrix[5] =	(*this)[0]  * (*this)[10] * (*this)[15] - 
						(*this)[0]  * (*this)[11] * (*this)[14] - 
						(*this)[8]  * (*this)[2] * (*this)[15] + 
						(*this)[8]  * (*this)[3] * (*this)[14] + 
						(*this)[12] * (*this)[2] * (*this)[11] - 
						(*this)[12] * (*this)[3] * (*this)[10];

	inverseMatrix[9] = -(*this)[0]  * (*this)[9] * (*this)[15] + 
						(*this)[0]  * (*this)[11] * (*this)[13] + 
						(*this)[8]  * (*this)[1] * (*this)[15] - 
						(*this)[8]  * (*this)[3] * (*this)[13] - 
						(*this)[12] * (*this)[1] * (*this)[11] + 
						(*this)[12] * (*this)[3] * (*this)[9];

	inverseMatrix[13]=	(*this)[0]  * (*this)[9] * (*this)[14] - 
						(*this)[0]  * (*this)[10] * (*this)[13] - 
						(*this)[8]  * (*this)[1] * (*this)[14] + 
						(*this)[8]  * (*this)[2] * (*this)[13] + 
						(*this)[12] * (*this)[1] * (*this)[10] - 
						(*this)[12] * (*this)[2] * (*this)[9];

	inverseMatrix[2] =	(*this)[1]  * (*this)[6] * (*this)[15] - 
						(*this)[1]  * (*this)[7] * (*this)[14] - 
						(*this)[5]  * (*this)[2] * (*this)[15] + 
						(*this)[5]  * (*this)[3] * (*this)[14] + 
						(*this)[13] * (*this)[2] * (*this)[7] - 
						(*this)[13] * (*this)[3] * (*this)[6];

	inverseMatrix[6] = -(*this)[0]  * (*this)[6] * (*this)[15] + 
						(*this)[0]  * (*this)[7] * (*this)[14] + 
						(*this)[4]  * (*this)[2] * (*this)[15] - 
						(*this)[4]  * (*this)[3] * (*this)[14] - 
						(*this)[12] * (*this)[2] * (*this)[7] + 
						(*this)[12] * (*this)[3] * (*this)[6];

	inverseMatrix[10] =	(*this)[0]  * (*this)[5] * (*this)[15] - 
						(*this)[0]  * (*this)[7] * (*this)[13] - 
						(*this)[4]  * (*this)[1] * (*this)[15] + 
						(*this)[4]  * (*this)[3] * (*this)[13] + 
						(*this)[12] * (*this)[1] * (*this)[7] - 
						(*this)[12] * (*this)[3] * (*this)[5];

	inverseMatrix[14] =-(*this)[0]  * (*this)[5] * (*this)[14] + 
						(*this)[0]  * (*this)[6] * (*this)[13] + 
						(*this)[4]  * (*this)[1] * (*this)[14] - 
						(*this)[4]  * (*this)[2] * (*this)[13] - 
						(*this)[12] * (*this)[1] * (*this)[6] + 
						(*this)[12] * (*this)[2] * (*this)[5];

	inverseMatrix[3] = -(*this)[1] * (*this)[6] * (*this)[11] + 
						(*this)[1] * (*this)[7] * (*this)[10] + 
						(*this)[5] * (*this)[2] * (*this)[11] - 
						(*this)[5] * (*this)[3] * (*this)[10] - 
						(*this)[9] * (*this)[2] * (*this)[7] + 
						(*this)[9] * (*this)[3] * (*this)[6];

	inverseMatrix[7] =	(*this)[0] * (*this)[6] * (*this)[11] - 
						(*this)[0] * (*this)[7] * (*this)[10] - 
						(*this)[4] * (*this)[2] * (*this)[11] + 
						(*this)[4] * (*this)[3] * (*this)[10] + 
						(*this)[8] * (*this)[2] * (*this)[7] - 
						(*this)[8] * (*this)[3] * (*this)[6];

	inverseMatrix[11] =-(*this)[0] * (*this)[5] * (*this)[11] + 
						(*this)[0] * (*this)[7] * (*this)[9] + 
						(*this)[4] * (*this)[1] * (*this)[11] - 
						(*this)[4] * (*this)[3] * (*this)[9] - 
						(*this)[8] * (*this)[1] * (*this)[7] + 
						(*this)[8] * (*this)[3] * (*this)[5];

	inverseMatrix[15] = (*this)[0] * (*this)[5] * (*this)[10] - 
						(*this)[0] * (*this)[6] * (*this)[9] - 
						(*this)[4] * (*this)[1] * (*this)[10] + 
						(*this)[4] * (*this)[2] * (*this)[9] + 
						(*this)[8] * (*this)[1] * (*this)[6] - 
						(*this)[8] * (*this)[2] * (*this)[5];

    determinant = (*this)[0] * inverseMatrix[0] + (*this)[1] * inverseMatrix[4] + (*this)[2] * inverseMatrix[8] + (*this)[3] * inverseMatrix[12];

    if(determinant == 0)
		return;

    determinant = 1.0f/determinant;

    for(int i=0; i<16; i++)
        inverseMatrix[i] *= determinant;

	setValue(inverseMatrix);
}

void Matrix::setView(Vector eye, Vector target, Vector userUp) {

	Matrix view;
    
	Vector lookAt = target;
	lookAt -= eye;
	lookAt.normalize();

	Vector right = Vector::crossProduct(lookAt,userUp);

	right.normalize();

	Vector up = Vector::crossProduct(right,lookAt);

	eye.negate();
	lookAt.negate();

	for(int i=0; i<3; i++) {

		view.setValue(0,i, right[i]);
		view.setValue(1,i, up[i]);
		view.setValue(2,i, lookAt[i]);
	}

	view.setValue(0,3,Vector::dotProduct(right,eye));
	view.setValue(1,3,Vector::dotProduct(up,eye));
	view.setValue(2,3,Vector::dotProduct(lookAt,eye));
	
	(*this) *= view;
}

void Matrix::setOrthogonalProjection(GLfloat left, GLfloat right, GLfloat top, GLfloat bottom, GLfloat nearZ, GLfloat farZ) {

	Matrix orthogonalProjection;

	orthogonalProjection.setValue(0,0, 2.0f/(right-left));
	orthogonalProjection.setValue(1,1, 2.0f/(top-bottom));
	orthogonalProjection.setValue(2,2,-2.0f/(farZ-nearZ));

	orthogonalProjection.setValue(0,3, -(right+left)/(right-left));
	orthogonalProjection.setValue(1,3, -(top+bottom)/(top-bottom));
	orthogonalProjection.setValue(2,3, -(farZ+nearZ)/(farZ-nearZ));

	(*this) *= orthogonalProjection;
}

void Matrix::setPerspectiveProjection(GLfloat fieldOfView, GLfloat aspectRatio, GLfloat nearZ, GLfloat farZ) {

	Matrix perspectiveProjection;

	GLfloat fieldOfViewRad = (GLfloat)(fieldOfView*DEGREES_TO_RADIANS);
	GLfloat cotangent = 1/tan(fieldOfViewRad/2.0f);

	perspectiveProjection.setValue(0,0, cotangent/aspectRatio);
	perspectiveProjection.setValue(1,1, cotangent);
	perspectiveProjection.setValue(2,2, (farZ+nearZ)/(nearZ-farZ));
	perspectiveProjection.setValue(3,3, 0);

	perspectiveProjection.setValue(2,3, -1);
	perspectiveProjection.setValue(3,2, (2*farZ*nearZ)/(nearZ-farZ));

	(*this) *= perspectiveProjection;
}

void Matrix::getValue(GLfloat* matrix) {
	
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			matrix[i*4+j] = this->matrix[i][j];
}

GLfloat Matrix::getValue(int row, int column) {

	return this->matrix[row][column];
}

void Matrix::setValue(const GLfloat value[16]) {

	for(int i=0; i<16; i++)
		this->matrix[i/4][i%4] = value[i];
}

void Matrix::setValue(int row, int column, GLfloat value) {

	this->matrix[row][column] = value;
}

GLfloat& Matrix::operator [] (int position) {

	return this->matrix[position/4][position%4];
}

Matrix Matrix::operator + (Matrix matrix){

	Matrix result;

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			result.setValue(i,j, getValue(i,j) + matrix.getValue(i,j));

	result.clean();

	return result;
}

Matrix Matrix::operator += (Matrix matrix) {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			setValue(i,j, getValue(i,j) + matrix.getValue(i,j));

	clean();

	return (*this);
}

Matrix Matrix::operator * (Matrix matrix) {
	
	Matrix multiplication;

	GLfloat accumulator = 0.0f;

	for(int i=0; i<16; i++) {

		accumulator = 0.0f;

		for(int j=0; j<4; j++)
			accumulator += getValue(i/4,j) * matrix.getValue(j,i%4);

		multiplication[i] = accumulator;
	}

	return multiplication;
}

Matrix Matrix::operator *= (Matrix matrix){

	Matrix multiplication;

	GLfloat accumulator = 0.0f;

	for(int i=0; i<16; i++) {

		accumulator = 0.0f;

		for(int j=0; j<4; j++)
			accumulator += getValue(i/4,j) * matrix.getValue(j,i%4);

		multiplication[i] = accumulator;
	}

	swap(*this,multiplication);

	return (*this);
}

Vector Matrix::operator * (Vector vector) {

	Vector temporary = vector;

	GLfloat accumulator = 0.0f;

	for(int i=0; i<4; i++) {

		accumulator = 0.0f;

		for(int j=0; j<4; j++)
			accumulator += getValue(i,j) * temporary[j];

		vector[i] = accumulator;
	}

	return vector;
}

bool Matrix::operator == (Matrix matrix) {

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			if(!(fabs(this->matrix[i][j] - matrix.getValue(i,j)) < threshold))
				return false;

	return true;
}

void Matrix::dump() {

	cout.precision(5);

	cout << "<Matrix Dump>" << endl << endl;

	cout << "   ";

	for(int j=0; j<4; j++)
		cout << "  " << j << " ";

	cout << endl;

	for(int i=0; i<4;i++) {

		cout << " " << i << " ";

		for(int j=0; j<4; j++)
			cout << " [" << this->matrix[i][j] << "]";

		cout << endl;
	}

	cout << endl;
}