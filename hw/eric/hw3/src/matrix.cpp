#include "matrix.h"


Matrix::Matrix(unsigned rows, unsigned cols)
{
  m = rows;
  n = cols;
  values = new int[rows*cols];
}



Matrix::Matrix(unsigned rows, unsigned cols, int* vals)
{
  m = rows;
  n = cols;
  values = vals;
}



Matrix::Matrix(const Matrix& mat)
{
  n = mat.n;
  m = mat.m;
  values = new int[m*n];
  std::copy(mat.values, mat.values + m*n, values);
}



// no error checking in here. Assumes file contains exactly what it says it does.
Matrix::Matrix(string filename)
{
  readFromFile(filename);
}



void Matrix::setValueBuffer(int* newValues)
{
  delete[] values;
  values = newValues;
}



void Matrix::readFromFile(string filename, bool vector)
{

  // open file
  ifstream infile(filename.c_str());

  // if vector, one pass through to get size
  if(vector)
  {
    m = 0;
    n = 1;
    int tmp;
    while( (infile >> tmp) )
      m++;
    infile.clear();
    infile.seekg(0);
  }
  else  // if NOT a vector, just read size of matrix
    infile >> m >> n;

  // read data
  values = new int[m*n];
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      infile >> values[j + n*i];

  // close file
  infile.close();

}



void Matrix::fill(int value)
{
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      values[j+i*n] = value;
}



Matrix Matrix::operator*(Matrix& B)
{
  Matrix A = *(this);

  if(A.n!=B.m)
    throw std::invalid_argument("Matrix dimension mismatch.");

  Matrix C(A.m,B.n);

  for(int irow=0; irow<A.m; irow++)
    for(int jcol=0; jcol<B.n; jcol++)
    {
      C(irow,jcol) = 0;
      for(int k=0; k<A.n; k++)
        C(irow,jcol) += A(irow,k)*B(k,jcol);
    }

    return C;
}



ColVector Matrix::operator*(ColVector& B)
{
  Matrix A = *(this);

  if(A.n!=B.m)
    throw std::invalid_argument("Matrix dimension mismatch.");

  ColVector C(A.m);

  for(int irow=0; irow<A.m; irow++)
    {
      C(irow) = 0;
      for(int k=0; k<A.n; k++)
        C(irow) += A(irow,k)*B(k);
    }

    return C;
}



void Matrix::print()
{
  for(int i=0; i<m; i++)
  {
    cout << "[";
    for(int j=0; j<n; j++)
      cout << values[j+n*i] << ", ";
    cout << "\b\b]\n";
  }
}



void Matrix::writeToFile(string filename)
{
  ofstream outFile;
  outFile.open(filename.c_str());
  bool isVector = (m==1 || n==1);
  if(!isVector)  // if not a vector
    outFile << m << " " << n;

  for(int irow=0; irow<m; irow++)
  {
      outFile << ( isVector ? " " : "\n" );
    for(int jcol=0; jcol<n; jcol++)
      outFile << values[jcol + n*irow] << " ";
  }

  outFile.close();
}



Matrix::~Matrix()
{
  delete[] values;
}



ColVector::ColVector(int n) : Matrix(n,1)
{}



ColVector::ColVector(int n, int* vals) : Matrix(n,1,vals)
{}



ColVector::ColVector(string filename) : Matrix(0,0)
{
  readFromFile(filename, true);
}
