#include "MatrixTests.hpp"

#include <utils/Matrix.hpp>

#include "PrintUtils.hpp"

#include <iostream>

namespace
{

using namespace test;

using utils::Dimension;
using utils::Matrix;

template<typename TDataType>
void
fillMatrix(Matrix<TDataType> &inMatrix)
{
    const Dimension &theMatrixDim = inMatrix.getDimension();
    Matrix<TDataType>::data_type *theData = inMatrix.getMutableData();
    for (int64_t w = 0; w < theMatrixDim.getW(); ++w) {
        for (int64_t z = 0; z < theMatrixDim.getZ(); ++z) {
            for (int64_t y = 0; y < theMatrixDim.getY(); ++y) {
                for (int64_t x = 0; x < theMatrixDim.getX(); ++x) {
                    theData[inMatrix.offset(x, y, z, w)] = static_cast<double>(w * 1000 + z * 100 + y * 10 + x);
                }
            }
        }
    }
}

bool
offset1()
{
    Dimension theOprandDim(5, 5);
    Matrix<double> theOprand(theOprandDim);
    Dimension thePosition(4, 0);
    int64_t theOffset = theOprand.offset(thePosition.getX(), thePosition.getY(), thePosition.getZ(), thePosition.getW());
    if (theOffset != 4) {
        std::cerr << "Matrix.offset1 test failed:"
                  << "oprand dimension: " << theOprandDim << std::endl
                  << "position: " << thePosition << std::endl
                  << "offset: " << theOffset << std::endl;
        return false;
    }
    return true;
}

bool
offset2()
{
    Dimension theOprandDim(5, 5);
    Matrix<double> theOprand(theOprandDim);
    Dimension thePosition(3, 2, 0);
    int64_t theOffset = theOprand.offset(thePosition.getX(), thePosition.getY(), thePosition.getZ(), thePosition.getW());
    if (theOffset != 13) {
        std::cerr << "Matrix.offset2 test failed:"
                  << "oprand dimension: " << theOprandDim << std::endl
                  << "position: " << thePosition << std::endl
                  << "offset: " << theOffset << std::endl;
        return false;
    }
    return true;
}

bool
offset3()
{
    Dimension theOprandDim(5, 10, 6);
    Matrix<double> theOprand(theOprandDim);
    Dimension thePosition(3, 2, 4, 0);
    int64_t theOffset = theOprand.offset(thePosition.getX(), thePosition.getY(), thePosition.getZ(), thePosition.getW());
    if (theOffset != 5 * 10 * 4 + 2 * 5 + 3) {
        std::cerr << "Matrix.offset3 test failed:"
                  << "oprand dimension: " << theOprandDim << std::endl
                  << "position: " << thePosition << std::endl
                  << "offset: " << theOffset << std::endl;
        return false;
    }
    return true;
}

bool
offset4()
{
    Dimension theOprandDim(5, 10, 6, 4);
    Matrix<double> theOprand(theOprandDim);
    Dimension thePosition(3, 2, 4, 2);
    int64_t theOffset = theOprand.offset(thePosition.getX(), thePosition.getY(), thePosition.getZ(), thePosition.getW());
    if (theOffset != 6 * 10 * 5 * 2 + 5 * 10 * 4 + 2 * 5 + 3) {
        std::cerr << "Matrix.offset4 test failed:"
                  << "oprand dimension: " << theOprandDim << std::endl
                  << "position: " << thePosition << std::endl
                  << "offset: " << theOffset << std::endl;
        return false;
    }
    return true;
}

bool
sub1()
{
    Dimension theOriginDim(10, 10);
    Matrix<double> theOrigin(theOriginDim);
    fillMatrix(theOrigin);
    const Dimension theStartDim(3, 3, 0);
    const Dimension theEndDim(7, 6, 0);
    Matrix<double> theSub = theOrigin.sub(theStartDim, theEndDim);
    const Matrix<double>::data_type *theSubData = theSub.getData();
    if (theSubData[theSub.offset(0,0)] != 33 ||
            theSubData[theSub.offset(3, 2)] != 56 ||
            theSubData[theSub.offset(2, 1)] != 45) {
        std::cerr << "Matrix.sub1 test failed. origin matrix: " << std::endl
                  << theOrigin << std::endl
                  << "sub matrix:" << std::endl
                  << theSub << std::endl;
        return false;
    }
    return true;
}

bool
reshape()
{
    Dimension theOriginDim(3, 5);
    Dimension theReshapedDim(5, 3);

    Matrix<double> theOrigin(theOriginDim);
    fillMatrix(theOrigin);
    Matrix<double> theReshaped(theOrigin);
    theReshaped.reshape(theReshapedDim);
    Matrix<double>::data_type *theReshapedData = theReshaped.getMutableData();
    if (theReshapedData[theReshaped.offset(0,0)] != 0 ||
            theReshapedData[theReshaped.offset(3, 2)] != 41 ||
            theReshapedData[theReshaped.offset(2, 1)] != 21) {
        std::cerr << "Matrix.reshape test failed. origin matrix: " << std::endl
                  << theOrigin << std::endl
                  << "reshaped matrix:" << std::endl
                  << theReshaped << std::endl;
        return false;
    }
    return true;
}
}

namespace test
{

void
MatrixTests::run()
{
    offset1();
    offset2();
    offset3();
    offset4();
    sub1();
    reshape();
}

}
