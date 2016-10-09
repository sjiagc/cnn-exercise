#include "DimensionTests.hpp"

#include <utils/Dimension.hpp>

#include "PrintUtils.hpp"

#include <iostream>

namespace
{

using namespace test;
using utils::Dimension;

bool
axisCount1()
{
    const Dimension theOprand(2, 2, 2, 2);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 4) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
axisCount2()
{
    const Dimension theOprand(2, 2, 2, 1);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 3) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
axisCount3()
{
    const Dimension theOprand(2, 2, 1, 1);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 2) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
axisCount4()
{
    const Dimension theOprand(2, 1, 1, 1);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 1) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
axisCount5()
{
    const Dimension theOprand(1, 1, 1, 1);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 1) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
axisCount6()
{
    const Dimension theOprand(2, 1, 0, 1);
    int64_t theResult = theOprand.axisCount();
    if (theResult != 1) {
        std::cerr << "axisCount test failed:" << std::endl
                  << "oprand: " << theOprand << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
addition()
{
    const Dimension theOprand1(1, -2, 8, 7);
    const Dimension theOprand2(1, 2, 3, -11);
    const Dimension theResult = theOprand1 + theOprand2;
    if (theResult.getX() != 2 || theResult.getY() != 0 ||
            theResult.getZ() != 11 || theResult.getW() != -4) {
        std::cerr << "Addition test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
substraction()
{
    const Dimension theOprand1(10, 9, 8, 7);
    const Dimension theOprand2(1, 2, 3, 11);
    const Dimension theResult = theOprand1 - theOprand2;
    if (theResult.getX() != 9 || theResult.getY() != 7 ||
            theResult.getZ() != 5 || theResult.getW() != -4) {
        std::cerr << "Substraction test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareEqual1()
{
    const Dimension theOprand1(1, 2, 1, 4);
    const Dimension theOprand2(1, 2, 1, 4);
    bool theResult = theOprand1 == theOprand2;
    if (!theResult) {
        std::cerr << "Equal test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareEqual2()
{
    const Dimension theOprand1(1, 2, 1, 4);
    const Dimension theOprand2(1, 1, 1, 4);
    bool theResult = theOprand1 == theOprand2;
    if (theResult) {
        std::cerr << "Equal test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLE1()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(2, 2, 3, 4);
    bool theResult = theOprand1 <= theOprand2;
    if (!theResult) {
        std::cerr << "LE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLE2()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(2, 2, 1, 4);
    bool theResult = theOprand1 <= theOprand2;
    if (theResult) {
        std::cerr << "LE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLE3()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 <= theOprand2;
    if (!theResult) {
        std::cerr << "LE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGE1()
{
    const Dimension theOprand1(2, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 >= theOprand2;
    if (!theResult) {
        std::cerr << "GE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGE2()
{
    const Dimension theOprand1(2, 2, 1, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 >= theOprand2;
    if (theResult) {
        std::cerr << "GE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGE3()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 <= theOprand2;
    if (!theResult) {
        std::cerr << "GE test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLT1()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(2, 2, 3, 4);
    bool theResult = theOprand1 < theOprand2;
    if (!theResult) {
        std::cerr << "LT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLT2()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(2, 2, 1, 4);
    bool theResult = theOprand1 < theOprand2;
    if (theResult) {
        std::cerr << "LT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareLT3()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 < theOprand2;
    if (theResult) {
        std::cerr << "LT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGT1()
{
    const Dimension theOprand1(2, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 > theOprand2;
    if (!theResult) {
        std::cerr << "GT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGT2()
{
    const Dimension theOprand1(2, 2, 1, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 > theOprand2;
    if (theResult) {
        std::cerr << "GT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

bool
compareGT3()
{
    const Dimension theOprand1(1, 2, 3, 4);
    const Dimension theOprand2(1, 2, 3, 4);
    bool theResult = theOprand1 > theOprand2;
    if (theResult) {
        std::cerr << "GT test failed:" << std::endl
                  << "oprand 1: " << theOprand1 << std::endl
                  << "oprand 2: " << theOprand2 << std::endl
                  << "result: " << theResult << std::endl;
        return false;
    }
    return true;
}

}

namespace test
{

void
DimensionTests::run()
{
    axisCount1();
    axisCount2();
    axisCount3();
    axisCount4();
    axisCount5();
    axisCount6();

    compareEqual1();
    compareEqual2();
    compareLE1();
    compareLE2();
    compareLE3();
    compareGE1();
    compareGE2();
    compareGE3();
    compareLT1();
    compareLT2();
    compareLT3();
    compareGT1();
    compareGT2();
    compareGT3();

    addition();
    substraction();
}

}
