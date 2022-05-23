// MathLibrary.cpp
// compile with: cl /c /EHsc MathLibrary.cpp
// post-build command: lib MathLibrary.obj

#include "MathLibrary.h"
#include <string>
#include <CL/cl.hpp>
#include <CL/opencl.h>
#include <string>

namespace MathLibrary
{
    double Arithmetic::Add(double a, double b)
    {
        return a + b;
    }

    double Arithmetic::Subtract(double a, double b)
    {
        return a - b;
    }

    double Arithmetic::Multiply(double a, double b)
    {
        return a * b;
    }

    double Arithmetic::Divide(double a, double b)
    {
        return a / b;
    }

    std::string Arithmetic::Test() {
        std::vector<cl::Platform> platform;
        cl_int result = cl::Platform::get(&platform);
        if (result != CL_SUCCESS) {
            return "ERRORE";
        }
        std::vector<cl::Device> device;
        result = platform[0].getDevices(CL_DEVICE_TYPE_GPU, &device);
        for (int i = 0; i < device.size(); i++) {
            return "DEVICE NAME: " + device[i].getInfo<CL_DEVICE_NAME>();
        }
    }


}