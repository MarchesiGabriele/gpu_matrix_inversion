# GPU Matrix Inversion with OpenCL

## Visual Studio Library Project Creation 
- The VS project/solution has been created using this guide. This allows to build the VS solution and get a .lib file.
- Then to be able to use OpenCL we imported the OpenCL library and Headers. You can find them in the OpenCL folder. 
Those are the VS project settings required to make OpenCL work:
  - C/C++ -> General -> Additional Include Directories (../OpenCL/include) 

  - C/C++ -> Optimization -> Whole Program Optimization -> NO 

  - C/C++ -> Code Generation -> Enable Function Level Linking -> Yes(/Gy) 

  - Librarian -> General -> Additional Dependencies -> add "OpenCL.lib" 

  - Librarian -> General -> Additional Library Directories -> (../OpenCL/lib) 

- The library has been built using "build solution" with the project in Release Mode. This produced a .lib that we then used inside Matlab.

## Matlab Library Import 
To import the library in Matlab and get a usable object you need to:
- Have a C++ compiler installec on your machine
- Download the Matlab folder that contains the .lib and .h files of the library.
- Open the folder in Matlab. 
- Create a Matlab file with this code:
'''
mex -setup cpp 

header = "./mat_inv_32.h"; 

libFile = "./mat_inv_32.lib"; 

libName = "MatrixInversion"; 

clibgen.generateLibraryDefinition(header, "Libraries", libFile, "PackageName",  libName); 

summary(defineMatrixInversion); 

build(defineMatrixInversion); 

inversaGPU = clib.matInv.matrix_inv_32(b, N).double 
'''


(This steps are for the (FP32 version + partial pivoting) version of the algorithm. To use the other versions you also need to create the .lib and .h files and change the Matlab code to utilize the new library and headers files)



