%% About definematInv.mlx
% This file defines the MATLAB interface to the library |matInv|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for &lt;SHAPE&gt;, &lt;DIRECTION&gt;, etc. For more
% information, see <matlab:helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') Define MATLAB Interface for C++ Library>.



%% Setup
% Do not edit this setup section.
function libDef = definematInv()
libDef = clibgen.LibraryDefinition("matInvData.xml");
%% OutputFolder and Libraries 
libDef.OutputFolder = "C:\TESI\matlab-parallel-computation-tool\FP32_matlab_version";
libDef.Libraries = "C:\TESI\matlab-parallel-computation-tool\FP32_matlab_version\mat_inv_32.lib";

%% C++ function |matrix_inv_32| with MATLAB name |clib.matInv.matrix_inv_32|
% C++ Signature: std::vector<float, std::allocator<float>> matrix_inv_32(std::vector<float, std::allocator<float>> matrix_vector,int matrix_order)
matrix_inv_32Definition = addFunction(libDef, ...
    "std::vector<float, std::allocator<float>> matrix_inv_32(std::vector<float, std::allocator<float>> matrix_vector,int matrix_order)", ...
    "MATLABName", "clib.matInv.matrix_inv_32", ...
    "Description", "clib.matInv.matrix_inv_32 Representation of C++ function matrix_inv_32."); % Modify help description values as needed.
defineArgument(matrix_inv_32Definition, "matrix_vector", "clib.array.matInv.Float");
defineArgument(matrix_inv_32Definition, "matrix_order", "int32");
defineOutput(matrix_inv_32Definition, "RetVal", "clib.array.matInv.Float");
validate(matrix_inv_32Definition);

%% Validate the library definition
validate(libDef);

end
