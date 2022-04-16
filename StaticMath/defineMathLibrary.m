%% About defineMathLibrary.mlx
% This file defines the MATLAB interface to the library |MathLibrary|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for &lt;SHAPE&gt;, &lt;DIRECTION&gt;, etc. For more
% information, see <matlab:helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') Define MATLAB Interface for C++ Library>.



%% Setup
% Do not edit this setup section.
function libDef = defineMathLibrary()
libDef = clibgen.LibraryDefinition("MathLibraryData.xml");
%% OutputFolder and Libraries 
libDef.OutputFolder = "C:\TESI\matlab-parallel-computation-tool\StaticMath";
libDef.Libraries = "C:\TESI\matlab-parallel-computation-tool\StaticMath\x64\Release\MathLibrary.lib";

%% C++ class |MathLibrary::Arithmetic| with MATLAB name |clib.MathLibrary.MathLibrary.Arithmetic| 
ArithmeticDefinition = addClass(libDef, "MathLibrary::Arithmetic", "MATLABName", "clib.MathLibrary.MathLibrary.Arithmetic", ...
    "Description", "clib.MathLibrary.MathLibrary.Arithmetic    Representation of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.

%% C++ class constructor for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: MathLibrary::Arithmetic::Arithmetic(MathLibrary::Arithmetic const & input1)
ArithmeticConstructor1Definition = addConstructor(ArithmeticDefinition, ...
    "MathLibrary::Arithmetic::Arithmetic(MathLibrary::Arithmetic const & input1)", ...
    "Description", "clib.MathLibrary.MathLibrary.Arithmetic Constructor of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineArgument(ArithmeticConstructor1Definition, "input1", "clib.MathLibrary.MathLibrary.Arithmetic", "input");
validate(ArithmeticConstructor1Definition);

%% C++ class constructor for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: MathLibrary::Arithmetic::Arithmetic()
ArithmeticConstructor2Definition = addConstructor(ArithmeticDefinition, ...
    "MathLibrary::Arithmetic::Arithmetic()", ...
    "Description", "clib.MathLibrary.MathLibrary.Arithmetic Constructor of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
validate(ArithmeticConstructor2Definition);

%% C++ class method |Add| for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: static double MathLibrary::Arithmetic::Add(double a,double b)
AddDefinition = addMethod(ArithmeticDefinition, ...
    "static double MathLibrary::Arithmetic::Add(double a,double b)", ...
    "MATLABName", "Add", ...
    "Description", "Add Method of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineArgument(AddDefinition, "a", "double");
defineArgument(AddDefinition, "b", "double");
defineOutput(AddDefinition, "RetVal", "double");
validate(AddDefinition);

%% C++ class method |Subtract| for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: static double MathLibrary::Arithmetic::Subtract(double a,double b)
SubtractDefinition = addMethod(ArithmeticDefinition, ...
    "static double MathLibrary::Arithmetic::Subtract(double a,double b)", ...
    "MATLABName", "Subtract", ...
    "Description", "Subtract Method of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineArgument(SubtractDefinition, "a", "double");
defineArgument(SubtractDefinition, "b", "double");
defineOutput(SubtractDefinition, "RetVal", "double");
validate(SubtractDefinition);

%% C++ class method |Multiply| for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: static double MathLibrary::Arithmetic::Multiply(double a,double b)
MultiplyDefinition = addMethod(ArithmeticDefinition, ...
    "static double MathLibrary::Arithmetic::Multiply(double a,double b)", ...
    "MATLABName", "Multiply", ...
    "Description", "Multiply Method of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineArgument(MultiplyDefinition, "a", "double");
defineArgument(MultiplyDefinition, "b", "double");
defineOutput(MultiplyDefinition, "RetVal", "double");
validate(MultiplyDefinition);

%% C++ class method |Divide| for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: static double MathLibrary::Arithmetic::Divide(double a,double b)
DivideDefinition = addMethod(ArithmeticDefinition, ...
    "static double MathLibrary::Arithmetic::Divide(double a,double b)", ...
    "MATLABName", "Divide", ...
    "Description", "Divide Method of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineArgument(DivideDefinition, "a", "double");
defineArgument(DivideDefinition, "b", "double");
defineOutput(DivideDefinition, "RetVal", "double");
validate(DivideDefinition);

%% C++ class method |Test| for C++ class |MathLibrary::Arithmetic| 
% C++ Signature: static std::string MathLibrary::Arithmetic::Test()
TestDefinition = addMethod(ArithmeticDefinition, ...
    "static std::string MathLibrary::Arithmetic::Test()", ...
    "MATLABName", "Test", ...
    "Description", "Test Method of C++ class MathLibrary::Arithmetic."); % Modify help description values as needed.
defineOutput(TestDefinition, "RetVal", "string");
validate(TestDefinition);

%% Validate the library definition
validate(libDef);

end
