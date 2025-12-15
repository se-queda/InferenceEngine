#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // CRITICAL: Automatically converts std::vector <-> Python list
#include "audioguard/Preprocessor.h"

namespace py = pybind11;

// The module name here (audioguard_core) MUST match the one in CMakeLists.txt
PYBIND11_MODULE(audioguard_core, m) {
    
    m.doc() = "AudioGuard C++ Core Module implemented with Pybind11";

    // Expose the Preprocessor class to Python
    py::class_<audioguard::Preprocessor>(m, "Preprocessor")
        // 1. Expose the Constructor
        .def(py::init<>())
        
        // 2. Expose the process function
        // Usage in Python: result_list = dsp.process(input_list)
        .def("process", &audioguard::Preprocessor::process, 
             "Takes a raw audio float list (16k), returns a flattened Mel-Spectrogram list.");
}