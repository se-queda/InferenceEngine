#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audioguard/Preprocessor.h"
#include "audioguard/AudioLoader.h"
#include "audioguard/InferenceEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(audioguard_core, m) {
    m.doc() = "AudioGuard C++ Core Module";

    // Expose Preprocessor 
    py::class_<audioguard::Preprocessor>(m, "Preprocessor")
        .def(py::init<>())
        .def("process", &audioguard::Preprocessor::process);

    // Expose AudioLoade
    py::class_<audioguard::AudioLoader>(m, "AudioLoader")
        .def_static("load_audio", &audioguard::AudioLoader::load_audio, 
                    "Loads audio file, resamples to 16kHz Mono, returns float list.");
    // Expose InferenceEngine 
    py::class_<audioguard::InferenceEngine>(m, "InferenceEngine")
        .def(py::init<const std::string&>(), "Load model from path")
        .def("predict", &audioguard::InferenceEngine::predict, 
             "Run inference on input vector",
             py::arg("input_data"), py::arg("input_shape"));
}
