#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audioguard/Preprocessor.h"
#include "audioguard/AudioLoader.h" // <--- Make sure this is included

namespace py = pybind11;

PYBIND11_MODULE(audioguard_core, m) {
    m.doc() = "AudioGuard C++ Core Module";

    // 1. Expose Preprocessor (The DSP Engine)
    py::class_<audioguard::Preprocessor>(m, "Preprocessor")
        .def(py::init<>())
        .def("process", &audioguard::Preprocessor::process);

    // 2. Expose AudioLoader (The FFMPEG Loader) - THIS WAS MISSING
    py::class_<audioguard::AudioLoader>(m, "AudioLoader")
        .def_static("load_audio", &audioguard::AudioLoader::load_audio, 
                    "Loads audio file, resamples to 16kHz Mono, returns float list.");
}