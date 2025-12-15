#include <iostream>
#include "audioguard/Preprocessor.h"

int main() {
    std::cout << "AudioGuard App: System Booting...\n";
    
    // Simple test to ensure linkage works
    audioguard::Preprocessor dsp;
    std::cout << "DSP Engine initialized successfully.\n";
    
    return 0;
}