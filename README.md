# KalmanCPP

Kalman filter library written in C++17 with an emphasis on templates. This library provides an easily extensible framework for implementing your own Kalman filters based on the provided base implementations.

Currently, only an extended Kalman filter (EKF) base implementation is available. The user implements their own prediction and update steps and passes to the filter. The library also allows the user to choose between using numerical and analytical methods for calculating Jacobians, further simplifying implementation. For further details, see the example folder.


## Features
- Extended Kalman filter


## Getting started

These instructions will get you a copy of the project up and running on your local
machine for development and testing purposes.

### Prerequisites

* **CMake v3.15+** - Found at [https://cmake.org/](https://cmake.org/)

* **C++ Compiler** - Needs to support at least the **C++17** standard. 
    This project has so far only been tested using *GCC* (v9.3) so it is not guaranteed to 
    work on other compilers such as *Clang* or *MSVC*. Additional compiler support might be
    implemented in the future and pull requests are always welcome.

> ***Note:*** *You also need to be able to provide ***CMake*** a supported [generator](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).*
   This project has mostly been tested with [Ninja](https://cmake.org/cmake/help/latest/generator/Ninja.html) so it is recommended to use that one
   unless you have a good reason to use something else.

* **Conan (package manager)** - Used for managing dependancies such as the Eigen library. More information [here](https://conan.io/).

### Building the library (as a standalone project)
If you are only interested in using the **KalmanCpp** library inside your own project, skip to the next section.
Building the library as a standalone-project should only be needed if you plan on contributing to the project or
if you want to run the examples.

We recommend building your library through an editor such as *CLion*, *Visual Studio* or *Visual Studio Code*.
If you want to build from the terminal however, you can run the following commands from the library root directory:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja .. && ninja
```

### Including KalmanCpp in your own project
To include **KalmanCpp** in your own project, you can clone the library as a submodule and use the 
`add_subdirectory(kalman_cpp)` command in your `CMakeLists.txt` file. You will also need to link the 
library to your project using the `target_link_libraries` command in *CMake*.


## Running the tests

This library makes use of [Google Test](https://github.com/google/googletest/) for unit 
testing. Unit testing can be disabled in the options, by setting the
`ENABLE_UNIT_TESTING` (from
[cmake/StandardSettings.cmake](cmake/StandardSettings.cmake)) to be false. To run
the tests, simply use CTest, from the build directory, passing the desire
configuration for which to run tests for. An example of this procedure is:

```bash
cd build/Release
ctest -C Release  # or `ctest -C Debug` or any other configuration you wish to test

# you can also run tests with the `-VV` flag for a more verbose output (i.e.
#GoogleTest output as well)
```


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our how you can
become a contributor and the process for submitting pull requests to us.

## Versioning

This project makes use of [SemVer](http://semver.org/) for versioning. A list of
existing versions can be found in the
[project's releases](https://github.com/filipdutescu/modern-cpp-template/releases).

## Authors

* **Thorsteinn Jonsson** - [@ThorsteinnJonsson](https://github.com/ThorsteinnJonsson)

## License
TBD
<!-- This project is licensed under the [Unlicense](https://unlicense.org/) - see the
[LICENSE](LICENSE) file for details -->
