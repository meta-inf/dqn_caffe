# Yet another CLion issue
cd ../src
protoc TrainerConfig.proto --cpp_out=.
cd ../bin
cmake ..
cmake --build . -- -j4

