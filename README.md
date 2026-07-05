Assuming you have the nVidia dev toolkit installed and you have Microsoft Studio 2022... do this to build

"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
nvcc -arch=sm_89 -O3 -use_fast_math -o freeway1 freeway1.cu
nvcc -arch=sm_89 -O3 -use_fast_math -o freeway2 freeway2.cu


