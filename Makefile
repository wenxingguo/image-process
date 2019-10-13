
CXX=g++
TARGET=demo.exe
INCLUDE_PATH=-IC:/opt/opencv/include -IC:/opt/FLTK/include -IC:/opt/FLTK/include/image -IC:/opt/fftw/include -I../src
CXX_FLAG=-D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_THREAD_SAFE
STATIC_LIB= C:/opt/FLTK/lib/libfltk.a C:/opt/fftw/lib/libfftw3.a 
LD_PATH= -LC:\opt\opencv\x64\mingw\lib\ 
SHARED_LIBS= -lole32 -luuid -lcomctl32 -lws2_32 -lgdi32 -lpthread -lopencv_world411.dll

release:img.o main.o
		$(CXX) img.o main.o $(STATIC_LIB)  $(LD_PATH) $(SHARED_LIBS) $(CXX_FLAG) -O3 -o $(TARGET)
		make clean

debug:img.o main.o
		$(CXX) img.o main.o $(STATIC_LIB) $(LD_PATH) $(SHARED_LIBS) $(CXX_FLAGS) -g

main.o:main.cpp
		$(CXX) main.cpp -c $(INCLUDE_PATH)

img.o:../src/img.cpp
		$(CXX) ../src/img.cpp -c $(INCLUDE_PATH)

reflash:
		del /f /q ..\src\img*
		del /f /q main.cpp
		wget.exe https://raw.githubusercontent.com/wenxingguo/image-process/master/main.cpp
		wget.exe https://raw.githubusercontent.com/wenxingguo/image-process/master/img.cpp
		wget.exe https://raw.githubusercontent.com/wenxingguo/image-process/master/img.h
		move img.h ../src/img.h
		move img.cpp ../src/img.cpp
clean:
		del /f /q ..\src\*.gch
		del /f /q *.o