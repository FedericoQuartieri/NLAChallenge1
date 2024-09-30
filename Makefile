CXX = g++
OPENMPFLAG = #-fopenmp
OPENMPLINKINGFLAG =
LIBS = #-Linclude/lis-2.0.32/BUILD/lib -llis
CXXFLAGS = -std=c++20 -O2 -fpic -I /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3 #-g 
CPPFLAGS = #-DNDEBUG 
PROGRAM = main
OBJ = main.o #utils_H.o



.PHONY : all clean distclean



main : $(OBJ)
	$(CXX) $(OPENMPLINKINGFLAG) $(LIBS) $(OBJ) $(OPENMPFLAG) -o main.exe


$(OBJ) : %.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMPFLAG) -c $<



clean :
	$(RM) *.o

distclean : clean
	$(RM) main.exe