# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/benjamin/LCN/Feux/workspace/Vehicle_Detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build

# Include any dependencies generated for this target.
include CMakeFiles/vehicle_detection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vehicle_detection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vehicle_detection.dir/flags.make

CMakeFiles/vehicle_detection.dir/src/main.cpp.o: CMakeFiles/vehicle_detection.dir/flags.make
CMakeFiles/vehicle_detection.dir/src/main.cpp.o: ../src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/vehicle_detection.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vehicle_detection.dir/src/main.cpp.o -c /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/main.cpp

CMakeFiles/vehicle_detection.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vehicle_detection.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/main.cpp > CMakeFiles/vehicle_detection.dir/src/main.cpp.i

CMakeFiles/vehicle_detection.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vehicle_detection.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/main.cpp -o CMakeFiles/vehicle_detection.dir/src/main.cpp.s

CMakeFiles/vehicle_detection.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/vehicle_detection.dir/src/main.cpp.o.requires

CMakeFiles/vehicle_detection.dir/src/main.cpp.o.provides: CMakeFiles/vehicle_detection.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/vehicle_detection.dir/build.make CMakeFiles/vehicle_detection.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/vehicle_detection.dir/src/main.cpp.o.provides

CMakeFiles/vehicle_detection.dir/src/main.cpp.o.provides.build: CMakeFiles/vehicle_detection.dir/src/main.cpp.o

CMakeFiles/vehicle_detection.dir/src/tools.cpp.o: CMakeFiles/vehicle_detection.dir/flags.make
CMakeFiles/vehicle_detection.dir/src/tools.cpp.o: ../src/tools.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/vehicle_detection.dir/src/tools.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vehicle_detection.dir/src/tools.cpp.o -c /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/tools.cpp

CMakeFiles/vehicle_detection.dir/src/tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vehicle_detection.dir/src/tools.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/tools.cpp > CMakeFiles/vehicle_detection.dir/src/tools.cpp.i

CMakeFiles/vehicle_detection.dir/src/tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vehicle_detection.dir/src/tools.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/src/tools.cpp -o CMakeFiles/vehicle_detection.dir/src/tools.cpp.s

CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.requires:
.PHONY : CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.requires

CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.provides: CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.requires
	$(MAKE) -f CMakeFiles/vehicle_detection.dir/build.make CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.provides.build
.PHONY : CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.provides

CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.provides.build: CMakeFiles/vehicle_detection.dir/src/tools.cpp.o

# Object files for target vehicle_detection
vehicle_detection_OBJECTS = \
"CMakeFiles/vehicle_detection.dir/src/main.cpp.o" \
"CMakeFiles/vehicle_detection.dir/src/tools.cpp.o"

# External object files for target vehicle_detection
vehicle_detection_EXTERNAL_OBJECTS =

vehicle_detection: CMakeFiles/vehicle_detection.dir/src/main.cpp.o
vehicle_detection: CMakeFiles/vehicle_detection.dir/src/tools.cpp.o
vehicle_detection: CMakeFiles/vehicle_detection.dir/build.make
vehicle_detection: /usr/local/lib/libopencv_dnn.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_ml.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_objdetect.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_shape.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_stitching.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_superres.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_videostab.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_viz.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_calib3d.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_features2d.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_flann.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_highgui.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_photo.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_video.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_videoio.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_imgproc.so.3.3.0
vehicle_detection: /usr/local/lib/libopencv_core.so.3.3.0
vehicle_detection: CMakeFiles/vehicle_detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable vehicle_detection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vehicle_detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vehicle_detection.dir/build: vehicle_detection
.PHONY : CMakeFiles/vehicle_detection.dir/build

CMakeFiles/vehicle_detection.dir/requires: CMakeFiles/vehicle_detection.dir/src/main.cpp.o.requires
CMakeFiles/vehicle_detection.dir/requires: CMakeFiles/vehicle_detection.dir/src/tools.cpp.o.requires
.PHONY : CMakeFiles/vehicle_detection.dir/requires

CMakeFiles/vehicle_detection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vehicle_detection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vehicle_detection.dir/clean

CMakeFiles/vehicle_detection.dir/depend:
	cd /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/benjamin/LCN/Feux/workspace/Vehicle_Detection /home/benjamin/LCN/Feux/workspace/Vehicle_Detection /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build /home/benjamin/LCN/Feux/workspace/Vehicle_Detection/build/CMakeFiles/vehicle_detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vehicle_detection.dir/depend

