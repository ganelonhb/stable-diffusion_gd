# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/stable-diffusion.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stable-diffusion.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stable-diffusion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stable-diffusion.dir/flags.make

CMakeFiles/stable-diffusion.dir/model.cpp.o: CMakeFiles/stable-diffusion.dir/flags.make
CMakeFiles/stable-diffusion.dir/model.cpp.o: /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/model.cpp
CMakeFiles/stable-diffusion.dir/model.cpp.o: CMakeFiles/stable-diffusion.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stable-diffusion.dir/model.cpp.o"
	ccache /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stable-diffusion.dir/model.cpp.o -MF CMakeFiles/stable-diffusion.dir/model.cpp.o.d -o CMakeFiles/stable-diffusion.dir/model.cpp.o -c /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/model.cpp

CMakeFiles/stable-diffusion.dir/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stable-diffusion.dir/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/model.cpp > CMakeFiles/stable-diffusion.dir/model.cpp.i

CMakeFiles/stable-diffusion.dir/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stable-diffusion.dir/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/model.cpp -o CMakeFiles/stable-diffusion.dir/model.cpp.s

CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o: CMakeFiles/stable-diffusion.dir/flags.make
CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o: /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/stable-diffusion.cpp
CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o: CMakeFiles/stable-diffusion.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o"
	ccache /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o -MF CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o.d -o CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o -c /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/stable-diffusion.cpp

CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/stable-diffusion.cpp > CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.i

CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/stable-diffusion.cpp -o CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.s

CMakeFiles/stable-diffusion.dir/upscaler.cpp.o: CMakeFiles/stable-diffusion.dir/flags.make
CMakeFiles/stable-diffusion.dir/upscaler.cpp.o: /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/upscaler.cpp
CMakeFiles/stable-diffusion.dir/upscaler.cpp.o: CMakeFiles/stable-diffusion.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/stable-diffusion.dir/upscaler.cpp.o"
	ccache /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stable-diffusion.dir/upscaler.cpp.o -MF CMakeFiles/stable-diffusion.dir/upscaler.cpp.o.d -o CMakeFiles/stable-diffusion.dir/upscaler.cpp.o -c /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/upscaler.cpp

CMakeFiles/stable-diffusion.dir/upscaler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stable-diffusion.dir/upscaler.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/upscaler.cpp > CMakeFiles/stable-diffusion.dir/upscaler.cpp.i

CMakeFiles/stable-diffusion.dir/upscaler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stable-diffusion.dir/upscaler.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/upscaler.cpp -o CMakeFiles/stable-diffusion.dir/upscaler.cpp.s

CMakeFiles/stable-diffusion.dir/util.cpp.o: CMakeFiles/stable-diffusion.dir/flags.make
CMakeFiles/stable-diffusion.dir/util.cpp.o: /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/util.cpp
CMakeFiles/stable-diffusion.dir/util.cpp.o: CMakeFiles/stable-diffusion.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/stable-diffusion.dir/util.cpp.o"
	ccache /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stable-diffusion.dir/util.cpp.o -MF CMakeFiles/stable-diffusion.dir/util.cpp.o.d -o CMakeFiles/stable-diffusion.dir/util.cpp.o -c /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/util.cpp

CMakeFiles/stable-diffusion.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stable-diffusion.dir/util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/util.cpp > CMakeFiles/stable-diffusion.dir/util.cpp.i

CMakeFiles/stable-diffusion.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stable-diffusion.dir/util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/util.cpp -o CMakeFiles/stable-diffusion.dir/util.cpp.s

# Object files for target stable-diffusion
stable__diffusion_OBJECTS = \
"CMakeFiles/stable-diffusion.dir/model.cpp.o" \
"CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o" \
"CMakeFiles/stable-diffusion.dir/upscaler.cpp.o" \
"CMakeFiles/stable-diffusion.dir/util.cpp.o"

# External object files for target stable-diffusion
stable__diffusion_EXTERNAL_OBJECTS = \
"/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/thirdparty/CMakeFiles/zip.dir/zip.c.o"

libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/model.cpp.o
libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.o
libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/upscaler.cpp.o
libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/util.cpp.o
libstable-diffusion.a: thirdparty/CMakeFiles/zip.dir/zip.c.o
libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/build.make
libstable-diffusion.a: CMakeFiles/stable-diffusion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libstable-diffusion.a"
	$(CMAKE_COMMAND) -P CMakeFiles/stable-diffusion.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stable-diffusion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stable-diffusion.dir/build: libstable-diffusion.a
.PHONY : CMakeFiles/stable-diffusion.dir/build

CMakeFiles/stable-diffusion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stable-diffusion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stable-diffusion.dir/clean

CMakeFiles/stable-diffusion.dir/depend:
	cd /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build /home/donquixote/git/godot/modules/stable_diffusion_gd/stable-diffusion_cpp/build/CMakeFiles/stable-diffusion.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/stable-diffusion.dir/depend
