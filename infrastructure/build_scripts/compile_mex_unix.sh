#!/bin/bash
#
# compile_mex_unix.sh
# 
# Script to compile MEX files for MFE Toolbox on Unix/Linux systems
# Generates optimized .mexa64 binaries from C source files
#
# MFE Toolbox Version 4.0 (28-Oct-2009)
#

# Initialize variables
MATLAB_BIN="$MATLAB_ROOT/bin"
MEX_SOURCE_DIR="../../src/backend/mex_source"
MEX_OUTPUT_DIR="../../src/backend/dlls"
MEX_FLAGS="-largeArrayDims -O"
INCLUDE_FLAGS="-I\"$MEX_SOURCE_DIR\""

# Function to check environment and prerequisites
check_environment() {
    # Check if MATLAB_ROOT environment variable is set
    if [ -z "$MATLAB_ROOT" ]; then
        echo "Error: MATLAB_ROOT environment variable is not set"
        echo "Please set MATLAB_ROOT to your MATLAB installation directory"
        return 1
    fi
    
    # Check if MEX compiler exists
    if [ ! -f "$MATLAB_BIN/mex" ]; then
        echo "Error: MEX compiler not found at $MATLAB_BIN/mex"
        echo "Please check your MATLAB installation"
        return 1
    fi
    
    # Check if source directory exists
    if [ ! -d "$MEX_SOURCE_DIR" ]; then
        echo "Error: Source directory not found: $MEX_SOURCE_DIR"
        return 1
    fi
    
    return 0
}

# Function to create output directory
create_output_directory() {
    mkdir -p "$MEX_OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory: $MEX_OUTPUT_DIR"
        return 1
    fi
    return 0
}

# Function to print formatted status message
print_build_status() {
    local file="$1"
    local status="$2"
    local timestamp=$(date "+%H:%M:%S")
    
    if [ "$status" == "SUCCESS" ]; then
        echo "[$timestamp] [SUCCESS] $file compiled successfully"
    else
        echo "[$timestamp] [FAILED]  $file compilation failed"
    fi
}

# Function to compile a single MEX file
compile_single_file() {
    local source_file="$1"
    local output_name="$2"
    local additional_sources="$3"
    
    echo "Compiling: $source_file -> $output_name.mexa64"
    
    # Check if source file exists
    if [ ! -f "$MEX_SOURCE_DIR/$source_file" ]; then
        echo "Error: Source file not found: $MEX_SOURCE_DIR/$source_file"
        return 1
    fi
    
    # Construct the command with all source files
    local sources="$MEX_SOURCE_DIR/$source_file $additional_sources"
    
    # Compile the MEX file
    "$MATLAB_BIN/mex" $MEX_FLAGS $INCLUDE_FLAGS $sources -output "$MEX_OUTPUT_DIR/$output_name"
    
    local compile_status=$?
    if [ $compile_status -ne 0 ]; then
        print_build_status "$source_file" "FAILED"
        return 1
    fi
    
    # Verify the output file was created
    if [ ! -f "$MEX_OUTPUT_DIR/$output_name.mexa64" ]; then
        echo "Error: Output file not created: $MEX_OUTPUT_DIR/$output_name.mexa64"
        print_build_status "$source_file" "FAILED"
        return 1
    fi
    
    print_build_status "$source_file" "SUCCESS"
    return 0
}

# Main function to orchestrate the compilation process
compile_mex_files() {
    # Check environment first
    check_environment
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Create output directory
    create_output_directory
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Print start message
    echo "==========================================================="
    echo "Compiling MEX files for MFE Toolbox (Unix/Linux)"
    echo "Source directory: $MEX_SOURCE_DIR"
    echo "Output directory: $MEX_OUTPUT_DIR"
    echo "MEX flags: $MEX_FLAGS"
    echo "==========================================================="
    
    # Utility source files that need to be included in each compilation
    local utility_sources="$MEX_SOURCE_DIR/matrix_operations.c $MEX_SOURCE_DIR/mex_utils.c"
    
    # Track compilation success
    local success_count=0
    local failure_count=0
    local total_count=6  # Total number of MEX files to compile
    
    # Compile model implementation files with utility sources
    if compile_single_file "agarch_core.c" "agarch_core" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    if compile_single_file "armaxerrors.c" "armaxerrors" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    if compile_single_file "composite_likelihood.c" "composite_likelihood" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    if compile_single_file "egarch_core.c" "egarch_core" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    if compile_single_file "igarch_core.c" "igarch_core" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    if compile_single_file "tarch_core.c" "tarch_core" "$utility_sources"; then
        ((success_count++))
    else
        ((failure_count++))
    fi
    
    # Print completion message
    echo "==========================================================="
    echo "MEX compilation summary:"
    echo "  Total files:   $total_count"
    echo "  Successful:    $success_count"
    echo "  Failed:        $failure_count"
    echo "==========================================================="
    
    # Set exit status based on compilation success
    if [ $success_count -eq $total_count ]; then
        echo "MEX compilation for Unix/Linux completed successfully"
        echo "Output files are located in: $MEX_OUTPUT_DIR"
        return 0
    else
        echo "MEX compilation completed with errors"
        return 1
    fi
}

# Run the compilation
compile_mex_files
exit $?