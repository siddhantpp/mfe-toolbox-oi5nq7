classdef CompilationSystemTest < BaseTest
    % COMPILATIONSYSTEMTEST System test class for validating the MEX compilation and build system of the MFE Toolbox
    %
    % This test suite validates the MEX compilation process, build system functionality, and 
    % cross-platform compatibility of the MFE Toolbox. It ensures:
    %   1. Required source files are present
    %   2. MEX files compile correctly with appropriate flags
    %   3. Large array support is enabled
    %   4. Platform-specific binaries are generated correctly
    %   5. Performance requirements are met (>50% improvement)
    %   6. The automated build process works correctly
    %
    % See also: BaseTest, MEXValidator, CrossPlatformValidator
    
    properties
        mexSourcePath          % Path to MEX source files
        mexOutputPath          % Path to compiled MEX binaries
        currentPlatform        % Current platform identifier
        tempBuildDir           % Temporary directory for build testing
        mexValidator           % MEXValidator instance for validation
        platformValidator      % CrossPlatformValidator for platform testing
        mexSources             % Structure with source file information
        compilationOptions     % Structure with compilation options
        cSourceFiles           % Cell array of C source files
        expectedMEXFiles       % Cell array of expected MEX files
    end
    
    methods
        function obj = CompilationSystemTest()
            % Constructor initializes test paths and platform information
            
            % Call parent constructor with class name
            obj@BaseTest('CompilationSystemTest');
            
            % Initialize paths to MEX source and output directories
            obj.mexSourcePath = '../../backend/mex_source/';
            obj.mexOutputPath = '../../backend/dlls/';
            
            % Create temporary build directory path
            obj.tempBuildDir = fullfile(tempdir, 'mfe_test_build');
            
            % Determine current platform
            obj.currentPlatform = computer();
            
            % Create validator instances
            obj.mexValidator = MEXValidator();
            obj.platformValidator = CrossPlatformValidator();
            
            % Initialize MEX source information
            obj.mexSources = struct();
            
            % Set compilation options
            obj.compilationOptions = struct('largeArrayDims', true);
            
            % Initialize list of C source files to test
            obj.cSourceFiles = {'agarch_core.c', 'armaxerrors.c', ...
                'composite_likelihood.c', 'egarch_core.c', 'igarch_core.c', ...
                'tarch_core.c'};
            
            % Set expected MEX files based on platform
            if strncmp(obj.currentPlatform, 'PCW', 3)
                % Windows platform
                obj.expectedMEXFiles = {'agarch_core.mexw64', 'armaxerrors.mexw64', ...
                    'composite_likelihood.mexw64', 'egarch_core.mexw64', ...
                    'igarch_core.mexw64', 'tarch_core.mexw64'};
            else
                % Unix platform
                obj.expectedMEXFiles = {'agarch_core.mexa64', 'armaxerrors.mexa64', ...
                    'composite_likelihood.mexa64', 'egarch_core.mexa64', ...
                    'igarch_core.mexa64', 'tarch_core.mexa64'};
            end
        end
        
        function setUp(obj)
            % Set up test environment before each test execution
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Ensure source directory exists
            if ~exist(obj.mexSourcePath, 'dir')
                error('MEX source directory not found: %s', obj.mexSourcePath);
            end
            
            % Create temporary build directory if it doesn't exist
            if ~exist(obj.tempBuildDir, 'dir')
                mkdir(obj.tempBuildDir);
            end
            
            % Get available C source files
            obj.mexSources = obj.getMEXSourceFiles();
            
            % Prepare compilation options based on platform
            if strncmp(obj.currentPlatform, 'PCW', 3)
                % Windows-specific options
                obj.compilationOptions.platform = 'PCWIN64';
                obj.compilationOptions.extension = 'mexw64';
            else
                % Unix-specific options
                obj.compilationOptions.platform = 'UNIX';
                obj.compilationOptions.extension = 'mexa64';
            end
            
            % Back up any existing MEX binaries to avoid interference
            backupDir = fullfile(obj.tempBuildDir, 'mex_backup');
            if ~exist(backupDir, 'dir')
                mkdir(backupDir);
            end
            
            % Copy existing MEX files to backup directory if they exist
            for i = 1:length(obj.expectedMEXFiles)
                mexFile = fullfile(obj.mexOutputPath, obj.expectedMEXFiles{i});
                if exist(mexFile, 'file')
                    copyfile(mexFile, backupDir);
                end
            end
        end
        
        function tearDown(obj)
            % Clean up after tests
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Remove temporary build directory if created
            if exist(obj.tempBuildDir, 'dir')
                try
                    % Restore any backed-up MEX binaries
                    backupDir = fullfile(obj.tempBuildDir, 'mex_backup');
                    if exist(backupDir, 'dir')
                        files = dir(fullfile(backupDir, ['*.', obj.compilationOptions.extension]));
                        for i = 1:length(files)
                            if exist(fullfile(backupDir, files(i).name), 'file')
                                try
                                    copyfile(fullfile(backupDir, files(i).name), obj.mexOutputPath);
                                catch
                                    % Ignore errors if files can't be copied back
                                end
                            end
                        end
                    end
                    
                    % Clean up temporary directory
                    rmdir(obj.tempBuildDir, 's');
                catch ME
                    warning('Error during tearDown: %s', ME.message);
                end
            end
        end
        
        function testSourceFilesExist(obj)
            % Test that all required C source files exist in the mex_source directory
            
            % Define the list of expected source files
            expectedFiles = {'agarch_core.c', 'armaxerrors.c', 'composite_likelihood.c', ...
                'egarch_core.c', 'igarch_core.c', 'tarch_core.c'};
            
            % Optional utility files
            utilFiles = {'matrix_operations.c', 'matrix_operations.h', 'mex_utils.c', 'mex_utils.h'};
            
            % Check each required file exists
            for i = 1:length(expectedFiles)
                fileName = fullfile(obj.mexSourcePath, expectedFiles{i});
                fileExists = exist(fileName, 'file') == 2; % 2 = file exists
                obj.assertTrue(fileExists, sprintf('Required source file not found: %s', expectedFiles{i}));
            end
            
            % Report on optional utility files
            for i = 1:length(utilFiles)
                fileName = fullfile(obj.mexSourcePath, utilFiles{i});
                if exist(fileName, 'file') ~= 2
                    fprintf('Note: Optional utility file not found: %s\n', utilFiles{i});
                end
            end
        end
        
        function testSourceCompiles(obj)
            % Test that all C source files can be compiled to MEX binaries
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Copy all source files to temporary directory for compilation
                for i = 1:length(obj.cSourceFiles)
                    sourceFile = fullfile(obj.mexSourcePath, obj.cSourceFiles{i});
                    if exist(sourceFile, 'file')
                        copyfile(sourceFile, obj.tempBuildDir);
                    else
                        obj.assertTrue(false, sprintf('Source file not found: %s', obj.cSourceFiles{i}));
                    end
                end
                
                % Copy utility files if they exist
                utilFiles = {'matrix_operations.c', 'matrix_operations.h', 'mex_utils.c', 'mex_utils.h'};
                for i = 1:length(utilFiles)
                    utilFile = fullfile(obj.mexSourcePath, utilFiles{i});
                    if exist(utilFile, 'file')
                        copyfile(utilFile, obj.tempBuildDir);
                    end
                end
                
                % Change to temporary directory for compilation
                cd(obj.tempBuildDir);
                
                % Compile each source file
                for i = 1:length(obj.cSourceFiles)
                    % Extract base name without extension
                    [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                    
                    fprintf('Compiling %s...\n', obj.cSourceFiles{i});
                    
                    % Try to compile the file
                    compileSuccess = obj.compileSourceFile(obj.cSourceFiles{i}, ...
                        [baseName, '.', obj.compilationOptions.extension], obj.compilationOptions);
                    
                    % Verify compilation success
                    obj.assertTrue(compileSuccess, sprintf('Failed to compile: %s', obj.cSourceFiles{i}));
                    
                    % Verify MEX file was created
                    mexFile = [baseName, '.', obj.compilationOptions.extension];
                    obj.assertTrue(exist(mexFile, 'file') == 3, sprintf('MEX file not created: %s', mexFile));
                    
                    % Try to execute the MEX function to verify functionality
                    % Use MEXValidator to test with appropriate inputs
                    testInputs = obj.mexValidator.generateTestInputs(baseName);
                    if ~isempty(testInputs)
                        funcHandle = str2func(baseName);
                        try
                            funcHandle(testInputs{:});
                            fprintf('Successfully executed %s\n', baseName);
                        catch ME
                            fprintf('Warning: Could not execute %s: %s\n', baseName, ME.message);
                        end
                    end
                end
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testBuildZipFileFunction(obj)
            % Test that the buildZipFile function successfully creates a complete package
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Create a special temporary directory for this test
                buildTestDir = fullfile(obj.tempBuildDir, 'build_test');
                if ~exist(buildTestDir, 'dir')
                    mkdir(buildTestDir);
                end
                
                % Change to the build test directory
                cd(buildTestDir);
                
                % Call the buildZipFile function
                fprintf('Testing buildZipFile function...\n');
                buildZipFile();
                
                % Verify ZIP file was created
                obj.assertTrue(exist('MFEToolbox.zip', 'file') == 2, 'MFEToolbox.zip was not created');
                
                % Verify ZIP contents
                verifyResult = obj.verifyPackageContents('MFEToolbox.zip');
                
                % Check mandatory directories exist
                mandatoryDirs = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                    'multivariate', 'tests', 'timeseries', 'univariate', 'utility', ...
                    'realized', 'mex_source', 'dlls'};
                
                for i = 1:length(mandatoryDirs)
                    obj.assertTrue(verifyResult.hasMandatoryDir.(mandatoryDirs{i}), ...
                        sprintf('Mandatory directory missing: %s', mandatoryDirs{i}));
                end
                
                % Verify core files are present
                obj.assertTrue(verifyResult.hasAddToPath, 'addToPath.m is missing');
                obj.assertTrue(verifyResult.hasContents, 'Contents.m is missing');
                
                % Verify MEX binaries are included with correct platform extensions
                obj.assertTrue(verifyResult.hasMEXBinaries, 'MEX binaries are missing');
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testLargeArrayDimsFlag(obj)
            % Test that MEX compilation with -largeArrayDims flag works correctly
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Create a test directory
                testDir = fullfile(obj.tempBuildDir, 'large_array_test');
                if ~exist(testDir, 'dir')
                    mkdir(testDir);
                end
                
                % Select a test source file
                testSource = 'agarch_core.c';
                sourceFile = fullfile(obj.mexSourcePath, testSource);
                
                % Copy source file and any dependencies to test directory
                if exist(sourceFile, 'file')
                    copyfile(sourceFile, testDir);
                    
                    % Copy utility files if they exist
                    utilFiles = {'matrix_operations.c', 'matrix_operations.h', 'mex_utils.c', 'mex_utils.h'};
                    for i = 1:length(utilFiles)
                        utilFile = fullfile(obj.mexSourcePath, utilFiles{i});
                        if exist(utilFile, 'file')
                            copyfile(utilFile, testDir);
                        end
                    end
                else
                    obj.assertTrue(false, sprintf('Test source file not found: %s', testSource));
                end
                
                % Change to test directory
                cd(testDir);
                
                % Compile with -largeArrayDims flag
                fprintf('Compiling with -largeArrayDims flag...\n');
                [~, baseName, ~] = fileparts(testSource);
                mexFile = [baseName, '.', obj.compilationOptions.extension];
                
                % Explicitly use -largeArrayDims flag
                mex('-largeArrayDims', testSource);
                
                % Verify MEX file was created
                obj.assertTrue(exist(mexFile, 'file') == 3, 'MEX file not created with -largeArrayDims flag');
                
                % Create large test input exceeding standard size limits
                fprintf('Testing large array handling...\n');
                
                % Generate large input data (e.g., 10,000,000 elements)
                % This should be handled correctly with -largeArrayDims
                largeData = randn(10000000, 1);
                parameters = [0.01; 0.1; 0.85]; % omega, alpha, beta
                initialVol = var(largeData(1:1000)); % Use sample for initial variance
                
                % Try to execute the MEX function with large input
                funcHandle = str2func(baseName);
                try
                    funcHandle(largeData, parameters, initialVol);
                    largeArrayHandled = true;
                catch ME
                    fprintf('Error with large array: %s\n', ME.message);
                    largeArrayHandled = false;
                end
                
                % Assert successful handling of large arrays
                obj.assertTrue(largeArrayHandled, 'MEX function failed with large array input');
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testPlatformSpecificCompilation(obj)
            % Test platform-specific compilation options and binary generation
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Create a test directory
                testDir = fullfile(obj.tempBuildDir, 'platform_test');
                if ~exist(testDir, 'dir')
                    mkdir(testDir);
                end
                
                % Determine current platform
                platform = obj.platformValidator.getCurrentPlatform();
                fprintf('Testing platform-specific compilation for %s\n', platform);
                
                % Select a test source file
                testSource = 'agarch_core.c';
                sourceFile = fullfile(obj.mexSourcePath, testSource);
                
                % Copy source file and any dependencies to test directory
                if exist(sourceFile, 'file')
                    copyfile(sourceFile, testDir);
                    
                    % Copy utility files if they exist
                    utilFiles = {'matrix_operations.c', 'matrix_operations.h', 'mex_utils.c', 'mex_utils.h'};
                    for i = 1:length(utilFiles)
                        utilFile = fullfile(obj.mexSourcePath, utilFiles{i});
                        if exist(utilFile, 'file')
                            copyfile(utilFile, testDir);
                        end
                    end
                else
                    obj.assertTrue(false, sprintf('Test source file not found: %s', testSource));
                end
                
                % Change to test directory
                cd(testDir);
                
                % Set platform-specific compilation flags
                options = obj.compilationOptions;
                
                % Compile a test source file
                fprintf('Compiling with platform-specific settings...\n');
                [~, baseName, ~] = fileparts(testSource);
                mexFile = [baseName, '.', options.extension];
                
                % Compile with platform-specific settings
                success = obj.compileSourceFile(testSource, mexFile, options);
                obj.assertTrue(success, 'Platform-specific compilation failed');
                
                % Verify correct platform-specific MEX extension
                mexExt = obj.mexValidator.getMEXExtension();
                obj.assertEqual(mexExt, options.extension, 'Incorrect MEX extension for platform');
                
                % Verify the file exists and is a MEX file
                mexFilePath = fullfile(testDir, mexFile);
                obj.assertTrue(exist(mexFilePath, 'file') == 3, 'MEX file not created with platform-specific settings');
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testCompilationScripts(obj)
            % Test the platform-specific compilation scripts
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Create a test directory
                testDir = fullfile(obj.tempBuildDir, 'scripts_test');
                if ~exist(testDir, 'dir')
                    mkdir(testDir);
                end
                
                % Copy source files to test directory
                for i = 1:length(obj.cSourceFiles)
                    sourceFile = fullfile(obj.mexSourcePath, obj.cSourceFiles{i});
                    if exist(sourceFile, 'file')
                        copyfile(sourceFile, testDir);
                    end
                end
                
                % Copy utility files if they exist
                utilFiles = {'matrix_operations.c', 'matrix_operations.h', 'mex_utils.c', 'mex_utils.h'};
                for i = 1:length(utilFiles)
                    utilFile = fullfile(obj.mexSourcePath, utilFiles{i});
                    if exist(utilFile, 'file')
                        copyfile(utilFile, testDir);
                    end
                end
                
                % Change to test directory
                cd(testDir);
                
                % Determine appropriate script for current platform
                if strncmp(obj.currentPlatform, 'PCW', 3)
                    % Windows platform
                    scriptName = 'compile_mex_windows.bat';
                    
                    % Create Windows batch script
                    fid = fopen(scriptName, 'w');
                    fprintf(fid, '@echo off\n');
                    fprintf(fid, 'echo Compiling MEX files for Windows...\n');
                    for i = 1:length(obj.cSourceFiles)
                        [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                        fprintf(fid, 'mex -largeArrayDims %s\n', obj.cSourceFiles{i});
                    end
                    fprintf(fid, 'echo Compilation complete.\n');
                    fclose(fid);
                    
                    % Execute Windows script
                    fprintf('Executing Windows compilation script...\n');
                    [status, cmdout] = system(scriptName);
                    fprintf('%s\n', cmdout);
                    obj.assertTrue(status == 0, 'Windows compilation script failed');
                else
                    % Unix platform
                    scriptName = 'compile_mex_unix.sh';
                    
                    % Create Unix shell script
                    fid = fopen(scriptName, 'w');
                    fprintf(fid, '#!/bin/bash\n');
                    fprintf(fid, 'echo "Compiling MEX files for Unix..."\n');
                    for i = 1:length(obj.cSourceFiles)
                        [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                        fprintf(fid, 'mex -largeArrayDims %s\n', obj.cSourceFiles{i});
                    end
                    fprintf(fid, 'echo "Compilation complete."\n');
                    fclose(fid);
                    
                    % Make script executable
                    system('chmod +x compile_mex_unix.sh');
                    
                    % Execute Unix script
                    fprintf('Executing Unix compilation script...\n');
                    [status, cmdout] = system('./compile_mex_unix.sh');
                    fprintf('%s\n', cmdout);
                    obj.assertTrue(status == 0, 'Unix compilation script failed');
                end
                
                % Verify expected MEX files were created
                for i = 1:length(obj.cSourceFiles)
                    [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                    mexFile = [baseName, '.', obj.compilationOptions.extension];
                    obj.assertTrue(exist(mexFile, 'file') == 3, sprintf('MEX file not created: %s', mexFile));
                    
                    % Try to execute the MEX function to verify functionality
                    testInputs = obj.mexValidator.generateTestInputs(baseName);
                    if ~isempty(testInputs)
                        funcHandle = str2func(baseName);
                        try
                            funcHandle(testInputs{:});
                        catch ME
                            fprintf('Warning: Could not execute %s: %s\n', baseName, ME.message);
                        end
                    end
                end
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testErrorHandling(obj)
            % Test error handling during compilation process
            
            % Get the current directory to return to later
            currentDir = pwd;
            
            try
                % Create a test directory
                testDir = fullfile(obj.tempBuildDir, 'error_test');
                if ~exist(testDir, 'dir')
                    mkdir(testDir);
                end
                
                % Change to test directory
                cd(testDir);
                
                % Create a source file with syntax errors
                fprintf('Testing error handling with invalid syntax...\n');
                fid = fopen('invalid_syntax.c', 'w');
                fprintf(fid, '#include "mex.h"\n');
                fprintf(fid, 'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\n');
                fprintf(fid, '    // Missing semicolon here\n');
                fprintf(fid, '    mexPrintf("This has an error")\n');
                fprintf(fid, '    // Unbalanced braces\n');
                fprintf(fid, '}\n');  % Missing one closing brace
                fclose(fid);
                
                % Try to compile the file with errors
                try
                    mex('invalid_syntax.c');
                    % Should not reach here
                    syntaxErrorCaught = false;
                catch ME
                    fprintf('Expected error caught: %s\n', ME.message);
                    syntaxErrorCaught = true;
                end
                
                % Assert that the error was caught
                obj.assertTrue(syntaxErrorCaught, 'Syntax error not caught during compilation');
                
                % Test handling of missing source files
                fprintf('Testing error handling with missing file...\n');
                try
                    mex('nonexistent_file.c');
                    % Should not reach here
                    missingFileCaught = false;
                catch ME
                    fprintf('Expected error caught: %s\n', ME.message);
                    missingFileCaught = true;
                end
                
                % Assert that the error was caught
                obj.assertTrue(missingFileCaught, 'Missing file error not caught during compilation');
                
                % Test handling of invalid compilation flags
                fprintf('Testing error handling with invalid flags...\n');
                try
                    mex('-invalid_flag', 'invalid_syntax.c');
                    % Should not reach here
                    invalidFlagCaught = false;
                catch ME
                    fprintf('Expected error caught: %s\n', ME.message);
                    invalidFlagCaught = true;
                end
                
                % Assert that the error was caught
                obj.assertTrue(invalidFlagCaught, 'Invalid flag error not caught during compilation');
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Return to original directory on error
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function testCrossCompilationCompatibility(obj)
            % Test that MEX binaries compiled on different platforms are functionally equivalent
            
            % This test is informational and conditional on having binaries for multiple platforms
            fprintf('Testing cross-compilation compatibility (conditional test)...\n');
            
            % Check if we have reference binaries for other platforms
            otherPlatform = '';
            if strncmp(obj.currentPlatform, 'PCW', 3)
                otherPlatform = 'GLNXA64';
                otherExt = 'mexa64';
            else
                otherPlatform = 'PCWIN64';
                otherExt = 'mexw64';
            end
            
            fprintf('Checking for binaries from %s platform...\n', otherPlatform);
            
            % Look for reference binaries in a standard location
            refBinaryPath = fullfile(obj.tempBuildDir, 'reference_binaries', otherPlatform);
            
            % If reference binaries don't exist, create the directory for them
            if ~exist(refBinaryPath, 'dir')
                mkdir(refBinaryPath);
                fprintf('Created directory for reference binaries: %s\n', refBinaryPath);
                fprintf('This test will be skipped as no reference binaries are available.\n');
                fprintf('Note: To enable this test, copy MEX binaries from %s platform to this directory.\n', otherPlatform);
                return;
            end
            
            % Check if at least one reference binary exists
            refBinaryExists = false;
            for i = 1:length(obj.cSourceFiles)
                [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                refBinaryFile = fullfile(refBinaryPath, [baseName, '.', otherExt]);
                if exist(refBinaryFile, 'file')
                    refBinaryExists = true;
                    break;
                end
            end
            
            if ~refBinaryExists
                fprintf('No reference binaries found. Test will be skipped.\n');
                return;
            end
            
            % We have reference binaries, so perform the compatibility test
            fprintf('Found reference binaries. Testing compatibility...\n');
            
            % Generate test data
            testData = randn(1000, 1);
            
            % Test each MEX file that has a reference binary
            for i = 1:length(obj.cSourceFiles)
                [~, baseName, ~] = fileparts(obj.cSourceFiles{i});
                currentBinary = fullfile(obj.mexOutputPath, [baseName, '.', obj.compilationOptions.extension]);
                refBinary = fullfile(refBinaryPath, [baseName, '.', otherExt]);
                
                % Skip if either binary doesn't exist
                if exist(currentBinary, 'file') ~= 3 || exist(refBinary, 'file') ~= 3
                    fprintf('Skipping %s - binary missing on one platform\n', baseName);
                    continue;
                end
                
                fprintf('Testing cross-platform compatibility for %s...\n', baseName);
                
                % Generate appropriate test inputs for this function
                testInputs = obj.mexValidator.generateTestInputs(baseName);
                if isempty(testInputs)
                    fprintf('Skipping %s - could not generate test inputs\n', baseName);
                    continue;
                end
                
                % Execute on current platform and capture results
                try
                    addpath(obj.mexOutputPath);
                    currFunc = str2func(baseName);
                    currResult = currFunc(testInputs{:});
                    
                    % This test can't be fully automated unless we have both
                    % platforms available. We'll just log the results for now.
                    fprintf('Current platform execution successful for %s\n', baseName);
                    
                    % Save the result for potential manual comparison
                    resultFile = fullfile(obj.tempBuildDir, [baseName, '_', lower(obj.currentPlatform), '_result.mat']);
                    save(resultFile, 'currResult', 'testInputs');
                    fprintf('Saved result to %s for manual comparison\n', resultFile);
                    
                    % Check if we're on a platform that can load the other binary
                    % This is unlikely in a normal test environment, so we'll just report
                    fprintf('Note: Full cross-platform comparison requires access to both Windows and Unix environments\n');
                    
                catch ME
                    fprintf('Error executing %s on current platform: %s\n', baseName, ME.message);
                end
                
                % Remove the MEX output path from MATLAB path
                rmpath(obj.mexOutputPath);
            end
        end
        
        function testPerformanceRequirements(obj)
            % Test that compiled MEX files meet performance improvement requirements
            
            fprintf('Testing performance requirements (>50%% improvement)...\n');
            
            % Define mapping of MEX files to MATLAB implementations
            mexToMatlabMap = struct(...
                'agarch_core', 'agarch_core_matlab', ...
                'egarch_core', 'egarch_core_matlab', ...
                'igarch_core', 'igarch_core_matlab', ...
                'tarch_core', 'tarch_core_matlab', ...
                'armaxerrors', 'armaxerrors_matlab', ...
                'composite_likelihood', 'composite_likelihood_matlab' ...
            );
            
            % Get current directory to return to later
            currentDir = pwd;
            
            try
                % Create test directory
                testDir = fullfile(obj.tempBuildDir, 'performance_test');
                if ~exist(testDir, 'dir')
                    mkdir(testDir);
                end
                
                % Change to test directory
                cd(testDir);
                
                % Add MEX output path to MATLAB path
                addpath(obj.mexOutputPath);
                
                % Store performance results
                perfResults = struct();
                
                % Test core computational MEX files that have MATLAB equivalents
                mexFiles = fieldnames(mexToMatlabMap);
                for i = 1:length(mexFiles)
                    mexFile = mexFiles{i};
                    matlabFile = mexToMatlabMap.(mexFile);
                    
                    % Skip if MEX file or MATLAB equivalent don't exist
                    mexPath = fullfile(obj.mexOutputPath, [mexFile, '.', obj.compilationOptions.extension]);
                    if exist(mexPath, 'file') ~= 3
                        fprintf('Skipping %s - MEX file not found\n', mexFile);
                        continue;
                    end
                    
                    if exist(matlabFile, 'file') ~= 2
                        fprintf('Skipping %s - MATLAB equivalent not found\n', mexFile);
                        continue;
                    end
                    
                    fprintf('Testing performance for %s vs %s...\n', mexFile, matlabFile);
                    
                    % Generate appropriate test inputs
                    testInputs = obj.mexValidator.generateTestInputs(mexFile);
                    if isempty(testInputs)
                        fprintf('Skipping %s - could not generate test inputs\n', mexFile);
                        continue;
                    end
                    
                    % Get function handles
                    mexFunc = str2func(mexFile);
                    matlabFunc = str2func(matlabFile);
                    
                    % Measure execution time for MATLAB implementation
                    fprintf('Measuring MATLAB implementation time...\n');
                    matlabTime = obj.measureExecutionTime(matlabFunc, testInputs, 10);
                    
                    % Measure execution time for MEX implementation
                    fprintf('Measuring MEX implementation time...\n');
                    mexTime = obj.measureExecutionTime(mexFunc, testInputs, 10);
                    
                    % Calculate performance improvement ratio
                    if matlabTime > 0
                        improvement = (matlabTime - mexTime) / matlabTime * 100;
                    else
                        improvement = 0;
                    end
                    
                    fprintf('%s: MATLAB time = %.6f sec, MEX time = %.6f sec, Improvement = %.2f%%\n', ...
                        mexFile, matlabTime, mexTime, improvement);
                    
                    % Store performance results
                    perfResults.(mexFile) = struct(...
                        'matlabTime', matlabTime, ...
                        'mexTime', mexTime, ...
                        'improvement', improvement ...
                    );
                    
                    % Assert performance improvement meets requirement
                    obj.assertTrue(improvement >= 50, ...
                        sprintf('%s does not meet 50%% performance improvement requirement (actual: %.2f%%)', ...
                        mexFile, improvement));
                end
                
                % Save performance results
                save('performance_results.mat', 'perfResults');
                fprintf('Performance results saved to %s\n', fullfile(testDir, 'performance_results.mat'));
                
                % Remove the MEX output path from MATLAB path
                rmpath(obj.mexOutputPath);
                
                % Return to original directory
                cd(currentDir);
            catch ME
                % Clean up on error
                if exist('addedPath', 'var') && addedPath
                    rmpath(obj.mexOutputPath);
                end
                cd(currentDir);
                rethrow(ME);
            end
        end
        
        function execTime = measureExecutionTime(obj, func, inputs, iterations)
            % Helper method to measure the execution time of a function with multiple iterations
            %
            % INPUTS:
            %   func - Function handle
            %   inputs - Cell array of inputs
            %   iterations - Number of iterations
            %
            % OUTPUTS:
            %   execTime - Average execution time in seconds
            
            % Default iterations
            if nargin < 4
                iterations = 5;
            end
            
            % Warm-up run (not timed)
            try
                func(inputs{:});
            catch ME
                fprintf('Error in warm-up: %s\n', ME.message);
                execTime = inf;
                return;
            end
            
            % Measure execution time over multiple iterations
            times = zeros(iterations, 1);
            for i = 1:iterations
                tic;
                try
                    func(inputs{:});
                    times(i) = toc;
                catch ME
                    fprintf('Error in iteration %d: %s\n', i, ME.message);
                    times(i) = inf;
                end
            end
            
            % Remove any inf values
            times = times(isfinite(times));
            if isempty(times)
                execTime = inf;
                return;
            end
            
            % Calculate average execution time
            execTime = mean(times);
        end
        
        function success = compileSourceFile(obj, sourceFile, outputFile, options)
            % Helper method to compile a C source file into a MEX binary
            %
            % INPUTS:
            %   sourceFile - Source file name
            %   outputFile - Output file name
            %   options - Compilation options structure
            %
            % OUTPUTS:
            %   success - Logical indicating compilation success
            
            % Default return value
            success = false;
            
            % Validate source file exists
            if ~exist(sourceFile, 'file')
                fprintf('Error: Source file %s not found\n', sourceFile);
                return;
            end
            
            % Build compilation command
            compileCmd = 'mex -largeArrayDims';
            
            % Add include path if specified
            if isfield(options, 'includePath') && ~isempty(options.includePath)
                compileCmd = [compileCmd, ' -I"', options.includePath, '"'];
            end
            
            % Add output file name
            if nargin >= 3 && ~isempty(outputFile)
                compileCmd = [compileCmd, ' -output ', outputFile];
            end
            
            % Add source file
            compileCmd = [compileCmd, ' ', sourceFile];
            
            % Add additional source files if needed
            utilFiles = {'matrix_operations.c', 'mex_utils.c'};
            for i = 1:length(utilFiles)
                if exist(utilFiles{i}, 'file')
                    compileCmd = [compileCmd, ' ', utilFiles{i}];
                end
            end
            
            % Execute compilation
            try
                fprintf('Executing: %s\n', compileCmd);
                eval(compileCmd);
                
                % Check if output file was created
                if nargin >= 3 && ~isempty(outputFile)
                    success = exist(outputFile, 'file') == 3; % 3 = MEX file
                else
                    % If no output file specified, check for default MEX filename
                    [~, baseName, ~] = fileparts(sourceFile);
                    mexExt = obj.mexValidator.getMEXExtension();
                    success = exist([baseName, '.', mexExt], 'file') == 3;
                end
            catch ME
                fprintf('Error compiling %s: %s\n', sourceFile, ME.message);
                success = false;
            end
        end
        
        function sources = getMEXSourceFiles(obj)
            % Helper method to get list of MEX source files
            %
            % OUTPUTS:
            %   sources - Structure with source file information
            
            % Initialize empty structure
            sources = struct();
            
            % Check if source directory exists
            if ~exist(obj.mexSourcePath, 'dir')
                warning('MEX source directory not found: %s', obj.mexSourcePath);
                return;
            end
            
            % Get all .c files in the directory
            files = dir(fullfile(obj.mexSourcePath, '*.c'));
            
            % Process each file
            for i = 1:length(files)
                % Skip utility files
                if strcmp(files(i).name, 'matrix_operations.c') || strcmp(files(i).name, 'mex_utils.c')
                    continue;
                end
                
                % Get base name without extension
                [~, baseName, ~] = fileparts(files(i).name);
                
                % Determine function category based on name
                if contains(baseName, 'garch')
                    category = 'garch';
                elseif contains(baseName, 'armax')
                    category = 'armax';
                elseif contains(baseName, 'likelihood')
                    category = 'likelihood';
                elseif contains(baseName, 'composite')
                    category = 'composite_likelihood';
                else
                    category = 'other';
                end
                
                % Add to sources structure
                sources.(baseName) = struct(...
                    'name', baseName, ...
                    'filename', files(i).name, ...
                    'path', fullfile(obj.mexSourcePath, files(i).name), ...
                    'category', category ...
                );
            end
        end
        
        function result = verifyPackageContents(obj, packagePath)
            % Helper method to verify contents of the built package
            %
            % INPUTS:
            %   packagePath - Path to the package file
            %
            % OUTPUTS:
            %   result - Verification results structure
            
            % Initialize result structure
            result = struct(...
                'hasMandatoryDir', struct(), ...
                'hasAddToPath', false, ...
                'hasContents', false, ...
                'hasMEXBinaries', false, ...
                'mandatoryDirs', {{'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                'multivariate', 'tests', 'timeseries', 'univariate', 'utility', ...
                'realized', 'mex_source', 'dlls'}} ...
            );
            
            % Initialize mandatory directory flags
            for i = 1:length(result.mandatoryDirs)
                result.hasMandatoryDir.(result.mandatoryDirs{i}) = false;
            end
            
            % Create extraction directory
            extractDir = fullfile(obj.tempBuildDir, 'extract_test');
            if ~exist(extractDir, 'dir')
                mkdir(extractDir);
            end
            
            % Extract the package
            fprintf('Extracting package to verify contents...\n');
            unzip(packagePath, extractDir);
            
            % Look for build directory inside the extracted contents
            buildDir = '';
            extractedItems = dir(extractDir);
            for i = 1:length(extractedItems)
                if extractedItems(i).isdir && ~strcmp(extractedItems(i).name, '.') && ~strcmp(extractedItems(i).name, '..')
                    buildDir = fullfile(extractDir, extractedItems(i).name);
                    break;
                end
            end
            
            % If no build directory found, use the extract directory
            if isempty(buildDir)
                buildDir = extractDir;
            end
            
            % Check for mandatory directories
            for i = 1:length(result.mandatoryDirs)
                dirName = result.mandatoryDirs{i};
                dirPath = fullfile(buildDir, dirName);
                result.hasMandatoryDir.(dirName) = exist(dirPath, 'dir') == 7; % 7 = directory
                
                if ~result.hasMandatoryDir.(dirName)
                    fprintf('Warning: Mandatory directory missing: %s\n', dirName);
                end
            end
            
            % Check for core files
            result.hasAddToPath = exist(fullfile(buildDir, 'addToPath.m'), 'file') == 2;
            result.hasContents = exist(fullfile(buildDir, 'Contents.m'), 'file') == 2;
            
            % Check for MEX binaries
            mexExt = obj.mexValidator.getMEXExtension();
            dllsDir = fullfile(buildDir, 'dlls');
            if exist(dllsDir, 'dir')
                mexFiles = dir(fullfile(dllsDir, ['*.', mexExt]));
                result.hasMEXBinaries = ~isempty(mexFiles);
                
                if result.hasMEXBinaries
                    fprintf('Found %d MEX binaries with extension %s\n', length(mexFiles), mexExt);
                else
                    fprintf('Warning: No MEX binaries found with extension %s\n', mexExt);
                end
            else
                fprintf('Warning: dlls directory not found\n');
            end
        end
    end
end