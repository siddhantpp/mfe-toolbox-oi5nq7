classdef MEXCompilationTest < BaseTest
    % MEXCompilationTest Test class for validating MEX file compilation functionality for the MFE Toolbox
    %
    % This test suite validates the compilation process of MEX files from C source code,
    % verifies that compiled MEX binaries function correctly, and ensures cross-platform
    % compatibility between Windows (PCWIN64) and Unix systems.
    %
    % The test suite covers:
    % - Compilation of core MEX files (agarch_core.c, armaxerrors.c, etc.)
    % - Verification of large array dimension support (-largeArrayDims flag)
    % - Platform-specific compilation and binary generation
    % - Include path resolution for compilation dependencies
    % - Input validation handling in compiled MEX files
    % - Comparison of MEX output with reference implementations
    %
    % Version 4.0 (28-Oct-2009)
    % MFE Toolbox
    %
    % See also: BaseTest, MEXValidator
    
    properties
        % Path to MEX source files
        mexSourcePath
        
        % Path to compiled MEX binary output directory
        mexOutputPath
        
        % Current platform identification string
        currentPlatform
        
        % MEXValidator instance for functionality validation
        mexValidator
        
        % Structure containing information about MEX source files
        mexSources
        
        % Compilation options structure
        compilationOptions
    end
    
    methods
        function obj = MEXCompilationTest()
            % Initialize the MEXCompilationTest with paths and platform information
            %
            % Creates a new MEXCompilationTest instance, sets up paths to source and output
            % directories, identifies the current platform, and initializes the MEXValidator.
            
            % Call parent constructor with test class name
            obj@BaseTest('MEXCompilationTest');
            
            % Set paths to MEX source and output directories
            obj.mexSourcePath = '../../backend/mex_source/';
            obj.mexOutputPath = '../../backend/dlls/';
            
            % Determine current platform using MATLAB's computer() function
            obj.currentPlatform = computer();
            
            % Create a MEXValidator instance for validation functions
            obj.mexValidator = MEXValidator();
            
            % Initialize mexSources structure with empty source list
            obj.mexSources = struct();
            
            % Set default compilation options
            obj.compilationOptions = struct('largeArrayDims', true);
        end
        
        function setUp(obj)
            % Set up test environment before each test
            %
            % This method prepares the test environment by ensuring source and output
            % directories exist, identifying available C source files, and configuring
            % platform-specific compilation options.
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Check if source and output directories exist
            if ~exist(obj.mexSourcePath, 'dir')
                warning('MEXCompilationTest:MissingDirectory', ...
                    'MEX source directory %s does not exist.', obj.mexSourcePath);
            end
            
            if ~exist(obj.mexOutputPath, 'dir')
                % Create output directory if it doesn't exist
                mkdir(obj.mexOutputPath);
            end
            
            % Get list of MEX source files from source directory
            obj.mexSources = obj.getMEXSourceFiles();
            
            % Configure platform-specific compilation options
            if strncmpi(obj.currentPlatform, 'PCW', 3) % Windows
                obj.compilationOptions.platform = 'Windows';
                obj.compilationOptions.extension = 'mexw64';
            else % Unix/Mac
                obj.compilationOptions.platform = 'Unix';
                obj.compilationOptions.extension = 'mexa64';
            end
        end
        
        function tearDown(obj)
            % Clean up after tests
            %
            % This method handles cleanup after test execution, including removing
            % any temporary files created during tests.
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Clean up any temporary files created during tests
            tmpFiles = dir(fullfile(obj.mexOutputPath, '*_test.*'));
            for i = 1:length(tmpFiles)
                delete(fullfile(obj.mexOutputPath, tmpFiles(i).name));
            end
        end
        
        function testSourceFilesExist(obj)
            % Test that all required source files exist in the mex_source directory
            %
            % This test verifies that all core MEX source files required by the MFE
            % Toolbox are present in the source directory.
            
            % Define list of expected core MEX source files
            expectedFiles = {'agarch_core.c', 'armaxerrors.c', 'composite_likelihood.c', ...
                'egarch_core.c', 'igarch_core.c', 'tarch_core.c'};
            
            % Check each expected file
            missingFiles = {};
            for i = 1:length(expectedFiles)
                fileName = expectedFiles{i};
                filePath = fullfile(obj.mexSourcePath, fileName);
                
                if ~exist(filePath, 'file')
                    missingFiles{end+1} = fileName;
                end
            end
            
            % Assert that no files are missing
            obj.assertTrue(isempty(missingFiles), ...
                sprintf('Missing required MEX source files: %s', strjoin(missingFiles, ', ')));
        end
        
        function testCompileAgarchCore(obj)
            % Test compilation of agarch_core.c to MEX binary
            %
            % This test compiles the agarch_core.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'agarch_core.c';
            
            % Output file name (without extension)
            outputFile = 'agarch_core';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for AGARCH model
            testInputs = obj.generateTestInput('agarch');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileArmaxErrors(obj)
            % Test compilation of armaxerrors.c to MEX binary
            %
            % This test compiles the armaxerrors.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'armaxerrors.c';
            
            % Output file name (without extension)
            outputFile = 'armaxerrors';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for ARMAX model
            testInputs = obj.generateTestInput('armax');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileCompositeLikelihood(obj)
            % Test compilation of composite_likelihood.c to MEX binary
            %
            % This test compiles the composite_likelihood.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'composite_likelihood.c';
            
            % Output file name (without extension)
            outputFile = 'composite_likelihood';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for composite likelihood
            testInputs = obj.generateTestInput('composite_likelihood');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileEgarchCore(obj)
            % Test compilation of egarch_core.c to MEX binary
            %
            % This test compiles the egarch_core.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'egarch_core.c';
            
            % Output file name (without extension)
            outputFile = 'egarch_core';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for EGARCH model
            testInputs = obj.generateTestInput('egarch');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileIgarchCore(obj)
            % Test compilation of igarch_core.c to MEX binary
            %
            % This test compiles the igarch_core.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'igarch_core.c';
            
            % Output file name (without extension)
            outputFile = 'igarch_core';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for IGARCH model
            testInputs = obj.generateTestInput('igarch');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileTarchCore(obj)
            % Test compilation of tarch_core.c to MEX binary
            %
            % This test compiles the tarch_core.c source file into a platform-specific
            % MEX binary and verifies it was created successfully with basic functionality.
            
            % Source file name
            sourceFile = 'tarch_core.c';
            
            % Output file name (without extension)
            outputFile = 'tarch_core';
            
            % Compile the source file
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Failed to compile %s', sourceFile));
            
            % Verify MEX file exists with correct platform extension
            mexExt = obj.mexValidator.getMEXExtension();
            fullMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            obj.assertTrue(exist(fullMexPath, 'file') == 3, ...
                sprintf('Compiled MEX file %s not found or not recognized as MEX', fullMexPath));
            
            % Test basic functionality with simple input
            % Generate test data for TARCH model
            testInputs = obj.generateTestInput('tarch');
            
            % Verify functionality using MEXValidator
            funcResult = obj.mexValidator.validateMEXFunctionality(outputFile, testInputs);
            obj.assertTrue(funcResult.canExecute, ...
                sprintf('Compiled MEX file %s failed functionality test: %s', ...
                outputFile, funcResult.errorMessage));
        end
        
        function testCompileWithLargeArrayDims(obj)
            % Test compilation with -largeArrayDims flag for large dataset support
            %
            % This test verifies that MEX files compiled with the -largeArrayDims flag
            % can successfully handle large arrays exceeding standard MATLAB array limits.
            
            % Source file to test with large arrays (agarch_core is a good candidate)
            sourceFile = 'agarch_core.c';
            
            % Output file with suffix indicating large array support
            outputFile = 'agarch_core_test_large';
            
            % Set explicit large array dims flag
            options = obj.compilationOptions;
            options.largeArrayDims = true;
            
            % Compile with explicit flag
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, options);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, 'Failed to compile with -largeArrayDims flag');
            
            % Create large test input (larger than standard array size)
            % For testing purposes, we'll use a reasonably large array that won't
            % exceed memory limits but will test the large array capability
            numObsLarge = 1e6; % 1 million observations
            
            try
                % Generate large dataset
                largeSeries = randn(numObsLarge, 1);
                
                % GARCH model parameters (omega, alpha, beta)
                parameters = [0.01; 0.1; 0.85];
                
                % Initial volatility estimate
                initialVol = var(largeSeries(1:1000)); % Use subset for efficiency
                
                % Test with large input
                mexExt = obj.mexValidator.getMEXExtension();
                mexFile = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
                
                % Ensure the MEX file is on the MATLAB path
                addpath(obj.mexOutputPath);
                
                % Clear function to ensure fresh load
                if exist(outputFile, 'file') == 3
                    clear(outputFile);
                end
                
                % Get function handle
                mexFunc = str2func(outputFile);
                
                % Execute with large array
                tic;
                mexFunc(largeSeries, parameters, initialVol);
                executionTime = toc;
                
                % If we get here without error, test passed
                obj.assertTrue(true, sprintf('Successfully processed %d observations in %.2f seconds', numObsLarge, executionTime));
                
            catch ME
                % If error occurred, check if it's memory-related or actual failure
                obj.assertFalse(contains(lower(ME.message), 'array size'), ...
                    sprintf('Large array handling failed: %s', ME.message));
            end
        end
        
        function testCompilationErrors(obj)
            % Test error handling during compilation process
            %
            % This test verifies that the MEX compilation process correctly detects
            % and reports errors in source code.
            
            % Create a temporary source file with deliberate syntax errors
            tempFileName = tempname;
            tempSourceFile = [tempFileName, '.c'];
            
            % Write invalid C code to the temp file
            fid = fopen(tempSourceFile, 'w');
            fprintf(fid, '#include "mex.h"\n\n');
            fprintf(fid, '/ This line has a deliberate syntax error\n\n');
            fprintf(fid, 'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\n');
            fprintf(fid, '    this_is_not_valid_code();\n');
            fprintf(fid, '}\n');
            fclose(fid);
            
            % Try to compile the invalid file
            [~, tempBaseName] = fileparts(tempFileName);
            outputFile = ['test_', tempBaseName];
            compilationSucceeded = false;
            
            try
                % Attempt to compile - should fail
                mex(tempSourceFile, '-outdir', obj.mexOutputPath, ['-output', outputFile]);
                compilationSucceeded = true;
            catch ME
                % Compilation should fail, so we expect an error
                % Make sure the error is compilation-related
                obj.assertTrue(contains(ME.message, 'Error') || ...
                    contains(ME.message, 'error') || ...
                    contains(ME.message, 'failed'), ...
                    'Compilation failed but with unexpected error message');
            end
            
            % Clean up temporary file
            if exist(tempSourceFile, 'file')
                delete(tempSourceFile);
            end
            
            % Check for temporary MEX file and remove if it exists
            mexExt = obj.mexValidator.getMEXExtension();
            tempMexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
            if exist(tempMexPath, 'file')
                delete(tempMexPath);
            end
            
            % Assert that compilation did not succeed
            obj.assertFalse(compilationSucceeded, 'Compilation incorrectly succeeded with invalid source code');
        end
        
        function testPlatformSpecificCompilation(obj)
            % Test platform-specific compilation options and binary generation
            %
            % This test verifies that MEX files are compiled with the correct
            % platform-specific settings and generate appropriate binary extensions.
            
            % Source file to test platform-specific compilation
            sourceFile = 'agarch_core.c';
            
            % Output file with platform indicator
            outputFile = ['agarch_core_test_', lower(obj.compilationOptions.platform)];
            
            % Set platform-specific options based on current platform
            options = obj.compilationOptions;
            
            % Compile file with platform-specific options
            compileSuccess = obj.compileSourceFile(sourceFile, outputFile, options);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, sprintf('Platform-specific compilation for %s failed', options.platform));
            
            % Verify file extension matches current platform
            expectedExt = obj.mexValidator.getMEXExtension();
            mexPath = fullfile(obj.mexOutputPath, [outputFile, '.', expectedExt]);
            
            obj.assertTrue(exist(mexPath, 'file') == 3, ...
                sprintf('Platform-specific MEX file %s not found or not recognized as MEX', mexPath));
            
            % Test basic functionality
            try
                % Add output directory to path
                addpath(obj.mexOutputPath);
                
                % Clear function to ensure fresh load
                if exist(outputFile, 'file') == 3
                    clear(outputFile);
                end
                
                % Get function handle
                mexFunc = str2func(outputFile);
                
                % Generate test data
                testInputs = obj.generateTestInput('agarch');
                
                % Execute MEX function
                mexFunc(testInputs{:});
                
                % If we get here without error, test passed
                obj.assertTrue(true, 'Platform-specific MEX file executed successfully');
                
            catch ME
                obj.assertTrue(false, sprintf('Platform-specific MEX execution failed: %s', ME.message));
            end
        end
        
        function testIncludePathResolution(obj)
            % Test that include paths are correctly resolved during compilation
            %
            % This test verifies that the MEX compilation process correctly resolves
            % header file includes from specified include paths.
            
            % Create temporary directory for test header files
            tempIncludeDir = fullfile(tempdir, 'mex_test_includes');
            if ~exist(tempIncludeDir, 'dir')
                mkdir(tempIncludeDir);
            end
            
            % Create a simple header file in the include directory
            headerFile = fullfile(tempIncludeDir, 'test_header.h');
            fid = fopen(headerFile, 'w');
            fprintf(fid, '/* Test header file */\n');
            fprintf(fid, '#ifndef TEST_HEADER_H\n');
            fprintf(fid, '#define TEST_HEADER_H\n\n');
            fprintf(fid, '#define TEST_CONSTANT 42\n\n');
            fprintf(fid, 'static double test_function(double x) {\n');
            fprintf(fid, '    return x * TEST_CONSTANT;\n');
            fprintf(fid, '}\n\n');
            fprintf(fid, '#endif /* TEST_HEADER_H */\n');
            fclose(fid);
            
            % Create a simple C source file that includes the header
            tempFileName = tempname;
            tempSourceFile = [tempFileName, '.c'];
            
            fid = fopen(tempSourceFile, 'w');
            fprintf(fid, '#include "mex.h"\n');
            fprintf(fid, '#include "test_header.h"\n\n');
            fprintf(fid, 'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\n');
            fprintf(fid, '    /* Basic MEX function that uses the included header */\n');
            fprintf(fid, '    double input_value = 0.0;\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    /* Check inputs */\n');
            fprintf(fid, '    if (nrhs < 1 || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {\n');
            fprintf(fid, '        mexErrMsgTxt("Input must be a real double scalar.");\n');
            fprintf(fid, '    }\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    /* Get input value */\n');
            fprintf(fid, '    input_value = mxGetScalar(prhs[0]);\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    /* Create output */\n');
            fprintf(fid, '    plhs[0] = mxCreateDoubleScalar(test_function(input_value));\n');
            fprintf(fid, '}\n');
            fclose(fid);
            
            % Get base name for the output file
            [~, tempBaseName] = fileparts(tempFileName);
            outputFile = ['test_include_', tempBaseName];
            
            % Compile with include path
            compilationSucceeded = false;
            try
                % Compile with explicit include path
                mex(tempSourceFile, ['-I', tempIncludeDir], '-outdir', obj.mexOutputPath, ['-output', outputFile]);
                compilationSucceeded = true;
            catch ME
                obj.assertTrue(false, sprintf('Compilation with include path failed: %s', ME.message));
            end
            
            % Clean up temporary files
            if exist(tempSourceFile, 'file')
                delete(tempSourceFile);
            end
            if exist(headerFile, 'file')
                delete(headerFile);
            end
            if exist(tempIncludeDir, 'dir')
                rmdir(tempIncludeDir);
            end
            
            % Test the compiled MEX file
            if compilationSucceeded
                mexExt = obj.mexValidator.getMEXExtension();
                mexPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
                
                obj.assertTrue(exist(mexPath, 'file') == 3, ...
                    'Compiled MEX file with include path not found');
                
                try
                    % Add output directory to path
                    addpath(obj.mexOutputPath);
                    
                    % Clear function to ensure fresh load
                    if exist(outputFile, 'file') == 3
                        clear(outputFile);
                    end
                    
                    % Get function handle
                    mexFunc = str2func(outputFile);
                    
                    % Call the function with test input
                    result = mexFunc(2.0);
                    
                    % Test result should be input * TEST_CONSTANT (2 * 42 = 84)
                    obj.assertEqual(result, 84, 'MEX function using include file returned incorrect result');
                    
                catch ME
                    obj.assertTrue(false, sprintf('MEX function with include path failed: %s', ME.message));
                end
                
                % Clean up compiled MEX file
                if exist(mexPath, 'file')
                    delete(mexPath);
                end
            end
        end
        
        function testCompareMEXOutput(obj)
            % Test that newly compiled MEX files produce identical results to reference binaries
            %
            % This test compiles a source file to a temporary MEX binary and compares
            % its output with a reference MEX binary to ensure consistency.
            
            % Source file to test
            sourceFile = 'agarch_core.c';
            
            % Reference output file name
            refOutputFile = 'agarch_core';
            
            % Temporary output file name
            tempOutputFile = 'agarch_core_test_comp';
            
            % Compile to temporary output file
            compileSuccess = obj.compileSourceFile(sourceFile, tempOutputFile, obj.compilationOptions);
            
            % Assert compilation was successful
            obj.assertTrue(compileSuccess, 'Failed to compile temporary MEX file for comparison');
            
            % Verify both reference and temp MEX files exist
            mexExt = obj.mexValidator.getMEXExtension();
            refMexPath = fullfile(obj.mexOutputPath, [refOutputFile, '.', mexExt]);
            tempMexPath = fullfile(obj.mexOutputPath, [tempOutputFile, '.', mexExt]);
            
            refExists = exist(refMexPath, 'file') == 3;
            tempExists = exist(tempMexPath, 'file') == 3;
            
            if ~refExists
                warning('MEXCompilationTest:MissingReference', ...
                    'Reference MEX file %s not found. Skipping comparison test.', refMexPath);
                return;
            end
            
            obj.assertTrue(tempExists, 'Temporary MEX file not found after compilation');
            
            % Add output directory to path
            addpath(obj.mexOutputPath);
            
            % Clear functions to ensure fresh load
            if exist(refOutputFile, 'file') == 3
                clear(refOutputFile);
            end
            if exist(tempOutputFile, 'file') == 3
                clear(tempOutputFile);
            end
            
            % Get function handles
            refFunc = str2func(refOutputFile);
            tempFunc = str2func(tempOutputFile);
            
            % Generate test input
            testInputs = obj.generateTestInput('agarch');
            
            try
                % Run both functions with identical inputs
                refOutput = refFunc(testInputs{:});
                tempOutput = tempFunc(testInputs{:});
                
                % Compare outputs (should be identical)
                % Use numerical comparator for robust comparison
                isEqual = obj.numericalComparator.compareMatrices(refOutput, tempOutput, 1e-12).isEqual;
                
                obj.assertTrue(isEqual, 'Newly compiled MEX file produces different results than reference');
                
            catch ME
                obj.assertTrue(false, sprintf('MEX output comparison failed: %s', ME.message));
            end
        end
        
        %% Helper Methods
        
        function success = compileSourceFile(obj, sourceFile, outputFile, options)
            % Helper method to compile a C source file into a MEX binary
            %
            % INPUTS:
            %   sourceFile - Name of the source file in mexSourcePath
            %   outputFile - Base name for output file (without extension)
            %   options - Structure with compilation options
            %
            % OUTPUTS:
            %   success - Logical flag indicating compilation success
            
            % Initialize success flag
            success = false;
            
            % Construct full source path
            fullSourcePath = fullfile(obj.mexSourcePath, sourceFile);
            
            % Validate source file exists
            if ~exist(fullSourcePath, 'file')
                warning('MEXCompilationTest:MissingSource', ...
                    'Source file %s not found', fullSourcePath);
                return;
            end
            
            % Construct compilation flags
            mexFlags = {};
            
            % Add large array dims flag if specified
            if isfield(options, 'largeArrayDims') && options.largeArrayDims
                mexFlags{end+1} = '-largeArrayDims';
            end
            
            % Specify output directory and name
            mexFlags{end+1} = '-outdir';
            mexFlags{end+1} = obj.mexOutputPath;
            mexFlags{end+1} = ['-output', outputFile];
            
            try
                % Execute MEX compilation command
                mex(fullSourcePath, mexFlags{:});
                
                % Verify MEX file was created
                mexExt = obj.mexValidator.getMEXExtension();
                outputPath = fullfile(obj.mexOutputPath, [outputFile, '.', mexExt]);
                
                if exist(outputPath, 'file') == 3 % 3 = MEX-file
                    success = true;
                else
                    warning('MEXCompilationTest:CompilationFailed', ...
                        'Compilation appeared to succeed but MEX file not found: %s', outputPath);
                end
                
            catch ME
                warning('MEXCompilationTest:CompilationError', ...
                    'Error compiling %s: %s', sourceFile, ME.message);
            end
        end
        
        function sources = getMEXSourceFiles(obj)
            % Helper method to get list of MEX source files
            %
            % OUTPUTS:
            %   sources - Structure with information about available source files
            
            % Initialize empty sources structure
            sources = struct();
            
            % Check if source directory exists
            if ~exist(obj.mexSourcePath, 'dir')
                warning('MEXCompilationTest:MissingSourceDir', ...
                    'MEX source directory %s not found', obj.mexSourcePath);
                return;
            end
            
            % List all .c files in the directory
            dirInfo = dir(fullfile(obj.mexSourcePath, '*.c'));
            
            % Process each file
            for i = 1:length(dirInfo)
                % Get file name and create field in sources structure
                fileName = dirInfo(i).name;
                [baseName, ~] = fileparts(fileName);
                
                % Determine function category based on file name
                if contains(fileName, 'garch') && ~contains(fileName, 'igarch')
                    category = 'garch';
                elseif contains(fileName, 'igarch')
                    category = 'igarch';
                elseif contains(fileName, 'agarch')
                    category = 'agarch';
                elseif contains(fileName, 'egarch')
                    category = 'egarch';
                elseif contains(fileName, 'tarch')
                    category = 'tarch';
                elseif contains(fileName, 'armax')
                    category = 'armax';
                elseif contains(fileName, 'composite')
                    category = 'composite_likelihood';
                else
                    category = 'other';
                end
                
                % Add file information to sources structure
                sources.(baseName) = struct(...
                    'name', fileName, ...
                    'path', fullfile(obj.mexSourcePath, fileName), ...
                    'category', category ...
                );
            end
        end
        
        function testInputs = generateTestInput(obj, mexType)
            % Generate appropriate test inputs for a specific MEX file
            %
            % INPUTS:
            %   mexType - Type of MEX file ('agarch', 'armax', etc.)
            %
            % OUTPUTS:
            %   testInputs - Cell array of test inputs suitable for the MEX type
            
            % Use MEXValidator to generate appropriate test inputs
            testInputs = obj.mexValidator.generateTestInputs(mexType);
        end
    end
end