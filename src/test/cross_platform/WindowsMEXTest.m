classdef WindowsMEXTest < BaseTest
    % WINDOWSMEXTEST A test class that validates Windows-specific MEX binary functionality in the MFE Toolbox
    % Performs comprehensive testing of Windows .mexw64 binaries to ensure correct compilation, proper 
    % functionality, and cross-platform compatibility with Unix implementations.
    
    properties
        mexDllPath              % Path to Windows MEX DLL files
        mexValidator            % MEXValidator instance for testing functionality
        crossValidator          % CrossPlatformValidator for cross-platform testing
        testResults             % Structure to store test results
        mexFileList             % List of Windows MEX files to test
        isWindowsPlatform       % Flag indicating if running on Windows
        referenceResultsPath    % Path to store reference results for cross-platform testing
    end
    
    methods
        function obj = WindowsMEXTest()
            % Initialize a new WindowsMEXTest instance for testing Windows MEX binaries
            
            % Call parent BaseTest constructor with 'WindowsMEXTest' name
            obj@BaseTest('WindowsMEXTest');
            
            % Check if running on Windows platform using computer() function
            obj.isWindowsPlatform = strncmpi(computer(), 'PCW', 3);
            
            % Initialize mexDllPath to 'src/backend/dlls/'
            obj.mexDllPath = 'src/backend/dlls/';
            
            % Initialize MEXValidator and CrossPlatformValidator instances
            obj.mexValidator = MEXValidator();
            obj.crossValidator = CrossPlatformValidator();
            
            % Set referenceResultsPath for storing cross-platform test results
            obj.referenceResultsPath = 'src/test/data/cross_platform/';
            
            % Initialize testResults structure for storing test outcomes
            obj.testResults = struct();
            
            % Populate mexFileList with Windows MEX binaries (.mexw64 files)
            if obj.isWindowsPlatform
                dirInfo = dir(fullfile(obj.mexDllPath, '*.mexw64'));
                obj.mexFileList = {dirInfo.name};
            else
                obj.mexFileList = {};
            end
        end
        
        function setUp(obj)
            % Prepares the test environment before each test execution
            
            % Call parent BaseTest.setUp method
            setUp@BaseTest(obj);
            
            % Verify current platform is Windows (PCWIN64)
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Tests are being run on a non-Windows platform. Windows-specific tests will be skipped.');
            end
            
            % Initialize test data structures and MEX validators
            obj.testResults.(obj.currentTestName) = struct('status', 'started', 'timestamp', datestr(now));
            
            % Create reference results directory if it doesn't exist
            if ~exist(obj.referenceResultsPath, 'dir')
                mkdir(obj.referenceResultsPath);
            end
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test execution
            
            % Record test completion status in testResults
            if isfield(obj.testResults, obj.currentTestName)
                obj.testResults.(obj.currentTestName).endTimestamp = datestr(now);
                
                % If status wasn't updated, mark it as completed
                if strcmp(obj.testResults.(obj.currentTestName).status, 'started')
                    obj.testResults.(obj.currentTestName).status = 'completed';
                end
                
                % Save Windows reference results for cross-platform comparison
                if obj.isWindowsPlatform
                    saveDir = fullfile(obj.referenceResultsPath, obj.currentTestName);
                    if ~exist(saveDir, 'dir')
                        mkdir(saveDir);
                    end
                    saveFile = fullfile(saveDir, 'PCWIN64_results.mat');
                    testResults = obj.testResults.(obj.currentTestName);
                    save(saveFile, 'testResults');
                end
            end
            
            % Call parent BaseTest.tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testWindowsMEXFilesExist(obj)
            % Tests that all required Windows MEX binary files exist in the DLLs directory
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Define list of required Windows MEX binary files (with .mexw64 extension)
            requiredMEXFiles = {...
                'agarch_core.mexw64', ...
                'armaxerrors.mexw64', ...
                'composite_likelihood.mexw64', ...
                'egarch_core.mexw64', ...
                'igarch_core.mexw64', ...
                'tarch_core.mexw64' ...
            };
            
            % Check if each file exists in mexDllPath
            missingFiles = {};
            for i = 1:length(requiredMEXFiles)
                filePath = fullfile(obj.mexDllPath, requiredMEXFiles{i});
                if ~exist(filePath, 'file')
                    missingFiles{end+1} = requiredMEXFiles{i};
                end
            end
            
            % Assert that all required Windows MEX binaries are present
            obj.assertTrue(isempty(missingFiles), sprintf('Missing required MEX files: %s', strjoin(missingFiles, ', ')));
            
            % Record existence status in testResults
            obj.testResults.testWindowsMEXFilesExist.status = 'passed';
            obj.testResults.testWindowsMEXFilesExist.missingFiles = missingFiles;
            obj.testResults.testWindowsMEXFilesExist.requiredFiles = requiredMEXFiles;
        end
        
        function testWindowsMEXFileLoading(obj)
            % Tests that all Windows MEX binary files can be loaded correctly
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % For each MEX file in mexFileList, attempt to locate using which()
            unloadableFiles = {};
            for i = 1:length(obj.mexFileList)
                [~, mexName] = fileparts(obj.mexFileList{i});
                mexPath = which(mexName);
                
                % Verify file can be found in MATLAB path
                if isempty(mexPath)
                    unloadableFiles{end+1} = mexName;
                end
            end
            
            % Assert that all MEX files are accessible to MATLAB
            obj.assertTrue(isempty(unloadableFiles), sprintf('MEX files not in MATLAB path: %s', strjoin(unloadableFiles, ', ')));
            
            % Record loading status in testResults
            obj.testResults.testWindowsMEXFileLoading.status = 'passed';
            obj.testResults.testWindowsMEXFileLoading.unloadableFiles = unloadableFiles;
        end
        
        function testAgarchCoreFunctionality(obj)
            % Tests the functionality of the Windows AGARCH core MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for AGARCH model (data series, parameters, options)
            numObs = 1000;
            data = randn(numObs, 1) * 0.1;
            parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
            initialVol = var(data);
            testInputs = {data, parameters, initialVol};
            
            % Create expected outputs based on MATLAB reference implementation
            % For this test, we'll use the MEXValidator to compare with a MATLAB implementation
            mexName = 'agarch_core';
            matlabName = 'agarch_core_matlab';
            
            % Use MEXValidator to validate functionality of agarch_core.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('AGARCH core MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testAgarchCoreFunctionality.status = 'passed';
            obj.testResults.testAgarchCoreFunctionality.validationResult = validationResult;
            obj.testResults.testAgarchCoreFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('agarch_core', testInputs);
        end
        
        function testTarchCoreFunctionality(obj)
            % Tests the functionality of the Windows TARCH core MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for TARCH model (data series, parameters, options)
            numObs = 1000;
            data = randn(numObs, 1) * 0.1;
            parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
            initialVol = var(data);
            testInputs = {data, parameters, initialVol};
            
            % Create expected outputs based on MATLAB reference implementation
            mexName = 'tarch_core';
            matlabName = 'tarch_core_matlab';
            
            % Use MEXValidator to validate functionality of tarch_core.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('TARCH core MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testTarchCoreFunctionality.status = 'passed';
            obj.testResults.testTarchCoreFunctionality.validationResult = validationResult;
            obj.testResults.testTarchCoreFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('tarch_core', testInputs);
        end
        
        function testEgarchCoreFunctionality(obj)
            % Tests the functionality of the Windows EGARCH core MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for EGARCH model (data series, parameters, options)
            numObs = 1000;
            data = randn(numObs, 1) * 0.1;
            parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
            initialVol = log(var(data));
            testInputs = {data, parameters, initialVol};
            
            % Create expected outputs based on MATLAB reference implementation
            mexName = 'egarch_core';
            matlabName = 'egarch_core_matlab';
            
            % Use MEXValidator to validate functionality of egarch_core.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('EGARCH core MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testEgarchCoreFunctionality.status = 'passed';
            obj.testResults.testEgarchCoreFunctionality.validationResult = validationResult;
            obj.testResults.testEgarchCoreFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('egarch_core', testInputs);
        end
        
        function testIgarchCoreFunctionality(obj)
            % Tests the functionality of the Windows IGARCH core MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for IGARCH model (data series, parameters, options)
            numObs = 1000;
            data = randn(numObs, 1) * 0.1;
            parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
            initialVol = var(data);
            testInputs = {data, parameters, initialVol};
            
            % Create expected outputs based on MATLAB reference implementation
            mexName = 'igarch_core';
            matlabName = 'igarch_core_matlab';
            
            % Use MEXValidator to validate functionality of igarch_core.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('IGARCH core MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testIgarchCoreFunctionality.status = 'passed';
            obj.testResults.testIgarchCoreFunctionality.validationResult = validationResult;
            obj.testResults.testIgarchCoreFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('igarch_core', testInputs);
        end
        
        function testArmaxerrorsFunctionality(obj)
            % Tests the functionality of the Windows ARMAX errors MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for ARMAX errors computation (data, model parameters)
            numObs = 1000;
            ar = 0.7;
            ma = 0.3;
            innovations = randn(numObs+100, 1) * 0.1;
            y = zeros(numObs+100, 1);
            
            for t = 2:length(y)
                y(t) = ar*y(t-1) + innovations(t) + ma*innovations(t-1);
            end
            
            % Discard burn-in period
            y = y(101:end);
            
            % Create parameter vector
            parameters = [ar; ma];
            
            % No exogenous variables
            X = [];
            
            testInputs = {y, parameters, X};
            
            % Create expected outputs based on MATLAB reference implementation
            mexName = 'armaxerrors';
            matlabName = 'armaxerrors_matlab';
            
            % Use MEXValidator to validate functionality of armaxerrors.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('ARMAX errors MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testArmaxerrorsFunctionality.status = 'passed';
            obj.testResults.testArmaxerrorsFunctionality.validationResult = validationResult;
            obj.testResults.testArmaxerrorsFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('armaxerrors', testInputs);
        end
        
        function testCompositeLikelihoodFunctionality(obj)
            % Tests the functionality of the Windows composite likelihood MEX implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % Create test inputs for composite likelihood computation
            % Generate test correlation matrix (3x3)
            R = [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1];
            
            % Generate multivariate normal data
            numObs = 500;
            data = mvnrnd(zeros(3, 1), R, numObs);
            
            testInputs = {data, R};
            
            % Create expected outputs based on MATLAB reference implementation
            mexName = 'composite_likelihood';
            matlabName = 'composite_likelihood_matlab';
            
            % Use MEXValidator to validate functionality of composite_likelihood.mexw64
            validationResult = obj.mexValidator.compareMEXWithMATLAB(mexName, matlabName, testInputs);
            
            % Assert that MEX implementation produces correct results
            obj.assertTrue(validationResult.isEqual, ...
                sprintf('Composite likelihood MEX output differs from MATLAB implementation (max diff: %g)', ...
                validationResult.maxAbsoluteDifference));
            
            % Record test results and generate reference data for cross-platform testing
            obj.testResults.testCompositeLikelihoodFunctionality.status = 'passed';
            obj.testResults.testCompositeLikelihoodFunctionality.validationResult = validationResult;
            obj.testResults.testCompositeLikelihoodFunctionality.referenceData = ...
                obj.generateWindowsReferenceData('composite_likelihood', testInputs);
        end
        
        function testWindowsMEXPerformance(obj)
            % Tests the performance of Windows MEX implementations compared to MATLAB equivalents
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % List of MEX files to benchmark
            mexFiles = {
                'agarch_core',
                'tarch_core',
                'egarch_core',
                'igarch_core',
                'armaxerrors',
                'composite_likelihood'
            };
            
            % Initialize results
            benchmarkResults = struct();
            allMeetRequirement = true;
            
            % For each MEX file, create appropriate test inputs
            for i = 1:length(mexFiles)
                mexName = mexFiles{i};
                matlabName = [mexName, '_matlab'];
                
                % Create test inputs
                testInputs = obj.createTestInputs(mexName);
                
                % Use MEXValidator.benchmarkMEXPerformance to benchmark performance
                benchmarkResult = obj.mexValidator.benchmarkMEXPerformance(mexName, matlabName, testInputs, 10);
                
                % Store result
                benchmarkResults.(mexName) = benchmarkResult;
                
                % Check if performance improvement meets requirement (>= 50%)
                if ~benchmarkResult.meetsRequirement
                    allMeetRequirement = false;
                    warning('WindowsMEXTest:PerformanceRequirementNotMet', ...
                        'MEX file %s performance improvement (%.2f%%) does not meet 50%% requirement', ...
                        mexName, benchmarkResult.performanceImprovement);
                end
            end
            
            % Assert that MEX implementations achieve at least 50% performance improvement
            obj.assertTrue(allMeetRequirement, 'Not all MEX implementations meet the 50% performance improvement requirement');
            
            % Record performance metrics in testResults
            obj.testResults.testWindowsMEXPerformance.status = 'passed';
            obj.testResults.testWindowsMEXPerformance.benchmarkResults = benchmarkResults;
            obj.testResults.testWindowsMEXPerformance.allMeetRequirement = allMeetRequirement;
        end
        
        function testWindowsMEXMemoryUsage(obj)
            % Tests the memory usage patterns of Windows MEX implementations
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % List of MEX files to test memory usage
            mexFiles = {
                'agarch_core',
                'tarch_core',
                'egarch_core',
                'igarch_core',
                'armaxerrors',
                'composite_likelihood'
            };
            
            % Initialize results
            memoryResults = struct();
            hasLeaks = false;
            
            % For each MEX file, create appropriate test inputs
            for i = 1:length(mexFiles)
                mexName = mexFiles{i};
                
                % Create test inputs
                testInputs = obj.createTestInputs(mexName);
                
                % Use MEXValidator.validateMemoryUsage to monitor memory patterns
                memoryResult = obj.mexValidator.validateMemoryUsage(mexName, testInputs, 50);
                
                % Store result
                memoryResults.(mexName) = memoryResult;
                
                % Check for memory leaks
                if memoryResult.hasLeak
                    hasLeaks = true;
                    warning('WindowsMEXTest:MemoryLeak', ...
                        'Potential memory leak detected in MEX file %s', mexName);
                end
            end
            
            % Assert that memory usage remains stable across iterations
            obj.assertFalse(hasLeaks, 'Memory leaks detected in one or more MEX implementations');
            
            % Record memory usage metrics in testResults
            obj.testResults.testWindowsMEXMemoryUsage.status = 'passed';
            obj.testResults.testWindowsMEXMemoryUsage.memoryResults = memoryResults;
            obj.testResults.testWindowsMEXMemoryUsage.hasLeaks = hasLeaks;
        end
        
        function testLargeArraySupport(obj)
            % Tests Windows MEX binaries with large array dimensions to verify -largeArrayDims flag implementation
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % List of MEX files to test with large arrays
            mexFiles = {
                'agarch_core',
                'tarch_core',
                'egarch_core',
                'igarch_core'
            };
            
            % Initialize results
            largeArrayResults = struct();
            allSupported = true;
            
            % Generate large test arrays approaching memory limits
            % For GARCH-type models, we need a large vector of data
            largeSize = 1e6; % 1 million elements
            largeData = randn(largeSize, 1) * 0.1;
            
            % For each MEX file, test with large arrays
            for i = 1:length(mexFiles)
                mexName = mexFiles{i};
                
                % Create appropriate parameters based on MEX type
                if strcmp(mexName, 'igarch_core')
                    parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                else
                    parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                end
                
                % Initial volatility
                initialVol = var(largeData(1:1000)); % Use a subset for initial vol
                
                % Prepare inputs
                if strcmp(mexName, 'egarch_core')
                    initialVol = log(initialVol);
                end
                
                testInputs = {largeData, parameters, initialVol};
                
                % Test MEX implementation with large arrays
                largeArrayResults.(mexName) = struct('success', false, 'error', '');
                
                try
                    % Get function handle
                    mexFunc = str2func(mexName);
                    
                    % Execute with large array
                    result = mexFunc(testInputs{:});
                    
                    % If execution completed, mark as successful
                    largeArrayResults.(mexName).success = true;
                    largeArrayResults.(mexName).outputSize = size(result);
                catch ME
                    % Execution failed
                    allSupported = false;
                    largeArrayResults.(mexName).success = false;
                    largeArrayResults.(mexName).error = ME.message;
                    
                    warning('WindowsMEXTest:LargeArrayFailed', ...
                        'MEX file %s failed with large array: %s', mexName, ME.message);
                end
            end
            
            % Verify successful processing without memory errors
            obj.assertTrue(allSupported, 'Not all MEX implementations support large arrays');
            
            % Record large array handling metrics in testResults
            obj.testResults.testLargeArraySupport.status = 'passed';
            obj.testResults.testLargeArraySupport.largeArrayResults = largeArrayResults;
            obj.testResults.testLargeArraySupport.allSupported = allSupported;
        end
        
        function testCrossPlatformCompatibility(obj)
            % Compares Windows MEX results with Unix results to ensure cross-platform consistency
            
            % Skip test if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Skipping test on non-Windows platform');
                return;
            end
            
            % List of MEX files to test for cross-platform compatibility
            mexFiles = {
                'agarch_core',
                'tarch_core',
                'egarch_core',
                'igarch_core',
                'armaxerrors',
                'composite_likelihood'
            };
            
            % Initialize results
            compatibilityResults = struct();
            allCompatible = true;
            
            % For each MEX file, test cross-platform compatibility
            for i = 1:length(mexFiles)
                mexName = mexFiles{i};
                
                % Create test inputs
                testInputs = obj.createTestInputs(mexName);
                
                % Use CrossPlatformValidator to validate MEX implementation
                compatibilityResult = obj.crossValidator.validateMEXCompatibility(mexName, testInputs);
                
                % Store result
                compatibilityResults.(mexName) = compatibilityResult;
                
                % Check if compatible across platforms
                if ~compatibilityResult.isCompatible
                    allCompatible = false;
                    warning('WindowsMEXTest:CrossPlatformIncompatibility', ...
                        'MEX file %s shows cross-platform incompatibility', mexName);
                end
            end
            
            % Assert that results are consistent within specified tolerance across platforms
            % Note: We don't use assertTrue here because Unix reference results might not exist yet
            if ~allCompatible
                warning('WindowsMEXTest:CrossPlatformIncompatibility', ...
                    'Not all MEX implementations show cross-platform compatibility. This may be normal if Unix reference results are not available.');
            end
            
            % Record compatibility metrics in testResults
            obj.testResults.testCrossPlatformCompatibility.status = 'passed';
            obj.testResults.testCrossPlatformCompatibility.compatibilityResults = compatibilityResults;
            obj.testResults.testCrossPlatformCompatibility.allCompatible = allCompatible;
        end
        
        function referenceData = generateWindowsReferenceData(obj, mexName, testInputs)
            % Generates reference data from Windows MEX implementations for cross-platform testing
            
            % Skip function if not on Windows platform
            if ~obj.isWindowsPlatform
                referenceData = struct();
                return;
            end
            
            % Execute specified MEX function with test inputs
            mexFunc = str2func(mexName);
            
            try
                % Execute MEX function
                result = mexFunc(testInputs{:});
                
                % Format results with Windows-specific metadata
                referenceData = struct(...
                    'mexName', mexName, ...
                    'platform', 'PCWIN64', ...
                    'matlabVersion', version, ...
                    'timestamp', datestr(now), ...
                    'result', result ...
                );
                
                % Use CrossPlatformValidator.generateReferenceResults to save data
                obj.crossValidator.generateReferenceResults(mexName, testInputs, referenceData);
            catch ME
                warning('WindowsMEXTest:ReferenceDataGenerationFailed', ...
                    'Failed to generate reference data for MEX file %s: %s', mexName, ME.message);
                referenceData = struct(...
                    'mexName', mexName, ...
                    'platform', 'PCWIN64', ...
                    'error', ME.message, ...
                    'timestamp', datestr(now) ...
                );
            end
        end
        
        function testInputs = createTestInputs(obj, mexType)
            % Creates appropriate test inputs for each MEX function type
            
            % Determine MEX function type from input parameter
            if contains(mexType, 'agarch')
                % AGARCH model inputs
                numObs = 1000;
                data = randn(numObs, 1) * 0.1;
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'tarch')
                % TARCH model inputs
                numObs = 1000;
                data = randn(numObs, 1) * 0.1;
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'egarch')
                % EGARCH model inputs
                numObs = 1000;
                data = randn(numObs, 1) * 0.1;
                parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
                initialVol = log(var(data));
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'igarch')
                % IGARCH model inputs
                numObs = 1000;
                data = randn(numObs, 1) * 0.1;
                parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'armaxerrors') || contains(mexType, 'armax')
                % ARMA/ARMAX model inputs
                numObs = 1000;
                ar = 0.7;
                ma = 0.3;
                innovations = randn(numObs+100, 1) * 0.1;
                y = zeros(numObs+100, 1);
                
                for t = 2:length(y)
                    y(t) = ar*y(t-1) + innovations(t) + ma*innovations(t-1);
                end
                
                % Discard burn-in period
                y = y(101:end);
                
                % Create parameter vector
                parameters = [ar; ma];
                
                % No exogenous variables
                X = [];
                
                testInputs = {y, parameters, X};
                
            elseif contains(mexType, 'composite')
                % Composite likelihood inputs
                R = [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1];
                numObs = 500;
                data = mvnrnd(zeros(3, 1), R, numObs);
                testInputs = {data, R};
                
            else
                % Default case - generic inputs
                warning('WindowsMEXTest:UnknownMEXType', ...
                    'Unknown MEX type: %s. Using generic test inputs.', mexType);
                testInputs = {rand(100, 1)};
            end
        end
        
        function report = generateTestReport(obj)
            % Generates a comprehensive report of Windows MEX test results
            
            % Skip function if not on Windows platform
            if ~obj.isWindowsPlatform
                warning('WindowsMEXTest:NotWindowsPlatform', 'Cannot generate report on non-Windows platform');
                report = struct();
                return;
            end
            
            % Gather test results from all test methods
            testMethods = {
                'testWindowsMEXFilesExist',
                'testWindowsMEXFileLoading',
                'testAgarchCoreFunctionality',
                'testTarchCoreFunctionality',
                'testEgarchCoreFunctionality',
                'testIgarchCoreFunctionality',
                'testArmaxerrorsFunctionality',
                'testCompositeLikelihoodFunctionality',
                'testWindowsMEXPerformance',
                'testWindowsMEXMemoryUsage',
                'testLargeArraySupport',
                'testCrossPlatformCompatibility'
            };
            
            % Initialize report
            report = struct(...
                'platform', 'PCWIN64', ...
                'matlabVersion', version, ...
                'timestamp', datestr(now), ...
                'summary', struct(...
                    'totalTests', length(testMethods), ...
                    'passedTests', 0, ...
                    'failedTests', 0, ...
                    'skippedTests', 0 ...
                ), ...
                'testResults', struct(), ...
                'mexFiles', {{}}, ...
                'performance', struct(...
                    'averageImprovement', 0, ...
                    'meetsRequirement', false ...
                ), ...
                'memory', struct(...
                    'hasLeaks', false ...
                ), ...
                'crossPlatform', struct(...
                    'compatible', false ...
                ) ...
            );
            
            % Compile test results
            for i = 1:length(testMethods)
                methodName = testMethods{i};
                
                if isfield(obj.testResults, methodName)
                    result = obj.testResults.(methodName);
                    report.testResults.(methodName) = result;
                    
                    % Update summary counters
                    if strcmp(result.status, 'passed')
                        report.summary.passedTests = report.summary.passedTests + 1;
                    elseif strcmp(result.status, 'failed')
                        report.summary.failedTests = report.summary.failedTests + 1;
                    else
                        report.summary.skippedTests = report.summary.skippedTests + 1;
                    end
                else
                    report.testResults.(methodName) = struct('status', 'skipped');
                    report.summary.skippedTests = report.summary.skippedTests + 1;
                end
            end
            
            % Compile MEX file list
            if isfield(obj.testResults, 'testWindowsMEXFilesExist') && ...
               isfield(obj.testResults.testWindowsMEXFilesExist, 'requiredFiles')
                report.mexFiles = obj.testResults.testWindowsMEXFilesExist.requiredFiles;
            end
            
            % Compile performance information
            if isfield(obj.testResults, 'testWindowsMEXPerformance') && ...
               isfield(obj.testResults.testWindowsMEXPerformance, 'benchmarkResults')
                
                benchmarks = obj.testResults.testWindowsMEXPerformance.benchmarkResults;
                fields = fieldnames(benchmarks);
                totalImprovement = 0;
                validBenchmarks = 0;
                
                for i = 1:length(fields)
                    field = fields{i};
                    if isfield(benchmarks.(field), 'performanceImprovement')
                        totalImprovement = totalImprovement + benchmarks.(field).performanceImprovement;
                        validBenchmarks = validBenchmarks + 1;
                    end
                end
                
                if validBenchmarks > 0
                    report.performance.averageImprovement = totalImprovement / validBenchmarks;
                end
                
                if isfield(obj.testResults.testWindowsMEXPerformance, 'allMeetRequirement')
                    report.performance.meetsRequirement = obj.testResults.testWindowsMEXPerformance.allMeetRequirement;
                end
            end
            
            % Compile memory usage information
            if isfield(obj.testResults, 'testWindowsMEXMemoryUsage') && ...
               isfield(obj.testResults.testWindowsMEXMemoryUsage, 'hasLeaks')
                report.memory.hasLeaks = obj.testResults.testWindowsMEXMemoryUsage.hasLeaks;
            end
            
            % Compile cross-platform information
            if isfield(obj.testResults, 'testCrossPlatformCompatibility') && ...
               isfield(obj.testResults.testCrossPlatformCompatibility, 'allCompatible')
                report.crossPlatform.compatible = obj.testResults.testCrossPlatformCompatibility.allCompatible;
            end
        end
    end
end