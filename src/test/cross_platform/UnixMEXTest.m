classdef UnixMEXTest < BaseTest
    % UNIXMEXTEST A test class that validates Unix-specific MEX binary functionality in the MFE Toolbox.
    %
    % This class performs Unix-platform validation, ensuring that Unix MEX binaries (.mexa64)
    % are compiled correctly, function as expected, and maintain numerical stability
    % and performance characteristics essential for high-performance financial econometrics.
    %
    % The class tests the following aspects of Unix MEX binaries:
    %   - Existence of all required MEX files
    %   - Loading capability of the MEX files
    %   - Functionality of each key MEX implementation (GARCH variants, ARMAX, etc.)
    %   - Performance improvement compared to MATLAB equivalents (>50% requirement)
    %   - Memory usage patterns to detect potential leaks
    %   - Support for large array dimensions (largeArrayDims flag)
    %   - Cross-platform compatibility with Windows implementations
    %
    % This class requires:
    %   - Running on a Unix platform (GLNXA64)
    %   - Access to MEX binaries in the specified path
    %   - MEXValidator and CrossPlatformValidator utility classes
    %
    % See also: BaseTest, MEXValidator, CrossPlatformValidator, WindowsMEXTest
    
    properties
        mexDllPath              % Path to MEX binary files
        mexValidator            % MEXValidator instance
        crossValidator          % CrossPlatformValidator instance
        testResults             % Structure to store test results
        mexFileList             % Cell array of MEX files to test
        isUnixPlatform          % Flag indicating if running on Unix
        referenceResultsPath    % Path to reference results
    end
    
    methods
        function obj = UnixMEXTest()
            % Initialize a new UnixMEXTest instance for testing Unix MEX binaries
            
            % Call parent constructor with class name
            obj@BaseTest('UnixMEXTest');
            
            % Check if running on Unix platform
            obj.isUnixPlatform = strcmpi(computer(), 'GLNXA64');
            
            % Initialize mexDllPath
            obj.mexDllPath = 'src/backend/dlls/';
            
            % Initialize MEXValidator and CrossPlatformValidator instances
            obj.mexValidator = MEXValidator(struct('mexBasePath', obj.mexDllPath));
            obj.crossValidator = CrossPlatformValidator();
            
            % Set reference results path
            obj.referenceResultsPath = 'src/test/data/cross_platform/';
            
            % Initialize testResults structure
            obj.testResults = struct();
            
            % Populate mexFileList with Unix MEX binaries (.mexa64 files)
            if obj.isUnixPlatform
                dirInfo = dir(fullfile(obj.mexDllPath, '*.mexa64'));
                obj.mexFileList = {dirInfo.name};
            else
                obj.mexFileList = {};
            end
        end
        
        function setUp(obj)
            % Prepares the test environment before each test execution
            
            % Call parent setUp method from BaseTest
            setUp@BaseTest(obj);
            
            % Verify current platform is Unix
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Tests skipped - not running on Unix platform');
            end
            
            % Initialize test data structures and MEX validators
            obj.testResults = struct();
            
            % Create reference results directory if it doesn't exist
            if ~exist(obj.referenceResultsPath, 'dir')
                mkdir(obj.referenceResultsPath);
            end
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test execution
            
            % Record test completion status
            obj.testResults.status = 'completed';
            
            % Save Unix reference results for cross-platform comparison
            if obj.isUnixPlatform && isfield(obj.testResults, 'referenceData')
                % Save reference data with platform-specific metadata
                timestamp = datestr(now);
                obj.testResults.referenceData.platform = 'GLNXA64';
                obj.testResults.referenceData.timestamp = timestamp;
                
                % Use CrossPlatformValidator to store reference results
                obj.crossValidator.generateReferenceResults(...
                    obj.testResults.testName, ...
                    obj.testResults.testInputs, ...
                    obj.testResults.referenceData);
            end
            
            % Free any allocated resources
            
            % Call parent tearDown method from BaseTest
            tearDown@BaseTest(obj);
        end
        
        function testUnixMEXFilesExist(obj)
            % Tests that all required Unix MEX binary files exist in the DLLs directory
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Define list of required Unix MEX binary files
            requiredMEXFiles = {'agarch_core.mexa64', 'tarch_core.mexa64', 'egarch_core.mexa64', ...
                                'igarch_core.mexa64', 'armaxerrors.mexa64', 'composite_likelihood.mexa64'};
            
            % Check if each file exists
            missingFiles = {};
            for i = 1:length(requiredMEXFiles)
                mexFile = requiredMEXFiles{i};
                if ~exist(fullfile(obj.mexDllPath, mexFile), 'file')
                    missingFiles{end+1} = mexFile;
                end
            end
            
            % Assert that all required Unix MEX binaries are present
            obj.assertTrue(isempty(missingFiles), ...
                sprintf('Missing Unix MEX files: %s', strjoin(missingFiles, ', ')));
            
            % Record results
            obj.testResults.existenceStatus = 'passed';
            obj.testResults.missingFiles = missingFiles;
        end
        
        function testUnixMEXFileLoading(obj)
            % Tests that all Unix MEX binary files can be loaded correctly
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % For each MEX file, check if it can be found in MATLAB path
            nonLoadableFiles = {};
            for i = 1:length(obj.mexFileList)
                mexFile = obj.mexFileList{i};
                [~, mexName] = fileparts(mexFile);
                
                % Try to locate MEX file using which()
                mexPath = which(mexName);
                if isempty(mexPath)
                    nonLoadableFiles{end+1} = mexName;
                end
            end
            
            % Assert that all MEX files are accessible to MATLAB
            obj.assertTrue(isempty(nonLoadableFiles), ...
                sprintf('Non-loadable Unix MEX files: %s', strjoin(nonLoadableFiles, ', ')));
            
            % Record results
            obj.testResults.loadingStatus = 'passed';
            obj.testResults.nonLoadableFiles = nonLoadableFiles;
        end
        
        function testAgarchCoreFunctionality(obj)
            % Tests the functionality of the Unix AGARCH core MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for AGARCH model
            testInputs = obj.createTestInputs('agarch');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('agarch_core', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('AGARCH core MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'agarch_core';
            obj.testResults.testInputs = testInputs;
            obj.testResults.agarchValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('agarch_core', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testTarchCoreFunctionality(obj)
            % Tests the functionality of the Unix TARCH core MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for TARCH model
            testInputs = obj.createTestInputs('tarch');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('tarch_core', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('TARCH core MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'tarch_core';
            obj.testResults.testInputs = testInputs;
            obj.testResults.tarchValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('tarch_core', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testEgarchCoreFunctionality(obj)
            % Tests the functionality of the Unix EGARCH core MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for EGARCH model
            testInputs = obj.createTestInputs('egarch');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('egarch_core', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('EGARCH core MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'egarch_core';
            obj.testResults.testInputs = testInputs;
            obj.testResults.egarchValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('egarch_core', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testIgarchCoreFunctionality(obj)
            % Tests the functionality of the Unix IGARCH core MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for IGARCH model
            testInputs = obj.createTestInputs('igarch');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('igarch_core', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('IGARCH core MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'igarch_core';
            obj.testResults.testInputs = testInputs;
            obj.testResults.igarchValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('igarch_core', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testArmaxerrorsFunctionality(obj)
            % Tests the functionality of the Unix ARMAX errors MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for ARMAX errors computation
            testInputs = obj.createTestInputs('armaxerrors');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('armaxerrors', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('ARMAX errors MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'armaxerrors';
            obj.testResults.testInputs = testInputs;
            obj.testResults.armaxerrorsValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('armaxerrors', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testCompositeLikelihoodFunctionality(obj)
            % Tests the functionality of the Unix composite likelihood MEX implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % Create test inputs for composite likelihood computation
            testInputs = obj.createTestInputs('composite_likelihood');
            
            % Validate functionality using MEXValidator
            result = obj.mexValidator.validateMEXFunctionality('composite_likelihood', testInputs);
            
            % Assert that MEX implementation functions correctly
            obj.assertTrue(result.canExecute, ...
                sprintf('Composite likelihood MEX validation failed: %s', result.errorMessage));
            
            % Record test results
            obj.testResults.testName = 'composite_likelihood';
            obj.testResults.testInputs = testInputs;
            obj.testResults.compositeLikelihoodValidation = result;
            
            % Generate reference data for cross-platform testing
            refData = obj.generateUnixReferenceData('composite_likelihood', testInputs);
            obj.testResults.referenceData = refData;
        end
        
        function testUnixMEXPerformance(obj)
            % Tests the performance of Unix MEX implementations compared to MATLAB equivalents
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % List of MEX files to benchmark with their MATLAB equivalents
            mexFunctions = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core', 'armaxerrors'};
            matlabFunctions = {'agarch_core_matlab', 'tarch_core_matlab', 'egarch_core_matlab', ...
                              'igarch_core_matlab', 'armaxerrors_matlab'};
            
            % Benchmark results
            benchmarkResults = struct();
            allMeetRequirement = true;
            
            % Run benchmarks for each MEX file
            for i = 1:length(mexFunctions)
                mexFunc = mexFunctions{i};
                matlabFunc = matlabFunctions{i};
                
                % Create appropriate test inputs
                testInputs = obj.createTestInputs(mexFunc);
                
                % Run benchmark using MEXValidator
                result = obj.mexValidator.benchmarkMEXPerformance(mexFunc, matlabFunc, testInputs);
                
                % Store results
                benchmarkResults.(mexFunc) = result;
                
                % Check if performance requirement is met (50% improvement)
                if ~result.meetsRequirement
                    allMeetRequirement = false;
                end
            end
            
            % Assert that all MEX implementations meet performance requirements
            obj.assertTrue(allMeetRequirement, ...
                'Not all Unix MEX implementations meet the 50% performance improvement requirement');
            
            % Record test results
            obj.testResults.performanceBenchmarks = benchmarkResults;
        end
        
        function testUnixMEXMemoryUsage(obj)
            % Tests the memory usage patterns of Unix MEX implementations
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % List of MEX files to test memory usage
            mexFiles = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core', 'armaxerrors', 'composite_likelihood'};
            
            % Memory test results
            memoryResults = struct();
            noMemoryLeaks = true;
            
            % Test memory usage for each MEX file
            for i = 1:length(mexFiles)
                mexFunc = mexFiles{i};
                
                % Create appropriate test inputs
                testInputs = obj.createTestInputs(mexFunc);
                
                % Test memory usage using MEXValidator
                result = obj.mexValidator.validateMemoryUsage(mexFunc, testInputs, 50);
                
                % Store results
                memoryResults.(mexFunc) = result;
                
                % Check for memory leaks
                if result.hasLeak
                    noMemoryLeaks = false;
                end
            end
            
            % Assert that no memory leaks were detected
            obj.assertTrue(noMemoryLeaks, 'Memory leaks detected in one or more Unix MEX implementations');
            
            % Record test results
            obj.testResults.memoryTests = memoryResults;
        end
        
        function testLargeArraySupport(obj)
            % Tests Unix MEX binaries with large array dimensions to verify -largeArrayDims flag implementation
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % List of MEX files to test with large arrays
            mexFiles = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core'};
            
            % Large array test results
            largeArrayResults = struct();
            allSupported = true;
            
            % Test each MEX file with large arrays
            for i = 1:length(mexFiles)
                mexFunc = mexFiles{i};
                
                % Create large test arrays (approx. 10^6 elements)
                numObs = 10^6;
                data = randn(numObs, 1);
                
                % Adjust test inputs based on function type
                switch mexFunc
                    case 'agarch_core'
                        parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                        initialVol = var(data);
                        testInputs = {data, parameters, initialVol};
                        
                    case 'tarch_core'
                        parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                        initialVol = var(data);
                        testInputs = {data, parameters, initialVol};
                        
                    case 'egarch_core'
                        parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
                        initialVol = log(var(data));
                        testInputs = {data, parameters, initialVol};
                        
                    case 'igarch_core'
                        parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                        initialVol = var(data);
                        testInputs = {data, parameters, initialVol};
                        
                    otherwise
                        continue; % Skip if not supported
                end
                
                % Test with large array
                try
                    % Get function handle
                    func = str2func(mexFunc);
                    
                    % Execute function with large array
                    tic;
                    result = func(testInputs{:});
                    execTime = toc;
                    
                    % Check if result is valid
                    largeArrayResults.(mexFunc) = struct(...
                        'status', 'passed', ...
                        'executionTime', execTime, ...
                        'dataSize', numObs, ...
                        'isValidResult', isstruct(result) || isnumeric(result) ...
                    );
                    
                catch ME
                    % Record failure
                    largeArrayResults.(mexFunc) = struct(...
                        'status', 'failed', ...
                        'errorMessage', ME.message, ...
                        'dataSize', numObs ...
                    );
                    
                    allSupported = false;
                end
            end
            
            % Assert that all MEX files support large arrays
            obj.assertTrue(allSupported, 'One or more Unix MEX implementations failed with large arrays');
            
            % Record test results
            obj.testResults.largeArrayTests = largeArrayResults;
        end
        
        function testCrossPlatformCompatibility(obj)
            % Compares Unix MEX results with Windows results to ensure cross-platform consistency
            
            % Skip test if not on Unix platform
            if ~obj.isUnixPlatform
                warning('UnixMEXTest:NotOnUnixPlatform', 'Test skipped - not running on Unix platform');
                return;
            end
            
            % List of MEX files to test cross-platform
            mexFiles = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core', 'armaxerrors', 'composite_likelihood'};
            
            % Cross-platform test results
            compatibilityResults = struct();
            allCompatible = true;
            
            % Test each MEX file for cross-platform compatibility
            for i = 1:length(mexFiles)
                mexFunc = mexFiles{i};
                
                % Create test inputs
                testInputs = obj.createTestInputs(mexFunc);
                
                % Use CrossPlatformValidator to check compatibility
                result = obj.crossValidator.validateMEXCompatibility(mexFunc, testInputs);
                
                % Store results
                compatibilityResults.(mexFunc) = result;
                
                % Check if compatible
                if ~result.isCompatible
                    allCompatible = false;
                end
            end
            
            % Assert that all MEX implementations are cross-platform compatible
            obj.assertTrue(allCompatible, 'One or more Unix MEX implementations are not cross-platform compatible');
            
            % Record test results
            obj.testResults.crossPlatformTests = compatibilityResults;
        end
        
        function refData = generateUnixReferenceData(obj, mexName, testInputs)
            % Generates reference data from Unix MEX implementations for cross-platform testing
            
            % Skip function if not on Unix platform
            if ~obj.isUnixPlatform
                refData = struct();
                return;
            end
            
            % Create reference data structure
            refData = struct(...
                'mexName', mexName, ...
                'platform', 'GLNXA64', ...
                'timestamp', datestr(now) ...
            );
            
            % Execute MEX function to get results
            try
                mexFunc = str2func(mexName);
                refData.results = mexFunc(testInputs{:});
                refData.status = 'success';
            catch ME
                refData.status = 'failed';
                refData.errorMessage = ME.message;
            end
            
            % Use CrossPlatformValidator to save reference results
            obj.crossValidator.generateReferenceResults(mexName, testInputs, refData);
        end
        
        function testInputs = createTestInputs(obj, mexType)
            % Creates appropriate test inputs for each MEX function type
            
            % Determine function type based on input
            if contains(mexType, 'agarch')
                % Generate test inputs for AGARCH model
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'tarch')
                % Generate test inputs for TARCH model
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'egarch')
                % Generate test inputs for EGARCH model
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
                initialVol = log(var(data));
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'igarch')
                % Generate test inputs for IGARCH model
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif contains(mexType, 'armaxerrors')
                % Generate test inputs for ARMAX model
                numObs = 1000;
                
                % ARMA(1,1) process
                ar = 0.7;
                ma = 0.3;
                innovations = randn(numObs+100, 1);
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
                
            elseif contains(mexType, 'composite_likelihood')
                % Generate test correlation matrix (3x3)
                R = [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1];
                
                % Generate multivariate normal data
                numObs = 500;
                data = mvnrnd(zeros(3, 1), R, numObs);
                
                testInputs = {data, R};
                
            else
                % Default generic inputs
                warning('UnixMEXTest:UnknownMEXType', 'Unknown MEX type. Using generic test inputs.');
                testInputs = {rand(100, 1), rand(3, 1)};
            end
        end
        
        function report = generateTestReport(obj)
            % Generates a comprehensive report of Unix MEX test results
            
            % Skip function if not on Unix platform
            if ~obj.isUnixPlatform
                report = struct('status', 'skipped', 'reason', 'Not running on Unix platform');
                return;
            end
            
            % Compile results from all test methods
            report = struct(...
                'platform', 'GLNXA64', ...
                'timestamp', datestr(now), ...
                'summary', struct(), ...
                'details', struct() ...
            );
            
            % Count tests by category
            numExistenceTests = isfield(obj.testResults, 'existenceStatus') && ...
                                strcmp(obj.testResults.existenceStatus, 'passed');
            
            numFunctionalityTests = 0;
            numPassedFunctionalityTests = 0;
            if isfield(obj.testResults, 'agarchValidation') && ...
                    strcmp(obj.testResults.agarchValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            if isfield(obj.testResults, 'tarchValidation') && ...
                    strcmp(obj.testResults.tarchValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            if isfield(obj.testResults, 'egarchValidation') && ...
                    strcmp(obj.testResults.egarchValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            if isfield(obj.testResults, 'igarchValidation') && ...
                    strcmp(obj.testResults.igarchValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            if isfield(obj.testResults, 'armaxerrorsValidation') && ...
                    strcmp(obj.testResults.armaxerrorsValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            if isfield(obj.testResults, 'compositeLikelihoodValidation') && ...
                    strcmp(obj.testResults.compositeLikelihoodValidation.status, 'passed')
                numFunctionalityTests = numFunctionalityTests + 1;
                numPassedFunctionalityTests = numPassedFunctionalityTests + 1;
            end
            
            % Performance metrics
            performancePassed = isfield(obj.testResults, 'performanceBenchmarks');
            performanceImprovement = 0;
            
            if performancePassed
                benchmarkFields = fieldnames(obj.testResults.performanceBenchmarks);
                for i = 1:length(benchmarkFields)
                    field = benchmarkFields{i};
                    if isfield(obj.testResults.performanceBenchmarks.(field), 'performanceImprovement')
                        performanceImprovement = performanceImprovement + ...
                            obj.testResults.performanceBenchmarks.(field).performanceImprovement;
                    end
                end
                
                if length(benchmarkFields) > 0
                    performanceImprovement = performanceImprovement / length(benchmarkFields);
                end
            end
            
            % Memory metrics
            memoryLeakFree = isfield(obj.testResults, 'memoryTests');
            
            % Large array support
            largeArraySupport = isfield(obj.testResults, 'largeArrayTests');
            
            % Cross-platform compatibility
            crossPlatformCompatible = isfield(obj.testResults, 'crossPlatformTests');
            
            % Build summary
            report.summary = struct(...
                'existenceTestsPassed', numExistenceTests, ...
                'functionalityTestsRun', numFunctionalityTests, ...
                'functionalityTestsPassed', numPassedFunctionalityTests, ...
                'performanceTestPassed', performancePassed, ...
                'averagePerformanceImprovement', performanceImprovement, ...
                'memoryLeakFree', memoryLeakFree, ...
                'largeArraySupport', largeArraySupport, ...
                'crossPlatformCompatible', crossPlatformCompatible, ...
                'overallStatus', 'passed' ...
            );
            
            % Set overall status
            if ~numExistenceTests || numPassedFunctionalityTests < numFunctionalityTests || ...
                    ~performancePassed || ~memoryLeakFree || ~largeArraySupport || ~crossPlatformCompatible
                report.summary.overallStatus = 'failed';
            end
            
            % Include detailed results if available
            if isfield(obj.testResults, 'existenceStatus')
                report.details.existenceTests = obj.testResults.existenceStatus;
            end
            
            if isfield(obj.testResults, 'agarchValidation')
                report.details.agarchTests = obj.testResults.agarchValidation;
            end
            
            if isfield(obj.testResults, 'tarchValidation')
                report.details.tarchTests = obj.testResults.tarchValidation;
            end
            
            if isfield(obj.testResults, 'egarchValidation')
                report.details.egarchTests = obj.testResults.egarchValidation;
            end
            
            if isfield(obj.testResults, 'igarchValidation')
                report.details.igarchTests = obj.testResults.igarchValidation;
            end
            
            if isfield(obj.testResults, 'armaxerrorsValidation')
                report.details.armaxerrorsTests = obj.testResults.armaxerrorsValidation;
            end
            
            if isfield(obj.testResults, 'compositeLikelihoodValidation')
                report.details.compositeLikelihoodTests = obj.testResults.compositeLikelihoodValidation;
            end
            
            if isfield(obj.testResults, 'performanceBenchmarks')
                report.details.performanceTests = obj.testResults.performanceBenchmarks;
            end
            
            if isfield(obj.testResults, 'memoryTests')
                report.details.memoryTests = obj.testResults.memoryTests;
            end
            
            if isfield(obj.testResults, 'largeArrayTests')
                report.details.largeArrayTests = obj.testResults.largeArrayTests;
            end
            
            if isfield(obj.testResults, 'crossPlatformTests')
                report.details.crossPlatformTests = obj.testResults.crossPlatformTests;
            end
        end
    end
end