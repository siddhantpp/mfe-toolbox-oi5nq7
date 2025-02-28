classdef MEXValidator < BaseTest
    % MEXVALIDATOR A class for validating MEX file functionality, performance, and cross-platform compatibility in the MFE Toolbox
    %
    % This validator class provides comprehensive tools for testing and
    % verifying MEX file functionality, performance, and cross-platform
    % compatibility within the MFE Toolbox. It includes methods for
    % validation, benchmarking, and memory usage analysis.
    %
    % The class enables systematic verification that MEX implementations
    % meet the performance requirements (>50% improvement over MATLAB),
    % function correctly across platforms, and handle memory efficiently.
    %
    % Example:
    %   % Create a validator for all MEX files
    %   validator = MEXValidator();
    %
    %   % Check if a specific MEX file exists
    %   exists = validator.validateMEXExists('agarch_core');
    %
    %   % Compare MEX implementation with MATLAB equivalent
    %   inputs = {data, parameters};
    %   result = validator.compareMEXWithMATLAB('agarch_core', 'agarch_core_matlab', inputs);
    %
    %   % Validate all MEX files
    %   report = validator.validateAllMEXFiles();
    %
    % See also: BaseTest, NumericalComparator, parametercheck
    
    properties
        mexBasePath         % Base path to MEX files
        platform            % Current platform identifier
        mexFiles            % Structure of available MEX files
        comparator          % NumericalComparator instance for result validation
        defaultTolerance    % Default tolerance for numerical comparisons
        validationResults   % Structure to store validation results
        verbose             % Flag to control output verbosity
        platformInfo        % Information about the current platform
    end
    
    methods
        function obj = MEXValidator(options)
            % Initialize a new MEXValidator instance with default settings
            %
            % INPUTS:
            %   options - Optional structure with configuration options:
            %       .mexBasePath - Custom path to MEX files
            %       .tolerance - Custom tolerance for numerical comparisons
            %       .verbose - Flag to enable verbose output
            %
            % OUTPUTS:
            %   obj - Initialized MEXValidator instance
            
            % Call parent constructor with class name
            obj@BaseTest('MEXValidator');
            
            % Set default options if not provided
            if nargin < 1
                options = struct();
            end
            
            % Set MEX base path (default to src/backend/dlls/)
            if isfield(options, 'mexBasePath')
                obj.mexBasePath = options.mexBasePath;
            else
                obj.mexBasePath = 'src/backend/dlls/';
            end
            
            % Determine current platform
            obj.platform = computer();
            
            % Initialize numerical comparator
            obj.comparator = NumericalComparator();
            
            % Set default tolerance
            if isfield(options, 'tolerance')
                obj.defaultTolerance = options.tolerance;
            else
                obj.defaultTolerance = 1e-10;
            end
            
            % Initialize validation results
            obj.validationResults = struct();
            
            % Set verbosity
            if isfield(options, 'verbose')
                obj.verbose = options.verbose;
            else
                obj.verbose = false;
            end
            
            % Collect platform information
            obj.platformInfo = struct(...
                'platform', obj.platform, ...
                'isWindows', strncmpi(obj.platform, 'PCW', 3), ...
                'isUnix', strncmpi(obj.platform, 'GLN', 3) || strncmpi(obj.platform, 'MAC', 3), ...
                'mexExtension', obj.getMEXExtension(), ...
                'compilerInfo', mex.getCompilerConfigurations('C', 'Selected') ...
            );
            
            % Initialize mexFiles structure with available MEX binaries
            obj.mexFiles = obj.getMEXFileList();
        end
        
        function result = validateMEXExists(obj, mexBaseName)
            % Validates that a MEX file exists in the expected location with the correct platform extension
            %
            % INPUTS:
            %   mexBaseName - Base name of the MEX file without extension
            %
            % OUTPUTS:
            %   result - Logical true if the MEX file exists
            
            % Get correct platform-specific extension
            mexExt = obj.getMEXExtension();
            
            % Construct full MEX filename
            mexFileName = fullfile(obj.mexBasePath, [mexBaseName, '.', mexExt]);
            
            % Check if file exists
            fileExists = (exist(mexFileName, 'file') == 3); % 3 = MEX-file
            
            % Record validation result
            timestamp = datestr(now);
            obj.validationResults.existence.(mexBaseName) = struct(...
                'exists', fileExists, ...
                'path', mexFileName, ...
                'timestamp', timestamp ...
            );
            
            % Return existence status
            result = fileExists;
        end
        
        function result = validateMEXFunctionality(obj, mexBaseName, testInputs)
            % Validates that a MEX file can be loaded and executed with basic test input
            %
            % INPUTS:
            %   mexBaseName - Base name of the MEX file without extension
            %   testInputs - Cell array of test inputs for the function
            %
            % OUTPUTS:
            %   result - Validation result structure
            
            % Check if MEX file exists
            mexExists = obj.validateMEXExists(mexBaseName);
            
            % Initialize result structure
            result = struct(...
                'name', mexBaseName, ...
                'exists', mexExists, ...
                'canExecute', false, ...
                'executionTime', NaN, ...
                'status', 'failed', ...
                'errorMessage', '', ...
                'timestamp', datestr(now) ...
            );
            
            % If MEX doesn't exist, return early
            if ~mexExists
                result.errorMessage = 'MEX file does not exist';
                obj.validationResults.functionality.(mexBaseName) = result;
                return;
            end
            
            % Try to clear and reload the MEX function
            try
                % Clear the function from memory
                clear(mexBaseName);
                
                % Get the function handle
                mexFunc = str2func(mexBaseName);
                
                % Execute with test inputs
                tic;
                try
                    mexOutput = mexFunc(testInputs{:});
                    executionTime = toc;
                    
                    % Check if output is valid
                    if isstruct(mexOutput) || isnumeric(mexOutput)
                        result.canExecute = true;
                        result.status = 'passed';
                        result.executionTime = executionTime;
                    else
                        result.errorMessage = 'MEX function did not return valid output';
                    end
                catch mexError
                    result.errorMessage = ['Execution error: ', mexError.message];
                end
            catch loadError
                result.errorMessage = ['Loading error: ', loadError.message];
            end
            
            % Record validation result
            obj.validationResults.functionality.(mexBaseName) = result;
            
            % Return result
            return;
        end
        
        function result = compareMEXWithMATLAB(obj, mexFunction, matlabFunction, testInputs, tolerance)
            % Compares MEX implementation results with equivalent MATLAB implementation
            %
            % INPUTS:
            %   mexFunction - Name of the MEX function
            %   matlabFunction - Name of the equivalent MATLAB function
            %   testInputs - Cell array of test inputs
            %   tolerance - Optional numerical tolerance for comparison
            %
            % OUTPUTS:
            %   result - Comparison result structure
            
            % Initialize result structure
            result = struct(...
                'mexFunction', mexFunction, ...
                'matlabFunction', matlabFunction, ...
                'isEqual', false, ...
                'performanceImprovement', NaN, ...
                'mexExecutionTime', NaN, ...
                'matlabExecutionTime', NaN, ...
                'maxAbsoluteDifference', NaN, ...
                'maxRelativeDifference', NaN, ...
                'status', 'failed', ...
                'errorMessage', '', ...
                'timestamp', datestr(now) ...
            );
            
            % Check if both functions exist
            mexExists = (exist(mexFunction, 'file') == 3); % 3 = MEX-file
            matlabExists = (exist(matlabFunction, 'file') == 2); % 2 = M-file
            
            if ~mexExists
                result.errorMessage = ['MEX function ', mexFunction, ' does not exist'];
                obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            if ~matlabExists
                result.errorMessage = ['MATLAB function ', matlabFunction, ' does not exist'];
                obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            % Get function handles
            mexFunc = str2func(mexFunction);
            matlabFunc = str2func(matlabFunction);
            
            % Use provided tolerance or default
            if nargin < 5 || isempty(tolerance)
                tolerance = obj.defaultTolerance;
            end
            
            % Execute MATLAB function and measure time
            try
                tic;
                matlabOutput = matlabFunc(testInputs{:});
                matlabTime = toc;
                result.matlabExecutionTime = matlabTime;
            catch matlabError
                result.errorMessage = ['MATLAB function error: ', matlabError.message];
                obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            % Execute MEX function and measure time
            try
                tic;
                mexOutput = mexFunc(testInputs{:});
                mexTime = toc;
                result.mexExecutionTime = mexTime;
            catch mexError
                result.errorMessage = ['MEX function error: ', mexError.message];
                obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            % Calculate performance improvement
            if matlabTime > 0
                result.performanceImprovement = (matlabTime - mexTime) / matlabTime * 100;
            end
            
            % Compare outputs
            try
                if isstruct(matlabOutput) && isstruct(mexOutput)
                    % Compare struct fields
                    matlabFields = fieldnames(matlabOutput);
                    mexFields = fieldnames(mexOutput);
                    
                    if ~isequal(sort(matlabFields), sort(mexFields))
                        result.errorMessage = 'Output structures have different fields';
                        obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                        return;
                    end
                    
                    % Compare each field
                    maxAbsDiff = 0;
                    maxRelDiff = 0;
                    
                    for i = 1:length(matlabFields)
                        field = matlabFields{i};
                        if isnumeric(matlabOutput.(field)) && isnumeric(mexOutput.(field))
                            % Compare numeric fields
                            compResult = obj.comparator.compareMatrices(...
                                matlabOutput.(field), mexOutput.(field), tolerance);
                            
                            if ~compResult.isEqual
                                result.errorMessage = ['Numeric field ', field, ' does not match'];
                                obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                                return;
                            end
                            
                            maxAbsDiff = max(maxAbsDiff, compResult.maxAbsoluteDifference);
                            maxRelDiff = max(maxRelDiff, compResult.maxRelativeDifference);
                        elseif ~isequal(matlabOutput.(field), mexOutput.(field))
                            result.errorMessage = ['Non-numeric field ', field, ' does not match'];
                            obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                            return;
                        end
                    end
                    
                    result.maxAbsoluteDifference = maxAbsDiff;
                    result.maxRelativeDifference = maxRelDiff;
                    
                elseif isnumeric(matlabOutput) && isnumeric(mexOutput)
                    % Compare numeric outputs directly
                    compResult = obj.comparator.compareMatrices(matlabOutput, mexOutput, tolerance);
                    
                    if ~compResult.isEqual
                        result.errorMessage = 'Numeric outputs do not match';
                        result.maxAbsoluteDifference = compResult.maxAbsoluteDifference;
                        result.maxRelativeDifference = compResult.maxRelativeDifference;
                        obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                        return;
                    end
                    
                    result.maxAbsoluteDifference = compResult.maxAbsoluteDifference;
                    result.maxRelativeDifference = compResult.maxRelativeDifference;
                    
                else
                    % Compare other output types using isequal
                    if ~isequal(matlabOutput, mexOutput)
                        result.errorMessage = 'Outputs do not match';
                        obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
                        return;
                    end
                end
                
                % If we got here, outputs match
                result.isEqual = true;
                result.status = 'passed';
            catch compError
                result.errorMessage = ['Comparison error: ', compError.message];
            end
            
            % Record validation result
            obj.validationResults.comparison.([mexFunction, '_vs_', matlabFunction]) = result;
        end
        
        function result = benchmarkMEXPerformance(obj, mexFunction, matlabFunction, testInputs, iterations)
            % Benchmarks MEX function performance against MATLAB implementation
            %
            % INPUTS:
            %   mexFunction - Name of the MEX function
            %   matlabFunction - Name of the equivalent MATLAB function
            %   testInputs - Cell array of test inputs
            %   iterations - Number of iterations for benchmarking
            %
            % OUTPUTS:
            %   result - Performance benchmark results
            
            % Default iterations
            if nargin < 5 || isempty(iterations)
                iterations = 10;
            end
            
            % Validate iterations parameter
            optStruct = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
            parametercheck(iterations, 'iterations', optStruct);
            
            % Initialize result structure
            result = struct(...
                'mexFunction', mexFunction, ...
                'matlabFunction', matlabFunction, ...
                'iterations', iterations, ...
                'mexTimes', zeros(1, iterations), ...
                'matlabTimes', zeros(1, iterations), ...
                'mexMeanTime', NaN, ...
                'matlabMeanTime', NaN, ...
                'mexStdTime', NaN, ...
                'matlabStdTime', NaN, ...
                'performanceImprovement', NaN, ...
                'meetsRequirement', false, ...
                'status', 'failed', ...
                'errorMessage', '', ...
                'timestamp', datestr(now) ...
            );
            
            % Check if both functions exist
            mexExists = (exist(mexFunction, 'file') == 3); % 3 = MEX-file
            matlabExists = (exist(matlabFunction, 'file') == 2); % 2 = M-file
            
            if ~mexExists
                result.errorMessage = ['MEX function ', mexFunction, ' does not exist'];
                obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            if ~matlabExists
                result.errorMessage = ['MATLAB function ', matlabFunction, ' does not exist'];
                obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            % Get function handles
            mexFunc = str2func(mexFunction);
            matlabFunc = str2func(matlabFunction);
            
            % Warm-up run (not timed)
            try
                mexFunc(testInputs{:});
                matlabFunc(testInputs{:});
            catch warmupError
                result.errorMessage = ['Warm-up error: ', warmupError.message];
                obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
                return;
            end
            
            % Benchmark MATLAB function
            for i = 1:iterations
                try
                    tic;
                    matlabFunc(testInputs{:});
                    result.matlabTimes(i) = toc;
                catch matlabError
                    result.errorMessage = ['MATLAB benchmark error: ', matlabError.message];
                    obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
                    return;
                end
            end
            
            % Benchmark MEX function
            for i = 1:iterations
                try
                    tic;
                    mexFunc(testInputs{:});
                    result.mexTimes(i) = toc;
                catch mexError
                    result.errorMessage = ['MEX benchmark error: ', mexError.message];
                    obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
                    return;
                end
            end
            
            % Calculate statistics
            result.mexMeanTime = mean(result.mexTimes);
            result.matlabMeanTime = mean(result.matlabTimes);
            result.mexStdTime = std(result.mexTimes);
            result.matlabStdTime = std(result.matlabTimes);
            
            % Calculate performance improvement
            if result.matlabMeanTime > 0
                result.performanceImprovement = (result.matlabMeanTime - result.mexMeanTime) / result.matlabMeanTime * 100;
                result.meetsRequirement = (result.performanceImprovement >= 50);
            end
            
            % Set status
            result.status = 'passed';
            
            % Record validation result
            obj.validationResults.performance.([mexFunction, '_vs_', matlabFunction]) = result;
        end
        
        function result = validateMemoryUsage(obj, mexFunction, testInputs, iterations)
            % Monitors MEX function memory usage to detect leaks or excessive consumption
            %
            % INPUTS:
            %   mexFunction - Name of the MEX function
            %   testInputs - Cell array of test inputs
            %   iterations - Number of iterations to run
            %
            % OUTPUTS:
            %   result - Memory usage analysis
            
            % Default iterations
            if nargin < 4 || isempty(iterations)
                iterations = 100;
            end
            
            % Initialize result structure
            result = struct(...
                'mexFunction', mexFunction, ...
                'iterations', iterations, ...
                'baselineMemory', NaN, ...
                'finalMemory', NaN, ...
                'memoryDelta', NaN, ...
                'memoryPerIteration', NaN, ...
                'hasLeak', false, ...
                'memoryProfile', zeros(1, iterations), ...
                'status', 'failed', ...
                'errorMessage', '', ...
                'timestamp', datestr(now) ...
            );
            
            % Check if MEX function exists
            if exist(mexFunction, 'file') ~= 3 % 3 = MEX-file
                result.errorMessage = ['MEX function ', mexFunction, ' does not exist'];
                obj.validationResults.memory.(mexFunction) = result;
                return;
            end
            
            % Get function handle
            mexFunc = str2func(mexFunction);
            
            % Baseline memory measurement
            try
                % Garbage collection to stabilize memory
                clear mexFunction;
                % Wait for garbage collection to complete
                pause(0.1);
                % Get baseline memory
                whos_output = whos();
                result.baselineMemory = sum([whos_output.bytes]);
            catch baselineError
                result.errorMessage = ['Baseline memory error: ', baselineError.message];
                obj.validationResults.memory.(mexFunction) = result;
                return;
            end
            
            % Execute MEX function repeatedly and monitor memory
            try
                for i = 1:iterations
                    % Execute function
                    mexFunc(testInputs{:});
                    
                    % Measure memory after execution
                    whos_output = whos();
                    currentMemory = sum([whos_output.bytes]);
                    result.memoryProfile(i) = currentMemory - result.baselineMemory;
                    
                    % Detect potential leak pattern
                    if i > 10 && (result.memoryProfile(i) - result.memoryProfile(i-10)) > 1e6
                        result.hasLeak = true;
                    end
                end
                
                % Final memory measurement
                whos_output = whos();
                result.finalMemory = sum([whos_output.bytes]);
                result.memoryDelta = result.finalMemory - result.baselineMemory;
                result.memoryPerIteration = result.memoryDelta / iterations;
                
                % Set status
                result.status = 'passed';
                
            catch memError
                result.errorMessage = ['Memory monitoring error: ', memError.message];
            end
            
            % Record validation result
            obj.validationResults.memory.(mexFunction) = result;
        end
        
        function result = validateInputHandling(obj, mexFunction, testCases)
            % Tests MEX function robustness with invalid or edge-case inputs
            %
            % INPUTS:
            %   mexFunction - Name of the MEX function
            %   testCases - Structure array with test cases:
            %       .name - Test case name
            %       .inputs - Cell array of inputs
            %       .expectedError - Expected error message or pattern (optional)
            %       .shouldFail - Boolean indicating if function should fail (default: true)
            %
            % OUTPUTS:
            %   result - Input validation results
            
            % Initialize result structure
            result = struct(...
                'mexFunction', mexFunction, ...
                'numTestCases', length(testCases), ...
                'passedCases', 0, ...
                'failedCases', 0, ...
                'caseResults', struct(), ...
                'status', 'failed', ...
                'errorMessage', '', ...
                'timestamp', datestr(now) ...
            );
            
            % Check if MEX function exists
            if exist(mexFunction, 'file') ~= 3 % 3 = MEX-file
                result.errorMessage = ['MEX function ', mexFunction, ' does not exist'];
                obj.validationResults.inputHandling.(mexFunction) = result;
                return;
            end
            
            % Get function handle
            mexFunc = str2func(mexFunction);
            
            % Process each test case
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                % Default shouldFail to true if not specified
                if ~isfield(testCase, 'shouldFail')
                    testCase.shouldFail = true;
                end
                
                % Initialize case result
                caseResult = struct(...
                    'name', testCase.name, ...
                    'shouldFail', testCase.shouldFail, ...
                    'didFail', false, ...
                    'errorMessage', '', ...
                    'passed', false ...
                );
                
                % Execute test case
                try
                    % Execute function with test inputs
                    mexFunc(testCase.inputs{:});
                    
                    % If we reach here, function did not fail
                    caseResult.didFail = false;
                    
                    % Check if this matches expected behavior
                    caseResult.passed = (caseResult.didFail == testCase.shouldFail);
                    
                catch testError
                    % Function failed
                    caseResult.didFail = true;
                    caseResult.errorMessage = testError.message;
                    
                    % Check if this matches expected behavior
                    caseResult.passed = (caseResult.didFail == testCase.shouldFail);
                    
                    % If expected error message is specified, check it
                    if isfield(testCase, 'expectedError') && ~isempty(testCase.expectedError)
                        if ischar(testCase.expectedError) && contains(testError.message, testCase.expectedError)
                            % Error message contains expected pattern
                            caseResult.passed = true;
                        elseif isa(testCase.expectedError, 'function_handle') && testCase.expectedError(testError.message)
                            % Error message matches custom validation function
                            caseResult.passed = true;
                        else
                            % Error doesn't match expected pattern
                            caseResult.passed = false;
                        end
                    end
                end
                
                % Update case counts
                if caseResult.passed
                    result.passedCases = result.passedCases + 1;
                else
                    result.failedCases = result.failedCases + 1;
                end
                
                % Store case result
                result.caseResults.(testCase.name) = caseResult;
            end
            
            % Set overall status
            if result.failedCases == 0
                result.status = 'passed';
            else
                result.status = 'failed';
            end
            
            % Record validation result
            obj.validationResults.inputHandling.(mexFunction) = result;
        end
        
        function result = validateAllMEXFiles(obj, options)
            % Validates all MEX files in the MFE Toolbox
            %
            % INPUTS:
            %   options - Optional structure with validation options:
            %       .testFunctionality - Test basic functionality [default: true]
            %       .compareMATLAB - Compare with MATLAB implementation [default: true]
            %       .benchmarkPerformance - Benchmark performance [default: true]
            %       .checkMemoryUsage - Check for memory leaks [default: true]
            %       .validateInputHandling - Test input validation [default: true]
            %
            % OUTPUTS:
            %   result - Comprehensive validation results
            
            % Default options
            if nargin < 2
                options = struct();
            end
            
            % Set default validation options
            if ~isfield(options, 'testFunctionality')
                options.testFunctionality = true;
            end
            
            if ~isfield(options, 'compareMATLAB')
                options.compareMATLAB = true;
            end
            
            if ~isfield(options, 'benchmarkPerformance')
                options.benchmarkPerformance = true;
            end
            
            if ~isfield(options, 'checkMemoryUsage')
                options.checkMemoryUsage = true;
            end
            
            if ~isfield(options, 'validateInputHandling')
                options.validateInputHandling = true;
            end
            
            % Initialize result structure
            result = struct(...
                'allMEXFiles', struct(), ...
                'summary', struct(...
                    'totalMEXFiles', 0, ...
                    'existenceChecks', struct('passed', 0, 'failed', 0), ...
                    'functionalityTests', struct('passed', 0, 'failed', 0), ...
                    'matlabComparisons', struct('passed', 0, 'failed', 0), ...
                    'performanceBenchmarks', struct('passed', 0, 'failed', 0, 'averageImprovement', 0), ...
                    'memoryChecks', struct('passed', 0, 'failed', 0, 'leaksDetected', 0), ...
                    'inputHandlingTests', struct('passed', 0, 'failed', 0) ...
                ), ...
                'timestamp', datestr(now) ...
            );
            
            % Get all MEX files
            mexFileList = obj.getMEXFileList();
            mexFileNames = fieldnames(mexFileList);
            result.summary.totalMEXFiles = length(mexFileNames);
            
            % Process each MEX file
            for i = 1:length(mexFileNames)
                mexName = mexFileNames{i};
                
                % Get MEX file info
                mexInfo = mexFileList.(mexName);
                
                % Initialize per-file result
                fileResult = struct(...
                    'name', mexName, ...
                    'path', mexInfo.path, ...
                    'category', mexInfo.category, ...
                    'exists', false, ...
                    'functionality', struct('status', 'skipped'), ...
                    'matlabComparison', struct('status', 'skipped'), ...
                    'performance', struct('status', 'skipped'), ...
                    'memory', struct('status', 'skipped'), ...
                    'inputHandling', struct('status', 'skipped') ...
                );
                
                % Check existence
                fileResult.exists = obj.validateMEXExists(mexName);
                if fileResult.exists
                    result.summary.existenceChecks.passed = result.summary.existenceChecks.passed + 1;
                else
                    result.summary.existenceChecks.failed = result.summary.existenceChecks.failed + 1;
                    % Skip further tests if file doesn't exist
                    result.allMEXFiles.(mexName) = fileResult;
                    continue;
                end
                
                % Generate appropriate test inputs based on function type
                testInputs = obj.generateTestInputs(mexInfo.category);
                
                % Test functionality
                if options.testFunctionality
                    funcResult = obj.validateMEXFunctionality(mexName, testInputs);
                    fileResult.functionality = funcResult;
                    
                    if strcmp(funcResult.status, 'passed')
                        result.summary.functionalityTests.passed = result.summary.functionalityTests.passed + 1;
                    else
                        result.summary.functionalityTests.failed = result.summary.functionalityTests.failed + 1;
                    end
                end
                
                % Compare with MATLAB implementation
                if options.compareMATLAB && isfield(mexInfo, 'matlabEquivalent')
                    matlabName = mexInfo.matlabEquivalent;
                    compResult = obj.compareMEXWithMATLAB(mexName, matlabName, testInputs);
                    fileResult.matlabComparison = compResult;
                    
                    if strcmp(compResult.status, 'passed')
                        result.summary.matlabComparisons.passed = result.summary.matlabComparisons.passed + 1;
                    else
                        result.summary.matlabComparisons.failed = result.summary.matlabComparisons.failed + 1;
                    end
                end
                
                % Benchmark performance
                if options.benchmarkPerformance && isfield(mexInfo, 'matlabEquivalent')
                    matlabName = mexInfo.matlabEquivalent;
                    perfResult = obj.benchmarkMEXPerformance(mexName, matlabName, testInputs);
                    fileResult.performance = perfResult;
                    
                    if strcmp(perfResult.status, 'passed')
                        result.summary.performanceBenchmarks.passed = result.summary.performanceBenchmarks.passed + 1;
                        
                        % Accumulate performance improvement for average calculation
                        result.summary.performanceBenchmarks.averageImprovement = ...
                            result.summary.performanceBenchmarks.averageImprovement + perfResult.performanceImprovement;
                    else
                        result.summary.performanceBenchmarks.failed = result.summary.performanceBenchmarks.failed + 1;
                    end
                end
                
                % Check memory usage
                if options.checkMemoryUsage
                    memResult = obj.validateMemoryUsage(mexName, testInputs);
                    fileResult.memory = memResult;
                    
                    if strcmp(memResult.status, 'passed')
                        result.summary.memoryChecks.passed = result.summary.memoryChecks.passed + 1;
                        
                        if memResult.hasLeak
                            result.summary.memoryChecks.leaksDetected = result.summary.memoryChecks.leaksDetected + 1;
                        end
                    else
                        result.summary.memoryChecks.failed = result.summary.memoryChecks.failed + 1;
                    end
                end
                
                % Test input handling
                if options.validateInputHandling
                    % Generate appropriate test cases based on function type
                    testCases = obj.generateInputTestCases(mexInfo.category);
                    
                    if ~isempty(testCases)
                        inputResult = obj.validateInputHandling(mexName, testCases);
                        fileResult.inputHandling = inputResult;
                        
                        if strcmp(inputResult.status, 'passed')
                            result.summary.inputHandlingTests.passed = result.summary.inputHandlingTests.passed + 1;
                        else
                            result.summary.inputHandlingTests.failed = result.summary.inputHandlingTests.failed + 1;
                        end
                    end
                end
                
                % Store per-file result
                result.allMEXFiles.(mexName) = fileResult;
            end
            
            % Calculate average performance improvement
            numPerfBenchmarks = result.summary.performanceBenchmarks.passed;
            if numPerfBenchmarks > 0
                result.summary.performanceBenchmarks.averageImprovement = ...
                    result.summary.performanceBenchmarks.averageImprovement / numPerfBenchmarks;
            end
            
            % Store overall validation result
            obj.validationResults.allMEXFiles = result;
        end
        
        function testInputs = generateTestInputs(obj, functionType, options)
            % Generates appropriate test inputs for specific MEX function types
            %
            % INPUTS:
            %   functionType - Type of function ('garch', 'armax', 'likelihood', etc.)
            %   options - Optional structure with generation options
            %
            % OUTPUTS:
            %   testInputs - Cell array of test inputs appropriate for the function type
            
            % Default options
            if nargin < 3
                options = struct();
            end
            
            % Initialize empty test inputs
            testInputs = {};
            
            % Generate inputs based on function type
            switch lower(functionType)
                case 'garch'
                    % Generate test time series data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Create parameter vector for GARCH(1,1)
                    parameters = [0.01; 0.1; 0.85]; % omega, alpha, beta
                    
                    % Initial volatility estimate
                    initialVol = var(data);
                    
                    % Pack inputs
                    testInputs = {data, parameters, initialVol};
                    
                case 'agarch'
                    % Generate test time series data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Create parameter vector for AGARCH(1,1)
                    parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                    
                    % Initial volatility estimate
                    initialVol = var(data);
                    
                    % Pack inputs
                    testInputs = {data, parameters, initialVol};
                    
                case 'egarch'
                    % Generate test time series data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Create parameter vector for EGARCH(1,1)
                    parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
                    
                    % Initial volatility estimate
                    initialVol = log(var(data));
                    
                    % Pack inputs
                    testInputs = {data, parameters, initialVol};
                    
                case 'tarch'
                    % Generate test time series data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Create parameter vector for TARCH(1,1)
                    parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                    
                    % Initial volatility estimate
                    initialVol = var(data);
                    
                    % Pack inputs
                    testInputs = {data, parameters, initialVol};
                    
                case 'igarch'
                    % Generate test time series data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Create parameter vector for IGARCH(1,1)
                    parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                    
                    % Initial volatility estimate
                    initialVol = var(data);
                    
                    % Pack inputs
                    testInputs = {data, parameters, initialVol};
                    
                case 'armax'
                    % Generate test time series data
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
                    
                    % Pack inputs
                    testInputs = {y, parameters, X};
                    
                case 'armaxerrors'
                    % Generate test time series data
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
                    
                    % Pack inputs
                    testInputs = {y, parameters, X};
                    
                case 'likelihood'
                    % Generate test data
                    numObs = 1000;
                    data = randn(numObs, 1);
                    
                    % Standard normal parameters
                    parameters = [0; 1]; % mean, std
                    
                    % Pack inputs
                    testInputs = {data, parameters};
                    
                case 'composite_likelihood'
                    % Generate test correlation matrix (3x3)
                    R = [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1];
                    
                    % Generate multivariate normal data
                    numObs = 500;
                    data = mvnrnd(zeros(3, 1), R, numObs);
                    
                    % Pack inputs
                    testInputs = {data, R};
                    
                otherwise
                    % Default case: create generic numeric inputs
                    warning('MEXValidator:UnknownFunctionType', ...
                        'Unknown function type "%s". Using generic test inputs.', functionType);
                    
                    % Generic matrix and vector inputs
                    A = rand(10, 10);
                    b = rand(10, 1);
                    
                    % Pack inputs
                    testInputs = {A, b};
            end
        end
        
        function testCases = generateInputTestCases(obj, functionType)
            % Generates test cases for input validation testing
            %
            % INPUTS:
            %   functionType - Type of function ('garch', 'armax', etc.)
            %
            % OUTPUTS:
            %   testCases - Structure array of test cases
            
            % Initialize empty test cases
            testCases = struct([]);
            
            % Generate test cases based on function type
            switch lower(functionType)
                case {'garch', 'agarch', 'egarch', 'tarch', 'igarch'}
                    % Base valid inputs
                    numObs = 1000;
                    data = randn(numObs, 1);
                    parameters = [0.01; 0.1; 0.85]; % omega, alpha, beta
                    initialVol = var(data);
                    
                    % Case 1: NaN in data
                    testCases(1).name = 'NaN_in_data';
                    badData = data;
                    badData(10) = NaN;
                    testCases(1).inputs = {badData, parameters, initialVol};
                    testCases(1).expectedError = 'NaN';
                    testCases(1).shouldFail = true;
                    
                    % Case 2: Inf in data
                    testCases(2).name = 'Inf_in_data';
                    badData = data;
                    badData(20) = Inf;
                    testCases(2).inputs = {badData, parameters, initialVol};
                    testCases(2).expectedError = 'infinite';
                    testCases(2).shouldFail = true;
                    
                    % Case 3: Non-positive volatility
                    testCases(3).name = 'Non_positive_vol';
                    testCases(3).inputs = {data, parameters, -1};
                    testCases(3).expectedError = 'positive';
                    testCases(3).shouldFail = true;
                    
                    % Case 4: Empty data
                    testCases(4).name = 'Empty_data';
                    testCases(4).inputs = {[], parameters, initialVol};
                    testCases(4).expectedError = 'empty';
                    testCases(4).shouldFail = true;
                    
                    % Case 5: Invalid parameter shape
                    testCases(5).name = 'Invalid_param_shape';
                    testCases(5).inputs = {data, parameters', initialVol};
                    testCases(5).expectedError = 'column';
                    testCases(5).shouldFail = true;
                    
                case 'armax'
                    % Base valid inputs
                    numObs = 1000;
                    ar = 0.7;
                    ma = 0.3;
                    innovations = randn(numObs+100, 1);
                    y = zeros(numObs+100, 1);
                    
                    for t = 2:length(y)
                        y(t) = ar*y(t-1) + innovations(t) + ma*innovations(t-1);
                    end
                    
                    y = y(101:end);
                    parameters = [ar; ma];
                    X = [];
                    
                    % Case 1: NaN in data
                    testCases(1).name = 'NaN_in_data';
                    badData = y;
                    badData(10) = NaN;
                    testCases(1).inputs = {badData, parameters, X};
                    testCases(1).expectedError = 'NaN';
                    testCases(1).shouldFail = true;
                    
                    % Case 2: Inf in data
                    testCases(2).name = 'Inf_in_data';
                    badData = y;
                    badData(20) = Inf;
                    testCases(2).inputs = {badData, parameters, X};
                    testCases(2).expectedError = 'infinite';
                    testCases(2).shouldFail = true;
                    
                    % Case 3: Empty data
                    testCases(3).name = 'Empty_data';
                    testCases(3).inputs = {[], parameters, X};
                    testCases(3).expectedError = 'empty';
                    testCases(3).shouldFail = true;
                    
                    % Case 4: Invalid parameter length
                    testCases(4).name = 'Invalid_param_length';
                    testCases(4).inputs = {y, [ar], X};
                    testCases(4).expectedError = 'parameter';
                    testCases(4).shouldFail = true;
                    
                case 'likelihood'
                    % Base valid inputs
                    numObs = 1000;
                    data = randn(numObs, 1);
                    parameters = [0; 1]; % mean, std
                    
                    % Case 1: NaN in data
                    testCases(1).name = 'NaN_in_data';
                    badData = data;
                    badData(10) = NaN;
                    testCases(1).inputs = {badData, parameters};
                    testCases(1).expectedError = 'NaN';
                    testCases(1).shouldFail = true;
                    
                    % Case 2: Inf in data
                    testCases(2).name = 'Inf_in_data';
                    badData = data;
                    badData(20) = Inf;
                    testCases(2).inputs = {badData, parameters};
                    testCases(2).expectedError = 'infinite';
                    testCases(2).shouldFail = true;
                    
                    % Case 3: Empty data
                    testCases(3).name = 'Empty_data';
                    testCases(3).inputs = {[], parameters};
                    testCases(3).expectedError = 'empty';
                    testCases(3).shouldFail = true;
                    
                    % Case 4: Invalid parameter values (negative std)
                    testCases(4).name = 'Negative_std';
                    testCases(4).inputs = {data, [0; -1]};
                    testCases(4).expectedError = 'positive';
                    testCases(4).shouldFail = true;
            end
        end
        
        function mexPath = getMEXPath(obj, mexBaseName)
            % Gets the full path to a MEX file with the correct platform-specific extension
            %
            % INPUTS:
            %   mexBaseName - Base name of the MEX file without extension
            %
            % OUTPUTS:
            %   mexPath - Full path to the MEX file
            
            % Get platform-specific extension
            mexExt = obj.getMEXExtension();
            
            % Construct full path
            mexPath = fullfile(obj.mexBasePath, [mexBaseName, '.', mexExt]);
            
            % Verify path exists
            if exist(mexPath, 'file') ~= 3 % 3 = MEX-file
                warning('MEXValidator:MEXNotFound', 'MEX file %s not found', mexPath);
            end
        end
        
        function mexExt = getMEXExtension(obj)
            % Gets the correct MEX file extension for the current platform
            %
            % OUTPUTS:
            %   mexExt - Platform-specific MEX extension
            
            % Determine extension based on platform
            if strncmpi(obj.platform, 'PCW', 3)
                % Windows
                mexExt = 'mexw64';
            elseif strncmpi(obj.platform, 'GLN', 3)
                % Linux
                mexExt = 'mexa64';
            elseif strncmpi(obj.platform, 'MAC', 3)
                % Mac
                mexExt = 'mexmaci64';
            else
                % Unknown platform
                warning('MEXValidator:UnknownPlatform', 'Unknown platform: %s', obj.platform);
                mexExt = 'mex';
            end
        end
        
        function mexList = getMEXFileList(obj)
            % Gets a list of all MEX files in the specified base path
            %
            % OUTPUTS:
            %   mexList - Structure containing MEX file information
            
            % Get platform-specific extension
            mexExt = obj.getMEXExtension();
            
            % Initialize empty list
            mexList = struct();
            
            % Check if mexBasePath exists
            if ~exist(obj.mexBasePath, 'dir')
                warning('MEXValidator:DirectoryNotFound', 'MEX directory %s not found', obj.mexBasePath);
                return;
            end
            
            % Get all MEX files
            dirInfo = dir(fullfile(obj.mexBasePath, ['*.', mexExt]));
            
            % Process each file
            for i = 1:length(dirInfo)
                % Get base name without extension
                [~, baseName, ~] = fileparts(dirInfo(i).name);
                
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
                
                % Build file info structure
                mexList.(baseName) = struct(...
                    'name', baseName, ...
                    'path', fullfile(obj.mexBasePath, dirInfo(i).name), ...
                    'extension', mexExt, ...
                    'category', category ...
                );
                
                % Try to find MATLAB equivalent
                % Convention: mexFile has a MATLAB equivalent named mexFile_matlab
                matlabEquiv = [baseName, '_matlab'];
                if exist(matlabEquiv, 'file') == 2 % 2 = M-file
                    mexList.(baseName).matlabEquivalent = matlabEquiv;
                end
            end
        end
        
        function report = generateValidationReport(obj)
            % Generates a comprehensive MEX validation report
            %
            % OUTPUTS:
            %   report - Detailed validation report
            
            % Initialize report structure
            report = struct(...
                'summary', struct(), ...
                'details', struct(), ...
                'recommendations', {}, ...
                'timestamp', datestr(now) ...
            );
            
            % Check if validation results exist
            if ~isfield(obj.validationResults, 'allMEXFiles')
                warning('MEXValidator:NoResults', 'No validation results available. Run validateAllMEXFiles first.');
                return;
            end
            
            % Copy validation results
            allResults = obj.validationResults.allMEXFiles;
            
            % Generate summary
            report.summary = allResults.summary;
            report.summary.overallStatus = 'passed';
            
            % Calculate success rates
            if report.summary.totalMEXFiles > 0
                report.summary.existenceRate = report.summary.existenceChecks.passed / report.summary.totalMEXFiles * 100;
            else
                report.summary.existenceRate = 0;
            end
            
            totalFunctionality = report.summary.functionalityTests.passed + report.summary.functionalityTests.failed;
            if totalFunctionality > 0
                report.summary.functionalityRate = report.summary.functionalityTests.passed / totalFunctionality * 100;
            else
                report.summary.functionalityRate = 0;
            end
            
            totalComparisons = report.summary.matlabComparisons.passed + report.summary.matlabComparisons.failed;
            if totalComparisons > 0
                report.summary.comparisonRate = report.summary.matlabComparisons.passed / totalComparisons * 100;
            else
                report.summary.comparisonRate = 0;
            end
            
            totalPerformance = report.summary.performanceBenchmarks.passed + report.summary.performanceBenchmarks.failed;
            if totalPerformance > 0
                report.summary.performanceRate = report.summary.performanceBenchmarks.passed / totalPerformance * 100;
                report.summary.averageImprovement = report.summary.performanceBenchmarks.averageImprovement;
            else
                report.summary.performanceRate = 0;
                report.summary.averageImprovement = 0;
            end
            
            totalMemory = report.summary.memoryChecks.passed + report.summary.memoryChecks.failed;
            if totalMemory > 0
                report.summary.memoryRate = report.summary.memoryChecks.passed / totalMemory * 100;
                report.summary.leakRate = report.summary.memoryChecks.leaksDetected / totalMemory * 100;
            else
                report.summary.memoryRate = 0;
                report.summary.leakRate = 0;
            end
            
            % Copy detailed results
            report.details = allResults.allMEXFiles;
            
            % Generate recommendations
            recommendations = {};
            
            % Check performance requirement
            if report.summary.averageImprovement < 50
                recommendations{end+1} = sprintf(['Performance improvement (%.1f%%) does not meet the 50%% requirement. ', ...
                    'Consider further optimization of critical MEX functions.'], report.summary.averageImprovement);
                report.summary.overallStatus = 'warning';
            end
            
            % Check for memory leaks
            if report.summary.leakRate > 0
                recommendations{end+1} = sprintf(['Memory leaks detected in %.1f%% of MEX functions. ', ...
                    'Review memory management in affected functions.'], report.summary.leakRate);
                report.summary.overallStatus = 'warning';
            end
            
            % Check for failed functionality tests
            if report.summary.functionalityRate < 100
                recommendations{end+1} = sprintf(['%.1f%% of MEX functions failed basic functionality tests. ', ...
                    'Debug affected functions for correct operation.'], 100 - report.summary.functionalityRate);
                report.summary.overallStatus = 'warning';
            end
            
            % Check for failed comparisons
            if report.summary.comparisonRate < 100
                recommendations{end+1} = sprintf(['%.1f%% of MEX functions do not match their MATLAB counterparts. ', ...
                    'Verify numerical precision and algorithm implementations.'], 100 - report.summary.comparisonRate);
                report.summary.overallStatus = 'warning';
            end
            
            % Store recommendations
            report.recommendations = recommendations;
        end
    end
end