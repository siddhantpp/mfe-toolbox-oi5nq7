classdef PlatformCompatibilityTest < BaseTest
    % PLATFORMCOMPATIBILITYTEST A test class that validates cross-platform compatibility of the MFE Toolbox by comparing results between Windows and Unix implementations to ensure numerical consistency and reliability.
    %
    % This test class validates that computational results, especially from MEX
    % implementations, remain numerically consistent regardless of the platform,
    % ensuring reliable financial econometric analysis in heterogeneous computing
    % environments.
    %
    % The class tests cross-platform compatibility of core MEX components:
    %   - agarch_core: AGARCH model implementation
    %   - tarch_core: TARCH/GARCH model implementation
    %   - egarch_core: EGARCH model implementation
    %   - igarch_core: IGARCH model implementation
    %   - armaxerrors: ARMAX error computation
    %   - composite_likelihood: Composite likelihood computation
    %
    % It validates that:
    %   - MEX binaries are available for all supported platforms
    %   - Results are numerically consistent within defined tolerance
    %   - Error handling is consistent across platforms
    %   - Performance characteristics are reasonably consistent
    %   - Large array handling behaves consistently
    %
    % Example:
    %   % Create and run cross-platform compatibility tests
    %   test = PlatformCompatibilityTest();
    %   results = test.runAllTests();
    %
    %   % Generate a comprehensive compatibility report
    %   report = test.generateCompatibilityReport();
    %
    % See also: BaseTest, CrossPlatformValidator, MEXValidator, NumericalComparator
    
    properties
        % Current platform identifier
        currentPlatform
        
        % Supported platform identifiers
        supportedPlatforms
        
        % Cross-platform validator instance
        crossValidator
        
        % MEX validator instance
        mexValidator
        
        % Numerical comparator for result validation
        numComparator
        
        % Structure to store compatibility results
        compatibilityResults
        
        % Tolerance for cross-platform comparisons
        crossPlatformTolerance
        
        % Path to reference results
        referenceResultsPath
        
        % Core MEX components to test
        mexComponents
    end
    
    methods
        function obj = PlatformCompatibilityTest()
            % Initialize a new PlatformCompatibilityTest instance with cross-platform validation capabilities
            
            % Call parent constructor with class name
            obj@BaseTest('PlatformCompatibilityTest');
            
            % Determine current platform
            obj.currentPlatform = computer();
            
            % Set supported platforms
            obj.supportedPlatforms = {'PCWIN64', 'GLNXA64'};
            
            % Set reference results path
            obj.referenceResultsPath = 'src/test/data/cross_platform';
            
            % Create CrossPlatformValidator instance for cross-platform testing
            obj.crossValidator = CrossPlatformValidator();
            
            % Create MEXValidator instance for MEX binary validation
            obj.mexValidator = MEXValidator();
            
            % Create NumericalComparator with appropriate tolerance for cross-platform comparison
            obj.numComparator = NumericalComparator();
            
            % Set cross-platform tolerance
            obj.crossPlatformTolerance = 1e-9;
            
            % Initialize empty compatibility results structure
            obj.compatibilityResults = struct();
            
            % Define MEX components to test
            obj.mexComponents = {'agarch_core', 'tarch_core', 'egarch_core', 'igarch_core', ...
                               'armaxerrors', 'composite_likelihood'};
        end
        
        function setUp(obj)
            % Prepare the test environment before each test execution
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Verify that validation objects are properly initialized
            if isempty(obj.crossValidator) || isempty(obj.mexValidator) || isempty(obj.numComparator)
                error('PlatformCompatibilityTest:SetupFailed', ...
                      'Validation objects not properly initialized');
            end
            
            % Ensure reference results directory exists or create it
            if ~exist(obj.referenceResultsPath, 'dir')
                mkdir(obj.referenceResultsPath);
            end
            
            % Reset compatibility results structure for new test
            obj.compatibilityResults = struct('testName', '', ...
                                          'timestamp', datestr(now), ...
                                          'platform', obj.currentPlatform, ...
                                          'status', 'running');
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test execution
            
            % Record test completion status in compatibilityResults
            obj.compatibilityResults.status = 'completed';
            obj.compatibilityResults.endTimestamp = datestr(now);
            
            % Clean up any temporary resources
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testMEXBinaryAvailability(obj)
            % Tests that MEX binaries for all supported platforms are available in the system
            
            % Initialize test results
            testResults = struct('componentResults', struct(), ...
                               'summary', struct('total', length(obj.mexComponents), ...
                                                'available', 0, ...
                                                'missing', 0));
            
            % For each MEX component
            for i = 1:length(obj.mexComponents)
                componentName = obj.mexComponents{i};
                componentResult = struct('name', componentName, 'platforms', struct());
                
                % Check binary availability for each supported platform
                for j = 1:length(obj.supportedPlatforms)
                    platform = obj.supportedPlatforms{j};
                    
                    % Get platform-specific MEX extension
                    if strcmp(platform, 'PCWIN64')
                        mexExt = 'mexw64';
                    else % GLNXA64
                        mexExt = 'mexa64';
                    end
                    
                    % Check if binary exists
                    mexPath = obj.mexValidator.getMEXPath(componentName);
                    platformPath = strrep(mexPath, obj.mexValidator.getMEXExtension(), mexExt);
                    
                    binaryExists = exist(platformPath, 'file') == 3; % 3 = MEX-file
                    
                    % Record result
                    componentResult.platforms.(platform) = struct('exists', binaryExists, ...
                                                               'path', platformPath);
                    
                    % Update summary
                    if binaryExists
                        testResults.summary.available = testResults.summary.available + 1;
                    else
                        testResults.summary.missing = testResults.summary.missing + 1;
                    end
                end
                
                % Store component result
                testResults.componentResults.(componentName) = componentResult;
            end
            
            % Assert that all MEX binaries are available for all platforms
            allAvailable = (testResults.summary.missing == 0);
            obj.assertTrue(allAvailable, 'Not all MEX binaries are available for all supported platforms');
            
            % Record availability status in compatibilityResults
            obj.compatibilityResults.binaryAvailability = testResults;
        end
        
        function testAgarchCoreCrossPlatformConsistency(obj)
            % Tests that AGARCH core MEX implementation produces consistent results across platforms
            
            % Generate test inputs for AGARCH model
            testInputs = obj.createTestInputs('agarch_core');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('agarch_core', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('AGARCH core results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.agarch_core = validationResult;
        end
        
        function testTarchCoreCrossPlatformConsistency(obj)
            % Tests that TARCH core MEX implementation produces consistent results across platforms
            
            % Generate test inputs for TARCH model
            testInputs = obj.createTestInputs('tarch_core');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('tarch_core', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('TARCH core results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.tarch_core = validationResult;
        end
        
        function testEgarchCoreCrossPlatformConsistency(obj)
            % Tests that EGARCH core MEX implementation produces consistent results across platforms
            
            % Generate test inputs for EGARCH model
            testInputs = obj.createTestInputs('egarch_core');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('egarch_core', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('EGARCH core results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.egarch_core = validationResult;
        end
        
        function testIgarchCoreCrossPlatformConsistency(obj)
            % Tests that IGARCH core MEX implementation produces consistent results across platforms
            
            % Generate test inputs for IGARCH model
            testInputs = obj.createTestInputs('igarch_core');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('igarch_core', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('IGARCH core results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.igarch_core = validationResult;
        end
        
        function testArmaxerrorsCrossPlatformConsistency(obj)
            % Tests that ARMAX errors MEX implementation produces consistent results across platforms
            
            % Generate test inputs for ARMAX error computation
            testInputs = obj.createTestInputs('armaxerrors');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('armaxerrors', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('ARMAX errors results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.armaxerrors = validationResult;
        end
        
        function testCompositeLikelihoodCrossPlatformConsistency(obj)
            % Tests that composite likelihood MEX implementation produces consistent results across platforms
            
            % Generate test inputs for composite likelihood computation
            testInputs = obj.createTestInputs('composite_likelihood');
            
            % Validate component across platforms
            validationResult = obj.validateComponentAcrossPlatforms('composite_likelihood', testInputs);
            
            % Assert that results are consistent within defined tolerance
            obj.assertTrue(validationResult.isCompatible, ...
                sprintf('Composite likelihood results are not consistent across platforms: %s', ...
                validationResult.details));
            
            % Record comparison metrics in compatibilityResults
            obj.compatibilityResults.composite_likelihood = validationResult;
        end
        
        function testFloatingPointConsistency(obj)
            % Tests floating-point behavior consistency between platforms using critical financial calculations
            
            % Create test cases specifically targeting floating-point edge cases
            testCases = struct();
            
            % Test case 1: Addition of very different magnitudes
            testCases(1).name = 'magnitude_addition';
            testCases(1).func = @() 1e20 + 1 - 1e20;
            testCases(1).expectedResult = 0;
            testCases(1).tolerance = 1e-10;
            
            % Test case 2: Subtraction near zero
            testCases(2).name = 'subtraction_near_zero';
            testCases(2).func = @() 1.0 - 0.9 - 0.1;
            testCases(2).expectedResult = 0;
            testCases(2).tolerance = 1e-14;
            
            % Test case 3: Accumulation (represents iterative algorithms)
            testCases(3).name = 'accumulation';
            testCases(3).func = @() sum(ones(1000, 1) * 0.1) - 100;
            testCases(3).expectedResult = 0;
            testCases(3).tolerance = 1e-12;
            
            % Test case 4: Special values (NaN handling)
            testCases(4).name = 'nan_handling';
            testCases(4).func = @() sum(isnan([1, NaN, 3]));
            testCases(4).expectedResult = 1;
            testCases(4).tolerance = 0;
            
            % Test case 5: Overflow handling
            testCases(5).name = 'overflow_handling';
            testCases(5).func = @() isfinite(exp(1000));
            testCases(5).expectedResult = false;
            testCases(5).tolerance = 0;
            
            % Initialize results
            fpResults = struct('testCases', struct(), ...
                            'summary', struct('total', length(testCases), ...
                                           'passed', 0, ...
                                           'failed', 0));
            
            % Execute test cases
            for i = 1:length(testCases)
                testCase = testCases(i);
                result = testCase.func();
                
                % Compare with expected result
                if isnumeric(result) && isnumeric(testCase.expectedResult)
                    isEqual = abs(result - testCase.expectedResult) <= testCase.tolerance;
                else
                    isEqual = isequal(result, testCase.expectedResult);
                end
                
                % Record result
                fpResults.testCases.(testCase.name) = struct('result', result, ...
                                                         'expected', testCase.expectedResult, ...
                                                         'isEqual', isEqual);
                
                % Update summary
                if isEqual
                    fpResults.summary.passed = fpResults.summary.passed + 1;
                else
                    fpResults.summary.failed = fpResults.summary.failed + 1;
                end
            end
            
            % Assert that all test cases passed
            obj.assertTrue(fpResults.summary.failed == 0, ...
                'Floating-point behavior is not consistent across platforms');
            
            % Record floating-point consistency results
            obj.compatibilityResults.floatingPoint = fpResults;
        end
        
        function testErrorHandlingConsistency(obj)
            % Tests that error handling is consistent across platforms for invalid inputs
            
            % Initialize results
            errorResults = struct('componentResults', struct(), ...
                               'summary', struct('total', length(obj.mexComponents), ...
                                              'consistent', 0, ...
                                              'inconsistent', 0));
            
            % For each MEX component
            for i = 1:length(obj.mexComponents)
                componentName = obj.mexComponents{i};
                
                % Create invalid inputs
                invalidInputs = obj.createInvalidTestInputs(componentName);
                
                % Initialize component result
                componentResult = struct('name', componentName, ...
                                      'errorConsistency', true, ...
                                      'testCases', struct());
                
                % Test each invalid input case
                for j = 1:length(invalidInputs)
                    testCase = invalidInputs(j);
                    
                    % Create test case result
                    caseResult = struct('name', testCase.name, ...
                                      'inputs', {testCase.inputs}, ...
                                      'currentPlatformError', '', ...
                                      'otherPlatformError', '', ...
                                      'isConsistent', false);
                    
                    % Execute on current platform
                    try
                        func = str2func(componentName);
                        func(testCase.inputs{:});
                        % If no error occurred
                        caseResult.currentPlatformError = 'No error';
                    catch ME
                        caseResult.currentPlatformError = ME.message;
                    end
                    
                    % Load reference errors from other platforms
                    if isfield(testCase, 'referenceErrors')
                        caseResult.otherPlatformError = testCase.referenceErrors;
                        
                        % Compare error messages (not expecting exact match, but similar content)
                        errorSimilarity = obj.getErrorSimilarity(caseResult.currentPlatformError, ...
                                                              caseResult.otherPlatformError);
                        
                        caseResult.errorSimilarity = errorSimilarity;
                        caseResult.isConsistent = (errorSimilarity > 0.7); % Threshold for similarity
                    else
                        % No reference errors available
                        caseResult.isConsistent = true; % Assume consistency if no reference
                    end
                    
                    % Update component consistency
                    componentResult.errorConsistency = componentResult.errorConsistency && caseResult.isConsistent;
                    
                    % Store test case result
                    componentResult.testCases.(testCase.name) = caseResult;
                end
                
                % Update summary
                if componentResult.errorConsistency
                    errorResults.summary.consistent = errorResults.summary.consistent + 1;
                else
                    errorResults.summary.inconsistent = errorResults.summary.inconsistent + 1;
                end
                
                % Store component result
                errorResults.componentResults.(componentName) = componentResult;
            end
            
            % Assert that error handling is consistent across platforms
            obj.assertTrue(errorResults.summary.inconsistent == 0, ...
                'Error handling is not consistent across platforms');
            
            % Record error handling consistency results
            obj.compatibilityResults.errorHandling = errorResults;
        end
        
        function testPerformanceConsistency(obj)
            % Tests that performance characteristics are reasonably consistent across platforms
            
            % Initialize results
            perfResults = struct('componentResults', struct(), ...
                              'summary', struct('total', length(obj.mexComponents), ...
                                             'consistent', 0, ...
                                             'inconsistent', 0));
            
            % Define acceptable performance variation (relative ratio)
            maxAcceptableRatio = 5.0; % Allow up to 5x difference between platforms
            
            % For each MEX component
            for i = 1:length(obj.mexComponents)
                componentName = obj.mexComponents{i};
                
                % Create test inputs
                testInputs = obj.createTestInputs(componentName);
                
                % Measure execution time on current platform
                func = str2func(componentName);
                
                % Warmup run
                func(testInputs{:});
                
                % Timed runs
                numRuns = 10;
                executionTimes = zeros(numRuns, 1);
                
                for run = 1:numRuns
                    tic;
                    func(testInputs{:});
                    executionTimes(run) = toc;
                end
                
                % Calculate statistics
                meanTime = mean(executionTimes);
                stdTime = std(executionTimes);
                
                % Initialize component result
                componentResult = struct('name', componentName, ...
                                      'currentPlatform', obj.currentPlatform, ...
                                      'meanTime', meanTime, ...
                                      'stdTime', stdTime, ...
                                      'isConsistent', true, ...
                                      'referenceTimes', struct());
                
                % Load reference performance metrics from other platforms
                referenceFound = false;
                
                for j = 1:length(obj.supportedPlatforms)
                    platform = obj.supportedPlatforms{j};
                    
                    % Skip current platform
                    if strcmp(platform, obj.currentPlatform)
                        continue;
                    end
                    
                    % Try to load reference data
                    refPath = fullfile(obj.referenceResultsPath, componentName, [platform, '_perf.mat']);
                    
                    if exist(refPath, 'file')
                        try
                            refData = load(refPath);
                            componentResult.referenceTimes.(platform) = refData.perfData;
                            
                            % Calculate performance ratio
                            ratio = meanTime / refData.perfData.meanTime;
                            componentResult.ratios.(platform) = ratio;
                            
                            % Check if performance is within acceptable range
                            if ratio > maxAcceptableRatio || ratio < 1/maxAcceptableRatio
                                componentResult.isConsistent = false;
                                componentResult.consistencyIssue = sprintf('Performance ratio with %s is %g (outside range [%g, %g])', ...
                                    platform, ratio, 1/maxAcceptableRatio, maxAcceptableRatio);
                            end
                            
                            referenceFound = true;
                        catch
                            warning('PlatformCompatibilityTest:LoadError', ...
                                'Failed to load reference performance data for %s on %s', ...
                                componentName, platform);
                        end
                    end
                end
                
                % Save current performance data for reference by other platforms
                perfData = struct('meanTime', meanTime, 'stdTime', stdTime, 'platform', obj.currentPlatform);
                perfDir = fullfile(obj.referenceResultsPath, componentName);
                if ~exist(perfDir, 'dir')
                    mkdir(perfDir);
                end
                save(fullfile(perfDir, [obj.currentPlatform, '_perf.mat']), 'perfData');
                
                % If no reference found, consider consistent (benefit of doubt)
                if ~referenceFound
                    componentResult.isConsistent = true;
                    componentResult.note = 'No reference data available from other platforms';
                end
                
                % Update summary
                if componentResult.isConsistent
                    perfResults.summary.consistent = perfResults.summary.consistent + 1;
                else
                    perfResults.summary.inconsistent = perfResults.summary.inconsistent + 1;
                end
                
                % Store component result
                perfResults.componentResults.(componentName) = componentResult;
            end
            
            % Assert that performance characteristics are reasonably consistent
            obj.assertTrue(perfResults.summary.inconsistent == 0, ...
                'Performance characteristics are not reasonably consistent across platforms');
            
            % Record performance consistency results
            obj.compatibilityResults.performance = perfResults;
        end
        
        function testLargeArrayHandling(obj)
            % Tests that large array handling is consistent across platforms
            
            % Initialize results
            largeArrayResults = struct('componentResults', struct(), ...
                                    'summary', struct('total', length(obj.mexComponents), ...
                                                   'consistent', 0, ...
                                                   'inconsistent', 0));
            
            % For each MEX component
            for i = 1:length(obj.mexComponents)
                componentName = obj.mexComponents{i};
                
                % Create large test data arrays
                largeInputs = obj.createLargeTestInputs(componentName);
                
                % Initialize component result
                componentResult = struct('name', componentName, ...
                                      'currentPlatform', obj.currentPlatform, ...
                                      'isConsistent', true, ...
                                      'errorMessage', '', ...
                                      'memoryUsed', 0);
                
                % Execute MEX function with large arrays on current platform
                func = str2func(componentName);
                
                try
                    % Measure memory before
                    memBefore = whos();
                    memBeforeBytes = sum([memBefore.bytes]);
                    
                    % Execute function
                    result = func(largeInputs{:});
                    
                    % Measure memory after
                    memAfter = whos();
                    memAfterBytes = sum([memAfter.bytes]);
                    
                    % Calculate memory used
                    componentResult.memoryUsed = memAfterBytes - memBeforeBytes;
                    componentResult.outputSize = whos('result');
                    componentResult.executionSuccess = true;
                catch ME
                    componentResult.executionSuccess = false;
                    componentResult.errorMessage = ME.message;
                end
                
                % Load reference results from other platforms
                referenceFound = false;
                
                for j = 1:length(obj.supportedPlatforms)
                    platform = obj.supportedPlatforms{j};
                    
                    % Skip current platform
                    if strcmp(platform, obj.currentPlatform)
                        continue;
                    end
                    
                    % Try to load reference data
                    refPath = fullfile(obj.referenceResultsPath, componentName, [platform, '_large.mat']);
                    
                    if exist(refPath, 'file')
                        try
                            refData = load(refPath);
                            componentResult.reference.(platform) = refData.largeArrayData;
                            
                            % Compare execution success
                            if componentResult.executionSuccess ~= refData.largeArrayData.executionSuccess
                                componentResult.isConsistent = false;
                                componentResult.consistencyIssue = sprintf('Execution success differs from %s', platform);
                            end
                            
                            % If both succeeded, compare memory usage pattern (not exact amount)
                            if componentResult.executionSuccess && refData.largeArrayData.executionSuccess
                                % Check if memory usage differs by more than 3x
                                memRatio = componentResult.memoryUsed / refData.largeArrayData.memoryUsed;
                                if memRatio > 3 || memRatio < 1/3
                                    componentResult.isConsistent = false;
                                    componentResult.consistencyIssue = sprintf('Memory usage ratio with %s is %g (outside range [%g, %g])', ...
                                        platform, memRatio, 1/3, 3);
                                end
                            end
                            
                            referenceFound = true;
                        catch
                            warning('PlatformCompatibilityTest:LoadError', ...
                                'Failed to load reference large array data for %s on %s', ...
                                componentName, platform);
                        end
                    end
                end
                
                % Save current large array data for reference by other platforms
                largeArrayData = struct('executionSuccess', componentResult.executionSuccess, ...
                                     'memoryUsed', componentResult.memoryUsed, ...
                                     'platform', obj.currentPlatform);
                
                if isfield(componentResult, 'errorMessage')
                    largeArrayData.errorMessage = componentResult.errorMessage;
                end
                
                largeDir = fullfile(obj.referenceResultsPath, componentName);
                if ~exist(largeDir, 'dir')
                    mkdir(largeDir);
                end
                save(fullfile(largeDir, [obj.currentPlatform, '_large.mat']), 'largeArrayData');
                
                % If no reference found, consider consistent (benefit of doubt)
                if ~referenceFound
                    componentResult.isConsistent = true;
                    componentResult.note = 'No reference data available from other platforms';
                end
                
                % Update summary
                if componentResult.isConsistent
                    largeArrayResults.summary.consistent = largeArrayResults.summary.consistent + 1;
                else
                    largeArrayResults.summary.inconsistent = largeArrayResults.summary.inconsistent + 1;
                end
                
                % Store component result
                largeArrayResults.componentResults.(componentName) = componentResult;
            end
            
            % Assert that large array handling is consistent across platforms
            obj.assertTrue(largeArrayResults.summary.inconsistent == 0, ...
                'Large array handling is not consistent across platforms');
            
            % Record large array handling results
            obj.compatibilityResults.largeArrayHandling = largeArrayResults;
        end
        
        function testInputs = createTestInputs(obj, componentName)
            % Creates appropriate test inputs for cross-platform testing of a specific MEX component
            %
            % INPUTS:
            %   componentName - Name of the component
            %
            % OUTPUTS:
            %   testInputs - Cell array of test inputs appropriate for the component
            
            % Determine component type from componentName
            if strncmpi(componentName, 'agarch', 6) || strcmp(componentName, 'agarch_core')
                % Generate test inputs for AGARCH
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif strncmpi(componentName, 'tarch', 5) || strcmp(componentName, 'tarch_core')
                % Generate test inputs for TARCH
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.05; 0.85; 0.1]; % omega, alpha, beta, gamma
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif strncmpi(componentName, 'egarch', 6) || strcmp(componentName, 'egarch_core')
                % Generate test inputs for EGARCH
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [-0.1; 0.1; 0.95; 0.1]; % omega, alpha, beta, gamma
                initialVol = log(var(data));
                testInputs = {data, parameters, initialVol};
                
            elseif strncmpi(componentName, 'igarch', 6) || strcmp(componentName, 'igarch_core')
                % Generate test inputs for IGARCH
                numObs = 1000;
                data = randn(numObs, 1);
                parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                initialVol = var(data);
                testInputs = {data, parameters, initialVol};
                
            elseif strncmpi(componentName, 'armax', 5) || strcmp(componentName, 'armaxerrors')
                % Generate test inputs for ARMAX
                numObs = 1000;
                
                % Generate an ARMA(1,1) process
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
                
            elseif strncmpi(componentName, 'composite', 9) || strcmp(componentName, 'composite_likelihood')
                % Generate test inputs for composite likelihood
                % Create a 3x3 correlation matrix
                R = [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1];
                
                % Generate multivariate normal data
                numObs = 500;
                data = mvnrnd(zeros(3, 1), R, numObs);
                
                testInputs = {data, R};
                
            else
                % Default case for unknown components
                warning('PlatformCompatibilityTest:UnknownComponent', ...
                    'Unknown component "%s". Using generic test inputs.', componentName);
                
                % Create generic test data
                testInputs = {randn(100, 1), [0.5; 0.3]};
            end
        end
        
        function validationResult = validateComponentAcrossPlatforms(obj, componentName, testInputs, tolerance)
            % Validates a specific component for cross-platform consistency
            %
            % INPUTS:
            %   componentName - Name of the component to validate
            %   testInputs - Cell array of inputs for testing the component
            %   tolerance - Numerical tolerance for comparison
            %
            % OUTPUTS:
            %   validationResult - Validation results structure
            
            % Generate test inputs if not provided
            if nargin < 3 || isempty(testInputs)
                testInputs = obj.createTestInputs(componentName);
            end
            
            % Use default tolerance if not provided
            if nargin < 4 || isempty(tolerance)
                tolerance = obj.crossPlatformTolerance;
            end
            
            % Use CrossPlatformValidator to validate component across platforms
            result = obj.crossValidator.validateComponent(componentName, testInputs, tolerance);
            
            % Initialize validation result structure
            validationResult = struct(...
                'componentName', componentName, ...
                'isCompatible', result.isCompatible, ...
                'platforms', {result.platforms}, ...
                'maxDifference', result.maxDifference, ...
                'details', '', ...
                'differences', struct(), ...
                'timestamp', datestr(now));
            
            % Add detailed information about differences
            if ~result.isCompatible && isfield(result, 'differences')
                diffFields = fieldnames(result.differences);
                
                % Compile details string
                details = '';
                for i = 1:length(diffFields)
                    diffField = diffFields{i};
                    diff = result.differences.(diffField);
                    
                    if isfield(diff, 'maxAbsoluteDifference')
                        details = [details, sprintf('%s: Max diff = %g; ', ...
                            diffField, diff.maxAbsoluteDifference)];
                    end
                    
                    % Store detailed difference information
                    validationResult.differences.(diffField) = diff;
                end
                
                validationResult.details = details;
            end
        end
        
        function report = generateCompatibilityReport(obj)
            % Generates a comprehensive cross-platform compatibility report
            %
            % OUTPUTS:
            %   report - Detailed compatibility report
            
            % Initialize report structure
            report = struct(...
                'title', 'Cross-Platform Compatibility Report', ...
                'timestamp', datestr(now), ...
                'platform', obj.currentPlatform, ...
                'supportedPlatforms', {obj.supportedPlatforms}, ...
                'summary', struct(), ...
                'componentResults', struct(), ...
                'recommendations', {{}});
            
            % Gather component status
            components = obj.mexComponents;
            
            % Initialize summary counters
            total = length(components);
            compatible = 0;
            incompatible = 0;
            untested = 0;
            
            % Process each component
            for i = 1:length(components)
                componentName = components{i};
                
                % Check if component was tested
                if isfield(obj.compatibilityResults, componentName)
                    result = obj.compatibilityResults.(componentName);
                    
                    % Count compatible/incompatible
                    if result.isCompatible
                        compatible = compatible + 1;
                    else
                        incompatible = incompatible + 1;
                    end
                    
                    % Store component result
                    report.componentResults.(componentName) = result;
                else
                    % Component not tested
                    untested = untested + 1;
                    report.componentResults.(componentName) = struct(...
                        'componentName', componentName, ...
                        'isCompatible', false, ...
                        'status', 'untested');
                end
            end
            
            % Create summary
            report.summary = struct(...
                'totalComponents', total, ...
                'compatibleComponents', compatible, ...
                'incompatibleComponents', incompatible, ...
                'untestedComponents', untested, ...
                'compatibilityRate', compatible / total * 100);
            
            % Generate platform-specific analysis
            for i = 1:length(obj.supportedPlatforms)
                platform = obj.supportedPlatforms{i};
                
                % Skip current platform
                if strcmp(platform, obj.currentPlatform)
                    continue;
                end
                
                % Analyze compatibility with this platform
                platformAnalysis = struct(...
                    'platform', platform, ...
                    'compatibleCount', 0, ...
                    'incompatibleCount', 0, ...
                    'untestableCount', 0, ...
                    'components', struct());
                
                % Check each component
                for j = 1:length(components)
                    componentName = components{j};
                    
                    if isfield(obj.compatibilityResults, componentName)
                        result = obj.compatibilityResults.(componentName);
                        
                        % Check if tested with this platform
                        if ismember(platform, result.platforms)
                            if result.isCompatible
                                platformAnalysis.compatibleCount = platformAnalysis.compatibleCount + 1;
                                platformAnalysis.components.(componentName) = 'compatible';
                            else
                                platformAnalysis.incompatibleCount = platformAnalysis.incompatibleCount + 1;
                                platformAnalysis.components.(componentName) = 'incompatible';
                            end
                        else
                            platformAnalysis.untestableCount = platformAnalysis.untestableCount + 1;
                            platformAnalysis.components.(componentName) = 'untestable';
                        end
                    else
                        platformAnalysis.untestableCount = platformAnalysis.untestableCount + 1;
                        platformAnalysis.components.(componentName) = 'untested';
                    end
                end
                
                % Calculate compatibility rate
                testableCount = platformAnalysis.compatibleCount + platformAnalysis.incompatibleCount;
                if testableCount > 0
                    platformAnalysis.compatibilityRate = platformAnalysis.compatibleCount / testableCount * 100;
                else
                    platformAnalysis.compatibilityRate = 0;
                end
                
                % Add platform analysis to report
                report.platformAnalysis.(platform) = platformAnalysis;
            end
            
            % Generate recommendations
            recommendations = {};
            
            % General compatibility recommendation
            if report.summary.compatibilityRate < 80
                recommendations{end+1} = sprintf(['Overall cross-platform compatibility rate (%.1f%%) is low. ', ...
                    'Focus on improving platform-specific numerical consistency.'], ...
                    report.summary.compatibilityRate);
            elseif report.summary.compatibilityRate < 95
                recommendations{end+1} = sprintf(['Cross-platform compatibility rate (%.1f%%) is moderate. ', ...
                    'Investigate and fix inconsistencies in the %d identified components.'], ...
                    report.summary.compatibilityRate, report.summary.incompatibleComponents);
            else
                recommendations{end+1} = sprintf(['Cross-platform compatibility rate (%.1f%%) is excellent. ', ...
                    'Continue monitoring for platform-specific differences in new components.'], ...
                    report.summary.compatibilityRate);
            end
            
            % Specific component recommendations
            if incompatible > 0
                % Get list of incompatible components
                incompatibleList = {};
                for i = 1:length(components)
                    componentName = components{i};
                    if isfield(obj.compatibilityResults, componentName) && ...
                       ~obj.compatibilityResults.(componentName).isCompatible
                        incompatibleList{end+1} = componentName;
                    end
                end
                
                % Add recommendation for incompatible components
                recommendations{end+1} = sprintf(['The following components have cross-platform inconsistencies: ', ...
                    '%s. Review MEX implementations for platform-specific numerical behavior.'], ...
                    strjoin(incompatibleList, ', '));
            end
            
            % Platform-specific recommendations
            for i = 1:length(obj.supportedPlatforms)
                platform = obj.supportedPlatforms{i};
                
                % Skip current platform
                if strcmp(platform, obj.currentPlatform)
                    continue;
                end
                
                if isfield(report, 'platformAnalysis') && isfield(report.platformAnalysis, platform)
                    platformReport = report.platformAnalysis.(platform);
                    
                    if platformReport.compatibilityRate < 80
                        recommendations{end+1} = sprintf(['Compatibility with %s platform is low (%.1f%%). ', ...
                            'Generate reference results on %s for comparison.'], ...
                            platform, platformReport.compatibilityRate, platform);
                    end
                else
                    recommendations{end+1} = sprintf(['No compatibility data available for %s platform. ', ...
                        'Generate reference results on %s for comparison.'], platform, platform);
                end
            end
            
            % Store recommendations
            report.recommendations = recommendations;
        end
        
        function invalidInputs = createInvalidTestInputs(obj, componentName)
            % Creates invalid test inputs for error handling consistency testing
            %
            % INPUTS:
            %   componentName - Name of the component
            %
            % OUTPUTS:
            %   invalidInputs - Structure array of invalid test cases
            
            % Determine component type from componentName
            if strncmpi(componentName, 'agarch', 6) || strcmp(componentName, 'agarch_core') || ...
               strncmpi(componentName, 'tarch', 5) || strcmp(componentName, 'tarch_core') || ...
               strncmpi(componentName, 'egarch', 6) || strcmp(componentName, 'egarch_core') || ...
               strncmpi(componentName, 'igarch', 6) || strcmp(componentName, 'igarch_core')
                % Invalid test cases for GARCH family models
                
                % Base valid data
                numObs = 100;
                data = randn(numObs, 1);
                
                % For IGARCH
                if strncmpi(componentName, 'igarch', 6) || strcmp(componentName, 'igarch_core')
                    parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                else
                    parameters = [0.01; 0.1; 0.85; 0.1]; % omega, alpha, beta, gamma
                end
                
                initialVol = var(data);
                
                % Create test cases
                invalidInputs = struct([]);
                
                % Case 1: NaN in data
                invalidInputs(1).name = 'nan_in_data';
                badData = data;
                badData(10) = NaN;
                invalidInputs(1).inputs = {badData, parameters, initialVol};
                invalidInputs(1).referenceErrors = 'NaN';
                
                % Case 2: Inf in data
                invalidInputs(2).name = 'inf_in_data';
                badData = data;
                badData(20) = Inf;
                invalidInputs(2).inputs = {badData, parameters, initialVol};
                invalidInputs(2).referenceErrors = 'Inf';
                
                % Case 3: Empty data
                invalidInputs(3).name = 'empty_data';
                invalidInputs(3).inputs = {[], parameters, initialVol};
                invalidInputs(3).referenceErrors = 'empty';
                
                % Case 4: Negative volatility
                invalidInputs(4).name = 'negative_vol';
                invalidInputs(4).inputs = {data, parameters, -1};
                invalidInputs(4).referenceErrors = 'positive';
                
                % Case 5: Wrong parameter size
                invalidInputs(5).name = 'wrong_param_size';
                invalidInputs(5).inputs = {data, parameters(1:end-1), initialVol};
                invalidInputs(5).referenceErrors = 'parameter';
                
            elseif strncmpi(componentName, 'armax', 5) || strcmp(componentName, 'armaxerrors')
                % Invalid test cases for ARMAX models
                
                % Base valid data
                numObs = 100;
                y = randn(numObs, 1);
                parameters = [0.5; 0.3]; % AR, MA
                X = [];
                
                % Create test cases
                invalidInputs = struct([]);
                
                % Case 1: NaN in data
                invalidInputs(1).name = 'nan_in_data';
                badData = y;
                badData(10) = NaN;
                invalidInputs(1).inputs = {badData, parameters, X};
                invalidInputs(1).referenceErrors = 'NaN';
                
                % Case 2: Inf in data
                invalidInputs(2).name = 'inf_in_data';
                badData = y;
                badData(20) = Inf;
                invalidInputs(2).inputs = {badData, parameters, X};
                invalidInputs(2).referenceErrors = 'Inf';
                
                % Case 3: Empty data
                invalidInputs(3).name = 'empty_data';
                invalidInputs(3).inputs = {[], parameters, X};
                invalidInputs(3).referenceErrors = 'empty';
                
                % Case 4: Wrong parameter size
                invalidInputs(4).name = 'wrong_param_size';
                invalidInputs(4).inputs = {y, [0.5], X};
                invalidInputs(4).referenceErrors = 'parameter';
                
            elseif strncmpi(componentName, 'composite', 9) || strcmp(componentName, 'composite_likelihood')
                % Invalid test cases for composite likelihood
                
                % Base valid data
                R = [1, 0.5; 0.5, 1];
                data = randn(50, 2);
                
                % Create test cases
                invalidInputs = struct([]);
                
                % Case 1: NaN in data
                invalidInputs(1).name = 'nan_in_data';
                badData = data;
                badData(10, 1) = NaN;
                invalidInputs(1).inputs = {badData, R};
                invalidInputs(1).referenceErrors = 'NaN';
                
                % Case 2: Inf in data
                invalidInputs(2).name = 'inf_in_data';
                badData = data;
                badData(20, 1) = Inf;
                invalidInputs(2).inputs = {badData, R};
                invalidInputs(2).referenceErrors = 'Inf';
                
                % Case 3: Empty data
                invalidInputs(3).name = 'empty_data';
                invalidInputs(3).inputs = {[], R};
                invalidInputs(3).referenceErrors = 'empty';
                
                % Case 4: Invalid correlation matrix
                invalidInputs(4).name = 'invalid_corr_matrix';
                badR = [1, 1.5; 0.5, 1]; % Invalid correlation (>1)
                invalidInputs(4).inputs = {data, badR};
                invalidInputs(4).referenceErrors = 'correlation';
                
                % Case 5: Dimension mismatch
                invalidInputs(5).name = 'dimension_mismatch';
                invalidInputs(5).inputs = {data, [1, 0.5, 0.3; 0.5, 1, 0.4; 0.3, 0.4, 1]};
                invalidInputs(5).referenceErrors = 'dimension';
                
            else
                % Default case for unknown components
                warning('PlatformCompatibilityTest:UnknownComponent', ...
                    'Unknown component "%s". Using generic invalid test inputs.', componentName);
                
                % Create generic test cases
                invalidInputs = struct([]);
                
                % Case 1: Empty data
                invalidInputs(1).name = 'empty_data';
                invalidInputs(1).inputs = {[], [0.5; 0.3]};
                invalidInputs(1).referenceErrors = 'empty';
                
                % Case 2: NaN data
                invalidInputs(2).name = 'nan_data';
                invalidInputs(2).inputs = {[NaN; 1; 2], [0.5; 0.3]};
                invalidInputs(2).referenceErrors = 'NaN';
            end
        end
        
        function largeInputs = createLargeTestInputs(obj, componentName)
            % Creates large test data arrays for testing memory handling
            %
            % INPUTS:
            %   componentName - Name of the component
            %
            % OUTPUTS:
            %   largeInputs - Cell array of large test inputs
            
            % Determine component type from componentName
            if strncmpi(componentName, 'agarch', 6) || strcmp(componentName, 'agarch_core') || ...
               strncmpi(componentName, 'tarch', 5) || strcmp(componentName, 'tarch_core') || ...
               strncmpi(componentName, 'egarch', 6) || strcmp(componentName, 'egarch_core') || ...
               strncmpi(componentName, 'igarch', 6) || strcmp(componentName, 'igarch_core')
                % Large inputs for GARCH family models
                
                % Large data array
                numObs = 100000; % 100K observations
                data = randn(numObs, 1);
                
                % For IGARCH
                if strncmpi(componentName, 'igarch', 6) || strcmp(componentName, 'igarch_core')
                    parameters = [0.01; 0.1]; % omega, alpha (beta = 1-alpha)
                else
                    parameters = [0.01; 0.1; 0.85; 0.1]; % omega, alpha, beta, gamma
                end
                
                initialVol = var(data);
                
                largeInputs = {data, parameters, initialVol};
                
            elseif strncmpi(componentName, 'armax', 5) || strcmp(componentName, 'armaxerrors')
                % Large inputs for ARMAX models
                
                % Large data array
                numObs = 100000; % 100K observations
                y = randn(numObs, 1);
                parameters = [0.5; 0.3]; % AR, MA
                
                % Large exogenous variables matrix
                X = randn(numObs, 5);
                
                largeInputs = {y, parameters, X};
                
            elseif strncmpi(componentName, 'composite', 9) || strcmp(componentName, 'composite_likelihood')
                % Large inputs for composite likelihood
                
                % Medium-sized correlation matrix (don't want to make it too large)
                dim = 20;
                R = eye(dim);
                
                % Fill with reasonable correlation values
                for i = 1:dim
                    for j = (i+1):dim
                        R(i,j) = 0.1 + 0.8 * rand(); % Random correlation between 0.1 and 0.9
                        R(j,i) = R(i,j); % Ensure symmetry
                    end
                end
                
                % Large data matrix
                numObs = 10000; % 10K observations
                data = randn(numObs, dim);
                
                largeInputs = {data, R};
                
            else
                % Default case for unknown components
                warning('PlatformCompatibilityTest:UnknownComponent', ...
                    'Unknown component "%s". Using generic large test inputs.', componentName);
                
                % Create generic large data
                largeInputs = {randn(50000, 1), [0.5; 0.3]};
            end
        end
        
        function similarity = getErrorSimilarity(obj, error1, error2)
            % Calculates the similarity between two error messages
            %
            % INPUTS:
            %   error1 - First error message
            %   error2 - Second error message or expected pattern
            %
            % OUTPUTS:
            %   similarity - Similarity score between 0 and 1
            
            % If error2 is just a pattern to check for
            if length(error2) < 20 && contains(error1, error2)
                similarity = 1.0;
                return;
            end
            
            % Calculate Levenshtein distance
            distance = levenshteinDistance(error1, error2);
            
            % Convert to similarity (0 to 1)
            maxLength = max(length(error1), length(error2));
            if maxLength > 0
                similarity = 1 - distance / maxLength;
            else
                similarity = 1; % Both empty strings are identical
            end
        end
    end
end

function distance = levenshteinDistance(s1, s2)
    % Calculate Levenshtein distance between two strings
    %
    % INPUTS:
    %   s1 - First string
    %   s2 - Second string
    %
    % OUTPUTS:
    %   distance - Edit distance between strings
    
    % Convert inputs to strings if they aren't already
    if ~ischar(s1)
        s1 = char(s1);
    end
    
    if ~ischar(s2)
        s2 = char(s2);
    end
    
    % Get string lengths
    m = length(s1);
    n = length(s2);
    
    % Initialize distance matrix
    d = zeros(m+1, n+1);
    
    % Source prefixes can be transformed into empty string by
    % deleting all characters
    for i = 1:m+1
        d(i, 1) = i-1;
    end
    
    % Target prefixes can be reached from empty source prefix
    % by inserting every character
    for j = 1:n+1
        d(1, j) = j-1;
    end
    
    % Fill in the rest of the matrix
    for j = 2:n+1
        for i = 2:m+1
            if s1(i-1) == s2(j-1)
                d(i, j) = d(i-1, j-1); % No operation required
            else
                d(i, j) = min([d(i-1, j) + 1,    % Deletion
                              d(i, j-1) + 1,    % Insertion
                              d(i-1, j-1) + 1]); % Substitution
            end
        end
    end
    
    % The last element contains the distance
    distance = d(m+1, n+1);
end