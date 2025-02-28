classdef MEXPerformanceTest < BaseTest
    % MEXPERFORMANCETEST A comprehensive test class that evaluates the performance 
    % characteristics of MEX implementations in the MFE Toolbox, focusing on execution 
    % speed, memory efficiency, numerical accuracy, and scalability with large datasets.
    
    properties
        mexValidator      MEXValidator     % MEX validation utility
        benchmarker       PerformanceBenchmark % Performance benchmarking utility
        dataGenerator     TestDataGenerator % Test data generation utility
        testCases         struct           % Structure of test cases
        testResults       struct           % Structure of test results
        mexFunctions      cell             % Cell array of MEX function names to test
        dataSizes         cell             % Array of data sizes for scalability testing
        requiredSpeedupFactor double       % Required performance improvement factor (e.g., 1.5 for 50%)
        generateVisualizations logical     % Whether to create performance visualization charts
        reportOutputPath  string           % Path for saving test results and visualizations
        verbose           logical          % Whether to display detailed output during testing
    end
    
    methods
        function obj = MEXPerformanceTest()
            % Initializes a new MEXPerformanceTest instance with default settings
            
            % Call parent BaseTest constructor with 'MEXPerformanceTest' name
            obj@BaseTest('MEXPerformanceTest');
            
            % Create MEXValidator instance for validating MEX functions
            obj.mexValidator = MEXValidator();
            
            % Create PerformanceBenchmark instance for performance measurements
            obj.benchmarker = PerformanceBenchmark();
            
            % Create TestDataGenerator instance for test data creation
            obj.dataGenerator = TestDataGenerator();
            
            % Set requiredSpeedupFactor to 1.5 (50% improvement) from Technical Specification
            obj.requiredSpeedupFactor = 1.5;
            obj.benchmarker.setSpeedupThreshold(obj.requiredSpeedupFactor);
            
            % Initialize mexFunctions array with all MEX functions to test
            obj.mexFunctions = {'agarch_core', 'armaxerrors', 'composite_likelihood', ...
                               'egarch_core', 'igarch_core', 'tarch_core'};
            
            % Initialize dataSizes for scalability testing with logarithmically increasing sizes
            obj.dataSizes = {100, 1000, 10000, 100000};
            
            % Set generateVisualizations to true for creating performance charts
            obj.generateVisualizations = true;
            
            % Set reportOutputPath to current directory for result output
            obj.reportOutputPath = pwd;
            
            % Set verbose to false for controlled output verbosity
            obj.verbose = false;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            
            % Call parent setUp method to initialize base test environment
            setUp@BaseTest(obj);
            
            % Verify all required MEX files exist and are accessible
            for i = 1:length(obj.mexFunctions)
                mexExists = obj.mexValidator.validateMEXExists(obj.mexFunctions{i});
                if ~mexExists
                    warning('MEX function %s not found - some tests may be skipped', obj.mexFunctions{i});
                end
            end
            
            % Generate test data structures for all MEX functions
            obj.testCases = struct();
            for i = 1:length(obj.mexFunctions)
                obj.testCases.(obj.mexFunctions{i}) = obj.generateMEXTestData(obj.mexFunctions{i}, 1000);
            end
            
            % Initialize test results structure for storing performance metrics
            obj.testResults = struct();
            
            % Configure benchmark parameters (iterations, warmup runs, etc.)
            obj.benchmarker.setIterations(100);
            obj.benchmarker.setWarmupIterations(10);
            
            % Set up visualization parameters if generateVisualizations is enabled
            if obj.generateVisualizations
                obj.benchmarker.enableVisualizationSaving(true, obj.reportOutputPath);
            end
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method execution
            
            % Save test results if any tests were executed
            if ~isempty(obj.testResults) && ~isempty(fieldnames(obj.testResults))
                resultFile = fullfile(obj.reportOutputPath, 'mex_performance_results.mat');
                save(resultFile, 'testResults');
            end
            
            % Generate performance visualizations if enabled
            if obj.generateVisualizations && isfield(obj.testResults, 'performance')
                obj.generatePerformanceReport(obj.testResults);
            end
            
            % Clean up any temporary test data to free memory
            obj.testCases = struct();
            
            % Call parent tearDown method for base cleanup operations
            tearDown@BaseTest(obj);
        end
        
        function results = testAgarchCorePerformance(obj)
            % Tests the performance of agarch_core MEX implementation against MATLAB implementation
            
            % Generate appropriate GARCH test data using dataGenerator
            testData = obj.generateMEXTestData('agarch_core', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @agarch_core;
            matlabFuncHandle = @agarch_core_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('agarch_core speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.agarch_core = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'AGARCH Core Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testArmaxErrorsPerformance(obj)
            % Tests the performance of armaxerrors MEX implementation against MATLAB implementation
            
            % Generate appropriate ARMAX test data using dataGenerator
            testData = obj.generateMEXTestData('armaxerrors', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @armaxerrors;
            matlabFuncHandle = @armaxerrors_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('armaxerrors speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.armaxerrors = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'ARMAX Errors Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testCompositeLikelihoodPerformance(obj)
            % Tests the performance of composite_likelihood MEX implementation against MATLAB implementation
            
            % Generate appropriate likelihood test data using dataGenerator
            testData = obj.generateMEXTestData('composite_likelihood', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @composite_likelihood;
            matlabFuncHandle = @composite_likelihood_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('composite_likelihood speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.composite_likelihood = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'Composite Likelihood Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testEgarchCorePerformance(obj)
            % Tests the performance of egarch_core MEX implementation against MATLAB implementation
            
            % Generate appropriate EGARCH test data using dataGenerator
            testData = obj.generateMEXTestData('egarch_core', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @egarch_core;
            matlabFuncHandle = @egarch_core_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('egarch_core speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.egarch_core = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'EGARCH Core Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testIgarchCorePerformance(obj)
            % Tests the performance of igarch_core MEX implementation against MATLAB implementation
            
            % Generate appropriate IGARCH test data using dataGenerator
            testData = obj.generateMEXTestData('igarch_core', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @igarch_core;
            matlabFuncHandle = @igarch_core_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('igarch_core speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.igarch_core = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'IGARCH Core Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testTarchCorePerformance(obj)
            % Tests the performance of tarch_core MEX implementation against MATLAB implementation
            
            % Generate appropriate TARCH test data using dataGenerator
            testData = obj.generateMEXTestData('tarch_core', 1000);
            
            % Create function handles for MEX and MATLAB implementations
            mexFuncHandle = @tarch_core;
            matlabFuncHandle = @tarch_core_matlab;
            
            % Set up test parameters (iterations, data size, model configuration)
            iterations = 50;
            
            % Use benchmarker.compareImplementations to measure performance of both implementations
            results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, iterations, testData{:});
            
            % Verify correctness of results between implementations using assertAlmostEqual
            mexOutput = mexFuncHandle(testData{:});
            matlabOutput = matlabFuncHandle(testData{:});
            obj.assertAlmostEqual(mexOutput, matlabOutput, 'MEX and MATLAB implementations produce different results');
            
            % Assert that speedup ratio meets or exceeds requiredSpeedupFactor
            obj.assertTrue(results.speedupRatio >= obj.requiredSpeedupFactor, ...
                sprintf('tarch_core speedup (%.2fx) does not meet required %.2fx improvement', ...
                results.speedupRatio, obj.requiredSpeedupFactor));
            
            % Record detailed performance metrics in testResults structure
            obj.testResults.tarch_core = results;
            
            % Generate performance visualization if enabled
            if obj.generateVisualizations
                obj.benchmarker.visualizeResults(results, 'comparison', 'TARCH Core Performance');
            end
            
            % Return comprehensive test results structure
            return;
        end
        
        function results = testMemoryEfficiency(obj)
            % Tests memory usage efficiency of MEX implementations compared to MATLAB implementations
            
            % For each MEX function, set up appropriate test data
            memResults = struct('function', {}, 'memoryRatio', {}, 'improvement', {});
            
            for i = 1:length(obj.mexFunctions)
                mexFunc = obj.mexFunctions{i};
                matlabFunc = [mexFunc '_matlab'];
                
                % Generate test data for this function
                testData = obj.generateMEXTestData(mexFunc, 1000);
                
                % Create function handles
                mexFuncHandle = str2func(mexFunc);
                matlabFuncHandle = str2func(matlabFunc);
                
                % Use benchmarker.compareMemoryUsage to compare MEX vs MATLAB implementations
                memoryComparison = obj.benchmarker.compareMemoryUsage(matlabFuncHandle, mexFuncHandle, testData{:});
                
                % Calculate memory efficiency ratios and percentage differences
                memResults(i).function = mexFunc;
                memResults(i).memoryRatio = memoryComparison.memoryRatio;
                memResults(i).improvement = memoryComparison.efficiencyPercent;
                
                % Assert that MEX implementations are at least as memory-efficient as MATLAB
                obj.assertTrue(memoryComparison.memoryRatio >= 1.0 || ...
                    (memoryComparison.results1.netChange < 0 && memoryComparison.results2.netChange < 0), ...
                    sprintf('%s memory usage (%.2f%%) is worse than MATLAB version', ...
                    mexFunc, memoryComparison.efficiencyPercent));
            end
            
            % Compile memory efficiency statistics across all implementations
            overallImprovement = mean([memResults.improvement]);
            
            % Generate memory usage visualizations if enabled
            if obj.generateVisualizations
                figure;
                bar([memResults.improvement]);
                set(gca, 'XTickLabel', {memResults.function});
                title('Memory Efficiency Improvement (%)');
                ylabel('Improvement (%)');
                grid on;
                savefig(fullfile(obj.reportOutputPath, 'memory_efficiency.fig'));
            end
            
            % Store results
            results = struct('memoryResults', memResults, 'overallImprovement', overallImprovement);
            obj.testResults.memoryEfficiency = results;
            
            % Return comprehensive memory efficiency metrics
            return;
        end
        
        function results = testScalability(obj)
            % Tests how MEX performance advantage scales with increasing data size
            
            % Select representative MEX functions for scalability testing
            testFuncs = {'armaxerrors', 'egarch_core', 'tarch_core'};
            
            % Initialize results structure
            scaleResults = struct('function', {}, 'dataSizes', {}, 'speedupRatios', {}, 'scalingBehavior', {});
            
            % For each function, test with progressively larger data sizes from dataSizes array
            for i = 1:length(testFuncs)
                mexFunc = testFuncs{i};
                matlabFunc = [mexFunc '_matlab'];
                
                % Create function handles
                mexFuncHandle = str2func(mexFunc);
                matlabFuncHandle = str2func(matlabFunc);
                
                % Initialize speedup ratio array
                speedups = zeros(length(obj.dataSizes), 1);
                
                % Test with each data size
                for j = 1:length(obj.dataSizes)
                    dataSize = obj.dataSizes{j};
                    
                    % Generate test data for this size
                    testData = obj.generateMEXTestData(mexFunc, dataSize);
                    
                    % Compare performance with this data size
                    if obj.verbose
                        fprintf('Testing %s with data size %d...\n', mexFunc, dataSize);
                    end
                    
                    % Use benchmarker.compareScalability to measure scaling behavior
                    results = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, 20, testData{:});
                    speedups(j) = results.speedupRatio;
                end
                
                % Analyze how speedup factor changes with increasing data size
                if speedups(end) >= speedups(1)
                    scalingBehavior = 'Improving - advantage increases with data size';
                elseif speedups(end) >= obj.requiredSpeedupFactor
                    scalingBehavior = 'Stable - maintains required advantage';
                else
                    scalingBehavior = 'Degrading - advantage decreases with data size';
                end
                
                % Store results for this function
                scaleResults(i).function = mexFunc;
                scaleResults(i).dataSizes = obj.dataSizes;
                scaleResults(i).speedupRatios = speedups;
                scaleResults(i).scalingBehavior = scalingBehavior;
                
                % Verify that performance advantage remains stable or improves with larger data
                obj.assertTrue(speedups(end) >= obj.requiredSpeedupFactor, ...
                    sprintf('%s does not maintain required speedup for large datasets', mexFunc));
            end
            
            % Generate scalability visualizations if enabled
            if obj.generateVisualizations
                figure;
                hold on;
                for i = 1:length(scaleResults)
                    loglog(cell2mat(obj.dataSizes), scaleResults(i).speedupRatios, 'o-', 'LineWidth', 2);
                end
                yline(obj.requiredSpeedupFactor, 'r--', 'Required Threshold');
                legend([{scaleResults.function}, {'Threshold'}], 'Location', 'Best');
                xlabel('Data Size (log scale)');
                ylabel('Speedup Ratio (log scale)');
                title('Scalability of MEX Performance Advantage');
                grid on;
                savefig(fullfile(obj.reportOutputPath, 'scalability.fig'));
            end
            
            % Store scalability results
            results = scaleResults;
            obj.testResults.scalability = results;
            
            % Return comprehensive scalability analysis results
            return;
        end
        
        function results = testNumericalConsistency(obj)
            % Tests numerical consistency between MEX and MATLAB implementations
            
            % Initialize results structure
            consistencyResults = struct('function', {}, 'maxAbsDifference', {}, 'relDifference', {}, 'isConsistent', {});
            
            % For each MEX function, generate test data across a range of scenarios
            for i = 1:length(obj.mexFunctions)
                mexFunc = obj.mexFunctions{i};
                matlabFunc = [mexFunc '_matlab'];
                
                % Create function handles
                mexFuncHandle = str2func(mexFunc);
                matlabFuncHandle = str2func(matlabFunc);
                
                % Test with different data sizes to ensure consistency across scenarios
                testSizes = [100, 1000, 5000];
                maxAbsDiff = 0;
                maxRelDiff = 0;
                
                for j = 1:length(testSizes)
                    % Generate test data for this size
                    testData = obj.generateMEXTestData(mexFunc, testSizes(j));
                    
                    % Compare output of MEX and MATLAB implementations using assertAlmostEqual
                    mexOutput = mexFuncHandle(testData{:});
                    matlabOutput = matlabFuncHandle(testData{:});
                    
                    % Calculate maximum absolute difference
                    if isnumeric(mexOutput) && isnumeric(matlabOutput)
                        if isscalar(mexOutput)
                            absDiff = abs(mexOutput - matlabOutput);
                            relDiff = absDiff / max(abs(matlabOutput), 1e-10);
                        else
                            absDiff = max(abs(mexOutput(:) - matlabOutput(:)));
                            relDiff = max(abs(mexOutput(:) - matlabOutput(:)) ./ max(abs(matlabOutput(:)), 1e-10));
                        end
                        
                        maxAbsDiff = max(maxAbsDiff, absDiff);
                        maxRelDiff = max(maxRelDiff, relDiff);
                    end
                end
                
                % Verify consistency across different parameter values and edge cases
                isConsistent = (maxAbsDiff < 1e-10);
                
                % Store results for this function
                consistencyResults(i).function = mexFunc;
                consistencyResults(i).maxAbsDifference = maxAbsDiff;
                consistencyResults(i).relDifference = maxRelDiff;
                consistencyResults(i).isConsistent = isConsistent;
                
                % Calculate maximum numerical differences and verify within tolerance
                obj.assertTrue(isConsistent, ...
                    sprintf('%s numerical results differ from MATLAB implementation (max diff: %g)', ...
                    mexFunc, maxAbsDiff));
            end
            
            % Store overall consistency results
            results = consistencyResults;
            obj.testResults.numericalConsistency = results;
            
            % Return comprehensive numerical consistency analysis
            return;
        end
        
        function results = testLargeScalePerformance(obj)
            % Tests performance with very large datasets to validate large-scale processing capability
            
            % Generate large-scale test data exceeding typical usage sizes
            largeDataSize = 100000;
            
            % Select representative MEX functions for large-scale testing
            testFuncs = {'armaxerrors', 'tarch_core'};
            
            % Initialize results structure
            largeScaleResults = struct('function', {}, 'dataSize', {}, 'mexTime', {}, 'matlabTime', {}, 'speedup', {});
            
            % For each function, test with large-scale data
            for i = 1:length(testFuncs)
                mexFunc = testFuncs{i};
                matlabFunc = [mexFunc '_matlab'];
                
                if obj.verbose
                    fprintf('Testing large-scale performance of %s with %d observations...\n', mexFunc, largeDataSize);
                end
                
                % Generate large-scale test data
                testData = obj.generateMEXTestData(mexFunc, largeDataSize);
                
                % Create function handles
                mexFuncHandle = str2func(mexFunc);
                matlabFuncHandle = str2func(matlabFunc);
                
                % Measure performance with large datasets using benchmarker
                largeScaleComp = obj.benchmarker.compareImplementations(matlabFuncHandle, mexFuncHandle, 10, testData{:});
                
                % Store results
                largeScaleResults(i).function = mexFunc;
                largeScaleResults(i).dataSize = largeDataSize;
                largeScaleResults(i).mexTime = largeScaleComp.results2.mean;
                largeScaleResults(i).matlabTime = largeScaleComp.results1.mean;
                largeScaleResults(i).speedup = largeScaleComp.speedupRatio;
                
                % Verify memory efficiency with large datasets
                obj.assertTrue(largeScaleComp.speedupRatio >= obj.requiredSpeedupFactor, ...
                    sprintf('%s large-scale speedup (%.2fx) does not meet requirement (%.2fx)', ...
                    mexFunc, largeScaleComp.speedupRatio, obj.requiredSpeedupFactor));
            end
            
            % Generate large-scale performance visualizations if enabled
            if obj.generateVisualizations
                figure;
                bar([largeScaleResults.speedup]);
                set(gca, 'XTickLabel', {largeScaleResults.function});
                yline(obj.requiredSpeedupFactor, 'r--', 'Required Threshold');
                title(sprintf('Large-Scale Performance (n=%d)', largeDataSize));
                ylabel('Speedup Ratio');
                grid on;
                savefig(fullfile(obj.reportOutputPath, 'large_scale_performance.fig'));
            end
            
            % Store large-scale performance results
            results = largeScaleResults;
            obj.testResults.largeScalePerformance = results;
            
            % Return comprehensive large-scale performance metrics
            return;
        end
        
        function results = runAllTests(obj)
            % Runs all performance tests and generates a comprehensive report
            
            % Execute all individual performance test methods
            if obj.verbose
                fprintf('Running all MEX performance tests...\n');
            end
            
            % Run individual function tests
            obj.testAgarchCorePerformance();
            obj.testArmaxErrorsPerformance();
            obj.testCompositeLikelihoodPerformance();
            obj.testEgarchCorePerformance();
            obj.testIgarchCorePerformance();
            obj.testTarchCorePerformance();
            
            % Run cross-cutting tests
            obj.testMemoryEfficiency();
            obj.testScalability();
            obj.testNumericalConsistency();
            obj.testLargeScalePerformance();
            
            % Aggregate performance metrics across all MEX functions
            speedups = zeros(length(obj.mexFunctions), 1);
            for i = 1:length(obj.mexFunctions)
                mexFunc = obj.mexFunctions{i};
                if isfield(obj.testResults, mexFunc)
                    speedups(i) = obj.testResults.(mexFunc).speedupRatio;
                end
            end
            
            % Calculate average, minimum, and maximum speedup across all tests
            avgSpeedup = mean(speedups);
            minSpeedup = min(speedups);
            maxSpeedup = max(speedups);
            
            % Verify overall average speedup meets or exceeds requiredSpeedupFactor
            passingTests = sum(speedups >= obj.requiredSpeedupFactor);
            
            % Generate comprehensive performance report with visualizations
            summary = struct(...
                'averageSpeedup', avgSpeedup, ...
                'minimumSpeedup', minSpeedup, ...
                'maximumSpeedup', maxSpeedup, ...
                'passingTests', passingTests, ...
                'totalTests', length(speedups), ...
                'overallResult', passingTests == length(speedups) ...
            );
            
            obj.testResults.summary = summary;
            
            % Generate final report
            reportPath = obj.generatePerformanceReport(obj.testResults);
            
            % Display summary statistics to console if verbose is true
            if obj.verbose
                fprintf('\nMEX Performance Test Summary:\n');
                fprintf('---------------------------\n');
                fprintf('Average speedup: %.2fx\n', avgSpeedup);
                fprintf('Minimum speedup: %.2fx\n', minSpeedup);
                fprintf('Maximum speedup: %.2fx\n', maxSpeedup);
                fprintf('Tests passing requirement: %d of %d\n', passingTests, length(speedups));
                
                if summary.overallResult
                    fprintf('OVERALL RESULT: PASS - All MEX functions exceed required %.2fx speedup\n', obj.requiredSpeedupFactor);
                else
                    fprintf('OVERALL RESULT: FAIL - Some MEX functions do not meet required %.2fx speedup\n', obj.requiredSpeedupFactor);
                end
                
                fprintf('Performance report saved to: %s\n', reportPath);
            end
            
            % Return aggregated test results with overall performance assessment
            results = obj.testResults;
        end
        
        function reportPath = generatePerformanceReport(obj, results)
            % Generates a detailed performance report with visualizations
            
            % Create performance summary from test results
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            reportName = ['mex_performance_report_', timestamp];
            reportPath = fullfile(obj.reportOutputPath, reportName);
            
            % Generate comparative bar charts for speedup factors
            figure('Name', 'MEX Performance Summary', 'Position', [100, 100, 800, 600]);
            
            % Extract function names and speedup ratios
            functionNames = obj.mexFunctions;
            speedups = zeros(length(functionNames), 1);
            
            for i = 1:length(functionNames)
                if isfield(results, functionNames{i})
                    speedups(i) = results.(functionNames{i}).speedupRatio;
                end
            end
            
            % Plot speedup comparison
            subplot(2, 1, 1);
            bar(speedups);
            hold on;
            yline(obj.requiredSpeedupFactor, 'r--', 'Required Threshold');
            hold off;
            set(gca, 'XTick', 1:length(functionNames), 'XTickLabel', functionNames);
            title('MEX Implementation Speedup Factors');
            ylabel('Speedup Ratio (x)');
            grid on;
            
            % Generate scalability charts showing performance vs data size
            if isfield(results, 'scalability')
                subplot(2, 1, 2);
                hold on;
                
                scalability = results.scalability;
                for i = 1:length(scalability)
                    semilogx(cell2mat(scalability(i).dataSizes), scalability(i).speedupRatios, ...
                        'o-', 'LineWidth', 2);
                end
                
                yline(obj.requiredSpeedupFactor, 'r--', 'Required Threshold');
                hold off;
                
                legend([{scalability.function}, {'Threshold'}], 'Location', 'Best');
                xlabel('Data Size (log scale)');
                ylabel('Speedup Ratio');
                title('Scalability of MEX Performance Advantage');
                grid on;
            end
            
            % Create summary tables with detailed metrics
            if isfield(results, 'summary')
                annotation('textbox', [0.1, 0.01, 0.8, 0.05], ...
                    'String', sprintf('Average Speedup: %.2fx | Min: %.2fx | Max: %.2fx | Passing: %d of %d', ...
                    results.summary.averageSpeedup, results.summary.minimumSpeedup, ...
                    results.summary.maximumSpeedup, results.summary.passingTests, ...
                    results.summary.totalTests), ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FitBoxToText', 'on');
            end
            
            % Save report and visualizations to reportOutputPath
            savefig([reportPath, '.fig']);
            
            % Return path to generated report
            reportPath = [reportPath, '.fig'];
        end
        
        function results = compareMEXvsReference(obj, mexName, matlabName, testInputs)
            % Utility method to compare MEX implementation against reference MATLAB implementation
            
            % Create function handles for MEX and MATLAB implementations
            mexFunc = str2func(mexName);
            matlabFunc = str2func(matlabName);
            
            % Configure benchmark parameters for fair comparison
            iterations = 50;
            
            % Use benchmarker.compareImplementations for performance measurement
            results = obj.benchmarker.compareImplementations(matlabFunc, mexFunc, iterations, testInputs{:});
            
            % Validate result correctness between implementations
            mexOutput = mexFunc(testInputs{:});
            matlabOutput = matlabFunc(testInputs{:});
            
            if isnumeric(mexOutput) && isnumeric(matlabOutput)
                if isscalar(mexOutput)
                    % For scalar outputs
                    diff = abs(mexOutput - matlabOutput);
                    results.numericalConsistency = (diff < 1e-10);
                else
                    % For vector/matrix outputs
                    diff = max(abs(mexOutput(:) - matlabOutput(:)));
                    results.numericalConsistency = (diff < 1e-10);
                end
            else
                % For non-numeric outputs
                results.numericalConsistency = isequal(mexOutput, matlabOutput);
            end
            
            % Calculate speedup ratio and performance metrics
            results.mexName = mexName;
            results.matlabName = matlabName;
            results.timestamp = datestr(now);
            
            % Return detailed comparison results
            return;
        end
        
        function testInputs = generateMEXTestData(obj, mexName, dataSize)
            % Generates appropriate test data for a specific MEX function
            
            % Determine appropriate data type based on MEX function
            if nargin < 3
                dataSize = 1000;
            end
            
            % Initialize cell array to hold test data
            testInputs = {};
            
            switch lower(mexName)
                case 'agarch_core'
                    % For GARCH-type functions, generate financial time series data
                    data = randn(dataSize, 1);
                    
                    % Define AGARCH parameters: omega, alpha, gamma, beta
                    parameters = [0.05; 0.1; 0.1; 0.8];
                    
                    % Initial variance
                    backcast = var(data);
                    
                    % GARCH order
                    p = 1;
                    q = 1;
                    
                    % Create parameter array
                    testInputs = {data, parameters, backcast, p, q, dataSize};
                    
                case 'armaxerrors'
                    % For ARMAX-type functions, generate time series with known properties
                    % Create an ARMA(1,1) process
                    data = zeros(dataSize, 1);
                    ar = 0.8;
                    ma = 0.3;
                    
                    % Generate with specified properties
                    noise = randn(dataSize+10, 1);
                    for t = 2:dataSize+10
                        data(t) = ar * data(t-1) + noise(t) + ma * noise(t-1);
                    end
                    data = data(11:end);
                    
                    % AR parameters
                    ar_params = [ar];
                    
                    % MA parameters
                    ma_params = [ma];
                    
                    % No exogenous variables
                    exog_data = [];
                    exog_params = [];
                    
                    % No constant
                    constant = 0;
                    
                    % Create parameter array
                    testInputs = {data, ar_params, ma_params, exog_data, exog_params, constant};
                    
                case 'composite_likelihood'
                    % For likelihood functions, generate distribution parameters and data
                    % Create correlation matrix for multivariate data
                    k = 5; % Dimension
                    R = eye(k);
                    
                    % Fill correlation matrix with reasonable values
                    for i = 1:k
                        for j = i+1:k
                            R(i,j) = 0.5 - 0.05 * abs(i-j);
                            R(j,i) = R(i,j);
                        end
                    end
                    
                    % Generate multivariate normal data with this correlation
                    X = randn(dataSize, k) * chol(R);
                    
                    % Create parameter array
                    testInputs = {X, R};
                    
                case 'egarch_core'
                    % Generate data for EGARCH model
                    data = randn(dataSize, 1);
                    
                    % Define EGARCH parameters: omega, alpha, gamma, beta
                    parameters = [-0.1; 0.1; 0.05; 0.9];
                    
                    % Initial variance
                    backcast = var(data);
                    
                    % GARCH order
                    p = 1;
                    q = 1;
                    
                    % Create parameter array
                    testInputs = {data, parameters, backcast, p, q, dataSize};
                    
                case 'igarch_core'
                    % Generate data for IGARCH model
                    data = randn(dataSize, 1);
                    
                    % Define IGARCH parameters: omega, alpha, beta
                    parameters = [0.05; 0.2; 0.8];
                    
                    % Initial variance
                    backcast = var(data);
                    
                    % GARCH order
                    p = 1;
                    q = 1;
                    
                    % Create parameter array
                    testInputs = {data, parameters, backcast, p, q, dataSize};
                    
                case 'tarch_core'
                    % Generate data for TARCH model
                    data = randn(dataSize, 1);
                    
                    % Define TARCH parameters: omega, alpha, gamma, beta
                    parameters = [0.05; 0.05; 0.1; 0.8];
                    
                    % Initial variance
                    backcast = var(data);
                    
                    % GARCH order
                    p = 1;
                    q = 1;
                    
                    % Create parameter array
                    testInputs = {data, parameters, backcast, p, q, dataSize};
                    
                otherwise
                    error('Unknown MEX function: %s', mexName);
            end
            
            % Scale data size according to the dataSize parameter
            return;
        end
        
        function setSpeedupThreshold(obj, factor)
            % Sets the required speedup factor threshold for tests to pass
            
            % Validate factor is greater than 1.0
            if factor <= 1.0
                error('Speedup threshold factor must be greater than 1.0');
            end
            
            % Update requiredSpeedupFactor property
            obj.requiredSpeedupFactor = factor;
            
            % Update benchmarker's speedupThreshold setting
            obj.benchmarker.setSpeedupThreshold(factor);
        end
        
        function setVisualizationOptions(obj, enable, outputPath)
            % Configures visualization options for performance test results
            
            % Set generateVisualizations property to enable value
            obj.generateVisualizations = enable;
            
            % If outputPath is provided, validate it is a valid directory
            if nargin > 2 && ~isempty(outputPath)
                if ~exist(outputPath, 'dir')
                    error('Output path does not exist: %s', outputPath);
                end
                
                % Set reportOutputPath property to specified path if provided
                obj.reportOutputPath = outputPath;
            end
            
            % Configure benchmarker visualization settings accordingly
            obj.benchmarker.enableVisualizationSaving(enable, obj.reportOutputPath);
        end
        
        function setVerbose(obj, verboseFlag)
            % Sets verbose mode for detailed console output during testing
            
            % Validate verboseFlag is a logical value
            if ~islogical(verboseFlag) && ~(isnumeric(verboseFlag) && (verboseFlag == 0 || verboseFlag == 1))
                error('verboseFlag must be a logical value');
            end
            
            % Set verbose property to specified value
            obj.verbose = logical(verboseFlag);
            
            % Configure benchmarker and validators to use the same verbosity setting
            obj.benchmarker.setVerbose(verboseFlag);
            obj.mexValidator.setVerbose(verboseFlag);
        end
    end
end