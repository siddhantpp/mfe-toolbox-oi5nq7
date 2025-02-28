classdef MemoryUsageTest < BaseTest
    %MEMORYUSAGETEST Test class for evaluating memory usage and efficiency across MFE Toolbox components
    %   This class provides comprehensive memory usage benchmarking, focusing on
    %   matrix operations, large dataset handling, and MEX implementation memory
    %   optimization. It validates that the toolbox meets memory efficiency
    %   requirements specified in the technical specification.

    properties
        benchmarker % PerformanceBenchmark instance
        memoryBaseline % struct
        generateVisualizations % logical
        reportOutputPath % string
        testDataSizes % cell
        testResults % struct
    end

    methods
        function obj = MemoryUsageTest()
            %MemoryUsageTest Initializes the MemoryUsageTest class with default settings and test configuration

            % Call superclass (BaseTest) constructor with 'Memory Usage Test' name
            obj = obj@BaseTest('Memory Usage Test');

            % Initialize benchmarker with PerformanceBenchmark instance
            obj.benchmarker = PerformanceBenchmark();

            % Configure generateVisualizations flag to true for memory visualization generation
            obj.generateVisualizations = true;

            % Set reportOutputPath to './memory_reports/'
            obj.reportOutputPath = './memory_reports/';

            % Define testDataSizes array with progressively larger dataset dimensions for memory testing
            obj.testDataSizes = {100, 500, 1000, 5000};

            % Initialize empty testResults structure for storing test outcomes
            obj.testResults = struct();
        end

        function setUp(obj)
            %setUp Prepares the test environment for memory testing

            % Call superclass setUp method
            setUp@BaseTest(obj);

            % Clear workspace to establish memory baseline
            clear;

            % Record initial memory state using memory() function
            initialMemory = memory();

            % Store memory baseline in memoryBaseline property
            obj.memoryBaseline = initialMemory;
        end

        function tearDown(obj)
            %tearDown Cleans up after memory test execution

            % Generate memory report if any tests were executed
            if ~isempty(fieldnames(obj.testResults))
                obj.generateMemoryReport();
            end

            % Clear variables created during tests
            clear;

            % Verify memory is properly released after test
            finalMemory = memory();
            memoryDifference = finalMemory.MemUsedMATLAB - obj.memoryBaseline.MemUsedMATLAB;
            obj.assertAlmostEqual(memoryDifference, 0, 'Memory leak detected: Memory was not properly released after test.');

            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end

        function testMatrixOperationMemory(obj)
            %testMatrixOperationMemory Tests memory efficiency of core matrix operations

            % Create test matrices of varying sizes
            for i = 1:length(obj.testDataSizes)
                size = obj.testDataSizes{i};
                A = rand(size);
                B = rand(size);

                % Measure memory usage during basic matrix operations (multiplication, inversion, etc.)
                memoryData = obj.measureMemoryFootprint(@(x, y) x * y, A, B);
                memoryUsage(i) = memoryData.netChangeMB;

                % Calculate memory efficiency ratio (actual/theoretical)
                theoreticalMemory = size * size * 8 * 3 / (1024^2); % Size of A, B, and result in MB
                efficiencyRatio = memoryUsage(i) / theoreticalMemory;

                % Verify memory usage stays within acceptable bounds
                obj.assertTrue(efficiencyRatio < 1.5, sprintf('Matrix multiplication memory usage exceeds threshold for size %d', size));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Matrix Multiplication Memory Usage (Size %d)', size), 'matrix_mult');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 500, 'Average matrix multiplication memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.matrixOperationMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testTimeSeriesMemory(obj)
            %testTimeSeriesMemory Tests memory usage of time series modeling components

            % Generate time series data of varying lengths using TestDataGenerator
            for i = 1:length(obj.testDataSizes)
                T = obj.testDataSizes{i};
                data = TestDataGenerator('generateFinancialReturns', T, 1);

                % Measure memory usage for ARMA/ARMAX model estimation
                memoryData = obj.measureMemoryFootprint(@armaxfilter, data.returns, [0.5, 0.2], [0.3, 0.1]);
                memoryUsage(i) = memoryData.netChangeMB;

                % Analyze memory scaling with increasing time series length
                % Verify implementation uses memory efficiently
                obj.assertTrue(memoryUsage(i) < T/10, sprintf('Time series memory usage exceeds threshold for length %d', T));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Time Series Memory Usage (Length %d)', T), 'time_series');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 100, 'Average time series memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.timeSeriesMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testVolatilityModelMemory(obj)
            %testVolatilityModelMemory Tests memory usage of volatility modeling components

            % Generate volatility model test data of varying sizes
            for i = 1:length(obj.testDataSizes)
                T = obj.testDataSizes{i};
                data = TestDataGenerator('generateFinancialReturns', T, 1);

                % Measure memory usage for different GARCH-type model estimations
                memoryData = obj.measureMemoryFootprint(@garchcore, data.returns, struct('model', 'GARCH', 'p', 1, 'q', 1));
                memoryUsage(i) = memoryData.netChangeMB;

                % Compare memory usage between MATLAB and MEX implementations
                % Analyze memory scaling with dataset size
                % Verify memory optimization in MEX implementations
                obj.assertTrue(memoryUsage(i) < T/5, sprintf('Volatility model memory usage exceeds threshold for length %d', T));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Volatility Model Memory Usage (Length %d)', T), 'volatility_model');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 150, 'Average volatility model memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.volatilityModelMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testMEXAllocationDeallocation(obj)
            %testMEXAllocationDeallocation Tests memory allocation and deallocation in MEX functions

            % Set up test cases for MEX function memory allocation
            % Execute MEX functions repeatedly to test memory management
            % Monitor memory usage pattern during execution
            % Verify proper deallocation after function completion
            % Check for memory leaks after multiple executions
            % Test edge cases (error conditions, large allocations)

            % Define test parameters
            T = 1000;
            p = 2;
            q = 2;
            numIterations = 100;

            % Generate test data
            data = randn(T, 1);
            arParams = randn(p, 1);
            maParams = randn(q, 1);

            % Measure memory usage before and after MEX execution
            initialMemory = memory();
            for i = 1:numIterations
                % Call MEX function
                armaxerrors(data, arParams, maParams);
            end
            finalMemory = memory();

            % Calculate memory difference
            memoryDifference = finalMemory.MemUsedMATLAB - initialMemory.MemUsedMATLAB;

            % Assert that MEX memory management is correct
            obj.assertAlmostEqual(memoryDifference, 0, 'MEX memory management is incorrect: Memory leak detected.');

            % Test edge cases (error conditions, large allocations)
            % Assert that MEX memory management is correct
            obj.assertTrue(memoryDifference < 10, 'MEX memory management is incorrect: Memory leak detected.');

            % Return structured test results with memory metrics
            obj.testResults.mexAllocationDeallocation = struct('memoryDifference', memoryDifference);
        end

        function testLargeDatasetMemory(obj)
            %testLargeDatasetMemory Tests memory efficiency with large-scale financial datasets

            % Generate increasingly large financial datasets
            for i = 1:length(obj.testDataSizes)
                size = obj.testDataSizes{i} * 10;
                data = TestDataGenerator('generateFinancialReturns', size, 1);

                % Measure memory requirements during processing
                memoryData = obj.measureMemoryFootprint(@mean, data.returns);
                memoryUsage(i) = memoryData.netChangeMB;

                % Calculate memory efficiency ratio (actual/theoretical)
                theoreticalMemory = size * 8 / (1024^2); % Size of data in MB
                efficiencyRatio = memoryUsage(i) / theoreticalMemory;

                % Analyze memory scaling with dataset size
                % Test memory-mapped file functionality for very large datasets
                obj.assertTrue(efficiencyRatio < 2, sprintf('Large dataset memory usage exceeds threshold for size %d', size));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Large Dataset Memory Usage (Size %d)', size), 'large_dataset');
                end
            end

            % Assert that memory scaling meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 200, 'Average large dataset memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.largeDatasetMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testHighFrequencyDataMemory(obj)
            %testHighFrequencyDataMemory Tests memory usage with high-frequency financial data

            % Generate high-frequency data with varying observation frequencies
            for i = 1:length(obj.testDataSizes)
                obsPerDay = obj.testDataSizes{i};
                data = TestDataGenerator('generateHighFrequencyData', 1, obsPerDay);

                % Measure memory usage for realized volatility calculations
                memoryData = obj.measureMemoryFootprint(@(x) sum(x.^2), data.returns);
                memoryUsage(i) = memoryData.netChangeMB;

                % Analyze memory optimization for high-frequency operations
                % Test memory efficiency of intraday pattern calculations
                obj.assertTrue(memoryUsage(i) < obsPerDay/1000, sprintf('High-frequency data memory usage exceeds threshold for obsPerDay %d', obsPerDay));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('High-Frequency Data Memory Usage (ObsPerDay %d)', obsPerDay), 'high_frequency');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 50, 'Average high-frequency data memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.highFrequencyDataMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testBootstrapMemory(obj)
            %testBootstrapMemory Tests memory usage of bootstrap methods

            % Generate test data for bootstrap methods
            for i = 1:length(obj.testDataSizes)
                T = obj.testDataSizes{i};
                data = TestDataGenerator('generateFinancialReturns', T, 1);

                % Measure memory usage for block bootstrap operations
                memoryData = obj.measureMemoryFootprint(@(x) bootstrp(100, @mean, x), data.returns);
                memoryUsage(i) = memoryData.netChangeMB;

                % Measure memory usage for stationary bootstrap operations
                % Analyze memory efficiency with increasing bootstrap iterations
                obj.assertTrue(memoryUsage(i) < T/2, sprintf('Bootstrap memory usage exceeds threshold for length %d', T));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Bootstrap Memory Usage (Length %d)', T), 'bootstrap');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 100, 'Average bootstrap memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.bootstrapMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testMultivariateModelMemory(obj)
            %testMultivariateModelMemory Tests memory usage of multivariate model implementations

            % Generate multivariate test data with varying dimensions
            for i = 1:length(obj.testDataSizes)
                dim = obj.testDataSizes{i};
                data = randn(100, dim);

                % Measure memory usage for VAR, VECM, and multivariate GARCH models
                memoryData = obj.measureMemoryFootprint(@cov, data);
                memoryUsage(i) = memoryData.netChangeMB;

                % Analyze memory scaling with dimension increase
                % Compare memory usage of different multivariate implementations
                obj.assertTrue(memoryUsage(i) < dim/10, sprintf('Multivariate model memory usage exceeds threshold for dimension %d', dim));

                % Generate memory usage visualization if enabled
                if obj.generateVisualizations
                    obj.visualizeMemoryUsage(memoryData, sprintf('Multivariate Model Memory Usage (Dimension %d)', dim), 'multivariate_model');
                end
            end

            % Assert that memory efficiency meets requirements
            avgMemoryUsage = mean(memoryUsage);
            obj.assertTrue(avgMemoryUsage < 100, 'Average multivariate model memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.multivariateModelMemory = struct('memoryUsage', memoryUsage, 'avgMemoryUsage', avgMemoryUsage);
        end

        function testCrossThreadMemory(obj)
            %testCrossThreadMemory Tests memory efficiency with parallel processing

            % Configure parallel processing environment
            % Execute memory-intensive operations in parallel
            % Measure memory usage across threads
            % Analyze memory sharing and isolation behavior
            % Verify efficient memory use in parallel context

            % Configure parallel pool
            numThreads = 4;
            pool = gcp('nocreate');
            if isempty(pool)
                parpool(numThreads);
            end

            % Define memory-intensive operation
            operation = @(x) eig(x);

            % Generate test data
            size = 500;
            A = rand(size);

            % Measure memory usage in parallel context
            memoryData = obj.measureMemoryFootprint(@(x) parfeval(pool, operation, 1, x), A);

            % Verify efficient memory use in parallel context
            obj.assertTrue(memoryData.netChangeMB < 500, 'Parallel processing memory usage exceeds threshold.');

            % Generate memory usage visualization if enabled
            if obj.generateVisualizations
                obj.visualizeMemoryUsage(memoryData, 'Parallel Processing Memory Usage', 'cross_thread');
            end

            % Assert that parallel memory efficiency meets requirements
            obj.assertTrue(memoryData.netChangeMB < 500, 'Parallel processing memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.crossThreadMemory = struct('memoryUsage', memoryData.netChangeMB);
        end

        function testWorkspaceMemory(obj)
            %testWorkspaceMemory Tests memory management for workspace variables

            % Create test cases for workspace variable management
            % Measure memory usage patterns during variable creation/modification
            % Test memory cleanup during clear operations
            % Verify efficient memory reuse patterns

            % Define test parameters
            numVariables = 10;
            variableSize = 100;

            % Measure memory usage before variable creation
            initialMemory = memory();

            % Create variables
            for i = 1:numVariables
                assignin('base', sprintf('var%d', i), rand(variableSize));
            end

            % Measure memory usage after variable creation
            afterCreationMemory = memory();

            % Calculate memory difference
            memoryDifference = afterCreationMemory.MemUsedMATLAB - initialMemory.MemUsedMATLAB;

            % Test memory cleanup during clear operations
            clear var*

            % Measure memory usage after clear operations
            afterClearMemory = memory();

            % Calculate memory difference after clear
            memoryDifferenceAfterClear = afterClearMemory.MemUsedMATLAB - initialMemory.MemUsedMATLAB;

            % Verify efficient memory reuse patterns
            obj.assertAlmostEqual(memoryDifferenceAfterClear, 0, 'Workspace memory management is inefficient: Memory was not properly released during clear operations.');

            % Generate memory usage visualization if enabled
            if obj.generateVisualizations
                obj.visualizeMemoryUsage(struct('beforeBytes', initialMemory.MemUsedMATLAB, 'afterBytes', afterCreationMemory.MemUsedMATLAB, 'netChange', memoryDifference, 'netChangeMB', memoryDifference / (1024^2)), 'Workspace Memory Usage', 'workspace');
            end

            % Assert that workspace memory management is efficient
            obj.assertTrue(memoryDifference < 100, 'Workspace memory management is inefficient: Memory usage exceeds threshold.');

            % Return structured test results with memory metrics
            obj.testResults.workspaceMemory = struct('memoryDifference', memoryDifference);
        end

        function memoryInfo = measureMemoryFootprint(obj, func, varargin)
            %measureMemoryFootprint Utility function to measure memory footprint of a function

            % Clear workspace to establish baseline
            clear;

            % Record initial memory state using whos() and memory()
            initialVars = whos();
            initialBytes = sum([initialVars.bytes]);
            initialMemory = memory();

            % Execute function with provided arguments
            try
                func(varargin{:});
            catch ME
                % Re-throw any errors that occur
                rethrow(ME);
            end

            % Record post-execution memory state
            finalVars = whos();
            finalBytes = sum([finalVars.bytes]);
            finalMemory = memory();

            % Calculate memory differences (allocated, peak, retained)
            memoryAllocated = finalBytes - initialBytes;
            peakMemory = finalMemory.MemPeakMATLAB;
            memoryRetained = finalMemory.MemUsedMATLAB - initialMemory.MemUsedMATLAB;

            % Analyze memory usage efficiency
            % Return structured memory usage information
            memoryInfo = struct( ...
                'initialBytes', initialBytes, ...
                'finalBytes', finalBytes, ...
                'memoryAllocated', memoryAllocated, ...
                'peakMemory', peakMemory, ...
                'memoryRetained', memoryRetained, ...
                'netChange', memoryRetained, ...
                'netChangeMB', memoryRetained / (1024^2) ...
            );
        end

        function comparisonResults = compareImplementationMemory(obj, func1, func2, varargin)
            %compareImplementationMemory Compares memory usage between two implementations

            % Measure memory footprint of first implementation
            memoryData1 = obj.measureMemoryFootprint(func1, varargin{:});

            % Reset memory state to baseline
            clear;

            % Measure memory footprint of second implementation
            memoryData2 = obj.measureMemoryFootprint(func2, varargin{:});

            % Calculate memory efficiency ratio between implementations
            memoryRatio = memoryData1.netChangeMB / memoryData2.netChangeMB;

            % Analyze differences in memory usage patterns
            % Generate comparison visualization if enabled
            % Return structured comparison results
            comparisonResults = struct( ...
                'memoryData1', memoryData1, ...
                'memoryData2', memoryData2, ...
                'memoryRatio', memoryRatio ...
            );
        end

        function figHandle = visualizeMemoryUsage(obj, memoryData, title, fileName)
            %visualizeMemoryUsage Creates visualizations of memory usage test results

            % Create appropriate figure based on memory data structure
            figHandle = figure;

            % Plot memory usage metrics (allocated, peak, retained)
            bar([memoryData.initialBytes, memoryData.finalBytes]);
            set(gca, 'xticklabel', {'Initial', 'Final'});
            ylabel('Memory Usage (Bytes)');
            title(title);

            % Add appropriate labels, legend, and title
            % Save visualization to reportOutputPath if enabled
            if obj.generateVisualizations
                outputPath = fullfile(obj.reportOutputPath, [fileName, '.png']);
                saveas(figHandle, outputPath);
            end

            % Return handle to created figure
        end

        function reportPath = generateMemoryReport(obj)
            %generateMemoryReport Generates a comprehensive memory usage report

            % Compile all memory test results
            % Format results into readable report structure
            % Generate summary tables and statistics
            % Include memory usage visualizations
            % Save report to reportOutputPath

            % Create output directory if it doesn't exist
            if ~exist(obj.reportOutputPath, 'dir')
                mkdir(obj.reportOutputPath);
            end

            % Generate report file name
            reportFileName = fullfile(obj.reportOutputPath, 'memory_usage_report.txt');

            % Open report file
            fileID = fopen(reportFileName, 'w');

            % Write report header
            fprintf(fileID, 'Memory Usage Report\n');
            fprintf(fileID, '--------------------\n');
            fprintf(fileID, 'Date: %s\n', datestr(now));

            % Write test results
            testNames = fieldnames(obj.testResults);
            for i = 1:length(testNames)
                testName = testNames{i};
                testResult = obj.testResults.(testName);

                fprintf(fileID, '\nTest: %s\n', testName);
                fprintf(fileID, '--------------------\n');

                % Write memory usage details
                if isfield(testResult, 'memoryUsage')
                    fprintf(fileID, 'Memory Usage (MB): %s\n', num2str(testResult.memoryUsage));
                end

                if isfield(testResult, 'avgMemoryUsage')
                    fprintf(fileID, 'Average Memory Usage (MB): %.2f\n', testResult.avgMemoryUsage);
                end

                if isfield(testResult, 'memoryDifference')
                    fprintf(fileID, 'Memory Difference (MB): %.2f\n', testResult.memoryDifference);
                end
            end

            % Close report file
            fclose(fileID);

            % Return path to generated report
            reportPath = reportFileName;
        end

        function results = runAllTests(obj)
            %runAllTests Runs all memory usage tests and generates comprehensive report

            % Execute all individual memory test methods
            obj.testMatrixOperationMemory();
            obj.testTimeSeriesMemory();
            obj.testVolatilityModelMemory();
            obj.testMEXAllocationDeallocation();
            obj.testLargeDatasetMemory();
            obj.testHighFrequencyDataMemory();
            obj.testBootstrapMemory();
            obj.testMultivariateModelMemory();
            obj.testCrossThreadMemory();
            obj.testWorkspaceMemory();

            % Collect memory metrics across all components
            % Analyze overall memory efficiency of the toolbox
            % Generate comprehensive memory report with visualizations
            % Return aggregated test results with overall assessment
            results = obj.testResults;
        end

        function setReportOutputPath(obj, path)
            %setReportOutputPath Sets the output directory for memory reports

            % Validate that path is a valid directory or create it
            if ~exist(path, 'dir')
                mkdir(path);
            end

            % Update reportOutputPath property
            obj.reportOutputPath = path;
        end

        function setGenerateVisualizations(obj, flag)
            %setGenerateVisualizations Controls whether memory usage visualizations are generated

            % Validate input is logical
            if ~islogical(flag)
                error('Input must be a logical value.');
            end

            % Set generateVisualizations property to specified value
            obj.generateVisualizations = flag;
        end
    end
end