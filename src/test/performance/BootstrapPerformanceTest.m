classdef BootstrapPerformanceTest < BaseTest
    % Test class for measuring and analyzing the performance of bootstrap methods in the MFE Toolbox
    
    properties
        benchmark           PerformanceBenchmark  % Benchmark utility for performance testing
        testData            matrix                % Test data for bootstrap operations
        testDataParameters  struct                % Parameters used for generating test data
        defaultIterations   double                % Default number of iterations for performance tests
        dataSizes           cell                  % Array of data sizes for scalability tests
        saveResults         logical               % Whether to save performance results
        resultsPath         char                  % Path for saving results
    end
    
    methods
        function obj = BootstrapPerformanceTest()
            % Initialize a new BootstrapPerformanceTest instance with test parameters
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Initialize benchmark utility
            obj.benchmark = PerformanceBenchmark();
            
            % Set default test parameters
            obj.defaultIterations = 50;
            obj.dataSizes = {100, 500, 1000, 5000, 10000};
            obj.saveResults = true;
            obj.resultsPath = '../results/performance';
            
            % Load test data
            obj.loadBootstrapTestData();
            
            % Configure benchmark parameters
            obj.benchmark.setIterations(obj.defaultIterations);
            obj.benchmark.setWarmupIterations(5);
            obj.benchmark.setVerbose(false);
            obj.benchmark.enableVisualizationSaving(obj.saveResults, obj.resultsPath);
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            
            % Call superclass setUp
            setUp@BaseTest(obj);
            
            % Ensure test data is available
            if isempty(obj.testData)
                obj.loadBootstrapTestData();
            end
            
            % Reset benchmark state
            obj.benchmark.setIterations(obj.defaultIterations);
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            
            % Call superclass tearDown
            tearDown@BaseTest(obj);
            
            % Save results if needed
            if obj.saveResults && ~isempty(obj.benchmark.getLastResult())
                % Create results directory if it doesn't exist
                if ~exist(obj.resultsPath, 'dir')
                    mkdir(obj.resultsPath);
                end
            end
        end
        
        function results = testBlockBootstrapPerformance(obj)
            % Test the execution time of the block_bootstrap function with various parameters
            
            % Configure test parameters
            blockSize = 10;
            numBootstraps = 500;
            
            % Create function handle for block bootstrap
            blockBootstrapFcn = @() block_bootstrap(obj.testData, blockSize, numBootstraps);
            
            % Benchmark function execution time
            results = obj.benchmark.benchmarkFunction(blockBootstrapFcn);
            
            % Generate bootstrap sample once to validate
            bsdata = block_bootstrap(obj.testData, blockSize, numBootstraps);
            
            % Verify bootstrap sample structure
            [T, N, B] = size(bsdata);
            obj.assertTrue(T == size(obj.testData, 1), 'Bootstrap sample should have same length as original data');
            obj.assertTrue(N == size(obj.testData, 2), 'Bootstrap sample should have same number of variables as original data');
            obj.assertTrue(B == numBootstraps, 'Number of bootstrap samples should match requested number');
            
            % Analyze performance results
            if obj.verbose
                fprintf('Block Bootstrap Performance:\n');
                fprintf('  Mean execution time: %.4f ms\n', results.mean * 1000);
                fprintf('  Median execution time: %.4f ms\n', results.median * 1000);
                fprintf('  Std dev: %.4f ms\n', results.std * 1000);
            end
            
            % Visualize results
            obj.benchmark.visualizeResults(results, 'timeseries', 'Block Bootstrap Execution Time');
            
            % Return performance metrics
            obj.testResults.blockBootstrapPerformance = results;
        end
        
        function results = testStationaryBootstrapPerformance(obj)
            % Test the execution time of the stationary_bootstrap function with various parameters
            
            % Configure test parameters
            p = 0.1; % Expected block length = 1/p = 10
            numBootstraps = 500;
            
            % Create function handle for stationary bootstrap
            stationaryBootstrapFcn = @() stationary_bootstrap(obj.testData, p, numBootstraps);
            
            % Benchmark function execution time
            results = obj.benchmark.benchmarkFunction(stationaryBootstrapFcn);
            
            % Generate bootstrap sample once to validate
            bsdata = stationary_bootstrap(obj.testData, p, numBootstraps);
            
            % Verify bootstrap sample structure
            [T, N, B] = size(bsdata);
            obj.assertTrue(T == size(obj.testData, 1), 'Bootstrap sample should have same length as original data');
            obj.assertTrue(N == size(obj.testData, 2), 'Bootstrap sample should have same number of variables as original data');
            obj.assertTrue(B == numBootstraps, 'Number of bootstrap samples should match requested number');
            
            % Analyze performance results
            if obj.verbose
                fprintf('Stationary Bootstrap Performance:\n');
                fprintf('  Mean execution time: %.4f ms\n', results.mean * 1000);
                fprintf('  Median execution time: %.4f ms\n', results.median * 1000);
                fprintf('  Std dev: %.4f ms\n', results.std * 1000);
            end
            
            % Visualize results
            obj.benchmark.visualizeResults(results, 'timeseries', 'Stationary Bootstrap Execution Time');
            
            % Return performance metrics
            obj.testResults.stationaryBootstrapPerformance = results;
        end
        
        function results = testBootstrapVariancePerformance(obj)
            % Test the execution time of the bootstrap_variance function with various parameters
            
            % Configure test parameters
            statistic_fn = @(x) mean(x); % Use mean as the test statistic
            options = struct('bootstrap_type', 'block', 'block_size', 10, 'replications', 500);
            
            % Create function handle for bootstrap variance
            bootstrapVarianceFcn = @() bootstrap_variance(obj.testData, statistic_fn, options);
            
            % Benchmark function execution time
            results = obj.benchmark.benchmarkFunction(bootstrapVarianceFcn);
            
            % Generate bootstrap variance estimate once to validate
            variance_results = bootstrap_variance(obj.testData, statistic_fn, options);
            
            % Verify variance estimate structure
            obj.assertTrue(isfield(variance_results, 'variance'), 'Variance estimate should be present');
            obj.assertTrue(isfield(variance_results, 'std_error'), 'Standard error should be present');
            obj.assertTrue(isfield(variance_results, 'bootstrap_stats'), 'Bootstrap statistics should be present');
            
            % Analyze performance results
            if obj.verbose
                fprintf('Bootstrap Variance Performance:\n');
                fprintf('  Mean execution time: %.4f ms\n', results.mean * 1000);
                fprintf('  Median execution time: %.4f ms\n', results.median * 1000);
                fprintf('  Std dev: %.4f ms\n', results.std * 1000);
            end
            
            % Visualize results
            obj.benchmark.visualizeResults(results, 'timeseries', 'Bootstrap Variance Execution Time');
            
            % Return performance metrics
            obj.testResults.bootstrapVariancePerformance = results;
        end
        
        function results = testBlockBootstrapMemoryUsage(obj)
            % Test the memory usage of the block_bootstrap function
            
            % Configure test parameters
            blockSize = 10;
            numBootstraps = 500;
            
            % Create function handle for block bootstrap
            blockBootstrapFcn = @() block_bootstrap(obj.testData, blockSize, numBootstraps);
            
            % Measure memory usage
            results = obj.benchmark.measureMemoryUsage(blockBootstrapFcn);
            
            % Analyze memory usage
            if obj.verbose
                fprintf('Block Bootstrap Memory Usage:\n');
                fprintf('  Memory change: %.2f MB\n', results.netChangeMB);
            end
            
            % Validate that memory usage is reasonable
            % The main memory usage should be proportional to data size * number of bootstraps
            expectedBytes = 8 * numel(obj.testData) * numBootstraps; % 8 bytes per double
            expectedMB = expectedBytes / (1024^2);
            
            % Allow 50% more for overhead
            obj.assertTrue(results.netChangeMB < expectedMB * 1.5, 'Memory usage should be reasonable');
            
            % Return memory usage statistics
            obj.testResults.blockBootstrapMemoryUsage = results;
        end
        
        function results = testStationaryBootstrapMemoryUsage(obj)
            % Test the memory usage of the stationary_bootstrap function
            
            % Configure test parameters
            p = 0.1; % Expected block length = 1/p = 10
            numBootstraps = 500;
            
            % Create function handle for stationary bootstrap
            stationaryBootstrapFcn = @() stationary_bootstrap(obj.testData, p, numBootstraps);
            
            % Measure memory usage
            results = obj.benchmark.measureMemoryUsage(stationaryBootstrapFcn);
            
            % Analyze memory usage
            if obj.verbose
                fprintf('Stationary Bootstrap Memory Usage:\n');
                fprintf('  Memory change: %.2f MB\n', results.netChangeMB);
            end
            
            % Validate that memory usage is reasonable
            % The main memory usage should be proportional to data size * number of bootstraps
            expectedBytes = 8 * numel(obj.testData) * numBootstraps; % 8 bytes per double
            expectedMB = expectedBytes / (1024^2);
            
            % Allow 50% more for overhead
            obj.assertTrue(results.netChangeMB < expectedMB * 1.5, 'Memory usage should be reasonable');
            
            % Return memory usage statistics
            obj.testResults.stationaryBootstrapMemoryUsage = results;
        end
        
        function results = testBootstrapScalability(obj)
            % Test how bootstrap performance scales with increasing data sizes
            
            % Define test parameters
            numBootstraps = 100;
            blockSize = 10;
            p = 0.1; % For stationary bootstrap
            
            % Create data generator function
            dataGenerator = @(size) obj.generateTestData(size);
            
            % Create function handles for bootstrap methods
            blockBootstrapFn = @(data) block_bootstrap(data, blockSize, numBootstraps);
            stationaryBootstrapFn = @(data) stationary_bootstrap(data, p, numBootstraps);
            
            % Perform scalability test for block bootstrap
            if obj.verbose
                fprintf('Testing Block Bootstrap scalability with data sizes: %s\n', ...
                    strjoin(cellfun(@num2str, obj.dataSizes, 'UniformOutput', false), ', '));
            end
            
            blockResults = obj.benchmark.scalabilityTest(blockBootstrapFn, obj.dataSizes, dataGenerator);
            
            % Perform scalability test for stationary bootstrap
            if obj.verbose
                fprintf('Testing Stationary Bootstrap scalability with data sizes: %s\n', ...
                    strjoin(cellfun(@num2str, obj.dataSizes, 'UniformOutput', false), ', '));
            end
            
            stationaryResults = obj.benchmark.scalabilityTest(stationaryBootstrapFn, obj.dataSizes, dataGenerator);
            
            % Compare scaling behavior
            figure;
            plot(cell2mat(obj.dataSizes), blockResults.timings * 1000, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;
            plot(cell2mat(obj.dataSizes), stationaryResults.timings * 1000, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
            hold off;
            legend('Block Bootstrap', 'Stationary Bootstrap', 'Location', 'NorthWest');
            title('Bootstrap Method Scalability Comparison');
            xlabel('Dataset Size (observations)');
            ylabel('Execution Time (ms)');
            grid on;
            set(gca, 'XScale', 'log');
            
            % Save the figure if needed
            if obj.saveResults
                if ~exist(obj.resultsPath, 'dir')
                    mkdir(obj.resultsPath);
                end
                saveas(gcf, fullfile(obj.resultsPath, 'bootstrap_scalability_comparison.fig'));
            end
            
            % Combine results
            results = struct(...
                'blockBootstrap', blockResults, ...
                'stationaryBootstrap', stationaryResults ...
            );
            
            % Analyze scaling behavior
            if obj.verbose
                fprintf('Scalability Test Results:\n');
                fprintf('  Block Bootstrap scaling behavior: %s (exponent = %.2f)\n', ...
                    blockResults.scalingBehavior, blockResults.scalingExponent);
                fprintf('  Stationary Bootstrap scaling behavior: %s (exponent = %.2f)\n', ...
                    stationaryResults.scalingBehavior, stationaryResults.scalingExponent);
            end
            
            % Return combined results
            obj.testResults.bootstrapScalability = results;
        end
        
        function results = testBootstrapParameterSensitivity(obj)
            % Test sensitivity of bootstrap performance to parameter variations
            
            % Define parameter ranges
            blockSizes = [5, 10, 20, 40, 80];
            probabilities = [0.2, 0.1, 0.05, 0.025, 0.0125]; % Equivalent expected block lengths
            numBootstraps = 100;
            
            % Initialize results arrays
            blockTimes = zeros(length(blockSizes), 1);
            stationaryTimes = zeros(length(probabilities), 1);
            
            % Test block bootstrap with varying block sizes
            for i = 1:length(blockSizes)
                blockSize = blockSizes(i);
                
                % Create function handle
                blockBootstrapFcn = @() block_bootstrap(obj.testData, blockSize, numBootstraps);
                
                % Benchmark execution time
                result = obj.benchmark.benchmarkFunction(blockBootstrapFcn);
                
                % Store mean execution time
                blockTimes(i) = result.mean;
                
                if obj.verbose
                    fprintf('Block Bootstrap with block size %d: %.4f ms\n', ...
                        blockSize, result.mean * 1000);
                end
            end
            
            % Test stationary bootstrap with varying probability parameters
            for i = 1:length(probabilities)
                p = probabilities(i);
                
                % Create function handle
                stationaryBootstrapFcn = @() stationary_bootstrap(obj.testData, p, numBootstraps);
                
                % Benchmark execution time
                result = obj.benchmark.benchmarkFunction(stationaryBootstrapFcn);
                
                % Store mean execution time
                stationaryTimes(i) = result.mean;
                
                if obj.verbose
                    fprintf('Stationary Bootstrap with p=%.4f (expected block length=%.1f): %.4f ms\n', ...
                        p, 1/p, result.mean * 1000);
                end
            end
            
            % Visualize parameter sensitivity
            figure;
            
            % Plot block bootstrap sensitivity
            subplot(2, 1, 1);
            plot(blockSizes, blockTimes * 1000, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
            title('Block Bootstrap Sensitivity to Block Size');
            xlabel('Block Size');
            ylabel('Execution Time (ms)');
            grid on;
            
            % Plot stationary bootstrap sensitivity
            subplot(2, 1, 2);
            plot(1 ./ probabilities, stationaryTimes * 1000, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
            title('Stationary Bootstrap Sensitivity to Expected Block Length');
            xlabel('Expected Block Length (1/p)');
            ylabel('Execution Time (ms)');
            grid on;
            
            % Save the figure if needed
            if obj.saveResults
                if ~exist(obj.resultsPath, 'dir')
                    mkdir(obj.resultsPath);
                end
                saveas(gcf, fullfile(obj.resultsPath, 'bootstrap_parameter_sensitivity.fig'));
            end
            
            % Create results structure
            results = struct(...
                'blockSizes', blockSizes, ...
                'blockTimes', blockTimes, ...
                'probabilities', probabilities, ...
                'stationaryTimes', stationaryTimes ...
            );
            
            % Return sensitivity analysis results
            obj.testResults.bootstrapParameterSensitivity = results;
        end
        
        function results = compareBootstrapMethods(obj)
            % Compare performance between block and stationary bootstrap methods
            
            % Configure test parameters to make methods comparable
            blockSize = 10;
            p = 0.1; % Expected block length = 1/p = 10 (matching block size)
            numBootstraps = 500;
            
            % Create function handles
            blockBootstrapFcn = @() block_bootstrap(obj.testData, blockSize, numBootstraps);
            stationaryBootstrapFcn = @() stationary_bootstrap(obj.testData, p, numBootstraps);
            
            % Benchmark both methods
            blockResults = obj.benchmark.benchmarkFunction(blockBootstrapFcn);
            stationaryResults = obj.benchmark.benchmarkFunction(stationaryBootstrapFcn);
            
            % Calculate performance ratio
            performanceRatio = blockResults.mean / stationaryResults.mean;
            
            % Display comparison results
            if obj.verbose
                fprintf('Bootstrap Methods Comparison:\n');
                fprintf('  Block Bootstrap (block size = %d): %.4f ms\n', ...
                    blockSize, blockResults.mean * 1000);
                fprintf('  Stationary Bootstrap (p = %.2f, expected block length = %.1f): %.4f ms\n', ...
                    p, 1/p, stationaryResults.mean * 1000);
                fprintf('  Performance ratio (block/stationary): %.2f\n', performanceRatio);
                
                if performanceRatio > 1
                    fprintf('  Stationary bootstrap is %.1f%% faster\n', (performanceRatio - 1) * 100);
                else
                    fprintf('  Block bootstrap is %.1f%% faster\n', (1 - performanceRatio) * 100);
                end
            end
            
            % Visualize comparison
            figure;
            bar([blockResults.mean, stationaryResults.mean] * 1000);
            set(gca, 'XTickLabel', {'Block Bootstrap', 'Stationary Bootstrap'});
            title('Performance Comparison of Bootstrap Methods');
            ylabel('Mean Execution Time (ms)');
            grid on;
            
            % Add text for performance ratio
            text(1.5, mean([blockResults.mean, stationaryResults.mean] * 1000), ...
                sprintf('Ratio: %.2f', performanceRatio), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            
            % Save the figure if needed
            if obj.saveResults
                if ~exist(obj.resultsPath, 'dir')
                    mkdir(obj.resultsPath);
                end
                saveas(gcf, fullfile(obj.resultsPath, 'bootstrap_methods_comparison.fig'));
            end
            
            % Create comparison results structure
            results = struct(...
                'blockBootstrap', blockResults, ...
                'stationaryBootstrap', stationaryResults, ...
                'performanceRatio', performanceRatio ...
            );
            
            % Return comparison results
            obj.testResults.bootstrapComparison = results;
        end
        
        function data = generateTestData(obj, size, parameters)
            % Generate or load test data for bootstrap performance testing
            
            % Use default size if not provided
            if nargin < 2 || isempty(size)
                size = 1000; % Default data size
            end
            
            % Use default parameters if not provided
            if nargin < 3 || isempty(parameters)
                % Default parameters for financial returns generation
                parameters = struct(...
                    'distribution', 't', ...
                    'distParams', 5, ...
                    'garchParams', [0.01, 0.1, 0.85] ...
                );
            end
            
            % Generate synthetic financial returns
            data = generateFinancialReturns(size, 1, parameters);
            
            % Store parameters for reference
            obj.testDataParameters = parameters;
        end
        
        function testData = loadBootstrapTestData(obj)
            % Load standard test datasets for bootstrap performance testing
            
            try
                % Try to load financial returns test data
                testData = obj.loadTestData('financial_returns.mat');
                
                % Extract normal returns from test data
                if isfield(testData, 'normal')
                    obj.testData = testData.normal;
                else
                    % Use the first field if 'normal' doesn't exist
                    fields = fieldnames(testData);
                    obj.testData = testData.(fields{1});
                end
                
            catch ME
                % If loading fails, generate synthetic data
                if obj.verbose
                    warning('Could not load test data: %s\nGenerating synthetic data instead.', ME.message);
                end
                
                % Generate financial returns data
                obj.testData = obj.generateTestData();
            end
            
            % Return the loaded test data
            testData = obj.testData;
        end
    end
end