classdef LargeScaleDataTest < BaseTest
    % LARGESCALEDATATEST Test class for evaluating the MFE Toolbox's performance,
    % memory efficiency, and numerical stability with large-scale financial datasets
    %
    % This class implements comprehensive tests to validate the MFE Toolbox's ability
    % to handle progressively larger financial datasets while maintaining computational
    % efficiency, numerical stability, and memory optimization.
    %
    % The tests cover various components:
    %   - Time series analysis (ARMA/ARMAX models)
    %   - Volatility models (GARCH family models)
    %   - Cross-sectional analysis
    %   - High-frequency data processing
    %   - Matrix operations with large matrices
    %   - Bootstrap methods with large samples
    %   - Multivariate models with high dimensions
    %
    % For each component, tests measure:
    %   - Computational efficiency (execution time vs data size)
    %   - Memory usage patterns
    %   - Numerical stability with increasing data size
    %   - Scaling behavior (linear, quadratic, etc.)
    %
    % Example:
    %   test = LargeScaleDataTest();
    %   results = test.runAllTests();
    %
    % See also: BaseTest, PerformanceBenchmark, TestDataGenerator, TestReporter
    
    properties
        % Data generator for creating test datasets
        dataGenerator TestDataGenerator
        
        % Performance benchmarking utility
        benchmarker PerformanceBenchmark
        
        % Test reporter for generating results
        reporter TestReporter
        
        % Structure for storing test data
        testData struct
        
        % Array of data sizes to test
        dataSizes cell
        
        % Flag to control visualization generation
        generateVisualizations logical
        
        % Path for report output
        reportOutputPath string
    end
    
    methods
        function obj = LargeScaleDataTest()
            % Initializes the LargeScaleDataTest class with default settings and test configuration
            %
            % USAGE:
            %   test = LargeScaleDataTest()
            %
            % OUTPUTS:
            %   test - Initialized LargeScaleDataTest object
            
            % Call superclass constructor with test name
            obj@BaseTest('Large Scale Data Test');
            
            % Initialize data generator
            obj.dataGenerator = TestDataGenerator();
            
            % Initialize benchmarker
            obj.benchmarker = PerformanceBenchmark();
            
            % Initialize test reporter
            obj.reporter = TestReporter('MFE Toolbox Large-Scale Performance Test');
            
            % Set up default data sizes for progressive testing
            % These represent sample sizes that increase exponentially
            obj.dataSizes = {100, 500, 1000, 5000, 10000, 50000, 100000};
            
            % Enable visualization generation by default
            obj.generateVisualizations = true;
            
            % Set default report output path to current directory
            obj.reportOutputPath = pwd;
            
            % Initialize test data structure
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepares the test environment before executing tests
            %
            % This method initializes the benchmarking environment, verifies
            % system resources, and prepares for large-scale testing.
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Configure benchmarker with appropriate iteration count
            obj.benchmarker.setIterations(10);  % Lower iteration count for large-scale tests
            
            % Verify system has sufficient memory for large-scale tests
            memInfo = memory;
            if memInfo.MemAvailableAllArrays < 1e9  % Check for at least 1GB available
                warning('Limited memory available (%0.2f GB). Some large-scale tests may fail.', ...
                    memInfo.MemAvailableAllArrays/1e9);
            end
            
            % Clear workspace to establish memory baseline
            evalin('caller', 'clear("variables")');
        end
        
        function tearDown(obj)
            % Cleans up after test execution
            %
            % This method generates a report if tests were run, cleans up test data,
            % and resets the test environment.
            
            % Generate performance report if tests were executed
            if ~isempty(fieldnames(obj.testData))
                obj.generateScalabilityReport(obj.testData);
            end
            
            % Clean up test data to free memory
            obj.testData = struct();
            
            % Reset benchmarker state
            obj.benchmarker = PerformanceBenchmark();
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function result = testTimeSeriesScalability(obj)
            % Tests how time series analysis performance scales with increasing dataset size
            %
            % OUTPUTS:
            %   result - Test result containing scalability metrics
            
            % Define array of increasing time series lengths to test
            timeSeries = obj.dataSizes;
            
            % Create data generator function for time series of varying sizes
            dataGenerator = @(size) obj.dataGenerator.generateFinancialReturns(size, 1, ...
                struct('mean', 0, 'variance', 1, 'arParameters', [0.5, -0.3], ...
                'maParameters', [0.2, 0.1], 'constant', 0.001));
            
            % Use benchmarker to measure ARMA/ARMAX performance across data sizes
            armaResults = obj.benchmarker.scalabilityTest(...
                @(data) armaxfilter([0.001; 0.5; -0.3; 0.2; 0.1], data, 2, 2, 1), ...
                timeSeries, dataGenerator);
            
            % Analyze scaling behavior (linear, quadratic, etc.)
            result = struct();
            result.component = 'Time Series Models';
            result.dataSizes = timeSeries;
            result.executionTimes = armaResults.timings;
            result.scalingBehavior = armaResults.scalingBehavior;
            result.scalingExponent = armaResults.scalingExponent;
            
            % Generate visualization if enabled
            if obj.generateVisualizations
                figure('Name', 'Time Series Scalability');
                loglog(cell2mat(timeSeries), result.executionTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Sample Size');
                ylabel('Execution Time (s)');
                title(sprintf('ARMA Model Scalability (%s: exponent = %.2f)', ...
                    result.scalingBehavior, result.scalingExponent));
            end
            
            % Assert that performance scales efficiently with time series length
            % For time series models, we typically expect linear or slightly superlinear scaling
            obj.assertTrue(result.scalingExponent < 2.0, ...
                'Time series models should scale better than quadratic');
            
            % Store result in test data
            obj.testData.timeSeriesScalability = result;
        end
        
        function result = testVolatilityModelScalability(obj)
            % Tests how volatility model estimation scales with dataset size
            %
            % OUTPUTS:
            %   result - Test result containing scalability metrics
            
            % Define array of increasing time series lengths for volatility models
            timeSeries = obj.dataSizes(1:min(5, numel(obj.dataSizes)));  % Limit to reasonable sizes for GARCH
            
            % Create data generator function for GARCH-type model data
            dataGenerator = @(size) obj.dataGenerator.generateVolatilitySeries(size, 'GARCH', ...
                struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85, 'p', 1, 'q', 1, ...
                'distribution', 'normal')).returns;
            
            % Setup GARCH parameter vector for testing
            garchParams = [0.05; 0.1; 0.85];  % omega, alpha, beta
            
            % Setup GARCH model options
            garchOptions = struct('model', 'GARCH', 'p', 1, 'q', 1);
            
            % Use benchmarker to measure volatility model performance
            % Testing MATLAB implementation
            matlabResults = obj.benchmarker.scalabilityTest(...
                @(data) garchcore(garchParams, data, garchOptions), ...
                timeSeries, dataGenerator);
            
            % Check if MEX implementation exists and compare if available
            mexImplementationExists = exist('tarch_core', 'file') == 3;  % 3 = MEX file exists
            if mexImplementationExists
                % Testing MEX implementation
                garchOptions.useMEX = true;
                mexResults = obj.benchmarker.scalabilityTest(...
                    @(data) garchcore(garchParams, data, garchOptions), ...
                    timeSeries, dataGenerator);
                
                % Compare scaling behavior
                result = struct();
                result.component = 'Volatility Models';
                result.dataSizes = timeSeries;
                result.matlabTimes = matlabResults.timings;
                result.matlabScalingBehavior = matlabResults.scalingBehavior;
                result.matlabScalingExponent = matlabResults.scalingExponent;
                result.mexTimes = mexResults.timings;
                result.mexScalingBehavior = mexResults.scalingBehavior;
                result.mexScalingExponent = mexResults.scalingExponent;
                result.speedupRatio = matlabResults.timings ./ mexResults.timings;
                
                % Generate comparison visualization if enabled
                if obj.generateVisualizations
                    figure('Name', 'Volatility Model Scalability');
                    
                    % Plot execution times
                    subplot(2, 1, 1);
                    loglog(cell2mat(timeSeries), matlabResults.timings, 'b-o', 'LineWidth', 2);
                    hold on;
                    loglog(cell2mat(timeSeries), mexResults.timings, 'r-s', 'LineWidth', 2);
                    grid on;
                    xlabel('Sample Size');
                    ylabel('Execution Time (s)');
                    legend('MATLAB Implementation', 'MEX Implementation', 'Location', 'NorthWest');
                    title('Volatility Model Scalability');
                    
                    % Plot speedup ratio
                    subplot(2, 1, 2);
                    semilogx(cell2mat(timeSeries), result.speedupRatio, 'g-d', 'LineWidth', 2);
                    grid on;
                    xlabel('Sample Size');
                    ylabel('Speedup Ratio (MATLAB/MEX)');
                    title('MEX Implementation Speedup');
                end
                
                % Assert MEX implementation provides significant speedup
                meanSpeedup = mean(result.speedupRatio);
                obj.assertTrue(meanSpeedup >= 1.5, ...
                    sprintf('MEX implementation should provide at least 50%% speedup. Actual: %.2f%%', ...
                    (meanSpeedup-1)*100));
            else
                % If MEX not available, just test MATLAB implementation
                result = struct();
                result.component = 'Volatility Models';
                result.dataSizes = timeSeries;
                result.executionTimes = matlabResults.timings;
                result.scalingBehavior = matlabResults.scalingBehavior;
                result.scalingExponent = matlabResults.scalingExponent;
                
                % Generate visualization if enabled
                if obj.generateVisualizations
                    figure('Name', 'Volatility Model Scalability');
                    loglog(cell2mat(timeSeries), result.executionTimes, 'b-o', 'LineWidth', 2);
                    grid on;
                    xlabel('Sample Size');
                    ylabel('Execution Time (s)');
                    title(sprintf('GARCH Model Scalability (%s: exponent = %.2f)', ...
                        result.scalingBehavior, result.scalingExponent));
                end
            end
            
            % Assert volatility models scale efficiently with data size
            obj.assertTrue(result.matlabScalingExponent < 2.2 || result.scalingExponent < 2.2, ...
                'Volatility models should scale better than O(n^2.2)');
            
            % Store result in test data
            obj.testData.volatilityScalability = result;
        end
        
        function result = testCrossSectionalScalability(obj)
            % Tests how cross-sectional analysis scales with increasing number of assets
            %
            % OUTPUTS:
            %   result - Test result containing scalability metrics
            
            % Define array of increasing asset counts and time periods
            assetCounts = {10, 50, 100, 500, 1000};
            timePeriods = 120;  % 10 years of monthly data
            
            % Create data generator function for cross-sectional data
            dataGenerator = @(numAssets) obj.dataGenerator.generateCrossSectionalData(...
                numAssets, timePeriods, 3, struct(...
                'factorMean', [0.001; 0.0005; 0.0008], ...
                'factorCov', [0.04, 0.01, 0.005; 0.01, 0.02, 0.003; 0.005, 0.003, 0.03], ...
                'loadingDist', 'normal', ...
                'loadingParams', struct('mean', 1.0, 'std', 0.3), ...
                'idioVol', 0.02)).returns;
            
            % Use benchmarker to measure cross-sectional analysis performance
            % Here we're testing a basic OLS regression as a representative cross-sectional operation
            csResults = obj.benchmarker.scalabilityTest(...
                @(data) data \ ones(size(data, 1), 1), ... % Simple OLS regression
                assetCounts, dataGenerator);
            
            % Analyze scaling behavior with respect to number of assets
            result = struct();
            result.component = 'Cross-Sectional Analysis';
            result.dataSizes = assetCounts;
            result.executionTimes = csResults.timings;
            result.scalingBehavior = csResults.scalingBehavior;
            result.scalingExponent = csResults.scalingExponent;
            
            % Generate visualization if enabled
            if obj.generateVisualizations
                figure('Name', 'Cross-Sectional Scalability');
                loglog(cell2mat(assetCounts), result.executionTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Number of Assets');
                ylabel('Execution Time (s)');
                title(sprintf('Cross-Sectional Analysis Scalability (%s: exponent = %.2f)', ...
                    result.scalingBehavior, result.scalingExponent));
            end
            
            % Assert performance scales efficiently with cross-sectional dimensions
            % For OLS regression, we typically expect O(n^3) scaling
            obj.assertTrue(result.scalingExponent < 3.5, ...
                'Cross-sectional analysis should scale better than O(n^3.5)');
            
            % Store result in test data
            obj.testData.crossSectionalScalability = result;
        end
        
        function result = testHighFrequencyScalability(obj)
            % Tests how high-frequency data analysis scales with increasing data frequency
            %
            % OUTPUTS:
            %   result - Test result containing scalability metrics
            
            % Define array of increasing observation frequencies for high-frequency data
            % These represent observations per day, e.g., 1-min, 5-min, tick data
            frequencies = {78, 156, 390, 780, 1560};  % e.g., 5-min, 2.5-min, 1-min, 30s, 15s
            numDays = 5;  % 1 trading week
            
            % Create data generator function for high-frequency data
            dataGenerator = @(frequency) obj.dataGenerator.generateHighFrequencyData(...
                numDays, frequency, struct(...
                'volatilityModel', 'garch', ...
                'volatilityParams', struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85), ...
                'intradayPattern', 'U-shape', ...
                'jumpProcess', 'poisson', ...
                'jumpParams', struct('intensity', 0.1, 'jumpSize', [0, 0.005]))).returns;
            
            % Use benchmarker to measure high-frequency analysis performance
            % Here we're testing realized volatility calculation as a representative operation
            hfResults = obj.benchmarker.scalabilityTest(...
                @(data) sqrt(sum(data.^2)), ... % Realized volatility calculation
                frequencies, dataGenerator);
            
            % Analyze scaling behavior with increasing observation frequency
            result = struct();
            result.component = 'High-Frequency Analysis';
            result.dataSizes = frequencies;
            result.executionTimes = hfResults.timings;
            result.scalingBehavior = hfResults.scalingBehavior;
            result.scalingExponent = hfResults.scalingExponent;
            
            % Generate visualization if enabled
            if obj.generateVisualizations
                figure('Name', 'High-Frequency Scalability');
                loglog(cell2mat(frequencies), result.executionTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Observations per Day');
                ylabel('Execution Time (s)');
                title(sprintf('High-Frequency Analysis Scalability (%s: exponent = %.2f)', ...
                    result.scalingBehavior, result.scalingExponent));
            end
            
            % Assert performance scales efficiently with data frequency
            % For summing squared returns, we expect O(n) scaling
            obj.assertTrue(result.scalingExponent < 1.5, ...
                'High-frequency analysis should scale linearly or near-linearly');
            
            % Store result in test data
            obj.testData.highFrequencyScalability = result;
        end
        
        function result = testLargeMatrixOperations(obj)
            % Tests performance and numerical stability of matrix operations with large matrices
            %
            % OUTPUTS:
            %   result - Test result containing performance and stability metrics
            
            % Create progressively larger matrices for testing
            % We'll test square matrices of increasing size
            matrixSizes = {10, 50, 100, 500, 1000, 2000};
            
            % Initialize results structure
            result = struct();
            result.component = 'Matrix Operations';
            result.matrixSizes = matrixSizes;
            result.multiplicationTimes = zeros(length(matrixSizes), 1);
            result.inversionTimes = zeros(length(matrixSizes), 1);
            result.conditionNumbers = zeros(length(matrixSizes), 1);
            result.numericalStability = zeros(length(matrixSizes), 1);
            
            % For each matrix size
            for i = 1:length(matrixSizes)
                size = matrixSizes{i};
                
                % Create well-conditioned test matrix
                % Using diagonal dominance to ensure invertibility
                A = randn(size);
                A = A + diag(size * ones(size, 1));
                
                % Measure performance of matrix multiplication
                multTime = obj.benchmarker.measureExecutionTime(@() A * A);
                result.multiplicationTimes(i) = multTime;
                
                % Only perform inversion for matrices up to 2000x2000 to avoid memory issues
                if size <= 2000
                    % Measure performance of matrix inversion
                    invTime = obj.benchmarker.measureExecutionTime(@() inv(A));
                    result.inversionTimes(i) = invTime;
                    
                    % Analyze numerical properties using matrixdiagnostics
                    diagnostics = matrixdiagnostics(A);
                    result.conditionNumbers(i) = diagnostics.ConditionNumber;
                    
                    % Compute numerical stability measure:
                    % |I - A*inv(A)|_F / |I|_F should be small for stable computation
                    Ainv = inv(A);
                    I = eye(size);
                    AAinv = A * Ainv;
                    relError = norm(I - AAinv, 'fro') / norm(I, 'fro');
                    result.numericalStability(i) = relError;
                else
                    % For very large matrices, skip inversion
                    result.inversionTimes(i) = NaN;
                    result.conditionNumbers(i) = NaN;
                    result.numericalStability(i) = NaN;
                end
            end
            
            % Generate visualization of performance vs. matrix size
            if obj.generateVisualizations
                figure('Name', 'Matrix Operations Performance');
                
                % Plot multiplication time
                subplot(2, 2, 1);
                loglog(cell2mat(matrixSizes), result.multiplicationTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size (n×n)');
                ylabel('Execution Time (s)');
                title('Matrix Multiplication Scaling');
                
                % Plot inversion time
                subplot(2, 2, 2);
                loglog(cell2mat(matrixSizes), result.inversionTimes, 'r-s', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size (n×n)');
                ylabel('Execution Time (s)');
                title('Matrix Inversion Scaling');
                
                % Plot condition number
                subplot(2, 2, 3);
                semilogx(cell2mat(matrixSizes), result.conditionNumbers, 'g-d', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size (n×n)');
                ylabel('Condition Number');
                title('Numerical Conditioning');
                
                % Plot numerical stability
                subplot(2, 2, 4);
                semilogx(cell2mat(matrixSizes), result.numericalStability, 'm-^', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size (n×n)');
                ylabel('Relative Error');
                title('Numerical Stability: |I - A*inv(A)|_F / |I|_F');
            end
            
            % Assert numerical stability is maintained with large matrices
            maxAllowedError = 1e-10;
            stableMatrices = result.numericalStability(~isnan(result.numericalStability));
            if ~isempty(stableMatrices)
                obj.assertTrue(max(stableMatrices) < maxAllowedError, ...
                    sprintf('Matrix operations should maintain numerical stability. Max error: %e', ...
                    max(stableMatrices)));
            end
            
            % Store result in test data
            obj.testData.matrixOperations = result;
        end
        
        function result = testLargeScaleMemoryUsage(obj)
            % Tests memory usage patterns with large-scale datasets
            %
            % OUTPUTS:
            %   result - Test result containing memory usage metrics
            
            % Initialize results structure
            result = struct();
            result.component = 'Memory Usage';
            
            % Test configurations - using smaller sizes to avoid memory exhaustion
            timeSeries = obj.dataSizes(1:min(5, numel(obj.dataSizes)));
            
            % Initialize result arrays
            result.dataSizes = timeSeries;
            result.theoreticalMemory = zeros(length(timeSeries), 1);  % Bytes
            result.actualMemory = zeros(length(timeSeries), 1);       % Bytes
            result.efficiency = zeros(length(timeSeries), 1);         % Ratio (theoretical/actual)
            
            % For each data size, measure memory usage
            for i = 1:length(timeSeries)
                size = timeSeries{i};
                
                % Calculate theoretical memory requirement for double precision array
                % 8 bytes per double value
                result.theoreticalMemory(i) = size * 8;
                
                % Generate test data and measure actual memory usage
                memoryInfo = obj.benchmarker.measureMemoryUsage(...
                    @() obj.dataGenerator.generateFinancialReturns(size, 1, struct()));
                
                % Record actual memory change
                result.actualMemory(i) = memoryInfo.memoryDifference;
                
                % Calculate efficiency ratio (closer to 1 is better)
                result.efficiency(i) = result.theoreticalMemory(i) / result.actualMemory(i);
            end
            
            % Generate visualization of memory usage patterns
            if obj.generateVisualizations
                figure('Name', 'Memory Usage Patterns');
                
                % Plot memory usage
                subplot(2, 1, 1);
                loglog(cell2mat(timeSeries), result.theoreticalMemory, 'b-o', 'LineWidth', 2);
                hold on;
                loglog(cell2mat(timeSeries), result.actualMemory, 'r-s', 'LineWidth', 2);
                grid on;
                xlabel('Data Size');
                ylabel('Memory Usage (bytes)');
                legend('Theoretical Memory', 'Actual Memory', 'Location', 'NorthWest');
                title('Memory Scaling with Data Size');
                
                % Plot efficiency ratio
                subplot(2, 1, 2);
                semilogx(cell2mat(timeSeries), result.efficiency, 'g-d', 'LineWidth', 2);
                grid on;
                xlabel('Data Size');
                ylabel('Efficiency Ratio (Theoretical/Actual)');
                title('Memory Efficiency');
            end
            
            % Assert memory efficiency meets requirements
            % Efficiency ratio should be reasonably close to 1
            avgEfficiency = mean(result.efficiency);
            obj.assertTrue(avgEfficiency > 0.5, ...
                sprintf('Memory efficiency should be at least 50%%. Actual: %.2f%%', ...
                avgEfficiency * 100));
            
            % Store result in test data
            obj.testData.memoryUsage = result;
        end
        
        function result = testBootstrapScalability(obj)
            % Tests how bootstrap methods scale with dataset size and resampling count
            %
            % OUTPUTS:
            %   result - Test result containing bootstrap scalability metrics
            
            % Define array of increasing dataset sizes
            timeSeries = obj.dataSizes(1:min(5, numel(obj.dataSizes)));
            
            % Set bootstrap parameters
            blockSizes = [10, 20];  % Block sizes for block bootstrap
            
            % Create data generator function for bootstrap-compatible data
            % Generate AR(1) process with moderate autocorrelation
            dataGenerator = @(size) obj.dataGenerator.generateTimeSeriesData(...
                struct('numObs', size, 'ar', 0.5, 'distribution', 'normal')).y;
            
            % Use benchmarker to measure bootstrap performance
            blockResults = obj.benchmarker.scalabilityTest(...
                @(data) blockBootstrap(data, blockSizes(1), 100), ...  % 100 bootstrap replications
                timeSeries, dataGenerator);
            
            % Analyze scaling behavior with respect to data size
            result = struct();
            result.component = 'Bootstrap Methods';
            result.dataSizes = timeSeries;
            result.executionTimes = blockResults.timings;
            result.scalingBehavior = blockResults.scalingBehavior;
            result.scalingExponent = blockResults.scalingExponent;
            
            % Generate visualization if enabled
            if obj.generateVisualizations
                figure('Name', 'Bootstrap Scalability');
                loglog(cell2mat(timeSeries), result.executionTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Sample Size');
                ylabel('Execution Time (s)');
                title(sprintf('Block Bootstrap Scalability (%s: exponent = %.2f)', ...
                    result.scalingBehavior, result.scalingExponent));
            end
            
            % Assert performance scales efficiently with bootstrap parameters
            % Bootstrap methods should scale linearly or near-linearly with data size
            obj.assertTrue(result.scalingExponent < 1.5, ...
                'Bootstrap methods should scale close to linearly with data size');
            
            % Store result in test data
            obj.testData.bootstrapScalability = result;
            
            % Simple helper function for block bootstrap
            function bootstrapData = blockBootstrap(data, blockSize, numBootstraps)
                T = length(data);
                bootstrapData = zeros(T, numBootstraps);
                
                % Generate block bootstrap samples
                for b = 1:numBootstraps
                    % Initialize bootstrap sample
                    sample = zeros(T, 1);
                    
                    % Fill sample with random blocks
                    t = 1;
                    while t <= T
                        % Select a random starting point
                        startIdx = randi(T - min(blockSize, T) + 1);
                        
                        % Get block length (may be truncated at the end)
                        blockEnd = min(startIdx + blockSize - 1, T);
                        blockLength = blockEnd - startIdx + 1;
                        
                        % Copy block to sample
                        endIdx = min(t + blockLength - 1, T);
                        copyLength = endIdx - t + 1;
                        sample(t:endIdx) = data(startIdx:startIdx + copyLength - 1);
                        
                        % Move to next position
                        t = t + copyLength;
                    end
                    
                    bootstrapData(:, b) = sample;
                end
            end
        end
        
        function result = testMultivariateScalability(obj)
            % Tests how multivariate models scale with increasing dimension
            %
            % OUTPUTS:
            %   result - Test result containing multivariate scalability metrics
            
            % Define array of increasing dimensions for multivariate models
            dimensions = {2, 5, 10, 20, 50};
            timeLength = 1000;  % Fixed time series length
            
            % Create data generator function for multivariate data
            dataGenerator = @(dim) obj.dataGenerator.generateCrossSectionalData(...
                dim, timeLength, dim, struct(...
                'factorMean', zeros(dim, 1), ...
                'factorCov', eye(dim), ...
                'loadingDist', 'normal', ...
                'loadingParams', struct('mean', 1.0, 'std', 0.1), ...
                'idioVol', 0.01)).returns;
            
            % Use benchmarker to measure multivariate model performance
            % Here we're using covariance estimation as a representative multivariate operation
            mvResults = obj.benchmarker.scalabilityTest(...
                @(data) cov(data), ...
                dimensions, dataGenerator);
            
            % Analyze scaling behavior with respect to dimension
            result = struct();
            result.component = 'Multivariate Models';
            result.dimensions = dimensions;
            result.executionTimes = mvResults.timings;
            result.scalingBehavior = mvResults.scalingBehavior;
            result.scalingExponent = mvResults.scalingExponent;
            
            % Generate visualization if enabled
            if obj.generateVisualizations
                figure('Name', 'Multivariate Scalability');
                loglog(cell2mat(dimensions), result.executionTimes, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Number of Variables');
                ylabel('Execution Time (s)');
                title(sprintf('Multivariate Analysis Scalability (%s: exponent = %.2f)', ...
                    result.scalingBehavior, result.scalingExponent));
            end
            
            % Assert performance scales efficiently with model dimension
            % For multivariate models, we expect O(n²) to O(n³) scaling depending on the operation
            obj.assertTrue(result.scalingExponent < 3.2, ...
                'Multivariate models should scale better than O(n^3.2)');
            
            % Store result in test data
            obj.testData.multivariateScalability = result;
        end
        
        function result = testNumericalStability(obj)
            % Tests numerical stability of algorithms with large-scale data
            %
            % OUTPUTS:
            %   result - Test result containing numerical stability metrics
            
            % Generate large-scale datasets with known numerical properties
            % We'll test with a condition number that increases with size
            matrixSizes = {10, 50, 100, 500, 1000};
            
            % Initialize results
            result = struct();
            result.component = 'Numerical Stability';
            result.matrixSizes = matrixSizes;
            result.conditionNumbers = zeros(length(matrixSizes), 1);
            result.relativeErrors = zeros(length(matrixSizes), 1);
            
            % For each size, generate test data and analyze stability
            for i = 1:length(matrixSizes)
                size = matrixSizes{i};
                
                % Generate a matrix with controlled condition number
                % We'll use a diagonal matrix with increasing condition number
                condNumber = 10^(ceil(size/200) + 1);  % Condition number increases with size
                
                % Create diagonal matrix with specific condition number
                s = logspace(-ceil(size/200) - 1, 0, size);  % Logarithmically spaced singular values
                D = diag(s);
                
                % Create random orthogonal matrices
                [U, ~] = qr(randn(size));
                [V, ~] = qr(randn(size));
                
                % Create test matrix with specified condition number
                A = U * D * V';
                
                % Get matrix diagnostics
                diagInfo = matrixdiagnostics(A);
                result.conditionNumbers(i) = diagInfo.ConditionNumber;
                
                % Test matrix inversion and measure error
                Ainv = inv(A);
                I = eye(size);
                relError = norm(I - A*Ainv, 'fro') / norm(I, 'fro');
                result.relativeErrors(i) = relError;
            end
            
            % Generate visualization of numerical stability
            if obj.generateVisualizations
                figure('Name', 'Numerical Stability Analysis');
                
                % Plot condition numbers
                subplot(2, 1, 1);
                loglog(cell2mat(matrixSizes), result.conditionNumbers, 'b-o', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size');
                ylabel('Condition Number');
                title('Matrix Condition Number vs. Size');
                
                % Plot relative errors
                subplot(2, 1, 2);
                loglog(cell2mat(matrixSizes), result.relativeErrors, 'r-s', 'LineWidth', 2);
                grid on;
                xlabel('Matrix Size');
                ylabel('Relative Error: ||I - A*A⁻¹||_F / ||I||_F');
                title('Numerical Stability with Increasing Size');
            end
            
            % Assert numerical stability is maintained with large datasets
            % Relative error should remain small even with large, ill-conditioned matrices
            maxAllowedError = 1e-8;
            obj.assertTrue(max(result.relativeErrors) < maxAllowedError, ...
                sprintf('Numerical stability should be maintained with large matrices. Max error: %e', ...
                max(result.relativeErrors)));
            
            % Store result in test data
            obj.testData.numericalStability = result;
        end
        
        function result = runAllTests(obj)
            % Runs all large-scale data tests and generates comprehensive report
            %
            % OUTPUTS:
            %   result - Aggregated test results with overall performance metrics
            
            % Initialize results structure
            result = struct();
            result.component = 'Complete Large-Scale Test Suite';
            result.startTime = now;
            result.testResults = struct();
            
            % Run all individual test methods
            disp('Running Time Series Scalability Test...');
            result.testResults.timeSeriesScalability = obj.testTimeSeriesScalability();
            
            disp('Running Volatility Model Scalability Test...');
            result.testResults.volatilityScalability = obj.testVolatilityModelScalability();
            
            disp('Running Cross-Sectional Scalability Test...');
            result.testResults.crossSectionalScalability = obj.testCrossSectionalScalability();
            
            disp('Running High-Frequency Scalability Test...');
            result.testResults.highFrequencyScalability = obj.testHighFrequencyScalability();
            
            disp('Running Matrix Operations Test...');
            result.testResults.matrixOperations = obj.testLargeMatrixOperations();
            
            disp('Running Memory Usage Test...');
            result.testResults.memoryUsage = obj.testLargeScaleMemoryUsage();
            
            disp('Running Bootstrap Scalability Test...');
            result.testResults.bootstrapScalability = obj.testBootstrapScalability();
            
            disp('Running Multivariate Scalability Test...');
            result.testResults.multivariateScalability = obj.testMultivariateScalability();
            
            disp('Running Numerical Stability Test...');
            result.testResults.numericalStability = obj.testNumericalStability();
            
            % Calculate runtime
            result.endTime = now;
            result.executionTime = (result.endTime - result.startTime) * 86400;  % Convert days to seconds
            
            % Analyze overall scaling behavior
            result.overallAssessment = obj.analyzeOverallPerformance(result.testResults);
            
            % Generate comprehensive report
            reportPath = obj.generateScalabilityReport(result);
            result.reportPath = reportPath;
            
            % Return aggregated test results
            return;
        end
        
        function assessment = analyzeOverallPerformance(obj, testResults)
            % Analyzes overall scaling behavior and performance bottlenecks
            %
            % INPUTS:
            %   testResults - Structure containing results from all tests
            %
            % OUTPUTS:
            %   assessment - Structure with overall performance assessment
            
            % Initialize assessment structure
            assessment = struct();
            
            % Extract scaling exponents from each test
            fields = fieldnames(testResults);
            exponents = [];
            components = {};
            
            for i = 1:length(fields)
                result = testResults.(fields{i});
                if isfield(result, 'scalingExponent')
                    exponents(end+1) = result.scalingExponent; %#ok<AGROW>
                    components{end+1} = result.component; %#ok<AGROW>
                elseif isfield(result, 'matlabScalingExponent')
                    exponents(end+1) = result.matlabScalingExponent; %#ok<AGROW>
                    components{end+1} = [result.component ' (MATLAB)']; %#ok<AGROW>
                end
            end
            
            % Calculate overall metrics
            assessment.meanScalingExponent = mean(exponents);
            assessment.minScalingExponent = min(exponents);
            assessment.maxScalingExponent = max(exponents);
            
            % Identify potential bottlenecks (components with highest scaling exponents)
            [sortedExponents, sortIdx] = sort(exponents, 'descend');
            sortedComponents = components(sortIdx);
            
            assessment.bottlenecks = struct();
            for i = 1:min(3, length(sortedComponents))
                fieldName = sprintf('bottleneck%d', i);
                assessment.bottlenecks.(fieldName) = struct(...
                    'component', sortedComponents{i}, ...
                    'scalingExponent', sortedExponents(i));
            end
            
            % Determine overall scaling classification
            if assessment.maxScalingExponent <= 1.2
                assessment.overallScaling = 'Linear';
            elseif assessment.maxScalingExponent <= 2.2
                assessment.overallScaling = 'Quadratic';
            elseif assessment.maxScalingExponent <= 3.2
                assessment.overallScaling = 'Cubic';
            else
                assessment.overallScaling = 'Polynomial';
            end
            
            % Overall conclusion
            if assessment.maxScalingExponent <= 3.0
                assessment.conclusion = 'The MFE Toolbox demonstrates good scalability for large-scale data processing.';
            else
                assessment.conclusion = 'The MFE Toolbox shows acceptable scalability, but some components may become performance bottlenecks with very large datasets.';
            end
        end
        
        function reportPath = generateScalabilityReport(obj, results)
            % Generates a detailed report showing performance scaling behavior
            %
            % INPUTS:
            %   results - Test results structure
            %
            % OUTPUTS:
            %   reportPath - Path to the generated report file
            
            % Configure reporter with scalability-focused settings
            obj.reporter.setReportTitle('MFE Toolbox Large-Scale Performance Analysis');
            obj.reporter.setReportOutputPath(obj.reportOutputPath);
            obj.reporter.setReportFormats({'html', 'text'});
            obj.reporter.setVerboseOutput(false);
            obj.reporter.setIncludePerformanceData(true);
            
            % Add all test results to reporter
            fields = fieldnames(results);
            
            % If a single result is passed, handle differently than complete test suite
            if isfield(results, 'component') && ~isfield(results, 'testResults')
                % Add a single test result
                testName = results.component;
                testCategory = 'Performance Scalability';
                isPassed = true;  % Performance tests don't have pass/fail criteria in the report
                
                details = struct();
                details.scalingBehavior = results.scalingBehavior;
                details.scalingExponent = results.scalingExponent;
                if isfield(results, 'executionTimes')
                    details.executionTimes = results.executionTimes;
                end
                details.message = sprintf('Component scales as O(n^%.2f) (%s)', ...
                    results.scalingExponent, results.scalingBehavior);
                
                obj.reporter.addTestResult(testName, testCategory, isPassed, details);
            elseif isfield(results, 'testResults')
                % Full test suite results
                resultFields = fieldnames(results.testResults);
                
                for i = 1:length(resultFields)
                    testResult = results.testResults.(resultFields{i});
                    testName = testResult.component;
                    testCategory = 'Large-Scale Performance';
                    isPassed = true;  % All tests technically "pass" as they're measurements
                    
                    % Create detailed result information
                    details = struct();
                    details.testType = resultFields{i};
                    
                    % Extract scalability metrics if available
                    if isfield(testResult, 'scalingExponent')
                        details.scalingExponent = testResult.scalingExponent;
                        details.scalingBehavior = testResult.scalingBehavior;
                        details.message = sprintf('Component scales as O(n^%.2f) (%s)', ...
                            testResult.scalingExponent, testResult.scalingBehavior);
                    elseif isfield(testResult, 'matlabScalingExponent')
                        details.matlabScalingExponent = testResult.matlabScalingExponent;
                        details.matlabScalingBehavior = testResult.matlabScalingBehavior;
                        details.mexScalingExponent = testResult.mexScalingExponent;
                        details.mexScalingBehavior = testResult.mexScalingBehavior;
                        details.speedupRatio = mean(testResult.speedupRatio);
                        details.message = sprintf('MATLAB: O(n^%.2f), MEX: O(n^%.2f), Speedup: %.2fx', ...
                            testResult.matlabScalingExponent, testResult.mexScalingExponent, details.speedupRatio);
                    end
                    
                    % Add relevant metrics to details
                    if isfield(testResult, 'executionTimes')
                        details.executionTimes = testResult.executionTimes;
                    end
                    if isfield(testResult, 'dataSizes')
                        details.dataSizes = testResult.dataSizes;
                    end
                    
                    % Add this test result to the reporter
                    obj.reporter.addTestResult(testName, testCategory, isPassed, details);
                end
                
                % Add overall assessment if available
                if isfield(results, 'overallAssessment')
                    assessment = results.overallAssessment;
                    obj.reporter.addTestResult('Overall Scalability', 'Summary', true, struct(...
                        'message', assessment.conclusion, ...
                        'overallScaling', assessment.overallScaling, ...
                        'meanScalingExponent', assessment.meanScalingExponent, ...
                        'maxScalingExponent', assessment.maxScalingExponent, ...
                        'bottlenecks', {fieldnames(assessment.bottlenecks)} ...
                    ));
                end
            end
            
            % Generate the report
            reportFiles = obj.reporter.generateReport();
            
            % Display summary to console
            obj.reporter.displaySummary();
            
            % Return the report path
            if isfield(reportFiles, 'html')
                reportPath = reportFiles.html;
            elseif isfield(reportFiles, 'text')
                reportPath = reportFiles.text;
            else
                reportPath = '';
            end
        end
        
        function setReportOutputPath(obj, path)
            % Sets the output directory for scalability reports
            %
            % INPUTS:
            %   path - Path to directory for report output
            
            % Validate the path
            if ~ischar(path) && ~isstring(path)
                error('Path must be a string or character array');
            end
            
            % Convert to character array if string
            if isstring(path)
                path = char(path);
            end
            
            % Check if directory exists, create if not
            if ~exist(path, 'dir')
                [success, message] = mkdir(path);
                if ~success
                    error('Failed to create directory: %s', message);
                end
            end
            
            % Update output path
            obj.reportOutputPath = path;
            
            % Configure reporter to use the specified output path
            obj.reporter.setReportOutputPath(path);
        end
    end
end