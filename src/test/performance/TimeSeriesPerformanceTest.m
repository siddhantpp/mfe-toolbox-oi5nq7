classdef TimeSeriesPerformanceTest < BaseTest
    % TimeSeriesPerformanceTest Performance testing class for time series analysis components of the MFE Toolbox.
    % Measures execution time, memory usage, and scalability across different sample sizes and model specifications.
    
    properties
        % Benchmark utility instance for performance testing
        benchmark PerformanceBenchmark
        
        % Data generator instance
        dataGenerator TestDataGenerator
        
        % Test data storage
        testData struct
        
        % Configuration for tests
        testConfig struct
        
        % Results storage
        testResults struct
        
        % Sample sizes for scalability testing
        sampleSizes cell
        
        % ARMA model orders for testing
        armaOrders cell
    end
    
    methods
        function obj = TimeSeriesPerformanceTest()
            % Initializes a new TimeSeriesPerformanceTest instance with performance testing configuration
            
            % Call superclass constructor with test name
            obj = obj@BaseTest('TimeSeriesPerformanceTest');
            
            % Initialize PerformanceBenchmark instance
            benchmarkOptions = struct();
            benchmarkOptions.iterations = 20;
            benchmarkOptions.warmupIterations = 3;
            benchmarkOptions.speedupThreshold = 1.5; % 50% improvement threshold
            benchmarkOptions.saveVisualizations = true;
            benchmarkOptions.visualizationPath = fullfile(pwd, 'results', 'timeseries');
            benchmarkOptions.verbose = false;
            obj.benchmark = PerformanceBenchmark(benchmarkOptions);
            
            % Initialize TestDataGenerator for generating time series data
            obj.dataGenerator = TestDataGenerator;
            
            % Set default sample sizes for scalability testing
            obj.sampleSizes = {100, 500, 1000, 5000, 10000};
            
            % Set default ARMA orders for testing
            obj.armaOrders = {
                [1, 0],  % AR(1)
                [0, 1],  % MA(1)
                [1, 1],  % ARMA(1,1)
                [2, 1],  % ARMA(2,1)
                [1, 2],  % ARMA(1,2)
                [2, 2]   % ARMA(2,2)
            };
            
            % Initialize test configuration
            obj.testConfig = struct();
            obj.testConfig.iterations = 10;
            obj.testConfig.warmup = 2;
            obj.testConfig.distributions = {'normal', 't', 'ged'};
            obj.testConfig.exogenousVariables = {0, 1, 2, 3};
            obj.testConfig.constants = [true, false];
            
            % Initialize results storage structure
            obj.testResults = struct();
        end
        
        function setUp(obj)
            % Prepares the test environment before each performance test
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load or generate test datasets with known time series properties
            obj.loadTestData();
            
            % Load financial_returns.mat and macroeconomic_data.mat for realistic tests
            try
                financial_data = load('financial_returns.mat');
                obj.testData.financial = financial_data;
            catch
                % Generate synthetic financial data if file doesn't exist
                params = struct();
                params.mean = 0;
                params.variance = 1;
                params.distribution = 't';
                params.distParams = 5;
                params.garch = struct('modelType', 'GARCH', 'p', 1, 'q', 1, ...
                                      'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                obj.testData.financial.returns = obj.dataGenerator('generateFinancialReturns', 5000, 1, params);
            end
            
            % Initialize results storage for current test
            obj.testResults.currentTest = struct();
            
            % Configure memory monitoring for time series functions
            obj.benchmark.setVerbose(false);
            
            % Set random seed for reproducible benchmarks
            rng(123456);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each performance test
            
            % Compile performance metrics from completed test
            testName = regexp(getStackTraceFunction(), '\.([^\.]+)$', 'tokens');
            if ~isempty(testName) && isfield(obj.testResults, 'currentTest')
                methodName = testName{1}{1};
                obj.testResults.(methodName) = obj.testResults.currentTest;
                obj.testResults = rmfield(obj.testResults, 'currentTest');
            end
            
            % Generate visualizations if enabled
            if isfield(obj.benchmark, 'saveVisualizations') && obj.benchmark.saveVisualizations
                obj.generatePerformanceVisualizations(obj.testResults, 'test_results');
            end
            
            % Free large data structures
            clear testData;
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function results = testARMAXFilterPerformance(obj)
            % Benchmarks the performance of ARMAX model estimation (armaxfilter function)
            
            % Create test configurations with different AR and MA orders
            testCases = struct();
            
            % Basic AR(1) model
            testCases(1).name = 'AR(1)';
            testCases(1).data = obj.testData.financial.returns;
            testCases(1).options = struct('p', 1, 'q', 0, 'constant', true);
            
            % ARMA(1,1) model
            testCases(2).name = 'ARMA(1,1)';
            testCases(2).data = obj.testData.financial.returns;
            testCases(2).options = struct('p', 1, 'q', 1, 'constant', true);
            
            % ARMA(2,2) model
            testCases(3).name = 'ARMA(2,2)';
            testCases(3).data = obj.testData.financial.returns;
            testCases(3).options = struct('p', 2, 'q', 2, 'constant', true);
            
            % Initialize results storage
            results = struct();
            results.testCases = testCases;
            results.timings = zeros(length(testCases), 1);
            results.memory = zeros(length(testCases), 1);
            
            % Generate test data samples of various sizes
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                % Benchmark armaxfilter execution time and memory usage
                metrics = obj.benchmark.benchmarkFunction(@armaxfilter, ...
                    obj.testConfig.iterations, testCase.data, [], testCase.options);
                results.timings(i) = metrics.mean;
                
                % Measure memory usage
                memInfo = obj.benchmark.measureMemoryUsage(@armaxfilter, ...
                    testCase.data, [], testCase.options);
                results.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMAX filter test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(testCases), testCase.name, results.timings(i), results.memory(i));
            end
            
            % Test with exogenous variables of different dimensions
            results.exogenous = struct();
            exogSizes = [1, 2, 3];
            results.exogenous.sizes = exogSizes;
            results.exogenous.timings = zeros(length(exogSizes), 1);
            results.exogenous.memory = zeros(length(exogSizes), 1);
            
            for i = 1:length(exogSizes)
                n = length(obj.testData.financial.returns);
                x = randn(n, exogSizes(i)); % Generate random exogenous variables
                
                metrics = obj.benchmark.benchmarkFunction(@armaxfilter, ...
                    obj.testConfig.iterations, obj.testData.financial.returns, x, ...
                    struct('p', 1, 'q', 1, 'constant', true));
                results.exogenous.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(@armaxfilter, ...
                    obj.testData.financial.returns, x, struct('p', 1, 'q', 1, 'constant', true));
                results.exogenous.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMAX filter exogenous test %d/%d: exog=%d - %.4f seconds, %.2f MB\n', ...
                    i, length(exogSizes), exogSizes(i), results.exogenous.timings(i), ...
                    results.exogenous.memory(i));
            end
            
            % Compare performance across different error distributions
            results.distributions = struct();
            distributions = {'normal', 't', 'ged'};
            results.distributions.types = distributions;
            results.distributions.timings = zeros(length(distributions), 1);
            results.distributions.memory = zeros(length(distributions), 1);
            
            for i = 1:length(distributions)
                options = struct('p', 1, 'q', 1, 'constant', true, ...
                                'distribution', distributions{i});
                
                metrics = obj.benchmark.benchmarkFunction(@armaxfilter, ...
                    obj.testConfig.iterations, obj.testData.financial.returns, [], options);
                results.distributions.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(@armaxfilter, ...
                    obj.testData.financial.returns, [], options);
                results.distributions.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMAX filter distribution test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(distributions), distributions{i}, ...
                    results.distributions.timings(i), results.distributions.memory(i));
            end
            
            % Analyze scalability with increasing sample size
            results.scalability = struct();
            results.scalability.sampleSizes = obj.sampleSizes;
            results.scalability.timings = zeros(length(obj.sampleSizes), 1);
            results.scalability.memory = zeros(length(obj.sampleSizes), 1);
            
            for i = 1:length(obj.sampleSizes)
                size = obj.sampleSizes{i};
                
                % Generate or retrieve test data of specified size
                if size <= 10000 % Skip very large sizes to avoid excessive test time
                    tsData = obj.dataGenerator('generateTimeSeriesData', struct('numObs', size));
                    
                    metrics = obj.benchmark.benchmarkFunction(@armaxfilter, 5, ... % Fewer iterations for large data
                        tsData.y, [], struct('p', 1, 'q', 1, 'constant', true));
                    results.scalability.timings(i) = metrics.mean;
                    
                    memInfo = obj.benchmark.measureMemoryUsage(@armaxfilter, ...
                        tsData.y, [], struct('p', 1, 'q', 1, 'constant', true));
                    results.scalability.memory(i) = memInfo.memoryDifferenceMB;
                    
                    fprintf('ARMAX filter scalability test %d/%d: n=%d - %.4f seconds, %.2f MB\n', ...
                        i, length(obj.sampleSizes), size, results.scalability.timings(i), ...
                        results.scalability.memory(i));
                else
                    results.scalability.timings(i) = NaN;
                    results.scalability.memory(i) = NaN;
                end
            end
            
            % Compile performance metrics and return results
            obj.testResults.currentTest = results;
            
            % Return results for direct inspection
            results = obj.testResults.currentTest;
        end
        
        function results = testARMAForPerformance(obj)
            % Benchmarks the performance of ARMA forecasting (armafor function)
            
            % Create test configurations with different forecast horizons
            testCases = struct();
            
            % First generate model parameters
            data = obj.testData.financial.returns;
            arModel = armaxfilter(data, [], struct('p', 1, 'q', 0, 'constant', true));
            armaModel = armaxfilter(data, [], struct('p', 1, 'q', 1, 'constant', true));
            
            % Use previously estimated models for forecasting
            testCases(1).name = 'AR(1) 10-step';
            testCases(1).params = arModel.parameters;
            testCases(1).data = data;
            testCases(1).p = 1;
            testCases(1).q = 0;
            testCases(1).constant = true;
            testCases(1).horizon = 10;
            
            testCases(2).name = 'ARMA(1,1) 20-step';
            testCases(2).params = armaModel.parameters;
            testCases(2).data = data;
            testCases(2).p = 1;
            testCases(2).q = 1;
            testCases(2).constant = true;
            testCases(2).horizon = 20;
            
            testCases(3).name = 'ARMA(1,1) 50-step';
            testCases(3).params = armaModel.parameters;
            testCases(3).data = data;
            testCases(3).p = 1;
            testCases(3).q = 1;
            testCases(3).constant = true;
            testCases(3).horizon = 50;
            
            % Initialize results storage
            results = struct();
            results.testCases = testCases;
            results.timings = zeros(length(testCases), 1);
            results.memory = zeros(length(testCases), 1);
            
            % Test both exact and simulation-based forecasting methods
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                forecastFn = @() armafor(testCase.params, testCase.data, ...
                    testCase.p, testCase.q, testCase.constant, [], testCase.horizon);
                
                metrics = obj.benchmark.benchmarkFunction(forecastFn, obj.testConfig.iterations);
                results.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(forecastFn);
                results.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMA forecast test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(testCases), testCase.name, results.timings(i), results.memory(i));
            end
            
            % Test different forecast methods
            results.methods = struct();
            methods = {'exact', 'simulation'};
            results.methods.methodNames = methods;
            results.methods.timings = zeros(length(methods), 1);
            results.methods.memory = zeros(length(methods), 1);
            
            for i = 1:length(methods)
                method = methods{i};
                
                forecastFn = @() armafor(armaModel.parameters, data, 1, 1, true, [], 20, [], method);
                
                metrics = obj.benchmark.benchmarkFunction(forecastFn, obj.testConfig.iterations);
                results.methods.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(forecastFn);
                results.methods.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMA forecast method test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(methods), method, results.methods.timings(i), results.methods.memory(i));
            end
            
            % Compare performance with different error distributions
            results.distributions = struct();
            distributions = {'normal', 't', 'ged'};
            results.distributions.types = distributions;
            results.distributions.timings = zeros(length(distributions), 1);
            
            for i = 1:length(distributions)
                dist = distributions{i};
                distParams = [];
                if strcmp(dist, 't')
                    distParams = 5;
                elseif strcmp(dist, 'ged')
                    distParams = 1.5;
                end
                
                forecastFn = @() armafor(armaModel.parameters, data, 1, 1, true, [], 20, ...
                    [], 'simulation', 1000, dist, struct('nu', distParams));
                
                metrics = obj.benchmark.benchmarkFunction(forecastFn, 5); % Fewer iterations
                results.distributions.timings(i) = metrics.mean;
                
                fprintf('ARMA forecast distribution test %d/%d: %s - %.4f seconds\n', ...
                    i, length(distributions), dist, results.distributions.timings(i));
            end
            
            % Analyze scalability with increasing forecast horizon
            horizons = [1, 5, 10, 20, 50, 100];
            results.horizons = struct();
            results.horizons.horizonValues = horizons;
            results.horizons.timings = zeros(length(horizons), 1);
            results.horizons.memory = zeros(length(horizons), 1);
            
            for i = 1:length(horizons)
                horizon = horizons(i);
                
                forecastFn = @() armafor(armaModel.parameters, data, 1, 1, true, [], horizon);
                
                metrics = obj.benchmark.benchmarkFunction(forecastFn, obj.testConfig.iterations);
                results.horizons.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(forecastFn);
                results.horizons.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('ARMA forecast horizon test %d/%d: h=%d - %.4f seconds, %.2f MB\n', ...
                    i, length(horizons), horizon, results.horizons.timings(i), ...
                    results.horizons.memory(i));
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testSACFPerformance(obj)
            % Benchmarks the performance of sample autocorrelation function (sacf)
            
            % Create test configurations with different lag specifications
            testCases = struct();
            
            % Test with default lags
            testCases(1).name = 'Default lags';
            testCases(1).data = obj.testData.financial.returns;
            testCases(1).lags = [];
            testCases(1).options = [];
            
            % Test with specific lags
            testCases(2).name = 'Custom lags';
            testCases(2).data = obj.testData.financial.returns;
            testCases(2).lags = [1, 2, 3, 5, 10, 20];
            testCases(2).options = [];
            
            % Test with standard errors and confidence intervals
            testCases(3).name = 'With SE & CI';
            testCases(3).data = obj.testData.financial.returns;
            testCases(3).lags = [];
            testCases(3).options = struct('alpha', 0.05);
            testCases(3).nargout = 3;
            
            % Initialize results storage
            results = struct();
            results.testCases = testCases;
            results.timings = zeros(length(testCases), 1);
            results.memory = zeros(length(testCases), 1);
            
            % Use both generated and real financial data
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                if isfield(testCase, 'nargout') && testCase.nargout > 1
                    sacfFn = @() nargout(testCase.nargout, @sacf, testCase.data, testCase.lags, testCase.options);
                else
                    sacfFn = @() sacf(testCase.data, testCase.lags, testCase.options);
                end
                
                metrics = obj.benchmark.benchmarkFunction(sacfFn, obj.testConfig.iterations);
                results.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(sacfFn);
                results.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('SACF test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(testCases), testCase.name, results.timings(i), results.memory(i));
            end
            
            % Test with various series lengths
            results.scalability = struct();
            results.scalability.sampleSizes = obj.sampleSizes;
            results.scalability.timings = zeros(length(obj.sampleSizes), 1);
            results.scalability.memory = zeros(length(obj.sampleSizes), 1);
            
            for i = 1:length(obj.sampleSizes)
                size = obj.sampleSizes{i};
                
                % Generate or retrieve data of specified size
                tsData = obj.dataGenerator('generateTimeSeriesData', struct('numObs', size));
                
                metrics = obj.benchmark.benchmarkFunction(@sacf, obj.testConfig.iterations, tsData.y);
                results.scalability.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(@sacf, tsData.y);
                results.scalability.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('SACF scalability test %d/%d: n=%d - %.4f seconds, %.2f MB\n', ...
                    i, length(obj.sampleSizes), size, results.scalability.timings(i), ...
                    results.scalability.memory(i));
            end
            
            % Compare performance with multivariate inputs
            results.multivariate = struct();
            dimensions = [1, 2, 5, 10];
            results.multivariate.dimensions = dimensions;
            results.multivariate.timings = zeros(length(dimensions), 1);
            
            for i = 1:length(dimensions)
                dim = dimensions(i);
                data = randn(1000, dim);
                
                metrics = obj.benchmark.benchmarkFunction(@sacf, obj.testConfig.iterations, data);
                results.multivariate.timings(i) = metrics.mean;
                
                fprintf('SACF multivariate test %d/%d: dim=%d - %.4f seconds\n', ...
                    i, length(dimensions), dim, results.multivariate.timings(i));
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testSPACFPerformance(obj)
            % Benchmarks the performance of sample partial autocorrelation function (spacf)
            
            % Create test configurations with different lag specifications
            testCases = struct();
            
            % Test with default lags
            testCases(1).name = 'Default lags';
            testCases(1).data = obj.testData.financial.returns;
            testCases(1).lags = [];
            testCases(1).options = [];
            
            % Test with specific lags
            testCases(2).name = 'Custom lags';
            testCases(2).data = obj.testData.financial.returns;
            testCases(2).lags = [1, 2, 3, 5, 10, 20];
            testCases(2).options = [];
            
            % Test with standard errors and confidence intervals
            testCases(3).name = 'With SE & CI';
            testCases(3).data = obj.testData.financial.returns;
            testCases(3).lags = [];
            testCases(3).options = struct('alpha', 0.05);
            testCases(3).nargout = 3;
            
            % Initialize results storage
            results = struct();
            results.testCases = testCases;
            results.timings = zeros(length(testCases), 1);
            results.memory = zeros(length(testCases), 1);
            
            % Use both generated and real financial data
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                if isfield(testCase, 'nargout') && testCase.nargout > 1
                    spacfFn = @() nargout(testCase.nargout, @spacf, testCase.data, testCase.lags, testCase.options);
                else
                    spacfFn = @() spacf(testCase.data, testCase.lags, testCase.options);
                end
                
                metrics = obj.benchmark.benchmarkFunction(spacfFn, obj.testConfig.iterations);
                results.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(spacfFn);
                results.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('SPACF test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(testCases), testCase.name, results.timings(i), results.memory(i));
            end
            
            % Test with various series lengths
            results.scalability = struct();
            results.scalability.sampleSizes = obj.sampleSizes;
            results.scalability.timings = zeros(length(obj.sampleSizes), 1);
            results.scalability.memory = zeros(length(obj.sampleSizes), 1);
            
            for i = 1:length(obj.sampleSizes)
                size = obj.sampleSizes{i};
                
                % Generate or retrieve data of specified size
                tsData = obj.dataGenerator('generateTimeSeriesData', struct('numObs', size));
                
                metrics = obj.benchmark.benchmarkFunction(@spacf, obj.testConfig.iterations, tsData.y);
                results.scalability.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(@spacf, tsData.y);
                results.scalability.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('SPACF scalability test %d/%d: n=%d - %.4f seconds, %.2f MB\n', ...
                    i, length(obj.sampleSizes), size, results.scalability.timings(i), ...
                    results.scalability.memory(i));
            end
            
            % Compare performance with multivariate inputs
            results.multivariate = struct();
            dimensions = [1, 2, 5, 10];
            results.multivariate.dimensions = dimensions;
            results.multivariate.timings = zeros(length(dimensions), 1);
            
            for i = 1:length(dimensions)
                dim = dimensions(i);
                data = randn(1000, dim);
                
                metrics = obj.benchmark.benchmarkFunction(@spacf, obj.testConfig.iterations, data);
                results.multivariate.timings(i) = metrics.mean;
                
                fprintf('SPACF multivariate test %d/%d: dim=%d - %.4f seconds\n', ...
                    i, length(dimensions), dim, results.multivariate.timings(i));
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testAICSBICPerformance(obj)
            % Benchmarks the performance of information criteria calculation (aicsbic)
            
            % Create test configurations with different model specifications
            testCases = struct();
            
            % Basic calculation
            testCases(1).name = 'Basic calculation';
            testCases(1).logL = -1000;
            testCases(1).k = 2;
            testCases(1).T = 1000;
            
            % Vector of log-likelihoods
            testCases(2).name = 'Vector logL';
            testCases(2).logL = -1000:-10:-1100;
            testCases(2).k = 2;
            testCases(2).T = 1000;
            
            % All vector inputs
            testCases(3).name = 'Vector inputs';
            testCases(3).logL = -1000:-10:-1100;
            testCases(3).k = 1:11;
            testCases(3).T = 1000:10:1100;
            
            % Initialize results storage
            results = struct();
            results.testCases = testCases;
            results.timings = zeros(length(testCases), 1);
            results.memory = zeros(length(testCases), 1);
            
            % Generate residuals from various models
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                aicsbicFn = @() aicsbic(testCase.logL, testCase.k, testCase.T);
                
                metrics = obj.benchmark.benchmarkFunction(aicsbicFn, obj.testConfig.iterations);
                results.timings(i) = metrics.mean;
                
                memInfo = obj.benchmark.measureMemoryUsage(aicsbicFn);
                results.memory(i) = memInfo.memoryDifferenceMB;
                
                fprintf('AICSBIC test %d/%d: %s - %.4f seconds, %.2f MB\n', ...
                    i, length(testCases), testCase.name, results.timings(i), results.memory(i));
            end
            
            % Test with varying degrees of freedom
            results.dofsTest = struct();
            dofs = [1, 5, 10, 20, 50];
            results.dofsTest.dofs = dofs;
            results.dofsTest.timings = zeros(length(dofs), 1);
            
            for i = 1:length(dofs)
                k = dofs(i);
                logL = -1000;
                T = 1000;
                
                aicsbicFn = @() aicsbic(logL, k, T);
                
                metrics = obj.benchmark.benchmarkFunction(aicsbicFn, obj.testConfig.iterations);
                results.dofsTest.timings(i) = metrics.mean;
                
                fprintf('AICSBIC dofs test %d/%d: k=%d - %.4f seconds\n', ...
                    i, length(dofs), k, results.dofsTest.timings(i));
            end
            
            % Analyze performance with different error distributions
            results.distributions = struct();
            distributions = {'normal', 't', 'ged'};
            results.distributions.types = distributions;
            results.distributions.timings = zeros(length(distributions), 1);
            
            % Create synthetic models with different distributions
            for i = 1:length(distributions)
                dist = distributions{i};
                
                % Create a small model to get log-likelihood
                options = struct('p', 1, 'q', 1, 'constant', true, 'distribution', dist);
                model = armaxfilter(obj.testData.financial.returns(1:200), [], options);
                logL = model.logL;
                k = 3; % p+q+const
                T = 200;
                
                aicsbicFn = @() aicsbic(logL, k, T);
                
                metrics = obj.benchmark.benchmarkFunction(aicsbicFn, obj.testConfig.iterations);
                results.distributions.timings(i) = metrics.mean;
                
                fprintf('AICSBIC distribution test %d/%d: %s - %.4f seconds\n', ...
                    i, length(distributions), dist, results.distributions.timings(i));
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testTimeSeriesScalability(obj)
            % Tests the scalability of time series functions with increasing sample size
            
            % Define scale factors for testing (powers of 10)
            functions = {
                @(data) armaxfilter(data, [], struct('p', 1, 'q', 1, 'constant', true)),
                @(data) armafor([0.5, 0.2], data, 1, 1, true, [], 10),
                @sacf,
                @spacf,
                @(data) aicsbic(-1000, 2, length(data))
            };
            
            functionNames = {
                'armaxfilter',
                'armafor',
                'sacf',
                'spacf',
                'aicsbic'
            };
            
            % Initialize results structure
            results = struct();
            results.functions = functionNames;
            results.sampleSizes = obj.sampleSizes;
            results.timings = zeros(length(functionNames), length(obj.sampleSizes));
            results.memory = zeros(length(functionNames), length(obj.sampleSizes));
            
            % Use benchmark.scalabilityTest to test with increasing sample sizes
            for f = 1:length(functions)
                func = functions{f};
                funcName = functionNames{f};
                
                fprintf('Testing scalability of %s...\n', funcName);
                
                for i = 1:length(obj.sampleSizes)
                    size = obj.sampleSizes{i};
                    
                    % Skip large sizes for armaxfilter to avoid excessive test time
                    if strcmp(funcName, 'armaxfilter') && size > 5000
                        results.timings(f, i) = NaN;
                        results.memory(f, i) = NaN;
                        continue;
                    end
                    
                    % Generate or get test data
                    tsData = obj.dataGenerator('generateTimeSeriesData', struct('numObs', size));
                    
                    % Use the benchmark utility for accurate measurement
                    scalability = obj.benchmark.scalabilityTest(@(x) func(x), size, ...
                        @(s) obj.dataGenerator('generateTimeSeriesData', struct('numObs', s)).y);
                    
                    % Record execution time and memory usage at each scale
                    results.timings(f, i) = scalability.executionTime;
                    results.memory(f, i) = scalability.memoryUsage;
                    
                    fprintf('  %s with n=%d: %.4f seconds, %.2f MB\n', ...
                        funcName, size, results.timings(f, i), results.memory(f, i));
                end
            end
            
            % Analyze computational complexity (linear, quadratic, etc.)
            results.complexity = cell(length(functionNames), 1);
            
            for f = 1:length(functionNames)
                % Get non-NaN timings
                timings = results.timings(f, :);
                validIdx = ~isnan(timings);
                if sum(validIdx) < 3
                    results.complexity{f} = 'Insufficient data';
                    continue;
                end
                
                validSizes = log(cell2mat(obj.sampleSizes(validIdx)));
                validTimings = log(timings(validIdx));
                
                % Fit linear model to log-log data
                p = polyfit(validSizes, validTimings, 1);
                exponent = p(1);
                
                % Determine complexity class based on exponent
                if exponent < 0.2
                    results.complexity{f} = 'O(1)';
                elseif exponent < 0.8
                    results.complexity{f} = 'O(log n)';
                elseif exponent < 1.2
                    results.complexity{f} = 'O(n)';
                elseif exponent < 1.8
                    results.complexity{f} = 'O(n log n)';
                elseif exponent < 2.2
                    results.complexity{f} = 'O(n²)';
                else
                    results.complexity{f} = sprintf('O(n^%.1f)', exponent);
                end
                
                fprintf('%s complexity: %s (exponent=%.2f)\n', ...
                    functionNames{f}, results.complexity{f}, exponent);
            end
            
            % Generate scalability plots and metrics
            if obj.benchmark.saveVisualizations
                obj.generateScalabilityPlots(results);
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testModelOrderSensitivity(obj)
            % Tests how ARMA model order affects performance
            
            % Test performance across a range of AR and MA orders
            arOrders = 0:4;
            maOrders = 0:4;
            
            % Initialize results storage
            results = struct();
            results.arOrders = arOrders;
            results.maOrders = maOrders;
            results.timings = zeros(length(arOrders), length(maOrders));
            results.memory = zeros(length(arOrders), length(maOrders));
            
            % Use financial returns data
            data = obj.testData.financial.returns;
            
            % Test each combination of AR and MA orders
            for i = 1:length(arOrders)
                for j = 1:length(maOrders)
                    p = arOrders(i);
                    q = maOrders(j);
                    
                    % Skip case where both p and q are zero
                    if p == 0 && q == 0
                        results.timings(i, j) = NaN;
                        results.memory(i, j) = NaN;
                        continue;
                    end
                    
                    fprintf('Testing ARMA(%d,%d) model...\n', p, q);
                    
                    % Create options for this model order
                    options = struct('p', p, 'q', q, 'constant', true);
                    
                    % Measure execution time variation with model complexity
                    try
                        metrics = obj.benchmark.benchmarkFunction(@armaxfilter, 5, data, [], options);
                        execTime = metrics.mean;
                        
                        memInfo = obj.benchmark.measureMemoryUsage(@armaxfilter, data, [], options);
                        memUsage = memInfo.memoryDifferenceMB;
                    catch
                        execTime = NaN;
                        memUsage = NaN;
                    end
                    
                    results.timings(i, j) = execTime;
                    results.memory(i, j) = memUsage;
                    
                    fprintf('  ARMA(%d,%d): %.4f seconds, %.2f MB\n', p, q, execTime, memUsage);
                end
            end
            
            % Identify model order combinations with potential performance issues
            [maxTime, maxTimeIdx] = max(results.timings(:));
            [maxMem, maxMemIdx] = max(results.memory(:));
            [r_time, c_time] = ind2sub(size(results.timings), maxTimeIdx);
            [r_mem, c_mem] = ind2sub(size(results.memory), maxMemIdx);
            
            results.worstTimeConfig = [arOrders(r_time), maOrders(c_time)];
            results.worstMemConfig = [arOrders(r_mem), maOrders(c_mem)];
            
            % Generate sensitivity analysis plots
            if obj.benchmark.saveVisualizations
                obj.generateSensitivityPlots(results);
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = testMEXOptimizationPerformance(obj)
            % Measures the performance improvement from MEX optimization in time series functions
            
            % Create reference MATLAB implementations without MEX
            matlab_armaxerrors = @(data, ar_params, ma_params, x, x_params, constant) ...
                obj.matlab_armaxerrors_impl(data, ar_params, ma_params, x, x_params, constant);
            
            % Define test cases
            testCases = struct();
            
            % AR(1) model
            testCases(1).name = 'AR(1)';
            testCases(1).data = obj.testData.financial.returns;
            testCases(1).ar_params = 0.5;
            testCases(1).ma_params = [];
            testCases(1).x = [];
            testCases(1).x_params = [];
            testCases(1).constant = 0;
            
            % ARMA(1,1) model
            testCases(2).name = 'ARMA(1,1)';
            testCases(2).data = obj.testData.financial.returns;
            testCases(2).ar_params = 0.5;
            testCases(2).ma_params = 0.3;
            testCases(2).x = [];
            testCases(2).x_params = [];
            testCases(2).constant = 0;
            
            % ARMAX(1,1,1) model
            testCases(3).name = 'ARMAX(1,1,1)';
            testCases(3).data = obj.testData.financial.returns;
            testCases(3).ar_params = 0.5;
            testCases(3).ma_params = 0.3;
            testCases(3).x = randn(length(obj.testData.financial.returns), 1);
            testCases(3).x_params = 0.2;
            testCases(3).constant = 0.001;
            
            % Initialize results
            results = struct();
            results.testCases = testCases;
            results.mexTimings = zeros(length(testCases), 1);
            results.matlabTimings = zeros(length(testCases), 1);
            results.speedup = zeros(length(testCases), 1);
            
            % Compare performance between MEX and non-MEX implementations
            for i = 1:length(testCases)
                testCase = testCases(i);
                
                mex_fn = @() armaxerrors(testCase.data, testCase.ar_params, testCase.ma_params, ...
                    testCase.x, testCase.x_params, testCase.constant);
                
                matlab_fn = @() matlab_armaxerrors(testCase.data, testCase.ar_params, testCase.ma_params, ...
                    testCase.x, testCase.x_params, testCase.constant);
                
                % Compare implementations using benchmark utility
                comparison = obj.benchmark.compareImplementations(matlab_fn, mex_fn, obj.testConfig.iterations);
                
                % Record results
                results.matlabTimings(i) = comparison.results1.mean;
                results.mexTimings(i) = comparison.results2.mean;
                results.speedup(i) = comparison.speedupRatio;
                
                fprintf('%s: MATLAB=%.4fs, MEX=%.4fs, Speedup=%.2fx\n', ...
                    testCase.name, results.matlabTimings(i), results.mexTimings(i), results.speedup(i));
            end
            
            % Calculate speedup ratios for different input sizes
            results.scalability = struct();
            results.scalability.sampleSizes = obj.sampleSizes;
            results.scalability.matlabTimings = zeros(length(obj.sampleSizes), 1);
            results.scalability.mexTimings = zeros(length(obj.sampleSizes), 1);
            results.scalability.speedup = zeros(length(obj.sampleSizes), 1);
            
            for i = 1:length(obj.sampleSizes)
                size = obj.sampleSizes{i};
                
                % Skip very large sizes to avoid excessive test time
                if size > 10000
                    results.scalability.matlabTimings(i) = NaN;
                    results.scalability.mexTimings(i) = NaN;
                    results.scalability.speedup(i) = NaN;
                    continue;
                end
                
                % Generate or get test data
                tsData = obj.dataGenerator('generateTimeSeriesData', struct('numObs', size));
                data = tsData.y;
                
                mex_fn = @() armaxerrors(data, 0.5, 0.3, [], [], 0);
                matlab_fn = @() matlab_armaxerrors(data, 0.5, 0.3, [], [], 0);
                
                % Compare implementations
                comparison = obj.benchmark.compareImplementations(matlab_fn, mex_fn, 5);
                
                results.scalability.matlabTimings(i) = comparison.results1.mean;
                results.scalability.mexTimings(i) = comparison.results2.mean;
                results.scalability.speedup(i) = comparison.speedupRatio;
                
                fprintf('n=%d: MATLAB=%.4fs, MEX=%.4fs, Speedup=%.2fx\n', ...
                    size, results.scalability.matlabTimings(i), ...
                    results.scalability.mexTimings(i), results.scalability.speedup(i));
            end
            
            % Verify at least 50% performance improvement with MEX
            results.meetsRequirement = all(results.speedup >= 1.5);
            
            % Filter out NaN values for valid scalability check
            validIdx = ~isnan(results.scalability.speedup);
            if any(validIdx)
                results.scaleRequirement = all(results.scalability.speedup(validIdx) >= 1.5);
            else
                results.scaleRequirement = false;
            end
            
            % Generate comparison visualizations
            if obj.benchmark.saveVisualizations
                obj.generateMEXComparisonPlots(results);
            end
            
            % Store and return results
            obj.testResults.currentTest = results;
            results = obj.testResults.currentTest;
        end
        
        function results = runAllTimeSeriesPerformanceTests(obj)
            % Runs all time series performance tests and generates comprehensive report
            
            fprintf('Running all time series performance tests...\n');
            
            % Run individual performance test methods
            fprintf('\n=== Testing ARMAX filter performance ===\n');
            armaxResults = obj.testARMAXFilterPerformance();
            
            fprintf('\n=== Testing ARMA forecasting performance ===\n');
            armaforResults = obj.testARMAForPerformance();
            
            fprintf('\n=== Testing SACF performance ===\n');
            sacfResults = obj.testSACFPerformance();
            
            fprintf('\n=== Testing SPACF performance ===\n');
            spacfResults = obj.testSPACFPerformance();
            
            fprintf('\n=== Testing AICSBIC performance ===\n');
            aicsbicResults = obj.testAICSBICPerformance();
            
            fprintf('\n=== Testing time series scalability ===\n');
            scalabilityResults = obj.testTimeSeriesScalability();
            
            fprintf('\n=== Testing model order sensitivity ===\n');
            sensitivityResults = obj.testModelOrderSensitivity();
            
            fprintf('\n=== Testing MEX optimization performance ===\n');
            mexResults = obj.testMEXOptimizationPerformance();
            
            % Compile results from all time series tests
            results = struct();
            results.armaxfilter = armaxResults;
            results.armafor = armaforResults;
            results.sacf = sacfResults;
            results.spacf = spacfResults;
            results.aicsbic = aicsbicResults;
            results.scalability = scalabilityResults;
            results.sensitivity = sensitivityResults;
            results.mex = mexResults;
            
            % Generate visualizations comparing all functions
            if obj.benchmark.saveVisualizations
                obj.generatePerformanceVisualizations(results, 'comprehensive_results');
            end
            
            % Prepare comprehensive performance report
            fprintf('\n=== Performance Test Summary ===\n');
            
            % MEX optimization summary
            fprintf('MEX Optimization Performance:\n');
            fprintf('  Average speedup: %.2fx\n', mean(mexResults.speedup));
            fprintf('  Maximum speedup: %.2fx\n', max(mexResults.speedup));
            fprintf('  Meets 50%% improvement requirement: %s\n', mat2str(mexResults.meetsRequirement));
            
            % Scalability summary
            fprintf('\nScalability Summary:\n');
            for i = 1:length(scalabilityResults.functions)
                fprintf('  %s: %s\n', scalabilityResults.functions{i}, ...
                    scalabilityResults.complexity{i});
            end
            
            % Identify performance bottlenecks and optimization opportunities
            fprintf('\nPerformance Bottlenecks:\n');
            fprintf('  Most time-intensive: ARMA(%d,%d)\n', ...
                sensitivityResults.worstTimeConfig(1), sensitivityResults.worstTimeConfig(2));
            fprintf('  Most memory-intensive: ARMA(%d,%d)\n', ...
                sensitivityResults.worstMemConfig(1), sensitivityResults.worstMemConfig(2));
            
            % Store consolidated results
            obj.testResults.consolidated = results;
            
            % Return aggregated performance metrics and analysis
            return results;
        end
        
        function success = generatePerformanceVisualizations(obj, results, visualizationDir)
            % Generates detailed performance visualizations for time series functions
            
            % Default success indicator
            success = false;
            
            % Create visualization directory if it doesn't exist
            if ~exist(visualizationDir, 'dir')
                [status, msg] = mkdir(visualizationDir);
                if ~status
                    warning('Failed to create visualization directory: %s', msg);
                    return;
                end
            end
            
            try
                % Format results for visualization
                if isfield(results, 'mex') && isfield(results.mex, 'testCases')
                    % MEX optimization comparison chart
                    figure('Name', 'MEX Optimization Performance');
                    barData = [results.mex.matlabTimings, results.mex.mexTimings];
                    bar(barData);
                    set(gca, 'XTickLabel', {results.mex.testCases.name});
                    legend('MATLAB', 'MEX');
                    title('MATLAB vs MEX Implementation Performance');
                    ylabel('Execution Time (seconds)');
                    grid on;
                    
                    % Add speedup values as text
                    for i = 1:length(results.mex.speedup)
                        text(i, max(barData(i,:))/2, sprintf('%.1fx', results.mex.speedup(i)), ...
                            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                            'FontWeight', 'bold');
                    end
                    
                    % Save visualization
                    saveas(gcf, fullfile(visualizationDir, 'mex_optimization.png'));
                end
                
                % Generate execution time comparison plots
                if isfield(results, 'scalability') && isfield(results.scalability, 'timings')
                    figure('Name', 'Scalability Analysis');
                    sampleSizes = cell2mat(results.scalability.sampleSizes);
                    
                    % Plot on log-log scale for better visualization
                    loglog(sampleSizes, results.scalability.timings', 'LineWidth', 2, 'Marker', 'o');
                    grid on;
                    title('Scalability of Time Series Functions');
                    xlabel('Sample Size (log scale)');
                    ylabel('Execution Time (seconds, log scale)');
                    legend(results.scalability.functions, 'Location', 'northwest');
                    
                    % Save visualization
                    saveas(gcf, fullfile(visualizationDir, 'scalability.png'));
                end
                
                % Generate memory usage comparison plots
                if isfield(results, 'scalability') && isfield(results.scalability, 'memory')
                    figure('Name', 'Memory Usage Analysis');
                    sampleSizes = cell2mat(results.scalability.sampleSizes);
                    
                    % Plot memory usage
                    loglog(sampleSizes, results.scalability.memory', 'LineWidth', 2, 'Marker', 's');
                    grid on;
                    title('Memory Usage of Time Series Functions');
                    xlabel('Sample Size (log scale)');
                    ylabel('Memory Usage (MB, log scale)');
                    legend(results.scalability.functions, 'Location', 'northwest');
                    
                    % Save visualization
                    saveas(gcf, fullfile(visualizationDir, 'memory_usage.png'));
                end
                
                % Generate model order sensitivity heatmaps
                if isfield(results, 'sensitivity') && isfield(results.sensitivity, 'timings')
                    figure('Name', 'Model Order Sensitivity');
                    imagesc(results.sensitivity.timings);
                    colorbar;
                    colormap('jet');
                    title('ARMA(p,q) Model Order Sensitivity - Execution Time');
                    xlabel('MA Order (q)');
                    ylabel('AR Order (p)');
                    set(gca, 'XTick', 1:length(results.sensitivity.maOrders));
                    set(gca, 'YTick', 1:length(results.sensitivity.arOrders));
                    set(gca, 'XTickLabel', results.sensitivity.maOrders);
                    set(gca, 'YTickLabel', results.sensitivity.arOrders);
                    
                    % Save visualization
                    saveas(gcf, fullfile(visualizationDir, 'model_sensitivity.png'));
                }
                
                % Indicate success
                success = true;
                
            catch e
                warning('Error generating visualizations: %s', e.message);
                success = false;
            end
        end
        
        function errors = matlab_armaxerrors_impl(obj, data, ar_params, ma_params, x, x_params, constant)
            % Pure MATLAB implementation of armaxerrors for performance comparison
            
            % Get dimensions
            T = length(data);
            
            % Process AR parameters
            if isempty(ar_params)
                p = 0;
            else
                p = length(ar_params);
            end
            
            % Process MA parameters
            if isempty(ma_params)
                q = 0;
            else
                q = length(ma_params);
            end
            
            % Process exogenous variables
            hasExog = ~isempty(x) && ~isempty(x_params);
            if hasExog
                [~, r] = size(x);
                if length(x_params) ~= r
                    error('Length of x_params must match number of columns in x');
                end
            end
            
            % Initialize errors vector
            errors = zeros(T, 1);
            
            % Main computation loop
            for t = 1:T
                % Start with constant term
                expectedValue = constant;
                
                % Add AR component
                for i = 1:p
                    if t > i
                        expectedValue = expectedValue + ar_params(i) * data(t-i);
                    end
                end
                
                % Add MA component
                for i = 1:q
                    if t > i
                        expectedValue = expectedValue + ma_params(i) * errors(t-i);
                    end
                end
                
                % Add exogenous component
                if hasExog
                    for j = 1:r
                        expectedValue = expectedValue + x_params(j) * x(t, j);
                    end
                end
                
                % Calculate error as difference between actual and predicted values
                errors(t) = data(t) - expectedValue;
            end
        end
    end
end