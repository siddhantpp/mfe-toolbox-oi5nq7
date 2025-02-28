classdef KpssTestTest < BaseTest
    % KpssTestTest Test class for validating the functionality and accuracy of the KPSS stationarity test implementation
    
    properties
        stationaryData
        nonStationaryData
        trendStationaryData
        financialReturns
        comparator
        dataGenerator
        expectedCriticalValues
    end
    
    methods
        function obj = KpssTestTest()
            % Initialize the KpssTestTest class with default configurations
            obj = obj@BaseTest('KpssTestTest');
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Initialize NumericalComparator for robust floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Initialize TestDataGenerator for creating test series
            obj.dataGenerator = TestDataGenerator;
            
            % Generate stationary test series using random normal data
            obj.stationaryData = obj.dataGenerator('generateStationaryData', 500, 1);
            
            % Generate non-stationary test series using cumulative sum of random data
            obj.nonStationaryData = obj.dataGenerator('generateNonStationaryData', 500, 1);
            
            % Generate trend-stationary series with deterministic trend component
            t = (1:500)';
            trend = 0.01 * t;
            obj.trendStationaryData = trend + 0.5 * randn(500, 1);
            
            % Load financial returns test data from financial_returns.mat
            loadedData = obj.loadTestData('financial_returns.mat');
            obj.financialReturns = loadedData.returns;
            
            % Define expected critical values for different significance levels
            obj.expectedCriticalValues = struct();
            obj.expectedCriticalValues.mu = [0.739, 0.574, 0.463, 0.347];   % 1%, 2.5%, 5%, 10%
            obj.expectedCriticalValues.tau = [0.216, 0.176, 0.146, 0.119];  % 1%, 2.5%, 5%, 10%
            
            % Set random number generator seed for reproducibility
            rng(123);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.stationaryData = [];
            obj.nonStationaryData = [];
            obj.trendStationaryData = [];
            obj.financialReturns = [];
            obj.comparator = [];
        end
        
        function testKpssLevelStationarity(obj)
            % Test KPSS test with level stationarity specification (mu option)
            
            % Run KPSS test on stationary data with regression_type='mu'
            options = struct('regression_type', 'mu');
            results = kpss_test(obj.stationaryData, options);
            
            % Verify test statistic is appropriately small for stationary series
            obj.assertTrue(results.stat < results.cv(3), 'KPSS test statistic should be small for stationary data');
            
            % Verify p-value is appropriately large for stationary series
            obj.assertTrue(results.pval > 0.05, 'P-value should be > 0.05 for stationary data');
            
            % Verify critical values match expected values for level stationarity
            obj.assertAlmostEqual(results.cv, obj.expectedCriticalValues.mu, 'Critical values do not match expected values');
            
            % Ensure stationary data correctly fails to reject null hypothesis of stationarity
            obj.assertEqual('mu', results.regression_type, 'Regression type should be mu');
            
            % Run KPSS test on non-stationary data with regression_type='mu'
            results = kpss_test(obj.nonStationaryData, options);
            
            % Verify test statistic is large for non-stationary series
            obj.assertTrue(results.stat > results.cv(3), 'KPSS test statistic should be large for non-stationary data');
            
            % Verify p-value is small for non-stationary series
            obj.assertTrue(results.pval < 0.05, 'P-value should be < 0.05 for non-stationary data');
            
            % Ensure non-stationary data correctly rejects null hypothesis of stationarity
        end
        
        function testKpssTrendStationarity(obj)
            % Test KPSS test with trend stationarity specification (tau option)
            
            % Run KPSS test on trend-stationary data with regression_type='tau'
            options = struct('regression_type', 'tau');
            results = kpss_test(obj.trendStationaryData, options);
            
            % Verify test statistic is appropriately small for trend-stationary series
            obj.assertTrue(results.stat < results.cv(3), 'KPSS test statistic should be small for trend stationary data');
            
            % Verify p-value is appropriately large for trend-stationary series
            obj.assertTrue(results.pval > 0.05, 'P-value should be > 0.05 for trend stationary data');
            
            % Verify critical values match expected values for trend stationarity
            obj.assertAlmostEqual(results.cv, obj.expectedCriticalValues.tau, 'Critical values do not match expected values');
            
            % Ensure trend-stationary data correctly fails to reject null hypothesis of stationarity
            obj.assertEqual('tau', results.regression_type, 'Regression type should be tau');
            
            % Run KPSS test on non-stationary data with regression_type='tau'
            results = kpss_test(obj.nonStationaryData, options);
            
            % Verify test statistic is large for non-stationary series
            obj.assertTrue(results.stat > results.cv(3), 'KPSS test statistic should be large for non-stationary data');
            
            % Verify p-value is small for non-stationary series
            obj.assertTrue(results.pval < 0.05, 'P-value should be < 0.05 for non-stationary data');
            
            % Ensure non-stationary data correctly rejects null hypothesis of stationarity
        end
        
        function testKpssAutomaticLags(obj)
            % Test KPSS test with automatic lag selection
            
            % Run KPSS test with automatic lag selection (default)
            options = struct();
            results = kpss_test(obj.stationaryData, options);
            
            % Verify lags are automatically determined based on sample size
            T = length(obj.stationaryData);
            expectedLags = floor(4 * (T/100)^0.25);
            obj.assertEqual(expectedLags, results.lags, 'Automatic lag selection did not use expected formula');
            
            % Compare results with expected values for automatic lag selection
            % Verify consistency of results with different sample sizes
            for T = [100, 200, 500, 1000]
                data = randn(T, 1);
                results = kpss_test(data);
                expectedLags = floor(4 * (T/100)^0.25);
                obj.assertEqual(expectedLags, results.lags, sprintf('Automatic lag for size %d incorrect', T));
            end
            
            % Confirm lag selection formula works correctly with default implementation
        end
        
        function testKpssFixedLags(obj)
            % Test KPSS test with manually specified lag parameters
            
            % Run KPSS test with lags=0 (no lag correction)
            options = struct('lags', 0);
            results0 = kpss_test(obj.stationaryData, options);
            obj.assertEqual(0, results0.lags, 'Fixed lag 0 not correctly used');
            
            % Run KPSS test with lags=1 (minimal lag correction)
            options.lags = 1;
            results1 = kpss_test(obj.stationaryData, options);
            obj.assertEqual(1, results1.lags, 'Fixed lag 1 not correctly used');
            
            % Run KPSS test with lags=8 (moderate lag correction)
            options.lags = 8;
            results8 = kpss_test(obj.stationaryData, options);
            obj.assertEqual(8, results8.lags, 'Fixed lag 8 not correctly used');
            
            % Run KPSS test with lags=20 (high lag correction)
            options.lags = 20;
            results20 = kpss_test(obj.stationaryData, options);
            obj.assertEqual(20, results20.lags, 'Fixed lag 20 not correctly used');
            
            % Verify test statistics differ appropriately with different lag specifications
            lagValues = [0, 1, 8, 20];
            allResults = [results0.stat, results1.stat, results8.stat, results20.stat];
            
            % Confirm lag parameter is correctly applied in long-run variance estimation
            obj.assertFalse(all(diff(allResults) == 0), 'Test statistics should vary with different lag values');
            
            % Verify that increasing lags generally decreases test statistic value
            % Note: This is not strictly guaranteed but often the case
            obj.testResults.lagEffect = allResults;
        end
        
        function testKpssFinancialData(obj)
            % Test KPSS test with real financial returns data
            
            % Run KPSS test on financial return series (expected to be stationary)
            returns = obj.financialReturns(:, 1);  % Use first series
            options = struct('regression_type', 'mu');
            results = kpss_test(returns, options);
            
            % Verify returns series fails to reject stationarity null hypothesis
            obj.assertTrue(results.pval > 0.05, 'P-value should be > 0.05 for financial returns');
            
            % Test with different regression specifications (mu and tau)
            options.regression_type = 'tau';
            resultsTau = kpss_test(returns, options);
            obj.assertTrue(resultsTau.pval > 0.05, 'P-value should be > 0.05 for financial returns with trend option');
            
            % Verify results are consistent with financial theory
            % Test on price levels data (expected to be non-stationary)
            prices = cumsum(returns) + 100;  % Construct price levels from returns
            options.regression_type = 'mu';
            priceResults = kpss_test(prices, options);
            
            % Verify price level series rejects stationarity null hypothesis
            obj.assertTrue(priceResults.stat > priceResults.cv(3), 'KPSS test statistic should be large for price levels');
            obj.assertTrue(priceResults.pval < 0.05, 'P-value should be < 0.05 for price levels');
            
            % Compare results with expected financial time series properties
        end
        
        function testKpssOutputStructure(obj)
            % Test that the KPSS test results structure contains all expected fields
            
            % Run KPSS test and capture results structure
            results = kpss_test(obj.stationaryData);
            
            % Verify presence of test statistic field
            obj.assertTrue(isfield(results, 'stat'), 'Results should contain stat field');
            obj.assertTrue(isnumeric(results.stat) && isscalar(results.stat), 'stat should be a numeric scalar');
            
            % Verify presence of critical values field at different significance levels
            obj.assertTrue(isfield(results, 'cv'), 'Results should contain cv field');
            obj.assertTrue(isnumeric(results.cv) && length(results.cv) == 4, 'cv should be a numeric vector of length 4');
            
            % Verify presence of p-value field
            obj.assertTrue(isfield(results, 'pval'), 'Results should contain pval field');
            obj.assertTrue(isnumeric(results.pval) && isscalar(results.pval), 'pval should be a numeric scalar');
            
            % Verify presence of regression type information
            obj.assertTrue(isfield(results, 'regression_type'), 'Results should contain regression_type field');
            obj.assertTrue(ischar(results.regression_type), 'regression_type should be a string');
            
            % Verify presence of lag information
            obj.assertTrue(isfield(results, 'lags'), 'Results should contain lags field');
            obj.assertTrue(isnumeric(results.lags) && isscalar(results.lags), 'lags should be a numeric scalar');
            
            % Ensure all expected fields are present and correctly formatted
            obj.assertTrue(isfield(results, 'significance_levels'), 'Results should contain significance_levels field');
            obj.assertTrue(isnumeric(results.significance_levels) && length(results.significance_levels) == 4, 
                'significance_levels should be a numeric vector of length 4');
        end
        
        function testKpssInvalidInputs(obj)
            % Test KPSS test with invalid inputs to verify proper error handling
            
            % Test with empty data and expect appropriate error
            obj.assertThrows(@() kpss_test([]), 'MATLAB:columncheck:EmptyMatrix', 'Empty data should throw error');
            
            % Test with non-numeric data and expect appropriate error
            obj.assertThrows(@() kpss_test('abc'), 'MATLAB:columncheck:InputTypeError', 'Non-numeric data should throw error');
            
            % Test with invalid regression_type value and expect appropriate error
            options = struct('regression_type', 'invalid');
            obj.assertThrows(@() kpss_test(obj.stationaryData, options), '', 'Invalid regression_type should throw error');
            
            % Test with invalid lag parameter (negative value) and expect appropriate error
            options = struct('lags', -1);
            obj.assertThrows(@() kpss_test(obj.stationaryData, options), '', 'Negative lag should throw error');
            
            % Test with NaN values in time series and expect appropriate error
            data = obj.stationaryData;
            data(10) = NaN;
            obj.assertThrows(@() kpss_test(data), '', 'Data with NaN should throw error');
            
            % Test with Inf values in time series and expect appropriate error
            data = obj.stationaryData;
            data(10) = Inf;
            obj.assertThrows(@() kpss_test(data), '', 'Data with Inf should throw error');
            
            % Verify appropriate error messages are generated for each case
        end
        
        function testKpssPerformance(obj)
            % Test performance characteristics of the KPSS test implementation
            
            % Measure execution time with varying data sizes
            dataSizes = [100, 500, 1000, 2000];
            execTimes = zeros(length(dataSizes), 1);
            
            for i = 1:length(dataSizes)
                T = dataSizes(i);
                data = randn(T, 1);
                
                % Measure execution time
                tic;
                kpss_test(data);
                execTimes(i) = toc;
            end
            
            % Store results for analysis
            obj.testResults.execTimes = execTimes;
            obj.testResults.dataSizes = dataSizes;
            
            % Measure execution time with varying lag orders
            T = 1000;
            data = randn(T, 1);
            lagOrders = [0, 5, 10, 20, 50];
            lagExecTimes = zeros(length(lagOrders), 1);
            
            for i = 1:length(lagOrders)
                lag = lagOrders(i);
                options = struct('lags', lag);
                
                % Measure execution time
                tic;
                kpss_test(data, options);
                lagExecTimes(i) = toc;
            end
            
            % Store results for analysis
            obj.testResults.lagExecTimes = lagExecTimes;
            obj.testResults.lagOrders = lagOrders;
            
            % Verify performance scales appropriately with data size
            if length(execTimes) >= 2
                scaling = execTimes(end) / execTimes(1);
                sizeScaling = dataSizes(end) / dataSizes(1);
                obj.testResults.scalingRatio = scaling / sizeScaling;
                
                % Very loose check that performance is reasonable
                obj.assertTrue(scaling < sizeScaling^3, 'Performance scaling worse than expected');
            end
            
            % Compare performance with and without automatic lag selection
            % Ensure efficient computation for large datasets
        end
        
        function testSeries = generateTestSeries(obj, seriesType, numObservations, parameters)
            % Helper method to generate test time series with known stationarity properties
            
            if nargin < 3
                numObservations = 500;
            end
            
            if nargin < 4
                parameters = struct();
            end
            
            % Generate white noise innovations using randn
            innovations = randn(numObservations, 1);
            
            switch lower(seriesType)
                case 'stationary'
                    % For 'stationary', return white noise innovations directly
                    testSeries = innovations;
                    
                case 'nonstationary'
                    % For 'nonstationary', compute cumulative sum of innovations
                    testSeries = cumsum(innovations);
                    
                case 'trend_stationary'
                    % For 'trend_stationary', add deterministic trend to stationary series
                    t = (1:numObservations)';
                    if isfield(parameters, 'trend_coef')
                        trendCoef = parameters.trend_coef;
                    else
                        trendCoef = 0.01;
                    end
                    
                    if isfield(parameters, 'noise_scale')
                        noiseScale = parameters.noise_scale;
                    else
                        noiseScale = 0.5;
                    end
                    
                    trend = trendCoef * t;
                    testSeries = trend + noiseScale * innovations;
                    
                otherwise
                    error('Unknown series type: %s', seriesType);
            end
            
            % Apply optional AR or MA components if specified in parameters
            if isfield(parameters, 'ar')
                ar = parameters.ar;
                arSeries = zeros(size(testSeries));
                arSeries(1) = testSeries(1);
                
                for t = 2:length(testSeries)
                    arSeries(t) = ar * arSeries(t-1) + testSeries(t);
                end
                
                testSeries = arSeries;
            end
            
            if isfield(parameters, 'ma')
                ma = parameters.ma;
                maSeries = zeros(size(testSeries));
                prevInnovation = 0;
                
                for t = 1:length(testSeries)
                    maSeries(t) = testSeries(t) + ma * prevInnovation;
                    prevInnovation = testSeries(t);
                end
                
                testSeries = maSeries;
            end
            
            % Return series with requested stationarity properties for testing
        end
    end
end