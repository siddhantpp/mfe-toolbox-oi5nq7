classdef ArmaforTest < BaseTest
    % Unit test class for the armafor.m function with comprehensive tests for ARMA/ARMAX forecasting functionality
    
    properties
        testTolerance       % Tolerance for numerical comparisons
        testData            % Structure to store test data
        financialReturns    % Matrix of financial returns for testing
        numObservations     % Number of observations in test data
        forecastHorizon     % Number of forecast periods
        numSimulations      % Number of simulations for simulation-based forecasts
    end
    
    methods
        function obj = ArmaforTest()
            % Initialize the ArmaforTest class with default test parameters
            obj = obj@BaseTest(); % Call the parent BaseTest constructor
            obj.testTolerance = 1e-10; % Set numerical tolerance for comparisons
            obj.forecastHorizon = 10; % Default forecast horizon
            obj.numSimulations = 1000; % Default number of simulations
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            obj.setUp@BaseTest(); % Call parent setUp method
            
            % Set random number generator seed for reproducibility
            rng(1234);
            
            % Load test data containing financial returns
            obj.testData = obj.loadTestData('financial_returns.mat');
            obj.financialReturns = obj.testData.returns;
            obj.numObservations = size(obj.financialReturns, 1);
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            obj.tearDown@BaseTest(); % Call parent tearDown method
            
            % Clear temporary variables created during tests
        end
        
        function testBasicARMAForecast(obj)
            % Test basic ARMA(1,1) model forecasting with exact method
            
            % Create a simple AR(1) model with coefficient 0.7
            arParams = 0.7;
            maParams = 0;
            
            % Generate synthetic data based on the model
            n = 100;
            data = obj.generateTestData(arParams, maParams, n, false, 0);
            
            % Forecast using exact method
            [forecasts, variances] = armafor(arParams, data, 1, 0, false);
            
            % Verify forecast has correct dimensions
            obj.assertEqual(size(forecasts), [1, 1], 'Forecast should have size [1,1]');
            
            % The 1-step ahead forecast for AR(1) should be arParams*data(end)
            expectedForecast = arParams * data(end);
            obj.assertAlmostEqual(forecasts, expectedForecast, 'One-step AR(1) forecast is incorrect');
            
            % Test multi-step forecasting
            horizon = obj.forecastHorizon;
            [forecasts, variances] = armafor(arParams, data, 1, 0, false, [], horizon);
            
            % Verify forecast dimensions
            obj.assertEqual(size(forecasts), [horizon, 1], 'Multi-step forecast has wrong dimensions');
            obj.assertEqual(size(variances), [horizon, 1], 'Variance array has wrong dimensions');
            
            % Check forecasts decrease exponentially toward mean
            for i = 2:horizon
                % For AR(1), each forecast should be arParams times the previous forecast
                expectedForecast = arParams * forecasts(i-1);
                obj.assertAlmostEqual(forecasts(i), expectedForecast, ...
                    sprintf('Forecast at horizon %d does not match expected value', i));
            end
            
            % Check variances increase with horizon (for stationary AR process)
            for i = 2:horizon
                obj.assertTrue(variances(i) > variances(i-1), ...
                    'Forecast variance should increase with horizon');
            end
        end
        
        function testARMAXForecast(obj)
            % Test ARMAX model forecasting with exogenous variables
            
            % Create ARMAX(1,1) model
            arParams = 0.5;
            maParams = 0.2;
            exogParams = 0.5; % Coefficient for exogenous variable
            
            % Generate test data
            n = 100;
            data = zeros(n, 1);
            x = randn(n, 1); % Exogenous variable
            
            % Initialize with random noise
            data(1) = randn();
            e = randn(n, 1);
            
            % Generate ARMAX process data
            for t = 2:n
                data(t) = arParams * data(t-1) + maParams * e(t-1) + exogParams * x(t) + e(t);
            end
            
            % Create future exogenous variable values for forecast period
            horizon = obj.forecastHorizon;
            futureX = randn(horizon, 1);
            
            % Set parameters for armafor
            parameters = [arParams, maParams, exogParams];
            
            % Forecast with exogenous variables
            [forecasts, variances] = armafor(parameters, data, 1, 1, false, futureX, horizon);
            
            % Verify forecast dimensions
            obj.assertEqual(size(forecasts), [horizon, 1], 'Forecast has wrong dimensions');
            obj.assertEqual(size(variances), [horizon, 1], 'Variance array has wrong dimensions');
            
            % Compare with forecast without exogenous variables
            [forecastsNoX, ~] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            
            % The forecasts should differ due to exogenous variables
            for i = 1:horizon
                obj.assertFalse(abs(forecasts(i) - forecastsNoX(i)) < obj.testTolerance, ...
                    'Forecasts with and without exogenous variables should differ');
            end
        end
        
        function testSimulationMethod(obj)
            % Test simulation-based forecasting method with normal errors
            
            % Create ARMA(1,1) model parameters
            arParams = 0.6;
            maParams = 0.3;
            
            % Generate synthetic data
            n = 200;
            data = obj.generateTestData([arParams], [maParams], n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Forecast using simulation method
            numSim = obj.numSimulations;
            [forecasts, variances, paths] = armafor([arParams, maParams], data, 1, 1, false, [], horizon, [], 'simulation', numSim);
            
            % Verify simulation paths dimensions
            obj.assertEqual(size(paths), [horizon, numSim], 'Simulation paths have wrong dimensions');
            
            % Also forecast using exact method for comparison
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            
            % Mean of simulation paths should approximate exact forecast
            for i = 1:horizon
                simulationMean = mean(paths(i, :));
                obj.assertAlmostEqual(simulationMean, exactForecasts(i), 'Simulation mean differs from exact forecast');
                
                % Standard deviation of paths should approximate theoretical standard error
                simulationStd = std(paths(i, :));
                theoreticalStd = sqrt(exactVariances(i));
                
                % Allow more tolerance for simulation variance comparison
                obj.assertMatrixEqualsWithTolerance(simulationStd, theoreticalStd, 0.05, ...
                    'Simulation standard deviation should approximate theoretical value');
            end
        end
        
        function testStudentTErrors(obj)
            % Test simulation-based forecasting with Student's t distributed errors
            
            % Create ARMA(1,1) model parameters
            arParams = 0.5;
            maParams = 0.2;
            
            % Generate synthetic data
            n = 200;
            data = obj.generateTestData([arParams], [maParams], n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Set up Student's t distribution parameters
            nu = 5; % Degrees of freedom
            dist_params = struct('nu', nu);
            
            % Forecast using simulation method with Student's t innovations
            numSim = obj.numSimulations;
            [forecasts, variances, paths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                horizon, [], 'simulation', numSim, 'student', dist_params);
            
            % Verify simulation paths dimensions
            obj.assertEqual(size(paths), [horizon, numSim], 'Simulation paths have wrong dimensions');
            
            % Also forecast using exact method for comparison
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            
            % Mean of simulation paths should approximate exact forecast
            for i = 1:horizon
                simulationMean = mean(paths(i, :));
                
                % Allow more tolerance for t-distribution comparison
                tolerance = 0.05;
                obj.assertMatrixEqualsWithTolerance(simulationMean, exactForecasts(i), tolerance, ...
                    'Simulation mean with t distribution differs from exact forecast');
                
                % Generate normal distribution simulation for comparison
                [~, ~, normalPaths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                    horizon, [], 'simulation', numSim, 'normal');
                
                % Get extreme quantiles for comparison
                tDistQ95 = quantile(paths(i, :), 0.95);
                normalQ95 = quantile(normalPaths(i, :), 0.95);
                
                % Student's t with nu=5 should have heavier tails than normal
                % Not a strict test, but should generally be true with enough simulations
                if i > 1 % Skip first horizon which might be similar
                    obj.assertTrue(abs(tDistQ95) >= abs(normalQ95) * 0.95, ...
                        'Student t distribution should generally have heavier tails than normal');
                end
            end
        end
        
        function testGEDErrors(obj)
            % Test simulation-based forecasting with GED distributed errors
            
            % Create ARMA(1,1) model parameters
            arParams = 0.5;
            maParams = 0.2;
            
            % Generate synthetic data
            n = 200;
            data = obj.generateTestData([arParams], [maParams], n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Set up GED distribution parameters
            nu = 1.5; % Shape parameter (nu<2 means heavier tails than normal)
            dist_params = struct('nu', nu);
            
            % Forecast using simulation method with GED innovations
            numSim = obj.numSimulations;
            [forecasts, variances, paths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                horizon, [], 'simulation', numSim, 'ged', dist_params);
            
            % Verify simulation paths dimensions
            obj.assertEqual(size(paths), [horizon, numSim], 'Simulation paths have wrong dimensions');
            
            % Also forecast using exact method for comparison
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            
            % Mean of simulation paths should approximate exact forecast
            for i = 1:horizon
                simulationMean = mean(paths(i, :));
                
                % Allow more tolerance for GED distribution comparison
                tolerance = 0.05;
                obj.assertMatrixEqualsWithTolerance(simulationMean, exactForecasts(i), tolerance, ...
                    'Simulation mean with GED distribution differs from exact forecast');
                
                % Generate normal distribution simulation for comparison
                [~, ~, normalPaths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                    horizon, [], 'simulation', numSim, 'normal');
                
                % Compare tail behavior - GED with nu=1.5 should have heavier tails than normal
                gedQ95 = quantile(paths(i, :), 0.95);
                normalQ95 = quantile(normalPaths(i, :), 0.95);
                
                % Not a strict test, but should generally be true with enough simulations
                if i > 1 % Skip first horizon which might be similar
                    obj.assertTrue(abs(gedQ95) >= abs(normalQ95) * 0.95, ...
                        'GED distribution with nu=1.5 should generally have heavier tails than normal');
                end
            end
        end
        
        function testSkewedTErrors(obj)
            % Test simulation-based forecasting with Hansen's Skewed t distributed errors
            
            % Create ARMA(1,1) model parameters
            arParams = 0.5;
            maParams = 0.2;
            
            % Generate synthetic data
            n = 200;
            data = obj.generateTestData([arParams], [maParams], n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Set up Skewed t distribution parameters
            nu = 5; % Degrees of freedom
            lambda = 0.2; % Skewness parameter
            dist_params = struct('nu', nu, 'lambda', lambda);
            
            % Forecast using simulation method with Skewed t innovations
            numSim = obj.numSimulations;
            [forecasts, variances, paths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                horizon, [], 'simulation', numSim, 'skewt', dist_params);
            
            % Verify simulation paths dimensions
            obj.assertEqual(size(paths), [horizon, numSim], 'Simulation paths have wrong dimensions');
            
            % Also forecast using exact method for comparison
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            
            % Mean of simulation paths should approximate exact forecast
            for i = 1:horizon
                simulationMean = mean(paths(i, :));
                
                % Allow more tolerance for skewed t distribution comparison
                tolerance = 0.05;
                obj.assertMatrixEqualsWithTolerance(simulationMean, exactForecasts(i), tolerance, ...
                    'Simulation mean with skewed t distribution differs from exact forecast');
                
                % Check for asymmetry in the simulated distribution
                % Calculate distance between median and mean as a simple skewness measure
                pathMedian = median(paths(i, :));
                pathMean = mean(paths(i, :));
                
                % Generate symmetric t distribution simulation for comparison
                symmetricParams = struct('nu', nu);
                [~, ~, symmetricPaths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                    horizon, [], 'simulation', numSim, 'student', symmetricParams);
                
                symMedian = median(symmetricPaths(i, :));
                symMean = mean(symmetricPaths(i, :));
                
                % For skewed distribution, the difference between mean and median
                % should be greater than for a symmetric distribution
                % Not a strict test, but should generally be true with enough simulations
                if i > 1 % Skip first horizon which might be similar
                    skewedDiff = abs(pathMean - pathMedian);
                    symDiff = abs(symMean - symMedian);
                    
                    % We don't use strict assertion here as it's a probabilistic test
                    % and could occasionally fail by chance
                    if lambda > 0.1 % Only check if lambda is sufficiently large
                        obj.assertTrue(skewedDiff > symDiff * 0.8, ...
                            'Skewed t should show greater difference between mean and median than symmetric t');
                    end
                end
            end
        end
        
        function testFinancialReturnsData(obj)
            % Test ARMA forecasting on real financial returns data
            
            % Use a single asset from the financial returns data
            returns = obj.financialReturns(:, 1);
            
            % Set up ARMA(1,1) model parameters
            % In practice, these would be estimated from the data
            arParams = 0.2; % Example parameter
            maParams = 0.1; % Example parameter
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Forecast using both exact and simulation methods
            [exactForecasts, exactVariances] = armafor([arParams, maParams], returns, 1, 1, false, [], horizon);
            [simForecasts, simVariances, paths] = armafor([arParams, maParams], returns, 1, 1, false, [], ...
                horizon, [], 'simulation', obj.numSimulations);
            
            % Verify forecast dimensions
            obj.assertEqual(size(exactForecasts), [horizon, 1], 'Exact forecast has wrong dimensions');
            obj.assertEqual(size(simForecasts), [horizon, 1], 'Simulation forecast has wrong dimensions');
            
            % Compare exact and simulation forecasts (should be close)
            for i = 1:horizon
                obj.assertMatrixEqualsWithTolerance(exactForecasts(i), simForecasts(i), 0.05, ...
                    sprintf('Exact and simulation forecasts differ at horizon %d', i));
            end
            
            % For financial time series, forecasts should typically revert toward the mean
            % Calculate the mean of the returns
            returnsMean = mean(returns);
            
            % Check if forecasts are approaching the mean as horizon increases
            distanceToMean = abs(exactForecasts - returnsMean);
            
            % For most financial series with stationary ARMA models, later forecasts
            % should be closer to the unconditional mean
            if abs(arParams) < 0.9 % Only check for clearly stationary models
                obj.assertTrue(distanceToMean(end) < distanceToMean(1) * 0.9, ...
                    'Forecasts should approach the unconditional mean over time');
            end
        end
        
        function testHigherOrderARMA(obj)
            % Test forecasting with higher-order ARMA(2,2) models
            
            % Create ARMA(2,2) model parameters
            arParams = [0.5, -0.2]; % AR(2) parameters
            maParams = [0.3, 0.1];  % MA(2) parameters
            
            % Generate synthetic data
            n = 200;
            data = obj.generateTestData(arParams, maParams, n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Forecast using both methods
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 2, 2, false, [], horizon);
            [simForecasts, simVariances, paths] = armafor([arParams, maParams], data, 2, 2, false, [], ...
                horizon, [], 'simulation', obj.numSimulations);
            
            % Verify forecast dimensions
            obj.assertEqual(size(exactForecasts), [horizon, 1], 'Exact forecast has wrong dimensions');
            obj.assertEqual(size(simForecasts), [horizon, 1], 'Simulation forecast has wrong dimensions');
            
            % Compare forecasts from different methods
            for i = 1:horizon
                obj.assertMatrixEqualsWithTolerance(exactForecasts(i), simForecasts(i), 0.05, ...
                    sprintf('Exact and simulation forecasts differ at horizon %d', i));
            end
            
            % For ARMA(2,2), verify the first forecast incorporates both AR lags
            expectedFirstForecast = arParams(1) * data(end) + arParams(2) * data(end-1);
            % We would also add MA components, but we don't have direct access to the errors
            % So this is just a rough check
            obj.assertTrue(abs(exactForecasts(1) - expectedFirstForecast) < 1.0, ...
                'First forecast should incorporate AR(2) effects');
        end
        
        function testParameterValidation(obj)
            % Test parameter validation and error handling in armafor function
            
            % Generate some test data
            n = 100;
            data = randn(n, 1);
            
            % Test with invalid AR order parameter (negative)
            obj.assertThrows(@() armafor(0.5, data, -1, 0, false), ...
                'PARAMETERCHECK:InvalidInput', 'Should reject negative AR order');
            
            % Test with invalid MA order parameter (negative)
            obj.assertThrows(@() armafor(0.5, data, 0, -1, false), ...
                'PARAMETERCHECK:InvalidInput', 'Should reject negative MA order');
            
            % Test with inconsistent data dimensions
            badData = data';  % Row vector instead of column vector
            obj.assertThrows(@() armafor(0.5, badData, 1, 0, false), ...
                'COLUMNCHECK:InvalidInput', 'Should reject row vector data');
            
            % Test with invalid forecast horizon (negative)
            obj.assertThrows(@() armafor(0.5, data, 1, 0, false, [], -5), ...
                'PARAMETERCHECK:InvalidInput', 'Should reject negative forecast horizon');
            
            % Test with invalid forecast horizon (zero)
            obj.assertThrows(@() armafor(0.5, data, 1, 0, false, [], 0), ...
                'PARAMETERCHECK:InvalidInput', 'Should reject zero forecast horizon');
            
            % Test with invalid distribution parameters for student distribution
            badDistParams = struct('wrong_param', 5);
            obj.assertThrows(@() armafor(0.5, data, 1, 0, false, [], 10, [], 'simulation', 100, 'student', badDistParams), ...
                'ARMAFOR:InvalidInput', 'Should reject invalid distribution parameters');
        end
        
        function testExtremeValues(obj)
            % Test ARMA forecasting with extreme parameter values
            
            % Test with AR coefficient near unity
            arParamsNearUnity = 0.99;
            n = 100;
            data = obj.generateTestData(arParamsNearUnity, 0, n, false, 0);
            
            % Forecast with near-unit-root process
            horizon = obj.forecastHorizon;
            [forecasts, variances] = armafor(arParamsNearUnity, data, 1, 0, false, [], horizon);
            
            % Verify forecast dimensions
            obj.assertEqual(size(forecasts), [horizon, 1], 'Forecast has wrong dimensions');
            
            % For near-unit-root process, forecasts should decay very slowly
            % Check if the last forecast is still significant compared to first
            obj.assertTrue(abs(forecasts(horizon) / forecasts(1)) > 0.9, ...
                'Near-unit-root process should show persistent forecasts');
            
            % Test with very small coefficients
            arParamsSmall = 0.01;
            dataSmall = obj.generateTestData(arParamsSmall, 0, n, false, 0);
            
            % Forecast with very small AR parameter
            [forecastsSmall, variancesSmall] = armafor(arParamsSmall, dataSmall, 1, 0, false, [], horizon);
            
            % For very small AR parameter, forecasts should decay very quickly to mean (0)
            obj.assertTrue(abs(forecastsSmall(2) / forecastsSmall(1)) < 0.05, ...
                'Process with small AR parameter should converge quickly to mean');
        end
        
        function testPerformanceComparison(obj)
            % Test and compare performance of exact vs. simulation methods
            
            % Generate a large dataset
            n = 1000;
            arParams = 0.5;
            maParams = 0.2;
            data = obj.generateTestData(arParams, maParams, n, false, 0);
            
            % Set forecast horizon
            horizon = obj.forecastHorizon;
            
            % Measure execution time for exact method
            tic;
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, 1, 1, false, [], horizon);
            exactTime = toc;
            
            % Measure execution time for simulation method with different simulation counts
            simCounts = [100, 1000, 5000];
            simTimes = zeros(length(simCounts), 1);
            
            for i = 1:length(simCounts)
                tic;
                [simForecasts, simVariances, paths] = armafor([arParams, maParams], data, 1, 1, false, [], ...
                    horizon, [], 'simulation', simCounts(i));
                simTimes(i) = toc;
            end
            
            % Verify that simulation time increases roughly proportionally with simulation count
            % This is not a strict assertion, just a benchmark
            ratios = simTimes ./ simCounts;
            avgRatio = mean(ratios);
            
            for i = 1:length(ratios)
                % Check if time per simulation is within reasonable range (50% tolerance)
                obj.assertTrue(abs(ratios(i) - avgRatio) < 0.5 * avgRatio, ...
                    'Simulation time should scale approximately linearly with simulation count');
            end
            
            % The exact method should generally be faster than simulation with many paths
            obj.assertTrue(exactTime < simTimes(end), ...
                'Exact method should be faster than simulation with many paths');
        end
        
        function data = generateTestData(obj, arParams, maParams, numObs, includeConstant, constantValue)
            % Helper method to generate synthetic ARMA process data
            
            % Validate inputs
            if nargin < 6
                constantValue = 0;
            end
            if nargin < 5
                includeConstant = false;
            end
            
            % Get AR and MA orders
            p = length(arParams);
            q = length(maParams);
            
            % Initialize data
            data = zeros(numObs, 1);
            
            % Initialize error terms
            e = randn(numObs, 1);
            
            % Generate first p observations with random noise
            data(1:p) = randn(p, 1);
            
            % Generate remaining observations
            for t = (max(p,q)+1):numObs
                % Start with constant if included
                data(t) = constantValue * includeConstant;
                
                % Add AR component
                for i = 1:p
                    data(t) = data(t) + arParams(i) * data(t-i);
                end
                
                % Add MA component
                for i = 1:q
                    data(t) = data(t) + maParams(i) * e(t-i);
                end
                
                % Add current error
                data(t) = data(t) + e(t);
            end
        end
        
        function results = compareForecastMethods(obj, arParams, maParams, data, horizon)
            % Helper method to compare exact and simulation forecast methods
            
            % Generate forecasts using exact method
            [exactForecasts, exactVariances] = armafor([arParams, maParams], data, length(arParams), length(maParams), false, [], horizon);
            
            % Generate forecasts using simulation method
            [simForecasts, simVariances, paths] = armafor([arParams, maParams], data, length(arParams), length(maParams), false, [], ...
                horizon, [], 'simulation', obj.numSimulations);
            
            % Calculate differences
            forecastDiff = exactForecasts - simForecasts;
            varianceDiff = exactVariances - simVariances;
            
            % Compute statistics
            results = struct();
            results.exactForecasts = exactForecasts;
            results.simForecasts = simForecasts;
            results.forecastDiff = forecastDiff;
            results.meanAbsDiff = mean(abs(forecastDiff));
            results.maxAbsDiff = max(abs(forecastDiff));
            results.varianceDiff = varianceDiff;
            results.paths = paths;
        end
    end
end