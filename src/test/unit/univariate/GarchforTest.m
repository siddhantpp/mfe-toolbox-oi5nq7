classdef GarchforTest < BaseTest
    % Test class for the garchfor function that validates forecasting functionality across different GARCH model variants
    
    properties
        testData     % Test data for forecasting
        garchModels  % Predefined GARCH model structures
        comparator   % NumericalComparator for precision-aware comparisons
    end
    
    methods
        function obj = GarchforTest()
            % Initialize the GarchforTest class
            obj@BaseTest();  % Call superclass constructor
            obj.comparator = NumericalComparator();  % Initialize comparator
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            obj.setUp@BaseTest();  % Call superclass setup
            
            % Load test data
            try
                % Try to load financial returns data
                financial_data = obj.loadTestData('financial_returns.mat');
                
                % Try to load simulated data
                simulated_data = obj.loadTestData('simulated_data.mat');
                
                % Combine data into testData structure
                obj.testData = struct(...
                    'returns', financial_data.returns, ...
                    'residuals', financial_data.residuals, ...
                    'variances', financial_data.variances, ...
                    'simulated', simulated_data ...
                );
            catch ME
                % If loading fails, create synthetic test data
                warning('Test data files not found. Creating synthetic test data.');
                
                % Create synthetic returns data
                T = 1000;
                rng(123);  % For reproducibility
                returns = 0.01 * randn(T, 1);
                
                % Create synthetic variances (GARCH-like pattern)
                variances = zeros(T, 1);
                variances(1) = 0.0001;
                for t = 2:T
                    variances(t) = 0.00001 + 0.1 * returns(t-1)^2 + 0.8 * variances(t-1);
                end
                
                % Rescale returns using variances
                residuals = returns .* sqrt(variances);
                
                % Store in testData structure
                obj.testData = struct(...
                    'returns', returns, ...
                    'residuals', residuals, ...
                    'variances', variances, ...
                    'simulated', struct('data', randn(500, 1)) ...
                );
            end
            
            % Set random seed for reproducibility
            rng(123);
            
            % Initialize GARCH model structures
            obj.garchModels = obj.createTestModels();
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            obj.tearDown@BaseTest();
            
            % Reset random number generator state
            rng('default');
        end
        
        function testGarchForBasicFunctionality(obj)
            % Verify basic forecasting functionality for a standard GARCH(1,1) model
            
            % Create a standard GARCH(1,1) model
            model = obj.garchModels.garch;
            
            % Generate a 10-step ahead forecast
            forecast = garchfor(model, 10);
            
            % Verify forecast structure contains expected fields
            obj.assertTrue(isfield(forecast, 'expectedVariances'), 'Forecast should contain expectedVariances field');
            obj.assertTrue(isfield(forecast, 'expectedReturns'), 'Forecast should contain expectedReturns field');
            obj.assertTrue(isfield(forecast, 'expectedVolatility'), 'Forecast should contain expectedVolatility field');
            
            % Verify dimensions of forecast
            obj.assertEqual(size(forecast.expectedVariances), [10, 1], 'Expected variance forecast should be 10x1');
            
            % Verify variances are positive
            obj.assertTrue(all(forecast.expectedVariances > 0), 'Variance forecasts should be positive');
            
            % Verify volatility is sqrt of variance
            obj.assertAlmostEqual(forecast.expectedVolatility, sqrt(forecast.expectedVariances), ...
                'Volatility forecast should be square root of variance forecast');
            
            % Verify returns for zero-mean model are zero
            obj.assertAlmostEqual(forecast.expectedReturns, zeros(10, 1), ...
                'Expected returns should be zero for zero-mean model');
            
            % Verify variances converge to long-run variance
            % First variance should depend on last observed values
            % Later variances should depend more on model parameters
            diff_var = diff(forecast.expectedVariances);
            obj.assertTrue(abs(diff_var(end)) < abs(diff_var(1)), ...
                'Variance changes should diminish as forecast horizon increases');
        end
        
        function testDeterministicForecasts(obj)
            % Test deterministic forecasting method for different GARCH model types
            
            % Model types to test
            modelTypes = {'garch', 'gjr', 'egarch', 'agarch', 'igarch', 'nagarch'};
            
            % Forecast horizon
            horizon = 20;
            
            for i = 1:length(modelTypes)
                if isfield(obj.garchModels, modelTypes{i})
                    % Get the current model
                    model = obj.garchModels.(modelTypes{i});
                    
                    % Generate deterministic forecast
                    forecast = garchfor(model, horizon);
                    
                    % Verify forecast structure
                    isValid = obj.verifyForecastStructure(forecast, false);
                    obj.assertTrue(isValid, ['Forecast structure is invalid for ' model.modelType]);
                    
                    % For stationary models, variance should tend toward unconditional variance
                    if strcmp(model.modelType, 'GARCH') || strcmp(model.modelType, 'GJR') || ...
                       strcmp(model.modelType, 'AGARCH') || strcmp(model.modelType, 'NAGARCH')
                        
                        % Check if model is stationary (simplified check)
                        isStationary = true;
                        if strcmp(model.modelType, 'GARCH')
                            alpha = model.parameters(2);
                            beta = model.parameters(3);
                            isStationary = (alpha + beta < 0.999);
                        end
                        
                        if isStationary
                            % For stationary models, variance should converge
                            % Check if late-horizon variances are close to each other
                            lateVar = forecast.expectedVariances(end-2:end);
                            diffs = diff(lateVar);
                            maxRelDiff = max(abs(diffs) ./ lateVar(1:end-1));
                            
                            obj.assertTrue(maxRelDiff < 0.05, ...
                                ['Variance forecasts should converge for stationary ' model.modelType]);
                        end
                    end
                    
                    % For integrated models, variance should increase or stay constant
                    if strcmp(model.modelType, 'IGARCH')
                        earlyVar = forecast.expectedVariances(5);
                        lateVar = forecast.expectedVariances(end);
                        
                        obj.assertTrue(lateVar >= earlyVar, ...
                            'IGARCH variance should not decrease over long horizons');
                    end
                    
                    % Variance should be positive for all models
                    obj.assertTrue(all(forecast.expectedVariances > 0), ...
                        ['Variance forecasts should be positive for ' model.modelType]);
                end
            end
        end
        
        function testSimulationForecasts(obj)
            % Test Monte Carlo simulation-based forecasting with different numbers of paths
            
            % Standard GARCH model
            model = obj.garchModels.garch;
            
            % Different simulation path counts
            pathCounts = [100, 500];
            
            for i = 1:length(pathCounts)
                % Create simulation options
                options = struct('simulate', true, 'numPaths', pathCounts(i), 'seed', 123);
                
                % Generate simulation forecast
                forecast = garchfor(model, 15, options);
                
                % Verify forecast structure
                isValid = obj.verifyForecastStructure(forecast, true);
                obj.assertTrue(isValid, 'Simulation forecast structure is invalid');
                
                % Verify simulation paths match expected dimensions
                obj.assertEqual(size(forecast.varPaths), [15, pathCounts(i)], ...
                    'Variance paths should match expected dimensions');
                obj.assertEqual(size(forecast.returnPaths), [15, pathCounts(i)], ...
                    'Return paths should match expected dimensions');
                
                % Verify mean simulated path is close to deterministic forecast
                detForecast = garchfor(model, 15);
                
                % Use relative comparison with adequate tolerance
                result = obj.comparator.compareMatrices(forecast.varMean, detForecast.expectedVariances, 0.1);
                obj.assertTrue(result.isEqual, 'Mean simulated path should be close to deterministic forecast');
                
                % Verify median is reasonable
                obj.assertTrue(all(forecast.varMedian > 0), 'Median variance should be positive');
                
                % Verify quantiles are ordered correctly
                for t = 1:15
                    quantiles = forecast.varQuantiles(t, :);
                    obj.assertTrue(all(diff(quantiles) >= -1e-10), 'Quantiles should be in ascending order');
                end
                
                % Verify confidence intervals have appropriate coverage
                % At least 90% of simulated paths should be within 99% confidence interval
                for t = 5:15  % Start from t=5 to allow for stabilization
                    lowerBound = forecast.varQuantiles(t, 1);  % 1% quantile
                    upperBound = forecast.varQuantiles(t, 5);  % 99% quantile
                    
                    % Count paths within bounds
                    withinBounds = (forecast.varPaths(t, :) >= lowerBound) & ...
                                  (forecast.varPaths(t, :) <= upperBound);
                    coverage = sum(withinBounds) / pathCounts(i);
                    
                    obj.assertTrue(coverage >= 0.9, 'Confidence intervals should have correct coverage');
                end
            end
        end
        
        function testDifferentDistributions(obj)
            % Test forecasting with different error distributions
            
            % Base GARCH model
            baseModel = obj.garchModels.garch;
            
            % Define distributions to test
            distributions = {
                {'normal', []},
                {'t', 5},
                {'ged', 1.5},
                {'skewt', [5, 0.2]}
            };
            
            for i = 1:length(distributions)
                % Create model with current distribution
                model = baseModel;
                model.distribution = distributions{i}{1};
                model.distParams = distributions{i}{2};
                
                % Create simulation options
                options = struct('simulate', true, 'numPaths', 500, 'seed', 123);
                
                % Generate simulation forecast
                forecast = garchfor(model, 15, options);
                
                % Verify forecast structure
                isValid = obj.verifyForecastStructure(forecast, true);
                obj.assertTrue(isValid, ['Forecast structure is invalid for ' model.distribution ' distribution']);
                
                % Verify return paths exist and have reasonable values
                obj.assertTrue(all(isfinite(forecast.returnPaths(:))), ...
                    ['Return paths should be finite for ' model.distribution ' distribution']);
                
                % Check distribution-specific properties when possible
                switch model.distribution
                    case 't'
                        % t distribution should have heavier tails than normal
                        % Check kurtosis of returns (should be > 3 for t)
                        kurt = kurtosis(forecast.returnPaths(10, :));
                        obj.assertTrue(kurt > 3, 'Student-t returns should have kurtosis > 3');
                        
                    case 'skewt'
                        % With positive lambda (0.2), should have negative skewness
                        skew = skewness(forecast.returnPaths(10, :));
                        
                        if model.distParams(2) > 0
                            obj.assertTrue(skew < 0, 'Skewed-t with positive lambda should have negative skewness');
                        elseif model.distParams(2) < 0
                            obj.assertTrue(skew > 0, 'Skewed-t with negative lambda should have positive skewness');
                        end
                end
            end
        end
        
        function testLongHorizonForecasts(obj)
            % Test forecasting performance over long horizons
            
            % Model to test
            model = obj.garchModels.garch;
            
            % Long horizons to test
            horizons = [100, 500];
            
            for j = 1:length(horizons)
                horizon = horizons(j);
                
                % Generate deterministic forecast
                forecast = garchfor(model, horizon);
                
                % Verify forecast structure and dimensions
                isValid = obj.verifyForecastStructure(forecast, false);
                obj.assertTrue(isValid, 'Long-horizon forecast structure is invalid');
                
                obj.assertEqual(size(forecast.expectedVariances), [horizon, 1], ...
                    'Expected variance forecast should match horizon');
                
                % Check if model is stationary
                alpha = model.parameters(2);
                beta = model.parameters(3);
                isStationary = (alpha + beta < 0.999);
                
                % For stationary models, verify convergence to unconditional variance
                if isStationary
                    % Get late forecasts
                    lateVar = forecast.expectedVariances(end-20:end);
                    
                    % Calculate relative changes
                    relChanges = abs(diff(lateVar) ./ lateVar(1:end-1));
                    
                    % Should be small relative changes in long-horizon forecasts
                    obj.assertTrue(max(relChanges) < 0.05, ...
                        'Long-horizon forecasts should converge to unconditional variance');
                    
                    % Calculate theoretical unconditional variance
                    omega = model.parameters(1);
                    uncondVar = omega / (1 - alpha - beta);
                    
                    % Verify that late forecasts are close to unconditional variance
                    relDiff = abs(lateVar(end) - uncondVar) / uncondVar;
                    obj.assertTrue(relDiff < 0.1, ...
                        'Long-horizon forecast should approach unconditional variance');
                end
                
                % Variance should remain positive
                obj.assertTrue(all(forecast.expectedVariances > 0), ...
                    'Variance forecasts should be positive');
                
                % Test with shorter simulation horizon to keep tests running in reasonable time
                if horizon <= 100
                    options = struct('simulate', true, 'numPaths', 100, 'seed', 123);
                    simForecast = garchfor(model, horizon, options);
                    
                    % Verify basic simulation structure
                    isValidSim = obj.verifyForecastStructure(simForecast, true);
                    obj.assertTrue(isValidSim, 'Long-horizon simulation forecast structure is invalid');
                end
            end
        end
        
        function testExtremeCases(obj)
            % Test forecasting behavior with extreme parameter values
            
            % Create GARCH model with high persistence (near unit root)
            highPersistModel = obj.createTestModel('GARCH', [0.01; 0.15; 0.84]);
            
            % Create model with very low alpha/beta ratio
            lowAlphaBetaModel = obj.createTestModel('GARCH', [0.01; 0.01; 0.90]);
            
            % Create model with extreme asymmetry
            extremeAsymModel = obj.createTestModel('GJR', [0.01; 0.05; 0.3; 0.6]);
            
            % Create IGARCH model (integrated)
            igarchModel = obj.createTestModel('IGARCH', [0.01; 0.1; 0.9]);
            
            % Test models array
            testModels = {highPersistModel, lowAlphaBetaModel, extremeAsymModel, igarchModel};
            modelDescriptions = {'high persistence', 'low alpha/beta', 'extreme asymmetry', 'integrated'};
            
            for i = 1:length(testModels)
                model = testModels{i};
                
                % Generate deterministic forecasts
                forecast = garchfor(model, 50);
                
                % Verify forecast structure
                isValid = obj.verifyForecastStructure(forecast, false);
                obj.assertTrue(isValid, ['Forecast structure is invalid for ' modelDescriptions{i} ' model']);
                
                % Verify variances remain positive
                obj.assertTrue(all(forecast.expectedVariances > 0), ...
                    ['Variance forecasts should be positive for ' modelDescriptions{i} ' model']);
                
                % For high persistence model, variances should not decrease much
                if i == 1
                    % Check if late variances are not decreasing significantly
                    lateVar = forecast.expectedVariances(40:end);
                    relDecreases = min(0, diff(lateVar) ./ lateVar(1:end-1));
                    
                    obj.assertTrue(all(relDecreases > -0.01), ...
                        'High persistence model should not have significantly decreasing variances');
                end
                
                % For IGARCH model, variances should grow with horizon
                if i == 4
                    % Get early and late variances
                    earlyVar = forecast.expectedVariances(5);
                    lateVar = forecast.expectedVariances(end);
                    
                    % Late variance should be larger than early variance
                    obj.assertTrue(lateVar > earlyVar, ...
                        'IGARCH model should have increasing forecast variances');
                end
            end
        end
        
        function testBadInputs(obj)
            % Test error handling for invalid inputs
            
            % Valid model for testing
            model = obj.garchModels.garch;
            
            % Test missing fields
            badModel1 = model;
            badModel1 = rmfield(badModel1, 'parameters');
            obj.assertThrows(@() garchfor(badModel1, 10), ...
                'GARCHMODEL is missing required field', ...
                'Should throw error for missing parameters field');
            
            % Test invalid model type
            badModel2 = model;
            badModel2.modelType = 'INVALID';
            obj.assertThrows(@() garchfor(badModel2, 10), ...
                'Unknown model type', ...
                'Should throw error for invalid model type');
            
            % Test negative forecast horizon
            obj.assertThrows(@() garchfor(model, -5), ...
                'must contain only positive values', ...
                'Should throw error for negative forecast horizon');
            
            % Test invalid distribution
            badModel3 = model;
            badModel3.distribution = 'invalid';
            obj.assertThrows(@() garchfor(badModel3, 10), ...
                'Unknown distribution type', ...
                'Should throw error for invalid distribution');
            
            % Test missing distribution parameters for t distribution
            badModel4 = model;
            badModel4.distribution = 't';
            badModel4.distParams = [];
            obj.assertThrows(@() garchfor(badModel4, 10), ...
                'Distribution parameters must be provided', ...
                'Should throw error for missing t distribution parameters');
            
            % Test invalid simulation options
            badOptions = struct('simulate', true, 'numPaths', -10);
            obj.assertThrows(@() garchfor(model, 10, badOptions), ...
                'must contain only positive values', ...
                'Should throw error for negative number of simulation paths');
        end
        
        function testPerformance(obj)
            % Test computational performance of forecasting with large datasets
            
            % Create a standard GARCH model
            model = obj.garchModels.garch;
            
            % Test forecasting with a moderate horizon
            horizon = 100;
            
            % Measure execution time for deterministic forecast
            tic;
            forecast = garchfor(model, horizon);
            detTime = toc;
            
            % Verify forecast structure
            isValid = obj.verifyForecastStructure(forecast, false);
            obj.assertTrue(isValid, 'Deterministic forecast structure is invalid');
            
            % Measure execution time for simulation forecast with fewer paths to keep test runtime reasonable
            options = struct('simulate', true, 'numPaths', 100, 'seed', 123);
            tic;
            simForecast = garchfor(model, horizon, options);
            simTime = toc;
            
            % Verify forecast structure
            isValidSim = obj.verifyForecastStructure(simForecast, true);
            obj.assertTrue(isValidSim, 'Simulation forecast structure is invalid');
            
            % Simulation should take longer than deterministic forecast
            obj.assertTrue(simTime > detTime, ...
                'Simulation forecast should take longer than deterministic forecast');
            
            % But simulation shouldn't be unreasonably slow
            % This is a loose test that may need adjustment based on system performance
            % We use a large upper bound to avoid false failures on slower systems
            obj.assertTrue(simTime < 100 * detTime, ...
                'Simulation forecast shouldn't be unreasonably slower than deterministic forecast');
            
            % Verify that MEX acceleration is working correctly if available
            % We check by ensuring performance is reasonable
            obj.assertTrue(detTime < 10, ...
                'Deterministic forecast should complete in reasonable time (possible MEX issue)');
        end
        
        % Helper methods
        
        function models = createTestModels(obj)
            % Helper method to create GARCH model structures for testing
            
            % Standard GARCH(1,1) model
            models.garch = struct(...
                'parameters', [0.01; 0.1; 0.8], ...  % omega, alpha, beta
                'modelType', 'GARCH', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
            
            % GJR/TARCH(1,1) model
            models.gjr = struct(...
                'parameters', [0.01; 0.05; 0.1; 0.8], ...  % omega, alpha, gamma, beta
                'modelType', 'GJR', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
            
            % EGARCH(1,1) model
            models.egarch = struct(...
                'parameters', [0.01; 0.1; 0.05; 0.8], ...  % omega, alpha, gamma, beta
                'modelType', 'EGARCH', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
            
            % AGARCH(1,1) model
            models.agarch = struct(...
                'parameters', [0.01; 0.1; 0.05; 0.8], ...  % omega, alpha, gamma, beta
                'modelType', 'AGARCH', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
            
            % IGARCH(1,1) model
            models.igarch = struct(...
                'parameters', [0.01; 0.1; 0.9], ...  % omega, alpha, beta
                'modelType', 'IGARCH', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
            
            % NAGARCH(1,1) model
            models.nagarch = struct(...
                'parameters', [0.01; 0.1; 0.05; 0.8], ...  % omega, alpha, gamma, beta
                'modelType', 'NAGARCH', ...
                'p', 1, ...
                'q', 1, ...
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...
                'distParams', [] ...
            );
        end
        
        function model = createTestModel(obj, modelType, parameters, options)
            % Helper method to create a GARCH model structure for testing
            
            % Default options
            if nargin < 4
                options = struct();
            end
            
            % Create base model
            model = struct(...
                'parameters', parameters, ...
                'modelType', modelType, ...
                'p', 1, ...  % Default GARCH order
                'q', 1, ...  % Default ARCH order
                'data', obj.testData.returns, ...
                'residuals', obj.testData.residuals, ...
                'ht', obj.testData.variances, ...
                'distribution', 'normal', ...  % Default distribution
                'distParams', [] ...
            );
            
            % Add custom fields from options if provided
            if isfield(options, 'p')
                model.p = options.p;
            end
            
            if isfield(options, 'q')
                model.q = options.q;
            end
            
            if isfield(options, 'distribution')
                model.distribution = options.distribution;
            end
            
            if isfield(options, 'distParams')
                model.distParams = options.distParams;
            end
        end
        
        function valid = verifyForecastStructure(obj, forecast, hasSimulation)
            % Helper method to validate forecast structure format
            
            % Check required fields for all forecasts
            obj.assertTrue(isfield(forecast, 'expectedVariances'), 'Forecast should contain expectedVariances field');
            obj.assertTrue(isfield(forecast, 'expectedReturns'), 'Forecast should contain expectedReturns field');
            obj.assertTrue(isfield(forecast, 'expectedVolatility'), 'Forecast should contain expectedVolatility field');
            
            % Get forecast horizon
            horizon = length(forecast.expectedVariances);
            
            % Verify dimensions
            obj.assertEqual(size(forecast.expectedVariances), [horizon, 1], 'Expected variance forecast should be Nx1');
            obj.assertEqual(size(forecast.expectedReturns), [horizon, 1], 'Expected returns forecast should be Nx1');
            obj.assertEqual(size(forecast.expectedVolatility), [horizon, 1], 'Expected volatility forecast should be Nx1');
            
            % Verify variance positivity
            obj.assertTrue(all(forecast.expectedVariances > 0), 'Variance forecasts should be positive');
            
            % Verify volatility is sqrt of variance with appropriate tolerance
            result = obj.comparator.compareMatrices(forecast.expectedVolatility, sqrt(forecast.expectedVariances), 1e-8);
            obj.assertTrue(result.isEqual, 'Volatility forecast should be square root of variance forecast');
            
            % If simulation forecast, check simulation fields
            if hasSimulation
                obj.assertTrue(isfield(forecast, 'varPaths'), 'Simulation forecast should contain varPaths field');
                obj.assertTrue(isfield(forecast, 'returnPaths'), 'Simulation forecast should contain returnPaths field');
                obj.assertTrue(isfield(forecast, 'volatilityPaths'), 'Simulation forecast should contain volatilityPaths field');
                obj.assertTrue(isfield(forecast, 'varQuantiles'), 'Simulation forecast should contain varQuantiles field');
                obj.assertTrue(isfield(forecast, 'returnQuantiles'), 'Simulation forecast should contain returnQuantiles field');
                obj.assertTrue(isfield(forecast, 'probLevels'), 'Simulation forecast should contain probLevels field');
                
                % Get number of paths
                [~, numPaths] = size(forecast.varPaths);
                
                % Verify dimensions
                obj.assertEqual(size(forecast.varPaths), [horizon, numPaths], 'Variance paths should be NxP');
                obj.assertEqual(size(forecast.returnPaths), [horizon, numPaths], 'Return paths should be NxP');
                obj.assertEqual(size(forecast.volatilityPaths), [horizon, numPaths], 'Volatility paths should be NxP');
                
                % Verify mean calculations with appropriate tolerance
                result = obj.comparator.compareMatrices(forecast.varMean, mean(forecast.varPaths, 2), 1e-8);
                obj.assertTrue(result.isEqual, 'varMean should be the mean of varPaths');
            end
            
            valid = true;
        end
    end
end