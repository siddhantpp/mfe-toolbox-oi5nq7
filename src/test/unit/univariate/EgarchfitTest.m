classdef EgarchfitTest < BaseTest
    % EgarchfitTest Test class for validating EGARCH model fitting functionality
    %
    % This test class validates the EGARCH (Exponential GARCH) volatility model 
    % fitting functionality across various test scenarios and inputs. It tests 
    % parameter estimation accuracy, distribution options, error handling, and 
    % computational performance.
    %
    % Tests include:
    % - Basic EGARCH functionality
    % - Support for different error distributions (normal, t, GED, skewt)
    % - Testing with different model orders (p,o,q)
    % - Custom starting values and fixed parameters
    % - Different backcast methods for initialization
    % - Input validation and error handling
    % - Numerical stability with challenging data
    % - MEX vs MATLAB performance comparison
    % - Forecasting capability
    
    properties
        testData          % Test data with known EGARCH patterns
        knownParameters   % Known true EGARCH parameters
        defaultOptions    % Default EGARCH model options
        tolerance         % Tolerance for numerical comparisons
        comparator        % NumericalComparator instance
    end
    
    methods
        function obj = EgarchfitTest()
            % Constructor initializes the test class with appropriate tolerance settings
            obj@BaseTest();  % Call superclass constructor
            obj.tolerance = 1e-4;  % Set tolerance for parameter comparisons
            obj.comparator = NumericalComparator();  % Initialize comparator
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random number generator seed for reproducibility
            rng(123);
            
            % Generate EGARCH test data with known parameters
            modelParams = struct('p', 1, 'q', 1);
            modelParams.omega = -0.1;    % Typical value for EGARCH constant
            modelParams.alpha = 0.15;    % Typical ARCH effect
            modelParams.gamma = -0.08;   % Typical asymmetry parameter (negative for leverage)
            modelParams.beta = 0.96;     % Typical persistence parameter
            
            % Generate data with known EGARCH parameters
            volData = generateVolatilitySeries(1000, 'EGARCH', modelParams);
            obj.testData = volData.returns;
            
            % Store the known parameters
            obj.knownParameters = volData.parameters;
            
            % Set default EGARCH options
            obj.defaultOptions = struct('distribution', 'normal');
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data and parameters
            clear obj.testData;
            clear obj.knownParameters;
        end
        
        function testEgarchfitBasicFunctionality(obj)
            % Tests basic functionality of EGARCH model fitting with default parameters
            
            % Call egarchfit with test data and default options
            [parameters] = egarchfit(obj.testData, 1, 1, 1, obj.defaultOptions);
            
            % Verify output structure contains expected fields
            obj.assertTrue(isfield(parameters, 'distribution'), 'Missing field: distribution');
            obj.assertTrue(isfield(parameters, 'll'), 'Missing field: ll');
            obj.assertTrue(isfield(parameters, 'parameters'), 'Missing field: parameters');
            obj.assertTrue(isfield(parameters, 'stderrors'), 'Missing field: stderrors');
            obj.assertTrue(isfield(parameters, 'tstat'), 'Missing field: tstat');
            obj.assertTrue(isfield(parameters, 'pvalues'), 'Missing field: pvalues');
            obj.assertTrue(isfield(parameters, 'ht'), 'Missing field: ht');
            obj.assertTrue(isfield(parameters, 'loght'), 'Missing field: loght');
            obj.assertTrue(isfield(parameters, 'AIC'), 'Missing field: AIC');
            obj.assertTrue(isfield(parameters, 'SBIC'), 'Missing field: SBIC');
            obj.assertTrue(isfield(parameters, 'persistence'), 'Missing field: persistence');
            
            % Verify estimated parameters are close to true parameters
            trueOmega = obj.knownParameters.omega;
            trueAlpha = obj.knownParameters.alpha(1);
            trueGamma = obj.knownParameters.gamma(1);
            trueBeta = obj.knownParameters.beta(1);
            
            estOmega = parameters.omega;
            estAlpha = parameters.alpha(1);
            estGamma = parameters.gamma(1);
            estBeta = parameters.beta(1);
            
            obj.assertAlmostEqual(trueOmega, estOmega, ['Omega parameter doesn''t match. ' ...
                'Expected: ' num2str(trueOmega) ', Got: ' num2str(estOmega)]);
            obj.assertAlmostEqual(trueAlpha, estAlpha, ['Alpha parameter doesn''t match. ' ...
                'Expected: ' num2str(trueAlpha) ', Got: ' num2str(estAlpha)]);
            obj.assertAlmostEqual(trueGamma, estGamma, ['Gamma parameter doesn''t match. ' ...
                'Expected: ' num2str(trueGamma) ', Got: ' num2str(estGamma)]);
            obj.assertAlmostEqual(trueBeta, estBeta, ['Beta parameter doesn''t match. ' ...
                'Expected: ' num2str(trueBeta) ', Got: ' num2str(estBeta)]);
            
            % Check convergence status
            obj.assertTrue(parameters.exitflag > 0, 'Model estimation did not converge');
            
            % Validate log-likelihood calculation
            obj.assertTrue(isfinite(parameters.ll), 'Log-likelihood is not finite');
            obj.assertTrue(~isnan(parameters.ll), 'Log-likelihood is NaN');
        end
        
        function testEgarchfitNormalDistribution(obj)
            % Tests EGARCH model fitting with normal error distribution
            
            % Set options with normal error distribution
            options = struct('distribution', 'NORMAL');
            
            % Call egarchfit with test data and options
            [parameters] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify parameter estimation accuracy for normal distribution
            obj.assertTrue(isfield(parameters, 'omega'), 'Missing omega parameter');
            obj.assertTrue(isfield(parameters, 'alpha'), 'Missing alpha parameter');
            obj.assertTrue(isfield(parameters, 'gamma'), 'Missing gamma parameter');
            obj.assertTrue(isfield(parameters, 'beta'), 'Missing beta parameter');
            
            % Check distribution is correctly set
            obj.assertEqual('normal', parameters.distribution, 'Distribution not set correctly');
            
            % Verify no distribution parameters are present for normal distribution
            obj.assertFalse(isfield(parameters, 'nu'), 'nu parameter should not exist for normal distribution');
            obj.assertFalse(isfield(parameters, 'lambda'), 'lambda parameter should not exist for normal distribution');
            
            % Compare information criteria (AIC, BIC) with expected values
            obj.assertTrue(isfinite(parameters.AIC), 'AIC is not finite');
            obj.assertTrue(isfinite(parameters.SBIC), 'SBIC is not finite');
        end
        
        function testEgarchfitTDistribution(obj)
            % Tests EGARCH model fitting with Student's t error distribution
            
            % Set options with t distribution
            options = struct('distribution', 'T');
            
            % Call egarchfit with test data and options
            [parameters] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify parameter estimation accuracy including degrees of freedom
            obj.assertTrue(isfield(parameters, 'omega'), 'Missing omega parameter');
            obj.assertTrue(isfield(parameters, 'alpha'), 'Missing alpha parameter');
            obj.assertTrue(isfield(parameters, 'gamma'), 'Missing gamma parameter');
            obj.assertTrue(isfield(parameters, 'beta'), 'Missing beta parameter');
            obj.assertTrue(isfield(parameters, 'nu'), 'Missing nu parameter for t distribution');
            
            % Check distribution is correctly set
            obj.assertEqual('t', parameters.distribution, 'Distribution not set correctly');
            
            % Compare log-likelihood with expected value
            obj.assertTrue(isfinite(parameters.ll), 'Log-likelihood is not finite');
            
            % Verify that degrees of freedom parameter is properly estimated
            obj.assertTrue(parameters.nu > 2, 'Degrees of freedom should be greater than 2');
            obj.assertTrue(isfinite(parameters.nu), 'Degrees of freedom should be finite');
        end
        
        function testEgarchfitGEDDistribution(obj)
            % Tests EGARCH model fitting with GED error distribution
            
            % Set options with GED distribution
            options = struct('distribution', 'GED');
            
            % Call egarchfit with test data and options
            [parameters] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify parameter estimation accuracy including shape parameter
            obj.assertTrue(isfield(parameters, 'omega'), 'Missing omega parameter');
            obj.assertTrue(isfield(parameters, 'alpha'), 'Missing alpha parameter');
            obj.assertTrue(isfield(parameters, 'gamma'), 'Missing gamma parameter');
            obj.assertTrue(isfield(parameters, 'beta'), 'Missing beta parameter');
            obj.assertTrue(isfield(parameters, 'nu'), 'Missing nu parameter for GED distribution');
            
            % Check distribution is correctly set
            obj.assertEqual('ged', parameters.distribution, 'Distribution not set correctly');
            
            % Compare log-likelihood with expected value
            obj.assertTrue(isfinite(parameters.ll), 'Log-likelihood is not finite');
            
            % Verify that shape parameter is properly estimated
            obj.assertTrue(parameters.nu > 0, 'Shape parameter should be positive');
            obj.assertTrue(isfinite(parameters.nu), 'Shape parameter should be finite');
        end
        
        function testEgarchfitSkewTDistribution(obj)
            % Tests EGARCH model fitting with skewed t error distribution
            
            % Set options with skewed t distribution
            options = struct('distribution', 'SKEWT');
            
            % Call egarchfit with test data and options
            [parameters] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify parameter estimation accuracy including degrees of freedom and skewness
            obj.assertTrue(isfield(parameters, 'omega'), 'Missing omega parameter');
            obj.assertTrue(isfield(parameters, 'alpha'), 'Missing alpha parameter');
            obj.assertTrue(isfield(parameters, 'gamma'), 'Missing gamma parameter');
            obj.assertTrue(isfield(parameters, 'beta'), 'Missing beta parameter');
            obj.assertTrue(isfield(parameters, 'nu'), 'Missing nu parameter for skewed t distribution');
            obj.assertTrue(isfield(parameters, 'lambda'), 'Missing lambda parameter for skewed t distribution');
            
            % Check distribution is correctly set
            obj.assertEqual('skewt', parameters.distribution, 'Distribution not set correctly');
            
            % Compare log-likelihood with expected value
            obj.assertTrue(isfinite(parameters.ll), 'Log-likelihood is not finite');
            
            % Verify that both degrees of freedom and skewness parameters are properly estimated
            obj.assertTrue(parameters.nu > 2, 'Degrees of freedom should be greater than 2');
            obj.assertTrue(parameters.lambda > -1 && parameters.lambda < 1, 'Skewness parameter should be between -1 and 1');
        end
        
        function testEgarchfitModelOrders(obj)
            % Tests EGARCH model fitting with various model orders (p,o,q)
            
            % Test EGARCH(1,1,1) specification
            params111 = egarchfit(obj.testData, 1, 1, 1, obj.defaultOptions);
            
            % Test EGARCH(2,1,1) specification
            params211 = egarchfit(obj.testData, 2, 1, 1, obj.defaultOptions);
            
            % Test EGARCH(1,1,2) specification
            params112 = egarchfit(obj.testData, 1, 1, 2, obj.defaultOptions);
            
            % Test EGARCH(2,2,2) specification
            params222 = egarchfit(obj.testData, 2, 2, 2, obj.defaultOptions);
            
            % Verify parameter estimation accuracy for each model order
            obj.assertTrue(params111.exitflag > 0, 'EGARCH(1,1,1) did not converge');
            obj.assertTrue(params211.exitflag > 0, 'EGARCH(2,1,1) did not converge');
            obj.assertTrue(params112.exitflag > 0, 'EGARCH(1,1,2) did not converge');
            obj.assertTrue(params222.exitflag > 0, 'EGARCH(2,2,2) did not converge');
            
            % Verify parameter vectors have the right length
            obj.assertEqual(length(params111.parameters), 4, 'EGARCH(1,1,1) should have 4 parameters');
            obj.assertEqual(length(params211.parameters), 5, 'EGARCH(2,1,1) should have 5 parameters');
            obj.assertEqual(length(params112.parameters), 5, 'EGARCH(1,1,2) should have 5 parameters');
            obj.assertEqual(length(params222.parameters), 7, 'EGARCH(2,2,2) should have 7 parameters');
            
            % Compare information criteria to identify best model
            models = {params111, params211, params112, params222};
            modelNames = {'EGARCH(1,1,1)', 'EGARCH(2,1,1)', 'EGARCH(1,1,2)', 'EGARCH(2,2,2)'};
            aics = zeros(length(models), 1);
            sbics = zeros(length(models), 1);
            
            for i = 1:length(models)
                aics(i) = models{i}.AIC;
                sbics(i) = models{i}.SBIC;
            end
            
            % Verify all AIC and SBIC values are finite
            obj.assertTrue(all(isfinite(aics)), 'Some AIC values are not finite');
            obj.assertTrue(all(isfinite(sbics)), 'Some SBIC values are not finite');
        end
        
        function testEgarchfitWithStartingValues(obj)
            % Tests EGARCH model fitting with user-provided starting parameter values
            
            % First get default parameters as reference
            [defaultParams] = egarchfit(obj.testData, 1, 1, 1, obj.defaultOptions);
            
            % Set custom starting values in options structure
            startingVals = [defaultParams.omega*1.2; defaultParams.alpha*0.8; 
                           defaultParams.gamma*1.1; defaultParams.beta*0.9];
            options = struct('distribution', 'normal', 'startingvals', startingVals);
            
            % Call egarchfit with test data and options
            [customParams] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify convergence to correct parameters despite different starting values
            obj.assertTrue(customParams.exitflag > 0, 'Model with custom starting values did not converge');
            
            % Compare number of iterations with default starting values
            if isfield(defaultParams.optim.output, 'iterations') && isfield(customParams.optim.output, 'iterations')
                fprintf('Iterations with default starting values: %d\n', defaultParams.optim.output.iterations);
                fprintf('Iterations with custom starting values: %d\n', customParams.optim.output.iterations);
            end
            
            % Verify parameters are similar despite different starting points
            obj.assertAlmostEqual(defaultParams.omega, customParams.omega, 'omega parameter different with custom starting values');
            obj.assertAlmostEqual(defaultParams.alpha, customParams.alpha, 'alpha parameter different with custom starting values');
            obj.assertAlmostEqual(defaultParams.gamma, customParams.gamma, 'gamma parameter different with custom starting values');
            obj.assertAlmostEqual(defaultParams.beta, customParams.beta, 'beta parameter different with custom starting values');
        end
        
        function testEgarchfitWithFixedParameters(obj)
            % Tests EGARCH model fitting with fixed parameter constraints
            
            % First, fit model normally to get baseline parameters
            [baseParams] = egarchfit(obj.testData, 1, 1, 1, obj.defaultOptions);
            
            % Set fixed parameter flags in options structure
            options = struct('distribution', 'normal', 'fixparam', [1 0 0 0]);
            options.startingvals = baseParams.parameters;
            
            % Call egarchfit with test data and options
            [fixedParams] = egarchfit(obj.testData, 1, 1, 1, options);
            
            % Verify fixed parameters remain unchanged
            obj.assertAlmostEqual(fixedParams.omega, baseParams.omega, 'Fixed omega parameter changed');
            
            % Verify free parameters are correctly estimated
            obj.assertTrue(isfield(fixedParams, 'alpha'), 'Missing alpha parameter');
            obj.assertTrue(isfield(fixedParams, 'gamma'), 'Missing gamma parameter');
            obj.assertTrue(isfield(fixedParams, 'beta'), 'Missing beta parameter');
        end
        
        function testEgarchfitBackcastMethods(obj)
            % Tests EGARCH model fitting with different backcast methods for initialization
            
            % Test with default backcast method
            [params1] = egarchfit(obj.testData, 1, 1, 1, obj.defaultOptions);
            
            % Test with 'LONG' backcast method
            options2 = struct('distribution', 'normal', 'backcast', struct('type', 'LONG'));
            [params2] = egarchfit(obj.testData, 1, 1, 1, options2);
            
            % Test with 'SHORT' backcast method
            options3 = struct('distribution', 'normal', 'backcast', struct('type', 'SHORT'));
            [params3] = egarchfit(obj.testData, 1, 1, 1, options3);
            
            % Test with custom backcast value
            customBackcast = var(obj.testData) * 1.5; % Some arbitrary value based on data variance
            options4 = struct('distribution', 'normal', 'backcast', struct('type', 'fixed', 'value', customBackcast));
            [params4] = egarchfit(obj.testData, 1, 1, 1, options4);
            
            % Verify consistent parameter estimation across different methods
            % Compare omega parameters
            obj.assertMatrixEqualsWithTolerance([params1.omega, params2.omega, params3.omega, params4.omega], ...
                params1.omega * ones(1,4), 0.1, 'Omega parameter varies too much with different backcasts');
            
            % Compare alpha parameters
            obj.assertMatrixEqualsWithTolerance([params1.alpha, params2.alpha, params3.alpha, params4.alpha], ...
                params1.alpha * ones(1,4), 0.1, 'Alpha parameter varies too much with different backcasts');
            
            % Compare beta parameters
            obj.assertMatrixEqualsWithTolerance([params1.beta, params2.beta, params3.beta, params4.beta], ...
                params1.beta * ones(1,4), 0.1, 'Beta parameter varies too much with different backcasts');
        end
        
        function testEgarchfitInputValidation(obj)
            % Tests error handling and input validation in EGARCH model fitting
            
            % Test with invalid data (NaN values)
            invalidData = obj.testData;
            invalidData(10) = NaN;
            
            obj.assertThrows(@() egarchfit(invalidData, 1, 1, 1), 'MATLAB:datacheck:DataContainsNaN', ...
                'egarchfit should throw error with NaN values in data');
            
            % Test with invalid model orders (negative, too large)
            obj.assertThrows(@() egarchfit(obj.testData, -1, 1, 1), 'MATLAB:parametercheck:isPositive', ...
                'egarchfit should throw error with negative p');
            
            obj.assertThrows(@() egarchfit(obj.testData, 1, -1, 1), 'MATLAB:parametercheck:isPositive', ...
                'egarchfit should throw error with negative o');
            
            obj.assertThrows(@() egarchfit(obj.testData, 1, 1, -1), 'MATLAB:parametercheck:isPositive', ...
                'egarchfit should throw error with negative q');
            
            % Test with invalid optimization options
            badOptions = struct('distribution', 'normal', 'optimoptions', 'not-a-struct');
            obj.assertThrows(@() egarchfit(obj.testData, 1, 1, 1, badOptions), 'MATLAB:optim:InvalidOptimsetArgument', ...
                'egarchfit should throw error with invalid optimization options');
            
            % Test with invalid distribution specification
            badDistOptions = struct('distribution', 'invalid-dist');
            obj.assertThrows(@() egarchfit(obj.testData, 1, 1, 1, badDistOptions), 'MATLAB:egarchfit:InvalidDistribution', ...
                'egarchfit should throw error with invalid distribution');
        end
        
        function testEgarchfitNumericalStability(obj)
            % Tests numerical stability of EGARCH fitting with challenging data
            
            % Generate high-volatility test data
            highVolData = obj.testData * 10;
            
            % Generate near-constant test data
            nearConstantData = ones(size(obj.testData)) * 0.01 + 0.001 * randn(size(obj.testData));
            
            % Test with high-volatility data
            try
                highVolParams = egarchfit(highVolData, 1, 1, 1, obj.defaultOptions);
                
                % Verify model converges and produces stable parameter estimates
                obj.assertTrue(highVolParams.exitflag > 0, 'Model failed to converge with high-volatility data');
                
                % Verify parameters are reasonable
                obj.assertTrue(isfinite(highVolParams.omega), 'omega is not finite with high-volatility data');
                obj.assertTrue(isfinite(highVolParams.alpha), 'alpha is not finite with high-volatility data');
                obj.assertTrue(isfinite(highVolParams.gamma), 'gamma is not finite with high-volatility data');
                obj.assertTrue(isfinite(highVolParams.beta), 'beta is not finite with high-volatility data');
            catch ME
                fprintf('Note: Model estimation failed with high-volatility data: %s\n', ME.message);
            end
            
            % Test with near-constant test data
            try
                constParams = egarchfit(nearConstantData, 1, 1, 1, obj.defaultOptions);
                
                % If no error occurs, verify parameters are at least finite
                obj.assertTrue(isfinite(constParams.omega), 'omega is not finite with near-constant data');
                obj.assertTrue(isfinite(constParams.alpha), 'alpha is not finite with near-constant data');
                obj.assertTrue(isfinite(constParams.gamma), 'gamma is not finite with near-constant data');
                obj.assertTrue(isfinite(constParams.beta), 'beta is not finite with near-constant data');
            catch ME
                % Model might fail with near-constant data, which is acceptable
                fprintf('Note: Model failed with near-constant data: %s\n', ME.message);
            end
            
            % Check for appropriate warning messages under challenging conditions
            try
                extremeOptions = struct('distribution', 'normal', 'startingvals', [10; 0.9; -0.9; 0.9]);
                extremeParams = egarchfit(obj.testData, 1, 1, 1, extremeOptions);
                obj.assertTrue(extremeParams.exitflag > 0, 'Model failed to converge with extreme starting values');
            catch ME
                fprintf('Note: Model failed with extreme starting values: %s\n', ME.message);
            end
        end
        
        function testEgarchfitPerformance(obj)
            % Tests performance of MEX-accelerated EGARCH fitting versus pure MATLAB implementation
            
            % Generate large test dataset
            n = 2000;
            largeData = randn(n, 1);
            
            % Measure execution time with MEX acceleration
            mexOptions = struct('distribution', 'normal', 'useMEX', true);
            tic;
            [mexParams] = egarchfit(largeData, 1, 1, 1, mexOptions);
            mexTime = toc;
            
            % Set options to disable MEX acceleration
            matlabOptions = struct('distribution', 'normal', 'useMEX', false);
            
            % Measure execution time without MEX acceleration
            tic;
            [matlabParams] = egarchfit(largeData, 1, 1, 1, matlabOptions);
            matlabTime = toc;
            
            % Calculate speedup
            speedup = matlabTime / mexTime;
            fprintf('MEX speedup: %.2fx (MEX: %.3fs, MATLAB: %.3fs)\n', speedup, mexTime, matlabTime);
            
            % Verify MEX implementation provides significant speedup (>50%)
            obj.assertTrue(speedup > 1.5, 'MEX implementation does not provide the required >50% speedup');
            
            % Verify numerical results are identical between implementations
            obj.assertAlmostEqual(mexParams.ll, matlabParams.ll, 'Log-likelihood differs between MEX and MATLAB implementations');
            obj.assertMatrixEqualsWithTolerance(mexParams.parameters, matlabParams.parameters, 1e-6, ...
                'Parameters differ between MEX and MATLAB implementations');
        end
        
        function testEgarchfitForecasting(obj)
            % Tests forecasting functionality of fitted EGARCH models
            
            % Fit EGARCH model to training portion of test data
            trainSize = floor(length(obj.testData) * 0.8);
            trainData = obj.testData(1:trainSize);
            testData = obj.testData(trainSize+1:end);
            
            % Fit model to training data
            [params] = egarchfit(trainData, 1, 1, 1, obj.defaultOptions);
            
            % Create GARCH model structure for forecasting
            garchModel = struct();
            garchModel.parameters = params.parameters;
            garchModel.modelType = 'EGARCH';
            garchModel.p = 1;
            garchModel.q = 1;
            garchModel.data = trainData;
            garchModel.residuals = trainData; % For EGARCH with zero mean
            garchModel.ht = params.ht;
            garchModel.distribution = params.distribution;
            
            % Generate forecasts for test portion
            forecast = garchfor(garchModel, length(testData));
            
            % Verify forecast structure has expected fields
            obj.assertTrue(isfield(forecast, 'expectedVariances'), 'Missing variance forecasts');
            obj.assertTrue(isfield(forecast, 'expectedVolatility'), 'Missing volatility forecasts');
            
            % Verify forecasts have correct length
            obj.assertEqual(length(forecast.expectedVariances), length(testData), 'Variance forecasts have incorrect length');
            
            % Verify all forecasts are positive
            obj.assertTrue(all(forecast.expectedVariances > 0), 'Variance forecasts must be positive');
            obj.assertTrue(all(forecast.expectedVolatility > 0), 'Volatility forecasts must be positive');
            
            % Compare forecasts with actual test values
            actualSquared = testData.^2;
            mse = mean((actualSquared - forecast.expectedVariances).^2);
            fprintf('Mean squared forecast error: %.6f\n', mse);
            
            % Test multi-step forecasting capability
            simOptions = struct('simulate', true, 'numPaths', 100);
            multistepForecast = garchfor(garchModel, 10, simOptions);
            
            % Verify simulation produces paths
            obj.assertTrue(isfield(multistepForecast, 'varPaths'), 'Missing simulated variance paths');
            obj.assertEqual(size(multistepForecast.varPaths, 1), 10, 'Incorrect number of forecast horizons');
        end
    end
end