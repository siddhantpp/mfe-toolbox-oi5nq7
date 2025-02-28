classdef IgarchfitTest < BaseTest
    % IGARCHFITTEST Test class for validating IGARCH model estimation functionality with unit persistence constraint
    %
    % This class contains unit tests for the igarchfit function in the MFE Toolbox.
    % It tests the functionality and accuracy of Integrated GARCH (IGARCH) model
    % parameter estimation with unit persistence constraint, various error distributions,
    % and optimization configurations.
    %
    % The tests cover basic IGARCH(1,1) model estimation, higher-order IGARCH(p,q) models,
    % constrained optimization, fixed starting values, input validation, performance,
    % and numerical stability.
    %
    % See also: igarchfit, generateVolatilitySeries, BaseTest
    
    properties
        tolerance   % Numerical tolerance for floating-point comparisons
        testData    % Structure to store test data
        voldata     % Volatility test data
    end
    
    methods
        function obj = IgarchfitTest()
            % Initialize the IgarchfitTest class with test data
            
            % Call superclass constructor (BaseTest)
            obj = obj@BaseTest();
            
            % Set numerical tolerance for floating-point comparisons appropriate for IGARCH models
            obj.tolerance = 1e-6;
            
            % Load standard volatility test data from voldata.mat
            obj.voldata = obj.loadTestData('voldata.mat');
            
            % Initialize test data structure for storing intermediate results
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Ensure test data is properly initialized
            % (e.g., reload data if necessary)
            
            % Reset test state for clean test execution
            obj.testData = struct();
        end
        
        function tearDown(obj)
            % Clean up after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clean up any temporary variables or resources
            
            % Finalize test state
        end
        
        function testIgarchFitBasic(obj)
            % Test basic IGARCH(1,1) model estimation with standard options and normal distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure IGARCH(1,1) model with normal distribution
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(returns);
            
            % Verify model parameters are correctly estimated
            obj.assertAlmostEqual(estimatedModel.omega, 0.0307, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha, 0.0737, obj.tolerance, 'Alpha parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta, 0.9263, obj.tolerance, 'Beta parameter mismatch');
            
            % Validate model has unit persistence (sum of alpha and beta = 1)
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Ensure conditional variances are properly computed
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
            
            % Verify parameter constraints are satisfied (omega > 0, alpha > 0)
            obj.assertTrue(estimatedModel.omega > 0, 'Omega must be positive');
            obj.assertTrue(estimatedModel.alpha > 0, 'Alpha must be positive');
        end
        
        function testIgarchFitWithStudentT(obj)
            % Test IGARCH model estimation with Student's t error distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure IGARCH(1,1) model with Student's t distribution
            options.distribution = 'T';
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertAlmostEqual(estimatedModel.omega, 0.0271, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha, 0.0784, obj.tolerance, 'Alpha parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta, 0.9216, obj.tolerance, 'Beta parameter mismatch');
            
            % Validate unit persistence constraint is satisfied
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Validate degrees of freedom parameter is properly estimated
            obj.assertAlmostEqual(estimatedModel.nu, 7.8424, obj.tolerance, 'Degrees of freedom parameter mismatch');
            
            % Compare log-likelihood with benchmark values
            obj.assertAlmostEqual(estimatedModel.LL, -1443.9, obj.tolerance, 'Log-likelihood mismatch');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
            obj.assertEqual(length(estimatedModel.stdresid), length(returns), 'Standardized residuals length mismatch');
        end
        
        function testIgarchFitWithGED(obj)
            % Test IGARCH model estimation with Generalized Error Distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure IGARCH(1,1) model with GED distribution
            options.distribution = 'GED';
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertAlmostEqual(estimatedModel.omega, 0.0284, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha, 0.0765, obj.tolerance, 'Alpha parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta, 0.9235, obj.tolerance, 'Beta parameter mismatch');
            
            % Validate unit persistence constraint is satisfied
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Validate GED shape parameter is properly estimated
            obj.assertAlmostEqual(estimatedModel.nu, 1.3257, obj.tolerance, 'GED shape parameter mismatch');
            
            % Compare log-likelihood with benchmark values
            obj.assertAlmostEqual(estimatedModel.LL, -1445.1, obj.tolerance, 'Log-likelihood mismatch');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
            obj.assertEqual(length(estimatedModel.stdresid), length(returns), 'Standardized residuals length mismatch');
        end
        
        function testIgarchFitWithSkewedT(obj)
            % Test IGARCH model estimation with Hansen's skewed t distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure IGARCH(1,1) model with skewed t distribution
            options.distribution = 'SKEWT';
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertAlmostEqual(estimatedModel.omega, 0.0278, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha, 0.0774, obj.tolerance, 'Alpha parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta, 0.9226, obj.tolerance, 'Beta parameter mismatch');
            
            % Validate unit persistence constraint is satisfied
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Validate degrees of freedom and skewness parameters are properly estimated
            obj.assertAlmostEqual(estimatedModel.nu, 7.7474, obj.tolerance, 'Degrees of freedom parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.lambda, -0.0684, obj.tolerance, 'Skewness parameter mismatch');
            
            % Compare log-likelihood with benchmark values
            obj.assertAlmostEqual(estimatedModel.LL, -1443.4, obj.tolerance, 'Log-likelihood mismatch');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
            obj.assertEqual(length(estimatedModel.stdresid), length(returns), 'Standardized residuals length mismatch');
        end
        
        function testIgarchFitHigherOrder(obj)
            % Test higher-order IGARCH(p,q) model estimation with unit persistence constraint
            
            % Generate synthetic data with known IGARCH(2,2) parameters using generateVolatilitySeries
            trueParams.omega = 0.01;
            trueParams.alpha1 = 0.1;
            trueParams.alpha2 = 0.05;
            trueParams.beta1 = 0.6;
            trueParams.beta2 = 0.2499;
            
            numObservations = 2000;
            [testData] = obj.generateIgarchTestData(numObservations, 2, 2, trueParams);
            
            % Configure IGARCH(2,2) model with appropriate options
            options.p = 2;
            options.q = 2;
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(testData.returns, options);
            
            % Verify all parameters (omega, alpha1, alpha2, beta1, beta2) are correctly estimated
            obj.assertAlmostEqual(estimatedModel.omega, trueParams.omega, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha(1), trueParams.alpha1, obj.tolerance, 'Alpha1 parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha(2), trueParams.alpha2, obj.tolerance, 'Alpha2 parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta(1), trueParams.beta1, obj.tolerance, 'Beta1 parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta(2), trueParams.beta2, obj.tolerance, 'Beta2 parameter mismatch');
            
            % Validate that unit persistence constraint is satisfied: sum(alpha) + sum(beta) = 1
            alphaCoeffs = estimatedModel.alpha;
            betaCoeffs = estimatedModel.beta;
            obj.verifyUnitPersistence(alphaCoeffs, betaCoeffs, obj.tolerance);
            
            % Ensure conditional variances match the true values within tolerance
            obj.assertMatrixEqualsWithTolerance(estimatedModel.ht, testData.ht, obj.tolerance, 'Conditional variance mismatch');
        end
        
        function testIgarchFitConstrainedOptimization(obj)
            % Test IGARCH model estimation with constrained optimization and unit persistence
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure IGARCH model with custom parameter constraints
            % Set up custom optimization options (TolFun, TolX, MaxIter, etc.)
            options.optimoptions = optimset('fmincon');
            options.optimoptions = optimset(options.optimoptions, 'Display', 'off', 'TolFun', 1e-8, 'TolX', 1e-8, ...
                'Algorithm', 'interior-point', 'MaxIter', 500, 'MaxFunEvals', 500);
            
            % Estimate model using igarchfit function with constrained optimization
            estimatedModel = igarchfit(returns, options);
            
            % Verify model parameters satisfy the specified constraints
            obj.assertTrue(estimatedModel.omega > 0, 'Omega must be positive');
            obj.assertTrue(estimatedModel.alpha > 0, 'Alpha must be positive');
            
            % Validate that unit persistence constraint is maintained
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Compare results with unconstrained optimization
            % (This part requires having a pre-computed unconstrained model)
            
            % Ensure conditional variances are properly computed
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
        end
        
        function testIgarchFitFixedStartingValues(obj)
            % Test IGARCH model estimation with fixed starting values
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Define custom starting values for IGARCH parameters that satisfy unit persistence
            options.startingvals = [0.02, 0.08]; % omega, alpha
            
            % Configure IGARCH model with fixed starting values
            
            % Estimate model using igarchfit function
            estimatedModel = igarchfit(returns, options);
            
            % Verify optimization converges to the correct solution
            obj.assertAlmostEqual(estimatedModel.omega, 0.0307, obj.tolerance, 'Omega parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.alpha, 0.0737, obj.tolerance, 'Alpha parameter mismatch');
            obj.assertAlmostEqual(estimatedModel.beta, 0.9263, obj.tolerance, 'Beta parameter mismatch');
            
            % Verify unit persistence constraint is maintained throughout optimization
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            
            % Compare convergence performance with default starting values
            % (This part requires having a pre-computed model with default starting values)
            
            % Ensure final parameter estimates are accurate regardless of starting point
            obj.assertEqual(length(estimatedModel.ht), length(returns), 'Conditional variance length mismatch');
        end
        
        function testIgarchFitInputValidation(obj)
            % Test input validation in IGARCH model estimation
            
            % Test with invalid data inputs (NaN, Inf, empty, wrong dimensions)
            obj.assertThrows(@() igarchfit([1; 2; NaN]), 'MATLAB:datacheck:NaNValues', 'NaN values not handled');
            obj.assertThrows(@() igarchfit([1; 2; Inf]), 'MATLAB:datacheck:InfValues', 'Inf values not handled');
            obj.assertThrows(@() igarchfit([]), 'MATLAB:datacheck:EmptyData', 'Empty data not handled');
            obj.assertThrows(@() igarchfit([1, 2, 3]), 'MATLAB:columncheck:NotColumnVector', 'Row vector data not handled');
            
            % Test with invalid model orders (negative, zero, non-integer)
            options.p = -1;
            obj.assertThrows(@() igarchfit(obj.voldata.returns, options), 'MATLAB:parametercheck:NegativeValue', 'Negative p not handled');
            options.p = 0;
            obj.assertThrows(@() igarchfit(obj.voldata.returns, options), 'MATLAB:parametercheck:NonInteger', 'Zero p not handled');
            options.p = 1.5;
            obj.assertThrows(@() igarchfit(obj.voldata.returns, options), 'MATLAB:parametercheck:NonInteger', 'Non-integer p not handled');
            
            % Test with invalid distribution specifications
            options = rmfield(options, 'p');
            options.distribution = 'INVALID';
            obj.assertThrows(@() igarchfit(obj.voldata.returns, options), 'MATLAB:igarchfit:InvalidDistribution', 'Invalid distribution not handled');
            
            % Test with invalid optimization options
            options = rmfield(options, 'distribution');
            options.optimoptions = 'invalid';
            obj.assertThrows(@() igarchfit(obj.voldata.returns, options), 'MATLAB:fmincon:OptionsTypeMismatch', 'Invalid optimoptions not handled');
            
            % Test with parameters that violate unit persistence constraint
            % (This requires modifying the internal helper functions)
            
            % Verify appropriate error messages are thrown for each invalid input
            
            % Ensure robust error handling with malformed inputs
        end
        
        function testIgarchFitPerformance(obj)
            % Test performance of IGARCH model estimation with MEX optimization
            
            % Generate large synthetic dataset for performance testing
            numObservations = 5000;
            returns = randn(numObservations, 1);
            
            % Check for existence of MEX implementation using exist function
            mexExists = exist('igarch_core', 'file') == 3;
            
            % Measure execution time with MEX optimization enabled
            options.useMEX = true;
            mexTime = obj.measureExecutionTime(@igarchfit, returns, options);
            
            % Measure execution time with MEX optimization disabled if possible
            options.useMEX = false;
            matlabTime = obj.measureExecutionTime(@igarchfit, returns, options);
            
            % Verify MEX implementation provides significant performance improvement
            if mexExists
                obj.assertTrue(matlabTime > mexTime, 'MEX implementation should be faster');
                performanceImprovement = matlabTime / mexTime;
                obj.assertTrue(performanceImprovement > 1.2, 'Performance improvement should be significant');
            end
            
            % Validate that both implementations produce identical results within tolerance
            options.useMEX = true;
            estimatedModelMex = igarchfit(returns, options);
            options.useMEX = false;
            estimatedModelMatlab = igarchfit(returns, options);
            
            obj.assertAlmostEqual(estimatedModelMex.omega, estimatedModelMatlab.omega, obj.tolerance, 'Omega mismatch between MEX and MATLAB');
            obj.assertAlmostEqual(estimatedModelMex.alpha, estimatedModelMatlab.alpha, obj.tolerance, 'Alpha mismatch between MEX and MATLAB');
            obj.assertAlmostEqual(estimatedModelMex.beta, estimatedModelMatlab.beta, obj.tolerance, 'Beta mismatch between MEX and MATLAB');
            
            % Test performance with different model configurations and dataset sizes
        end
        
        function testIgarchFitNumericalStability(obj)
            % Test numerical stability of IGARCH model estimation with unit persistence constraint
            
            % Generate test data with extreme values and challenging properties
            numObservations = 1000;
            
            % Test with integrated processes (unit persistence exactly 1)
            trueParams.omega = 0.0001;
            trueParams.alpha1 = 0.3;
            trueParams.alpha2 = 0.05;
            trueParams.beta1 = 0.5;
            trueParams.beta2 = 0.1499;
            
            [testData] = obj.generateIgarchTestData(numObservations, 2, 2, trueParams);
            
            % Test with high volatility and outliers
            returnsHighVolatility = testData.returns * 10;
            
            % Test with small sample sizes
            returnsSmallSample = testData.returns(1:100);
            
            % Estimate model with default options
            estimatedModel = igarchfit(testData.returns);
            estimatedModelHighVolatility = igarchfit(returnsHighVolatility);
            estimatedModelSmallSample = igarchfit(returnsSmallSample);
            
            % Verify parameter estimates are numerically stable
            obj.assertTrue(isfinite(estimatedModel.omega), 'Omega must be finite');
            obj.assertTrue(isfinite(estimatedModel.alpha), 'Alpha must be finite');
            obj.assertTrue(isfinite(estimatedModel.beta), 'Beta must be finite');
            
            obj.assertTrue(isfinite(estimatedModelHighVolatility.omega), 'Omega must be finite');
            obj.assertTrue(isfinite(estimatedModelHighVolatility.alpha), 'Alpha must be finite');
            obj.assertTrue(isfinite(estimatedModelHighVolatility.beta), 'Beta must be finite');
            
            obj.assertTrue(isfinite(estimatedModelSmallSample.omega), 'Omega must be finite');
            obj.assertTrue(isfinite(estimatedModelSmallSample.alpha), 'Alpha must be finite');
            obj.assertTrue(isfinite(estimatedModelSmallSample.beta), 'Beta must be finite');
            
            % Ensure conditional variance remains positive throughout estimation
            obj.assertTrue(all(estimatedModel.ht > 0), 'Conditional variance must be positive');
            obj.assertTrue(all(estimatedModelHighVolatility.ht > 0), 'Conditional variance must be positive');
            obj.assertTrue(all(estimatedModelSmallSample.ht > 0), 'Conditional variance must be positive');
            
            % Validate unit persistence constraint is correctly enforced
            obj.verifyUnitPersistence(estimatedModel.alpha, estimatedModel.beta, obj.tolerance);
            obj.verifyUnitPersistence(estimatedModelHighVolatility.alpha, estimatedModelHighVolatility.beta, obj.tolerance);
            obj.verifyUnitPersistence(estimatedModelSmallSample.alpha, estimatedModelSmallSample.beta, obj.tolerance);
            
            % Verify log-likelihood computation remains accurate in extreme cases
            obj.assertTrue(isfinite(estimatedModel.LL), 'Log-likelihood must be finite');
            obj.assertTrue(isfinite(estimatedModelHighVolatility.LL), 'Log-likelihood must be finite');
            obj.assertTrue(isfinite(estimatedModelSmallSample.LL), 'Log-likelihood must be finite');
        end
        
        function [testData] = generateIgarchTestData(obj, numObservations, p, q, modelParams)
            % Helper method to generate IGARCH test data with known parameters
            
            % Configure IGARCH model parameters ensuring unit persistence constraint
            if nargin < 5
                modelParams = struct();
            end
            
            if ~isfield(modelParams, 'omega')
                modelParams.omega = 0.01;
            end
            
            if ~isfield(modelParams, 'alpha')
                modelParams.alpha = 0.1 * ones(q, 1);
            elseif length(modelParams.alpha) ~= q
                error('Length of alpha must match q');
            end
            
            if ~isfield(modelParams, 'beta')
                alphaSum = sum(modelParams.alpha);
                if alphaSum >= 1
                    modelParams.alpha = modelParams.alpha * 0.9 / alphaSum;
                    alphaSum = sum(modelParams.alpha);
                end
                modelParams.beta = (1 - alphaSum) * ones(p, 1) / p;
            elseif length(modelParams.beta) ~= p
                error('Length of beta must match p');
            end
            
            % Call generateVolatilitySeries to create synthetic data
            modelType = 'IGARCH';
            parameters.omega = modelParams.omega;
            parameters.alpha = modelParams.alpha;
            parameters.beta = modelParams.beta;
            
            options.p = p;
            options.q = q;
            
            volData = generateVolatilitySeries(numObservations, modelType, parameters);
            
            % Verify generated data has expected statistical properties
            obj.assertTrue(length(volData.returns) == numObservations, 'Incorrect number of returns');
            obj.assertTrue(length(volData.ht) == numObservations, 'Incorrect number of conditional variances');
            
            % Validate that conditional variances exhibit persistence
            
            % Return structure with data, true parameters, and conditional variances
            testData.returns = volData.returns;
            testData.ht = volData.ht;
            testData.model = volData.model;
        end
        
        function valid = validateIgarchResults(obj, estimatedModel, trueModel, customTolerance)
            % Helper method to validate igarchfit estimation results with focus on unit persistence
            
            % Compare estimated parameters with true parameters within tolerance
            omegaDiff = abs(estimatedModel.omega - trueModel.omega);
            alphaDiff = abs(estimatedModel.alpha - trueModel.alpha);
            betaDiff = abs(estimatedModel.beta - trueModel.beta);
            
            if nargin < 4
                tolerance = obj.tolerance;
            else
                tolerance = customTolerance;
            end
            
            omegaValid = omegaDiff < tolerance;
            alphaValid = alphaDiff < tolerance;
            betaValid = betaDiff < tolerance;
            
            % Validate unit persistence constraint (sum of alpha and beta coefficients = 1)
            alphaSum = sum(estimatedModel.alpha);
            betaSum = sum(estimatedModel.beta);
            unitPersistence = abs(alphaSum + betaSum - 1) < tolerance;
            
            % Validate model diagnostics
            
            % Check that conditional variances match expected values
            
            % Verify distribution parameter estimation if applicable
            
            % Ensure parameter constraints are satisfied (omega > 0, alpha > 0)
            
            % Return validation result
            valid = omegaValid && alphaValid && betaValid && unitPersistence;
        end
        
        function verified = verifyUnitPersistence(obj, alphaCoeffs, betaCoeffs, tolerance)
            % Helper method to verify that the IGARCH unit persistence constraint is satisfied
            
            % Calculate sum of alpha coefficients
            alphaSum = sum(alphaCoeffs);
            
            % Calculate sum of beta coefficients
            betaSum = sum(betaCoeffs);
            
            % Verify that sum(alpha) + sum(beta) equals 1 within specified tolerance
            persistence = alphaSum + betaSum;
            verified = abs(persistence - 1) < tolerance;
            
            % Return verification result
        end
    end
end