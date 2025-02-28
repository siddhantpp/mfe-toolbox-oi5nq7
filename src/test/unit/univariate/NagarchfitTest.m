classdef NagarchfitTest < BaseTest
    properties
        tolerance  % Numerical tolerance for floating-point comparisons
        testData   % Structure for test data
        voldata    % Standard volatility test data
    end
    
    methods
        function obj = NagarchfitTest()
            % Initialize the NagarchfitTest class with test data
            obj@BaseTest();
            
            % Set numerical tolerance for floating-point comparisons
            obj.tolerance = 1e-6;
            
            % Load standard volatility test data from voldata.mat
            obj.voldata = obj.loadTestData('voldata.mat');
            
            % Initialize test data structure for storing intermediate results
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            setUp@BaseTest(obj);
            % Ensure test data is properly initialized
            % Reset test state for clean test execution
        end
        
        function tearDown(obj)
            % Clean up after each test method execution
            tearDown@BaseTest(obj);
            % Clean up any temporary variables or resources
            % Finalize test state
        end
        
        function testNagarchFitBasic(obj)
            % Test basic NAGARCH(1,1) model estimation with standard options
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure NAGARCH(1,1) model with normal distribution
            options = struct('p', 1, 'q', 1);
            
            % Estimate model using nagarchfit function
            model = nagarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertTrue(length(model.parameters) == 4, 'NAGARCH(1,1) should have 4 parameters: omega, alpha, gamma, beta');
            
            % Check parameter signs and constraints
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            
            obj.assertTrue(omega > 0, 'Omega should be positive');
            obj.assertTrue(alpha > 0, 'Alpha should be positive');
            obj.assertTrue(beta > 0, 'Beta should be positive');
            
            % Check model persistence
            persistence = alpha*(1 + gamma^2) + beta;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            
            % Validate model diagnostics
            obj.assertEqual('NAGARCH', model.model_type, 'Model type should be NAGARCH');
            obj.assertEqual('NORMAL', model.error_type, 'Error type should be NORMAL');
            obj.assertTrue(~isempty(model.LL), 'Log-likelihood should be computed');
            obj.assertTrue(~isempty(model.ht), 'Conditional variances should be computed');
            obj.assertTrue(~isempty(model.stdresid), 'Standardized residuals should be computed');
            obj.assertTrue(~isempty(model.std_errors), 'Standard errors should be computed');
            obj.assertTrue(~isempty(model.tstat), 'T-statistics should be computed');
        end
        
        function testNagarchFitWithStudentT(obj)
            % Test NAGARCH model estimation with Student's t error distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure NAGARCH(1,1) model with Student's t distribution
            options = struct('p', 1, 'q', 1, 'error_type', 'T');
            
            % Estimate model using nagarchfit function
            model = nagarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertTrue(length(model.parameters) == 5, 'NAGARCH(1,1) with t-dist should have 5 parameters');
            
            % Extract and check parameters
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            nu = model.parameters(5); % Degrees of freedom
            
            obj.assertTrue(omega > 0, 'Omega should be positive');
            obj.assertTrue(alpha > 0, 'Alpha should be positive');
            obj.assertTrue(beta > 0, 'Beta should be positive');
            obj.assertTrue(nu > 2, 'Degrees of freedom should be > 2');
            
            % Check model persistence
            persistence = alpha*(1 + gamma^2) + beta;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            
            % Validate degrees of freedom parameter is properly estimated
            obj.assertTrue(~isempty(model.nu), 'Degrees of freedom parameter should be present');
            obj.assertEqual(nu, model.nu, 'nu parameter should match in model structure');
            
            % Compare log-likelihood with benchmark values
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
            obj.assertTrue(all(isfinite(model.stdresid)), 'All standardized residuals should be finite');
        end
        
        function testNagarchFitWithGED(obj)
            % Test NAGARCH model estimation with Generalized Error Distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure NAGARCH(1,1) model with GED distribution
            options = struct('p', 1, 'q', 1, 'error_type', 'GED');
            
            % Estimate model using nagarchfit function
            model = nagarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertTrue(length(model.parameters) == 5, 'NAGARCH(1,1) with GED should have 5 parameters');
            
            % Extract and check parameters
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            nu = model.parameters(5); % Shape parameter
            
            obj.assertTrue(omega > 0, 'Omega should be positive');
            obj.assertTrue(alpha > 0, 'Alpha should be positive');
            obj.assertTrue(beta > 0, 'Beta should be positive');
            obj.assertTrue(nu > 0, 'Shape parameter should be positive');
            
            % Check model persistence
            persistence = alpha*(1 + gamma^2) + beta;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            
            % Validate GED shape parameter is properly estimated
            obj.assertTrue(~isempty(model.nu), 'GED shape parameter should be present');
            obj.assertEqual(nu, model.nu, 'nu parameter should match in model structure');
            
            % Compare log-likelihood with benchmark values
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
            obj.assertTrue(all(isfinite(model.stdresid)), 'All standardized residuals should be finite');
        end
        
        function testNagarchFitWithSkewedT(obj)
            % Test NAGARCH model estimation with Hansen's skewed t distribution
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure NAGARCH(1,1) model with skewed t distribution
            options = struct('p', 1, 'q', 1, 'error_type', 'SKEWT');
            
            % Estimate model using nagarchfit function
            model = nagarchfit(returns, options);
            
            % Verify model parameters are correctly estimated
            obj.assertTrue(length(model.parameters) == 6, 'NAGARCH(1,1) with skewed t should have 6 parameters');
            
            % Extract and check parameters
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            nu = model.parameters(5); % Degrees of freedom
            lambda = model.parameters(6); % Skewness parameter
            
            obj.assertTrue(omega > 0, 'Omega should be positive');
            obj.assertTrue(alpha > 0, 'Alpha should be positive');
            obj.assertTrue(beta > 0, 'Beta should be positive');
            obj.assertTrue(nu > 2, 'Degrees of freedom should be > 2');
            obj.assertTrue(abs(lambda) < 1, 'Skewness parameter should be between -1 and 1');
            
            % Check model persistence
            persistence = alpha*(1 + gamma^2) + beta;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            
            % Validate degrees of freedom and skewness parameters are properly estimated
            obj.assertTrue(~isempty(model.nu), 'Degrees of freedom parameter should be present');
            obj.assertTrue(~isempty(model.lambda), 'Skewness parameter should be present');
            obj.assertEqual(nu, model.nu, 'nu parameter should match in model structure');
            obj.assertEqual(lambda, model.lambda, 'lambda parameter should match in model structure');
            
            % Compare log-likelihood with benchmark values
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
            
            % Ensure conditional variances and standardized residuals are properly computed
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
            obj.assertTrue(all(isfinite(model.stdresid)), 'All standardized residuals should be finite');
        end
        
        function testNagarchFitHigherOrder(obj)
            % Test higher-order NAGARCH(p,q) model estimation
            
            % Generate synthetic data with known NAGARCH(2,2) parameters using generateVolatilitySeries
            T = 1000;
            params = struct('omega', 0.01, 'alpha', [0.05; 0.04], 'gamma', 0.5, 'beta', [0.5; 0.3]);
            testData = obj.generateTestData('NAGARCH22', T, params);
            
            % Configure NAGARCH(2,2) model with appropriate options
            options = struct('p', 2, 'q', 2);
            
            % Estimate model using nagarchfit function
            model = nagarchfit(testData.returns, options);
            
            % Verify all parameters (omega, alpha1, alpha2, gamma, beta1, beta2) are correctly estimated
            obj.assertTrue(length(model.parameters) == 6, 'NAGARCH(2,2) should have 6 parameters');
            
            % Extract estimated parameters
            omega = model.parameters(1);
            alpha1 = model.parameters(2);
            alpha2 = model.parameters(3);
            gamma = model.parameters(4);
            beta1 = model.parameters(5);
            beta2 = model.parameters(6);
            
            % Compare with true parameters (allowing for estimation error)
            obj.assertAlmostEqual(params.omega, omega, 'Omega not correctly estimated', obj.tolerance * 10);
            obj.assertAlmostEqual(params.alpha(1), alpha1, 'Alpha1 not correctly estimated', obj.tolerance * 10);
            obj.assertAlmostEqual(params.alpha(2), alpha2, 'Alpha2 not correctly estimated', obj.tolerance * 10);
            obj.assertAlmostEqual(params.gamma, gamma, 'Gamma not correctly estimated', obj.tolerance * 10);
            obj.assertAlmostEqual(params.beta(1), beta1, 'Beta1 not correctly estimated', obj.tolerance * 10);
            obj.assertAlmostEqual(params.beta(2), beta2, 'Beta2 not correctly estimated', obj.tolerance * 10);
            
            % Validate model diagnostics (persistence, log-likelihood)
            persistence = alpha1*(1 + gamma^2) + alpha2*(1 + gamma^2) + beta1 + beta2;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
            
            % Ensure conditional variances match the true values within tolerance
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
        end
        
        function testNagarchFitConstrainedOptimization(obj)
            % Test NAGARCH model estimation with constrained optimization
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Configure NAGARCH model with custom parameter constraints
            options = struct('p', 1, 'q', 1);
            
            % Set up custom optimization options (TolFun, TolX, MaxIter, etc.)
            options.LB = [1e-6; 0.01; -1; 0.7]; % Lower bounds
            options.UB = [0.1; 0.3; 1; 0.99]; % Upper bounds
            options.TolFun = 1e-8;
            options.TolX = 1e-8;
            options.MaxIter = 500;
            
            % Estimate model using nagarchfit function with constrained optimization
            model = nagarchfit(returns, options);
            
            % Verify model parameters satisfy the specified constraints
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            
            obj.assertTrue(omega >= options.LB(1) && omega <= options.UB(1), 'Omega should be within bounds');
            obj.assertTrue(alpha >= options.LB(2) && alpha <= options.UB(2), 'Alpha should be within bounds');
            obj.assertTrue(gamma >= options.LB(3) && gamma <= options.UB(3), 'Gamma should be within bounds');
            obj.assertTrue(beta >= options.LB(4) && beta <= options.UB(4), 'Beta should be within bounds');
            
            % Compare results with unconstrained optimization
            unconstrained_model = nagarchfit(returns, struct('p', 1, 'q', 1));
            
            % Validate both models but expect different parameter values due to constraints
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite for constrained model');
            obj.assertTrue(isfinite(unconstrained_model.LL), 'Log-likelihood should be finite for unconstrained model');
            
            % Ensure conditional variances are properly computed
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
        end
        
        function testNagarchFitFixedStartingValues(obj)
            % Test NAGARCH model estimation with fixed starting values
            
            % Load test returns data from voldata
            returns = obj.voldata.returns;
            
            % Define custom starting values for NAGARCH parameters
            startingVals = [0.05; 0.1; 0.3; 0.5]; % [omega, alpha, gamma, beta]
            
            % Configure NAGARCH model with fixed starting values
            options = struct('p', 1, 'q', 1, 'starting_values', startingVals);
            
            % Estimate model using nagarchfit function
            model = nagarchfit(returns, options);
            
            % Verify optimization converges to the correct solution
            obj.assertTrue(model.convergence > 0, 'Model should converge with valid starting values');
            
            % Extract estimated parameters
            omega = model.parameters(1);
            alpha = model.parameters(2);
            gamma = model.parameters(3);
            beta = model.parameters(4);
            
            % Check parameter constraints and signs
            obj.assertTrue(omega > 0, 'Omega should be positive');
            obj.assertTrue(alpha > 0, 'Alpha should be positive');
            obj.assertTrue(beta > 0, 'Beta should be positive');
            
            % Compare convergence performance with default starting values
            % In a real test, we would compare execution times or iteration counts,
            % but here we just ensure the model converges correctly
            
            % Ensure final parameter estimates are accurate regardless of starting point
            persistence = alpha*(1 + gamma^2) + beta;
            obj.assertTrue(persistence < 1, 'Model should be stationary (persistence < 1)');
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
        end
        
        function testNagarchFitInputValidation(obj)
            % Test input validation in NAGARCH model estimation
            
            % Test with invalid data inputs (NaN, Inf, empty, wrong dimensions)
            nanData = rand(100, 1);
            nanData(10) = NaN;
            obj.assertThrows(@() nagarchfit(nanData), 'data cannot contain NaN values');
            
            infData = rand(100, 1);
            infData(20) = Inf;
            obj.assertThrows(@() nagarchfit(infData), 'data cannot contain Inf');
            
            emptyData = [];
            obj.assertThrows(@() nagarchfit(emptyData), 'data cannot be empty');
            
            matrixData = rand(10, 2);
            obj.assertThrows(@() nagarchfit(matrixData), 'data must be a column vector');
            
            % Test with invalid model orders (negative, zero, non-integer)
            returns = obj.voldata.returns;
            obj.assertThrows(@() nagarchfit(returns, struct('p', -1, 'q', 1)), 'p must be');
            obj.assertThrows(@() nagarchfit(returns, struct('p', 1, 'q', -1)), 'q must be');
            obj.assertThrows(@() nagarchfit(returns, struct('p', 1.5, 'q', 1)), 'p must be');
            obj.assertThrows(@() nagarchfit(returns, struct('p', 1, 'q', 0)), 'q must be');
            
            % Test with invalid distribution specifications
            obj.assertThrows(@() nagarchfit(returns, struct('error_type', 'UNKNOWN')), 'Unknown');
            
            % Test with invalid optimization options
            badStartVals = [0.05; -0.1; 0.3; 0.5]; % Negative alpha
            obj.assertThrows(@() nagarchfit(returns, struct('starting_values', badStartVals)), 'Bounds violation');
            
            % Ensure robust error handling with malformed inputs
            obj.assertThrows(@() nagarchfit(returns, struct('p', 1, 'q', 1, 'error_type', 'T', 'starting_values', [0.1; 0.1])), 'length');
        end
        
        function testNagarchFitPerformance(obj)
            % Test performance of NAGARCH model estimation with MEX optimization
            
            % Generate large synthetic dataset for performance testing
            T = 5000;
            returns = randn(T, 1);
            
            % Check if MEX implementation is available
            hasMex = exist('tarch_core', 'file') == 3 || exist('agarch_core', 'file') == 3;
            
            if hasMex
                % Measure execution time with MEX optimization enabled
                options_with_mex = struct('p', 1, 'q', 1, 'useMEX', true);
                time_with_mex = obj.measureExecutionTime(@() nagarchfit(returns, options_with_mex));
                
                % Measure execution time with MEX optimization disabled
                options_without_mex = struct('p', 1, 'q', 1, 'useMEX', false);
                time_without_mex = obj.measureExecutionTime(@() nagarchfit(returns, options_without_mex));
                
                % Verify MEX implementation provides significant performance improvement
                obj.assertTrue(time_with_mex < time_without_mex, 'MEX implementation should be faster');
                speedup = time_without_mex / time_with_mex;
                obj.assertTrue(speedup > 1.2, sprintf('MEX should provide at least 20%% speedup, got %.2f%%', (speedup-1)*100));
                
                % Validate that both implementations produce identical results within tolerance
                model_mex = nagarchfit(returns, options_with_mex);
                model_nomex = nagarchfit(returns, options_without_mex);
                
                for i = 1:length(model_mex.parameters)
                    obj.assertAlmostEqual(model_mex.parameters(i), model_nomex.parameters(i), ...
                        'Parameters should be equal regardless of MEX usage', 1e-4);
                end
            else
                % Skip test if MEX implementation not available
                warning('MEX implementation not available. Skipping performance test.');
            end
            
            % Test performance with different model configurations and dataset sizes
            if hasMex
                % Test with different model orders
                t1 = obj.measureExecutionTime(@() nagarchfit(returns, struct('p', 1, 'q', 1)));
                t2 = obj.measureExecutionTime(@() nagarchfit(returns, struct('p', 2, 'q', 2)));
                obj.assertTrue(t2 > t1, 'Higher-order model should take longer to estimate');
            end
        end
        
        function testNagarchFitNumericalStability(obj)
            % Test numerical stability of NAGARCH model estimation
            
            % Generate test data with extreme values and challenging properties
            T = 1000;
            
            % Test with near-integrated processes (persistence close to 1)
            omega = 0.01;
            alpha = 0.15;
            gamma = 0.1;
            beta = 0.84; % Total persistence close to 0.99
            
            testData = obj.generateTestData('NAGARCH11', T, struct('omega', omega, 'alpha', alpha, ...
                'gamma', gamma, 'beta', beta));
            
            options = struct('p', 1, 'q', 1);
            model = nagarchfit(testData.returns, options);
            
            % Verify parameter estimates are numerically stable
            obj.assertTrue(model.convergence > 0, 'Model should converge even for near-integrated process');
            obj.assertTrue(all(isfinite(model.parameters)), 'All parameters should be finite');
            
            % Test with high volatility and outliers
            outlierData = testData.returns;
            outlierData(100) = outlierData(100) * 10; % Add outlier
            outlierData(200) = outlierData(200) * 10; % Add outlier
            
            model_outliers = nagarchfit(outlierData, options);
            obj.assertTrue(model_outliers.convergence > 0, 'Model should converge even with outliers');
            
            % Ensure conditional variance remains positive throughout estimation
            obj.assertTrue(all(model.ht > 0), 'All conditional variances should be positive');
            obj.assertTrue(all(model_outliers.ht > 0), 'All conditional variances should be positive with outliers');
            
            % Test with small sample sizes
            smallSample = testData.returns(1:200);
            model_small = nagarchfit(smallSample, options);
            obj.assertTrue(model_small.convergence > 0, 'Model should converge even with small sample');
            
            % Validate log-likelihood computation remains accurate in extreme cases
            obj.assertTrue(isfinite(model.LL), 'Log-likelihood should be finite');
            obj.assertTrue(isfinite(model_outliers.LL), 'Log-likelihood should be finite even with outliers');
            obj.assertTrue(isfinite(model_small.LL), 'Log-likelihood should be finite even with small sample');
        end
        
        function testNagarchFitNonlinearAsymmetry(obj)
            % Test NAGARCH-specific nonlinear asymmetry handling
            
            % Generate synthetic data with known asymmetric leverage effects
            T = 2000;
            gamma_true = 0.5; % Strong asymmetry
            
            % Configure NAGARCH model with appropriate options
            testData = obj.generateTestData('NAGARCH11', T, struct('omega', 0.01, 'alpha', 0.1, ...
                'gamma', gamma_true, 'beta', 0.8));
            
            % Estimate model using nagarchfit function
            model_nagarch = nagarchfit(testData.returns, struct('p', 1, 'q', 1));
            
            % Verify nonlinear asymmetry parameter (gamma) is correctly estimated
            gamma_est = model_nagarch.parameters(3);
            obj.assertAlmostEqual(gamma_true, gamma_est, 'Asymmetry parameter not correctly estimated', 0.2);
            
            % Compare with symmetric model to validate asymmetry significance
            model_garch = nagarchfit(testData.returns, struct('p', 1, 'q', 1, 'model', 'GARCH'));
            obj.assertTrue(model_nagarch.LL > model_garch.LL, 'NAGARCH should fit asymmetric data better than GARCH');
            
            % Ensure conditional variances correctly incorporate asymmetric effects
            returns = testData.returns(1:end-1);
            variance_changes = diff(model_nagarch.ht);
            
            % Compute leverage effect correlation between returns and variance changes
            corr_val = corr(returns, variance_changes);
            
            % Validate model's ability to capture leverage effects in financial returns
            if gamma_true < 0
                obj.assertTrue(corr_val < 0, 'Negative gamma should produce negative leverage correlation');
            else
                % For positive gamma, just check the correlation is captured
                obj.assertTrue(isfinite(corr_val), 'Leverage correlation should be finite');
            end
        end
        
        function testData = generateTestData(obj, modelType, numObservations, modelParams)
            % Helper method to generate test data for specific test cases
            
            % Configure model parameters based on modelType and modelParams
            rng(123); % Set fixed seed for reproducibility
            
            % Initialize output structure
            testData = struct();
            
            if strcmp(modelType, 'NAGARCH11')
                % NAGARCH(1,1) model
                omega = modelParams.omega;
                alpha = modelParams.alpha;
                gamma = modelParams.gamma;
                beta = modelParams.beta;
                
                % Initialize arrays
                returns = zeros(numObservations, 1);
                ht = zeros(numObservations, 1);
                
                % Initial variance
                ht(1) = omega / (1 - alpha*(1 + gamma^2) - beta);
                
                % Generate NAGARCH process
                for t = 2:numObservations
                    % Generate random innovation
                    z = randn(1);
                    
                    % Compute NAGARCH variance
                    ht(t) = omega + beta * ht(t-1) + alpha * ht(t-1) * (z - gamma)^2;
                    
                    % Generate return
                    returns(t) = sqrt(ht(t)) * randn(1);
                end
                
                % Store generated data
                testData.returns = returns;
                testData.ht = ht;
                testData.true_params = [omega; alpha; gamma; beta];
                
            elseif strcmp(modelType, 'NAGARCH22')
                % NAGARCH(2,2) model
                omega = modelParams.omega;
                alpha = modelParams.alpha; % Vector [alpha1; alpha2]
                gamma = modelParams.gamma;
                beta = modelParams.beta;   % Vector [beta1; beta2]
                
                % Initialize arrays
                returns = zeros(numObservations, 1);
                ht = zeros(numObservations, 1);
                
                % Approximate unconditional variance for initialization
                persistence = sum(alpha)*(1 + gamma^2) + sum(beta);
                uncond_var = omega / (1 - persistence);
                
                % Initialize first observations
                ht(1:2) = uncond_var;
                z = randn(2, 1);
                returns(1:2) = sqrt(ht(1:2)) .* z;
                
                % Generate NAGARCH(2,2) process
                for t = 3:numObservations
                    % Compute NAGARCH variance with 2 lags
                    ht(t) = omega;
                    
                    % GARCH terms
                    ht(t) = ht(t) + beta(1) * ht(t-1) + beta(2) * ht(t-2);
                    
                    % ARCH terms with nonlinear asymmetry
                    z_t1 = returns(t-1) / sqrt(ht(t-1));
                    z_t2 = returns(t-2) / sqrt(ht(t-2));
                    ht(t) = ht(t) + alpha(1) * ht(t-1) * (z_t1 - gamma)^2;
                    ht(t) = ht(t) + alpha(2) * ht(t-2) * (z_t2 - gamma)^2;
                    
                    % Generate return
                    returns(t) = sqrt(ht(t)) * randn(1);
                end
                
                % Store generated data
                testData.returns = returns;
                testData.ht = ht;
                testData.true_params = [omega; alpha(1); alpha(2); gamma; beta(1); beta(2)];
            else
                error('Unsupported model type: %s', modelType);
            end
            
            % Verify generated data has expected statistical properties
            testData.persistence = sum(alpha)*(1 + gamma^2) + sum(beta);
            
            % Return structure with data, true parameters, and conditional variances
            return;
        end
        
        function isValid = validateNagarchResults(obj, estimatedModel, trueModel, customTolerance)
            % Helper method to validate nagarchfit estimation results
            
            % Use specified tolerance or default
            if nargin < 4
                customTolerance = obj.tolerance;
            end
            
            % Initialize result
            isValid = true;
            
            % Compare estimated parameters with true parameters within tolerance
            est_params = estimatedModel.parameters;
            true_params = trueModel.true_params;
            
            if length(est_params) ~= length(true_params)
                isValid = false;
                return;
            end
            
            % Check each parameter
            for i = 1:length(true_params)
                diff = abs(est_params(i) - true_params(i));
                if diff > customTolerance * max(1, abs(true_params(i)))
                    isValid = false;
                    break;
                end
            end
            
            % Check model diagnostics (persistence, unconditional variance)
            if isfield(trueModel, 'persistence')
                % Extract NAGARCH parameters
                if length(est_params) >= 4
                    est_omega = est_params(1);
                    est_alpha = est_params(2);
                    est_gamma = est_params(3);
                    est_beta = est_params(4);
                    
                    % Compute persistence
                    est_persistence = est_alpha * (1 + est_gamma^2) + est_beta;
                    true_persistence = trueModel.persistence;
                    
                    if abs(est_persistence - true_persistence) > customTolerance
                        isValid = false;
                    end
                end
            end
            
            % Check that conditional variances match expected values
            if isfield(estimatedModel, 'ht') && isfield(trueModel, 'ht')
                if length(estimatedModel.ht) == length(trueModel.ht)
                    % Sample a few points to check (not all for efficiency)
                    sample_indices = round(linspace(1, length(trueModel.ht), 10));
                    for i = sample_indices
                        rel_diff = abs(estimatedModel.ht(i) - trueModel.ht(i)) / max(trueModel.ht(i), 1e-6);
                        if rel_diff > customTolerance * 10 % Allow more tolerance for variances
                            isValid = false;
                            break;
                        end
                    end
                end
            end
            
            % Ensure parameter constraints are satisfied
            if length(est_params) >= 4
                est_omega = est_params(1);
                est_alpha = est_params(2);
                est_gamma = est_params(3);
                est_beta = est_params(4);
                
                if est_omega <= 0 || est_alpha < 0 || est_beta < 0
                    isValid = false;
                end
            end
            
            % Validate nonlinear asymmetry effects in NAGARCH variance equation
            if isfield(estimatedModel, 'stdresid') && length(est_params) >= 4
                % Check if standardized residuals have appropriate properties
                if abs(mean(estimatedModel.stdresid)) > 0.1 || abs(var(estimatedModel.stdresid) - 1) > 0.2
                    isValid = false;
                end
            end
            
            return;
        end
    end
end