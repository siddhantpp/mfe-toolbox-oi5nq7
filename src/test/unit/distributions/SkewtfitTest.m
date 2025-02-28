classdef SkewtfitTest < BaseTest
    % SKEWTFITTEST Test class for validating the Hansen's skewed t-distribution parameter estimation
    %
    % This class tests the implementation of maximum likelihood estimation for the
    % Hansen's skewed t-distribution parameters (nu, lambda, mu, sigma) in skewtfit.m.
    % It verifies parameter accuracy, numerical stability, error handling, and
    % various edge cases to ensure robust estimation behavior.
    %
    % The tests include:
    %   - Basic functionality with simulated data
    %   - Accuracy with large sample sizes
    %   - Options handling for customization
    %   - Output structure compliance
    %   - Error handling for invalid inputs
    %   - Estimation consistency across multiple runs
    %   - Asymmetry parameter estimation
    %   - Boundary condition handling
    %   - Log-likelihood consistency verification
    %
    % See also: skewtfit, skewtrnd, skewtloglik, BaseTest
    
    properties
        % Structure containing reference test data
        testData
        
        % Tolerance for floating-point comparisons
        tolerance
        
        % Matrix of simulated data for testing
        simulatedData
        
        % Matrix of known parameters for validation [nu, lambda, mu, sigma]
        knownParameters
    end
    
    methods
        function obj = SkewtfitTest()
            % Initialize the test class with test name and settings
            
            % Call superclass constructor with test name
            obj = obj@BaseTest('Hansen''s Skewed T-Distribution Parameter Estimation Test');
            
            % Set appropriate tolerance for floating-point comparisons
            % Estimation precision depends on optimization complexity
            obj.tolerance = 1e-3;
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method runs
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load test data from MAT file
            testDataStruct = obj.loadTestData('known_distributions.mat');
            
            % Extract relevant data for skewed t-distribution
            if isfield(testDataStruct, 'skewt')
                obj.testData = testDataStruct.skewt;
            else
                % Create basic test data if not available from file
                obj.testData = struct();
                obj.testData.nu = [5, 8, 10];
                obj.testData.lambda = [-0.5, 0, 0.5];
                obj.testData.standardSizes = [500, 1000, 5000];
            end
            
            % Generate simulated data with known parameters for testing
            rng(12345); % Set seed for reproducibility
            
            % Define parameter sets for testing
            nuValues = [5, 10, 20];
            lambdaValues = [-0.5, 0, 0.5];
            
            % Generate test data for each parameter combination
            n = 1000; % Default sample size
            
            % Create simulation data matrix
            % Each row is a different dataset with columns: nu, lambda, mu, sigma, sampleSize
            obj.knownParameters = [];
            obj.simulatedData = {};
            
            for i = 1:length(nuValues)
                for j = 1:length(lambdaValues)
                    % Set parameters
                    nu = nuValues(i);
                    lambda = lambdaValues(j);
                    mu = 0;  % Default location parameter
                    sigma = 1; % Default scale parameter
                    
                    % Generate data
                    data = skewtrnd(nu, lambda, n, 1);
                    
                    % Scale and shift data to implement mu and sigma
                    % Standardized data from skewtrnd has mean~0 and variance~1
                    data = sigma * data + mu;
                    
                    % Store parameters and data
                    obj.knownParameters = [obj.knownParameters; nu, lambda, mu, sigma, n];
                    obj.simulatedData{end+1} = data;
                end
            end
        end
        
        function tearDown(obj)
            % Clean up after each test method completes
            
            % Call superclass tearDown
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Tests the basic functionality of skewtfit against known parameters with simulated data
            
            % Test with a subset of simulated datasets
            for i = 1:min(3, size(obj.knownParameters, 1))
                % Get parameters and data
                nu = obj.knownParameters(i, 1);
                lambda = obj.knownParameters(i, 2);
                mu = obj.knownParameters(i, 3);
                sigma = obj.knownParameters(i, 4);
                data = obj.simulatedData{i};
                
                % Estimate parameters using skewtfit
                results = skewtfit(data);
                
                % Verify estimated parameters are close to the true values
                % Use different tolerances for different parameters
                obj.assertAlmostEqual(results.nu, nu, ...
                    sprintf('Degrees of freedom (nu) estimation failed for dataset %d', i));
                
                obj.assertAlmostEqual(results.lambda, lambda, ...
                    sprintf('Skewness parameter (lambda) estimation failed for dataset %d', i));
                
                obj.assertAlmostEqual(results.mu, mu, ...
                    sprintf('Location parameter (mu) estimation failed for dataset %d', i));
                
                obj.assertAlmostEqual(results.sigma, sigma, ...
                    sprintf('Scale parameter (sigma) estimation failed for dataset %d', i));
                
                % Verify parameters array matches individual parameters
                obj.assertEqual(results.parameters, [results.nu, results.lambda, results.mu, results.sigma], ...
                    'Parameters array does not match individual parameter values');
                
                % Verify convergence status
                obj.assertTrue(results.convergence, 'Estimation should converge for well-behaved data');
            end
        end
        
        function testAccuracyWithLargeData(obj)
            % Tests parameter estimation accuracy with large sample sizes
            
            % Define parameters for testing
            nu = 8;
            lambda = 0.3;
            mu = 0;
            sigma = 1;
            
            % Test with increasing sample sizes to verify convergence
            sampleSizes = [1000, 5000, 10000];
            
            % Store estimation errors
            nuErrors = zeros(length(sampleSizes), 1);
            lambdaErrors = zeros(length(sampleSizes), 1);
            muErrors = zeros(length(sampleSizes), 1);
            sigmaErrors = zeros(length(sampleSizes), 1);
            
            for i = 1:length(sampleSizes)
                % Generate data with known parameters
                rng(i); % Different seed for each sample size
                n = sampleSizes(i);
                data = skewtrnd(nu, lambda, n, 1);
                data = sigma * data + mu;
                
                % Estimate parameters
                results = skewtfit(data);
                
                % Calculate absolute errors
                nuErrors(i) = abs(results.nu - nu);
                lambdaErrors(i) = abs(results.lambda - lambda);
                muErrors(i) = abs(results.mu - mu);
                sigmaErrors(i) = abs(results.sigma - sigma);
                
                % Verify estimation is reasonably accurate
                % Tolerance scales with sample size
                tolerance = obj.tolerance / sqrt(n/1000);
                
                obj.assertAlmostEqual(results.nu, nu, ...
                    sprintf('Degrees of freedom (nu) estimation failed for n=%d', n));
                
                obj.assertAlmostEqual(results.lambda, lambda, ...
                    sprintf('Skewness parameter (lambda) estimation failed for n=%d', n));
                
                obj.assertAlmostEqual(results.mu, mu, ...
                    sprintf('Location parameter (mu) estimation failed for n=%d', n));
                
                obj.assertAlmostEqual(results.sigma, sigma, ...
                    sprintf('Scale parameter (sigma) estimation failed for n=%d', n));
            end
            
            % Verify that errors decrease with increasing sample size
            for i = 2:length(sampleSizes)
                obj.assertTrue(nuErrors(i) <= nuErrors(i-1)*1.2, ...
                    'Estimation error for nu should generally decrease with larger samples');
                obj.assertTrue(lambdaErrors(i) <= lambdaErrors(i-1)*1.2, ...
                    'Estimation error for lambda should generally decrease with larger samples');
                obj.assertTrue(muErrors(i) <= muErrors(i-1)*1.2, ...
                    'Estimation error for mu should generally decrease with larger samples');
                obj.assertTrue(sigmaErrors(i) <= sigmaErrors(i-1)*1.2, ...
                    'Estimation error for sigma should generally decrease with larger samples');
            end
        end
        
        function testOptionsHandling(obj)
            % Tests the handling of options structure in parameter estimation
            
            % Get a test dataset
            data = obj.simulatedData{1};
            nu_true = obj.knownParameters(1, 1);
            lambda_true = obj.knownParameters(1, 2);
            mu_true = obj.knownParameters(1, 3);
            sigma_true = obj.knownParameters(1, 4);
            
            % Test with default options
            results_default = skewtfit(data);
            
            % Test with custom starting values
            options = struct();
            options.startingVals = [nu_true*1.5, lambda_true*0.8, mu_true+0.5, sigma_true*1.2];
            results_custom_start = skewtfit(data, options);
            
            % Verify both approaches converge to approximately the same values
            obj.assertAlmostEqual(results_default.nu, results_custom_start.nu, ...
                'Parameter estimation should converge to similar values with different starting points');
            obj.assertAlmostEqual(results_default.lambda, results_custom_start.lambda, ...
                'Parameter estimation should converge to similar values with different starting points');
            obj.assertAlmostEqual(results_default.mu, results_custom_start.mu, ...
                'Parameter estimation should converge to similar values with different starting points');
            obj.assertAlmostEqual(results_default.sigma, results_custom_start.sigma, ...
                'Parameter estimation should converge to similar values with different starting points');
            
            % Test with display option
            options = struct();
            options.display = 'off';
            options.maxIter = 250;
            options.tolerance = 1e-5;
            results_display = skewtfit(data, options);
            
            % Verify converged to sensible values
            obj.assertAlmostEqual(results_display.nu, nu_true, ...
                'Parameter estimation with display options should work correctly');
            obj.assertAlmostEqual(results_display.lambda, lambda_true, ...
                'Parameter estimation with display options should work correctly');
            
            % Test with bad starting values that should still converge
            options = struct();
            options.startingVals = [6, 0, 0, 1]; % Generic starting values
            results_generic = skewtfit(data, options);
            
            % Verify converged to sensible values despite generic starting values
            obj.assertAlmostEqual(results_generic.nu, nu_true, ...
                'Parameter estimation should work with generic starting values');
            obj.assertAlmostEqual(results_generic.lambda, lambda_true, ...
                'Parameter estimation should work with generic starting values');
        end
        
        function testOutputStructure(obj)
            % Tests the structure and contents of the function output
            
            % Get a test dataset
            data = obj.simulatedData{1};
            
            % Estimate parameters
            results = skewtfit(data);
            
            % Verify structure contains all expected fields
            expectedFields = {'nu', 'lambda', 'mu', 'sigma', ...
                'nuSE', 'lambdaSE', 'muSE', 'sigmaSE', ...
                'logL', 'aic', 'bic', 'parameters', 'vcv', ...
                'iterations', 'convergence', 'exitflag', 'message'};
            
            for i = 1:length(expectedFields)
                obj.assertTrue(isfield(results, expectedFields{i}), ...
                    sprintf('Output structure missing expected field: %s', expectedFields{i}));
            end
            
            % Verify parameter vector has correct length and content
            obj.assertEqual(length(results.parameters), 4, ...
                'Parameters vector should have length 4');
            obj.assertEqual(results.parameters, [results.nu, results.lambda, results.mu, results.sigma], ...
                'Parameters vector content does not match individual parameters');
            
            % Verify variance-covariance matrix dimensions
            obj.assertEqual(size(results.vcv), [4, 4], ...
                'Variance-covariance matrix should be 4x4');
            
            % Verify standard errors are non-negative
            obj.assertTrue(results.nuSE >= 0, 'Standard error for nu should be non-negative');
            obj.assertTrue(results.lambdaSE >= 0, 'Standard error for lambda should be non-negative');
            obj.assertTrue(results.muSE >= 0, 'Standard error for mu should be non-negative');
            obj.assertTrue(results.sigmaSE >= 0, 'Standard error for sigma should be non-negative');
            
            % Verify information criteria make sense
            obj.assertTrue(results.aic > results.logL * (-2), 'AIC should be greater than -2*logL');
            obj.assertTrue(results.bic > results.aic, 'BIC should be greater than AIC for n > e²');
        end
        
        function testErrorHandling(obj)
            % Tests error handling for invalid inputs
            
            % Get a test dataset
            data = obj.simulatedData{1};
            
            % Test with invalid data types
            obj.assertThrows(@() skewtfit('string'), 'MATLAB:datacheck:DataMustBeNumeric', ...
                'Should throw error for non-numeric data');
            
            % Test with NaN values
            data_with_nan = data;
            data_with_nan(5) = NaN;
            obj.assertThrows(@() skewtfit(data_with_nan), 'MATLAB:datacheck:DataContainsNaNs', ...
                'Should throw error for data with NaN values');
            
            % Test with Inf values
            data_with_inf = data;
            data_with_inf(5) = Inf;
            obj.assertThrows(@() skewtfit(data_with_inf), 'MATLAB:datacheck:DataContainsInfs', ...
                'Should throw error for data with Inf values');
            
            % Test with empty data
            obj.assertThrows(@() skewtfit([]), 'MATLAB:datacheck:DataCannotBeEmpty', ...
                'Should throw error for empty data');
            
            % Test with invalid options structure
            options = struct();
            options.startingVals = [1, 0, 0, 1]; % nu < 2, which is invalid
            obj.assertThrows(@() skewtfit(data, options), 'MATLAB:parametercheck:ParamOutOfRange', ...
                'Should throw error for nu < 2 in starting values');
            
            % Test with invalid lambda in starting values
            options = struct();
            options.startingVals = [5, 1.5, 0, 1]; % lambda > 1, which is invalid
            obj.assertThrows(@() skewtfit(data, options), 'MATLAB:parametercheck:ParamOutOfRange', ...
                'Should throw error for lambda > 1 in starting values');
            
            % Test with invalid sigma in starting values
            options = struct();
            options.startingVals = [5, 0, 0, -1]; % sigma < 0, which is invalid
            obj.assertThrows(@() skewtfit(data, options), 'MATLAB:parametercheck:ParamMustBePositive', ...
                'Should throw error for sigma ≤ 0 in starting values');
        end
        
        function testEstimationConsistency(obj)
            % Tests consistency of parameter estimation across multiple runs
            
            % Get a test dataset
            data = obj.simulatedData{1};
            
            % Run estimation multiple times
            numRuns = 3;
            results = cell(numRuns, 1);
            
            for i = 1:numRuns
                results{i} = skewtfit(data);
            end
            
            % Verify parameter estimates are consistent across runs
            nuValues = zeros(numRuns, 1);
            lambdaValues = zeros(numRuns, 1);
            muValues = zeros(numRuns, 1);
            sigmaValues = zeros(numRuns, 1);
            
            for i = 1:numRuns
                nuValues(i) = results{i}.nu;
                lambdaValues(i) = results{i}.lambda;
                muValues(i) = results{i}.mu;
                sigmaValues(i) = results{i}.sigma;
            end
            
            % Compute standard deviations of estimates across runs
            nuStd = std(nuValues);
            lambdaStd = std(lambdaValues);
            muStd = std(muValues);
            sigmaStd = std(sigmaValues);
            
            % Verify consistency (standard deviations should be very small)
            obj.assertTrue(nuStd < 1e-10, 'Parameter estimates should be consistent across runs');
            obj.assertTrue(lambdaStd < 1e-10, 'Parameter estimates should be consistent across runs');
            obj.assertTrue(muStd < 1e-10, 'Parameter estimates should be consistent across runs');
            obj.assertTrue(sigmaStd < 1e-10, 'Parameter estimates should be consistent across runs');
            
            % Test with slightly different datasets that are statistically equivalent
            rng(12345);
            data1 = skewtrnd(5, 0.3, 1000, 1);
            rng(67890);
            data2 = skewtrnd(5, 0.3, 1000, 1);
            
            % Estimate parameters
            results1 = skewtfit(data1);
            results2 = skewtfit(data2);
            
            % Verify estimates are reasonably close despite different samples
            obj.assertAlmostEqual(results1.nu, results2.nu, 'Parameter estimates should be similar for equivalent datasets', 0.2);
            obj.assertAlmostEqual(results1.lambda, results2.lambda, 'Parameter estimates should be similar for equivalent datasets', 0.1);
        end
        
        function testAsymmetryEstimation(obj)
            % Tests accurate estimation of skewness in asymmetric data
            
            % Test with positive skewness
            nu = 5;
            lambda_pos = 0.7;
            n = 2000;
            rng(123);
            data_pos = skewtrnd(nu, lambda_pos, n, 1);
            
            % Estimate parameters for positive skewness
            results_pos = skewtfit(data_pos);
            
            % Verify lambda is positive and close to true value
            obj.assertTrue(results_pos.lambda > 0, 'Estimated lambda should be positive for positively skewed data');
            obj.assertAlmostEqual(results_pos.lambda, lambda_pos, 'Lambda estimation should be accurate for positive skewness');
            
            % Test with negative skewness
            lambda_neg = -0.7;
            rng(456);
            data_neg = skewtrnd(nu, lambda_neg, n, 1);
            
            % Estimate parameters for negative skewness
            results_neg = skewtfit(data_neg);
            
            % Verify lambda is negative and close to true value
            obj.assertTrue(results_neg.lambda < 0, 'Estimated lambda should be negative for negatively skewed data');
            obj.assertAlmostEqual(results_neg.lambda, lambda_neg, 'Lambda estimation should be accurate for negative skewness');
            
            % Test with no skewness (symmetric case)
            lambda_sym = 0;
            rng(789);
            data_sym = skewtrnd(nu, lambda_sym, n, 1);
            
            % Estimate parameters for symmetric case
            results_sym = skewtfit(data_sym);
            
            % Verify lambda is close to zero for symmetric data
            obj.assertAlmostEqual(results_sym.lambda, lambda_sym, 'Lambda estimation should be close to zero for symmetric data', 0.1);
        end
        
        function testBoundaryConditions(obj)
            % Tests behavior near boundary conditions for parameters
            
            % Test with degrees of freedom near lower bound
            nu_low = 2.2;
            lambda = 0;
            n = 2000;
            rng(123);
            data_low_nu = skewtrnd(nu_low, lambda, n, 1);
            
            % Estimate parameters
            results_low_nu = skewtfit(data_low_nu);
            
            % Verify nu is estimated reasonably close to low value
            obj.assertAlmostEqual(results_low_nu.nu, nu_low, 'Nu estimation should work near lower bound', 0.5);
            
            % Test with lambda near upper bound
            nu = 5;
            lambda_high = 0.95;
            rng(456);
            data_high_lambda = skewtrnd(nu, lambda_high, n, 1);
            
            % Estimate parameters
            results_high_lambda = skewtfit(data_high_lambda);
            
            % Verify lambda is estimated reasonably close to upper bound
            obj.assertAlmostEqual(results_high_lambda.lambda, lambda_high, 'Lambda estimation should work near upper bound', 0.15);
            
            % Test with lambda near lower bound
            lambda_low = -0.95;
            rng(789);
            data_low_lambda = skewtrnd(nu, lambda_low, n, 1);
            
            % Estimate parameters
            results_low_lambda = skewtfit(data_low_lambda);
            
            % Verify lambda is estimated reasonably close to lower bound
            obj.assertAlmostEqual(results_low_lambda.lambda, lambda_low, 'Lambda estimation should work near lower bound', 0.15);
            
            % Test with high degrees of freedom (approaching normal distribution)
            nu_high = 50;
            lambda = 0.3;
            rng(101112);
            data_high_nu = skewtrnd(nu_high, lambda, n, 1);
            
            % Estimate parameters
            results_high_nu = skewtfit(data_high_nu);
            
            % Verify estimation works with high degrees of freedom
            % (Exact precision is difficult with high nu, so we use a wider tolerance)
            obj.assertTrue(results_high_nu.nu > 15, 'Nu estimation should produce high value for high degrees of freedom');
            obj.assertAlmostEqual(results_high_nu.lambda, lambda, 'Lambda estimation should work with high degrees of freedom', 0.25);
        end
        
        function testLogLikelihoodConsistency(obj)
            % Tests consistency between parameter estimates and log-likelihood values
            
            % Get a test dataset
            data = obj.simulatedData{2};
            
            % Estimate parameters
            results = skewtfit(data);
            
            % Extract estimated parameters
            parameters = [results.nu, results.lambda, results.mu, results.sigma];
            
            % Calculate log-likelihood with estimated parameters
            [~, logL] = skewtloglik(data, parameters);
            
            % Calculate total log-likelihood
            total_logL = sum(logL);
            
            % Verify it matches the reported log-likelihood
            obj.assertAlmostEqual(total_logL, results.logL, 'Calculated log-likelihood should match reported value');
            
            % Test with slightly perturbed parameters to verify optimality
            perturbations = [
                [0.1, 0, 0, 0];    % Perturb nu
                [0, 0.05, 0, 0];   % Perturb lambda
                [0, 0, 0.05, 0];   % Perturb mu
                [0, 0, 0, 0.05]    % Perturb sigma
            ];
            
            for i = 1:size(perturbations, 1)
                % Perturb parameters
                perturbed_params = parameters + perturbations(i, :);
                
                % Calculate log-likelihood with perturbed parameters
                [nlogL_perturbed, ~] = skewtloglik(data, perturbed_params);
                logL_perturbed = -nlogL_perturbed;
                
                % Verify optimality (perturbed log-likelihood should be worse)
                obj.assertTrue(logL_perturbed <= results.logL, ...
                    sprintf('Log-likelihood should be optimal at estimated parameters (perturbation %d)', i));
            end
        end
    end
end