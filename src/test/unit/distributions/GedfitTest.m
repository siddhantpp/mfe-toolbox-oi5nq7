classdef GedfitTest < BaseTest
    % GEDFITTEST Unit test for the Generalized Error Distribution parameter estimation function
    %
    % This test class validates the implementation of maximum likelihood estimation
    % for GED parameters by testing against known reference values, simulated data,
    % and edge cases to ensure accurate and numerically stable estimation.
    %
    % The tests verify that the gedfit function:
    %   * Accurately estimates parameters from data with known distributions
    %   * Shows improved accuracy with larger sample sizes
    %   * Handles options and configuration correctly
    %   * Produces correctly structured output
    %   * Properly handles invalid inputs and edge cases
    %   * Maintains estimation consistency
    %   * Functions correctly near boundary conditions
    %   * Produces log-likelihood values consistent with direct calculation
    
    properties
        testData        % Structure holding test data from known_distributions.mat
        tolerance       % Numerical tolerance for floating-point comparisons
        simulatedData   % Matrix of simulated data for testing
        knownParameters % Matrix of known parameters corresponding to simulatedData
    end
    
    methods
        function obj = GedfitTest()
            % Initialize the test class with test name and configuration
            obj = obj@BaseTest('Generalized Error Distribution Parameter Estimation Test');
            
            % Set numerical tolerance for floating-point comparisons
            obj.tolerance = 1e-4;
        end
        
        function setUp(obj)
            % Set up test environment before each test
            setUp@BaseTest(obj);
            
            % Load test data from known distributions
            try
                dataFile = fullfile(obj.testDataPath, 'known_distributions.mat');
                obj.testData = load(dataFile);
            catch ME
                warning('Unable to load test data: %s. Using simulated data only.', ME.message);
                obj.testData = struct();
            end
            
            % Generate simulated data with known parameters
            rng(1234); % Set seed for reproducibility
            
            % Generate data with different shape parameters
            nu_values = [1.0, 1.5, 2.0, 3.0];
            n_samples = 2000;
            
            % Initialize data structure
            obj.simulatedData = cell(length(nu_values), 1);
            obj.knownParameters = zeros(length(nu_values), 3);
            
            % Generate data for each shape parameter
            for i = 1:length(nu_values)
                nu = nu_values(i);
                mu = 0;
                sigma = 1;
                
                % Generate random data using gedrnd function
                obj.simulatedData{i} = gedrnd(nu, n_samples, 1, mu, sigma);
                obj.knownParameters(i, :) = [nu, mu, sigma];
            end
        end
        
        function tearDown(obj)
            % Clean up after each test
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Tests the basic functionality of gedfit against known parameters with simulated data
            
            % Test with different shape parameters
            for i = 1:size(obj.knownParameters, 1)
                nu_true = obj.knownParameters(i, 1);
                mu_true = obj.knownParameters(i, 2);
                sigma_true = obj.knownParameters(i, 3);
                
                data = obj.simulatedData{i};
                
                % Estimate parameters using gedfit
                params = gedfit(data);
                
                % Verify estimated parameters are close to true values
                % Shape parameter (nu) is harder to estimate accurately, so use higher tolerance
                nu_tolerance = 0.15;
                abs_diff_nu = abs(params.nu - nu_true);
                obj.assertTrue(abs_diff_nu < nu_tolerance, ...
                    sprintf('Shape parameter (nu) estimate incorrect for nu=%g (error: %g)', ...
                    nu_true, abs_diff_nu));
                
                % Location parameter (mu) should be very accurate
                mu_tolerance = 0.05;
                abs_diff_mu = abs(params.mu - mu_true);
                obj.assertTrue(abs_diff_mu < mu_tolerance, ...
                    sprintf('Location parameter (mu) estimate incorrect for nu=%g (error: %g)', ...
                    nu_true, abs_diff_mu));
                
                % Scale parameter (sigma) should be fairly accurate
                sigma_tolerance = 0.1;
                abs_diff_sigma = abs(params.sigma - sigma_true);
                obj.assertTrue(abs_diff_sigma < sigma_tolerance, ...
                    sprintf('Scale parameter (sigma) estimate incorrect for nu=%g (error: %g)', ...
                    nu_true, abs_diff_sigma));
            end
        end
        
        function testAccuracyWithLargeData(obj)
            % Tests parameter estimation accuracy with large sample sizes
            
            % Test parameters
            nu = 1.5;
            mu = 0;
            sigma = 1;
            
            % Test with different sample sizes
            sample_sizes = [1000, 5000, 10000];
            nu_errors = zeros(length(sample_sizes), 1);
            mu_errors = zeros(length(sample_sizes), 1);
            sigma_errors = zeros(length(sample_sizes), 1);
            std_errors = zeros(length(sample_sizes), 3);
            
            % Set seed for reproducibility
            rng(5678);
            
            for i = 1:length(sample_sizes)
                n = sample_sizes(i);
                
                % Generate large sample data
                data = gedrnd(nu, n, 1, mu, sigma);
                
                % Estimate parameters
                params = gedfit(data);
                
                % Calculate absolute errors
                nu_errors(i) = abs(params.nu - nu);
                mu_errors(i) = abs(params.mu - mu);
                sigma_errors(i) = abs(params.sigma - sigma);
                std_errors(i, :) = params.stderrors';
                
                % Verify estimates are close to true values with appropriate tolerance
                % for sample size
                nu_tolerance = 0.2 / sqrt(n/1000);
                mu_tolerance = 0.05 / sqrt(n/1000);
                sigma_tolerance = 0.1 / sqrt(n/1000);
                
                obj.assertTrue(nu_errors(i) < nu_tolerance, ...
                    sprintf('Shape parameter (nu) estimate incorrect with %d samples (error: %g)', ...
                    n, nu_errors(i)));
                
                obj.assertTrue(mu_errors(i) < mu_tolerance, ...
                    sprintf('Location parameter (mu) estimate incorrect with %d samples (error: %g)', ...
                    n, mu_errors(i)));
                
                obj.assertTrue(sigma_errors(i) < sigma_tolerance, ...
                    sprintf('Scale parameter (sigma) estimate incorrect with %d samples (error: %g)', ...
                    n, sigma_errors(i)));
            end
            
            % Verify errors tend to decrease with larger sample sizes
            % This is a stochastic test, so use a relaxed condition
            obj.assertTrue(mean(nu_errors(2:3)) < 1.2*nu_errors(1), ...
                sprintf('Estimation error for nu did not decrease with larger sample size: %s', ...
                mat2str(nu_errors)));
                
            obj.assertTrue(mean(mu_errors(2:3)) < 1.2*mu_errors(1), ...
                sprintf('Estimation error for mu did not decrease with larger sample size: %s', ...
                mat2str(mu_errors)));
                
            obj.assertTrue(mean(sigma_errors(2:3)) < 1.2*sigma_errors(1), ...
                sprintf('Estimation error for sigma did not decrease with larger sample size: %s', ...
                mat2str(sigma_errors)));
            
            % Verify standard errors decrease with larger sample sizes
            % Standard errors should be approximately proportional to 1/sqrt(n)
            for j = 1:3
                ratio_10k_to_1k = std_errors(3,j) / std_errors(1,j);
                expected_ratio = sqrt(sample_sizes(1)/sample_sizes(3));
                ratio_tolerance = 0.3; % Allow some deviation due to randomness
                
                obj.assertTrue(abs(ratio_10k_to_1k - expected_ratio) < ratio_tolerance, ...
                    sprintf('Standard errors did not scale correctly with sample size for parameter %d', j));
            end
        end
        
        function testOptionsHandling(obj)
            % Tests the handling of options structure in parameter estimation
            
            % Get test data
            data = obj.simulatedData{2};  % Use data with nu=1.5
            
            % Test default options
            params_default = gedfit(data);
            
            % Test with custom starting values
            options1 = struct();
            options1.startingvals = [1.2, 0.1, 0.9];  % Different starting values
            params1 = gedfit(data, options1);
            
            % Results should be similar despite different starting values
            obj.assertAlmostEqual(params1.nu, params_default.nu, ...
                'Parameter estimates differ with custom starting values');
            
            % Test with display option
            options2 = struct();
            options2.display = 'off';
            params2 = gedfit(data, options2);
            
            % Results should be the same with display option
            obj.assertEqual(params2.nu, params_default.nu, ...
                'Parameter estimates differ with display option');
            
            % Test with algorithm option
            options3 = struct();
            options3.algorithm = 'interior-point';
            options3.MaxIter = 300;
            options3.MaxFunEvals = 600;
            params3 = gedfit(data, options3);
            
            % Results should be similar with algorithm option
            % (may not be identical due to different optimization path)
            nu_diff = abs(params3.nu - params_default.nu);
            obj.assertTrue(nu_diff < 0.1, ...
                sprintf('Parameter estimates differ too much with algorithm option (diff: %g)', nu_diff));
            
            % Test with tolerance options
            options4 = struct();
            options4.TolFun = 1e-9;
            options4.TolX = 1e-9;
            params4 = gedfit(data, options4);
            
            % Results should be similar with tolerance options
            nu_diff = abs(params4.nu - params_default.nu);
            obj.assertTrue(nu_diff < 0.1, ...
                sprintf('Parameter estimates differ too much with tolerance options (diff: %g)', nu_diff));
        end
        
        function testOutputStructure(obj)
            % Tests the structure and contents of the function output
            
            % Use simulated data with known parameters
            data = obj.simulatedData{2};  % Use data with nu=1.5
            
            % Get the output structure
            params = gedfit(data);
            
            % Verify output structure has all expected fields
            expectedFields = {'nu', 'mu', 'sigma', 'loglik', 'vcv', 'stderrors', 'exitflag', 'output'};
            fields = fieldnames(params);
            
            for i = 1:length(expectedFields)
                field = expectedFields{i};
                obj.assertTrue(ismember(field, fields), ...
                    sprintf('Output structure missing field: %s', field));
            end
            
            % Verify parameter types and dimensions
            obj.assertTrue(isscalar(params.nu), 'Shape parameter (nu) is not a scalar');
            obj.assertTrue(isscalar(params.mu), 'Location parameter (mu) is not a scalar');
            obj.assertTrue(isscalar(params.sigma), 'Scale parameter (sigma) is not a scalar');
            obj.assertTrue(isscalar(params.loglik), 'Log-likelihood is not a scalar');
            
            % Verify variance-covariance matrix dimensions
            obj.assertTrue(all(size(params.vcv) == [3, 3]), ...
                'Variance-covariance matrix has incorrect dimensions');
            
            % Verify standard errors vector
            obj.assertTrue(length(params.stderrors) == 3, ...
                'Standard errors vector has incorrect length');
            
            % Verify exitflag is an integer
            obj.assertTrue(isscalar(params.exitflag) && ...
                params.exitflag == floor(params.exitflag), ...
                'exitflag is not an integer scalar');
            
            % Verify output structure has expected fields
            obj.assertTrue(isstruct(params.output), 'output is not a structure');
            
            % Verify vcv matrix is positive semi-definite (diagonals > 0)
            obj.assertTrue(all(diag(params.vcv) > 0), ...
                'Variance-covariance matrix has non-positive diagonal elements');
            
            % Verify standard errors are positive
            obj.assertTrue(all(params.stderrors > 0), ...
                'Standard errors are not all positive');
        end
        
        function testErrorHandling(obj)
            % Tests error handling for invalid inputs
            
            % Test with invalid data (NaN values)
            data_nan = [1; 2; NaN; 4];
            obj.assertThrows(@() gedfit(data_nan), 'data', ...
                'Function did not detect NaN values properly');
            
            % Test with invalid data (Inf values)
            data_inf = [1; 2; Inf; 4];
            obj.assertThrows(@() gedfit(data_inf), 'data', ...
                'Function did not detect Inf values properly');
            
            % Test with empty data
            data_empty = [];
            obj.assertThrows(@() gedfit(data_empty), 'data', ...
                'Function did not detect empty data properly');
            
            % Test with non-numeric data
            data_cell = {'a', 'b', 'c'};
            obj.assertThrows(@() gedfit(data_cell), 'data', ...
                'Function did not detect non-numeric data properly');
            
            % Test with invalid options
            validData = obj.simulatedData{1};
            
            % Invalid startingvals (negative shape parameter)
            badOptions1 = struct('startingvals', [-1, 0, 1]);
            obj.assertThrows(@() gedfit(validData, badOptions1), 'options', ...
                'Function did not detect invalid starting values');
            
            % Invalid display option
            badOptions2 = struct('display', 'invalid');
            obj.assertThrows(@() gedfit(validData, badOptions2), 'options', ...
                'Function did not detect invalid display option');
        end
        
        function testEstimationConsistency(obj)
            % Tests consistency of parameter estimation across multiple runs
            
            % Use simulated data with known parameters
            data = obj.simulatedData{2};  % Use data with nu=1.5
            
            % Run estimation multiple times with the same data
            num_runs = 5;
            nu_estimates = zeros(num_runs, 1);
            mu_estimates = zeros(num_runs, 1);
            sigma_estimates = zeros(num_runs, 1);
            
            for i = 1:num_runs
                params = gedfit(data);
                nu_estimates(i) = params.nu;
                mu_estimates(i) = params.mu;
                sigma_estimates(i) = params.sigma;
            end
            
            % Verify all estimates are identical across runs
            for i = 2:num_runs
                obj.assertEqual(nu_estimates(1), nu_estimates(i), ...
                    'Shape parameter estimates vary across identical runs');
                obj.assertEqual(mu_estimates(1), mu_estimates(i), ...
                    'Location parameter estimates vary across identical runs');
                obj.assertEqual(sigma_estimates(1), sigma_estimates(i), ...
                    'Scale parameter estimates vary across identical runs');
            end
            
            % Test with statistically equivalent datasets
            % Create multiple datasets with the same parameters
            rng(9876);
            n_samples = 5000;
            nu_true = 1.5;
            mu_true = 0;
            sigma_true = 1;
            
            num_datasets = 3;
            datasets = cell(num_datasets, 1);
            nu_dataset_estimates = zeros(num_datasets, 1);
            
            for i = 1:num_datasets
                datasets{i} = gedrnd(nu_true, n_samples, 1, mu_true, sigma_true);
                params = gedfit(datasets{i});
                nu_dataset_estimates(i) = params.nu;
            end
            
            % Verify estimates are close across different but equivalent datasets
            % Use higher tolerance for statistical variation
            dataset_tolerance = 0.1;
            for i = 2:num_datasets
                abs_diff = abs(nu_dataset_estimates(1) - nu_dataset_estimates(i));
                obj.assertTrue(abs_diff < dataset_tolerance, ...
                    sprintf('Estimates vary excessively across equivalent datasets: %g vs %g', ...
                    nu_dataset_estimates(1), nu_dataset_estimates(i)));
            end
        end
        
        function testBoundaryConditions(obj)
            % Tests behavior near boundary conditions for shape parameter
            
            % Test with very small shape parameter (heavy tails)
            rng(1111);
            n_samples = 5000;
            
            % Generate data with small nu (heavy tails)
            nu_small = 1.0;  % At the lower bound of 1.0 in optimization
            data_heavy = gedrnd(nu_small, n_samples, 1);
            
            % Estimate parameters
            params_heavy = gedfit(data_heavy);
            
            % Verify estimate is reasonably close to true value
            heavy_tail_tolerance = 0.3;
            abs_diff_small = abs(params_heavy.nu - nu_small);
            obj.assertTrue(abs_diff_small < heavy_tail_tolerance, ...
                sprintf('Failed to estimate small shape parameter: true=%g, est=%g, diff=%g', ...
                nu_small, params_heavy.nu, abs_diff_small));
            
            % Test with large shape parameter (approaching normal distribution)
            nu_large = 5.0;
            data_light = gedrnd(nu_large, n_samples, 1);
            
            % Estimate parameters
            params_light = gedfit(data_light);
            
            % Verify estimate is reasonably close to true value
            % Use higher tolerance as large shape is harder to estimate
            light_tail_tolerance = 1.0;
            abs_diff_large = abs(params_light.nu - nu_large);
            obj.assertTrue(abs_diff_large < light_tail_tolerance, ...
                sprintf('Failed to estimate large shape parameter: true=%g, est=%g, diff=%g', ...
                nu_large, params_light.nu, abs_diff_large));
            
            % Test with shape parameter = 2 (normal distribution)
            nu_normal = 2.0;
            data_normal = gedrnd(nu_normal, n_samples, 1);
            
            % Estimate parameters
            params_normal = gedfit(data_normal);
            
            % Verify estimate is reasonably close to true value
            normal_tolerance = 0.3;
            abs_diff_normal = abs(params_normal.nu - nu_normal);
            obj.assertTrue(abs_diff_normal < normal_tolerance, ...
                sprintf('Failed to estimate normal distribution shape parameter: true=%g, est=%g, diff=%g', ...
                nu_normal, params_normal.nu, abs_diff_normal));
        end
        
        function testLogLikelihoodConsistency(obj)
            % Tests consistency between parameter estimates and log-likelihood values
            
            % Use simulated data with known parameters
            data = obj.simulatedData{2};  % Use data with nu=1.5
            nu_true = obj.knownParameters(2, 1);
            mu_true = obj.knownParameters(2, 2);
            sigma_true = obj.knownParameters(2, 3);
            
            % Estimate parameters
            params = gedfit(data);
            
            % Calculate log-likelihood using estimated parameters
            loglik_estimated = gedloglik(data, params.nu, params.mu, params.sigma);
            
            % Verify log-likelihood matches the value reported by gedfit
            obj.assertAlmostEqual(loglik_estimated, params.loglik, ...
                'Log-likelihood from gedloglik does not match the value reported by gedfit');
            
            % Calculate log-likelihood using true parameters
            loglik_true = gedloglik(data, nu_true, mu_true, sigma_true);
            
            % Verify estimated parameters give higher or approximately equal log-likelihood 
            % compared to true parameters (accounting for numerical precision)
            obj.assertTrue(loglik_estimated >= loglik_true - obj.tolerance, ...
                sprintf(['Estimated parameters do not yield higher log-likelihood than true parameters:', ...
                ' est=%g, true=%g, diff=%g'], ...
                loglik_estimated, loglik_true, loglik_estimated - loglik_true));
            
            % Test with various parameter combinations
            % Slightly perturb each parameter and verify log-likelihood decreases
            
            % Perturb nu
            perturb_factor = 1.05;  % 5% perturbation
            loglik_nu_perturbed = gedloglik(data, params.nu * perturb_factor, params.mu, params.sigma);
            obj.assertTrue(loglik_estimated >= loglik_nu_perturbed - obj.tolerance, ...
                sprintf(['Log-likelihood did not decrease when perturbing shape parameter:', ...
                ' original=%g, perturbed=%g, diff=%g'], ...
                loglik_estimated, loglik_nu_perturbed, loglik_estimated - loglik_nu_perturbed));
            
            % Perturb mu
            mu_perturb = 0.05;  % Small perturbation
            loglik_mu_perturbed = gedloglik(data, params.nu, params.mu + mu_perturb, params.sigma);
            obj.assertTrue(loglik_estimated >= loglik_mu_perturbed - obj.tolerance, ...
                sprintf(['Log-likelihood did not decrease when perturbing location parameter:', ...
                ' original=%g, perturbed=%g, diff=%g'], ...
                loglik_estimated, loglik_mu_perturbed, loglik_estimated - loglik_mu_perturbed));
            
            % Perturb sigma
            loglik_sigma_perturbed = gedloglik(data, params.nu, params.mu, params.sigma * perturb_factor);
            obj.assertTrue(loglik_estimated >= loglik_sigma_perturbed - obj.tolerance, ...
                sprintf(['Log-likelihood did not decrease when perturbing scale parameter:', ...
                ' original=%g, perturbed=%g, diff=%g'], ...
                loglik_estimated, loglik_sigma_perturbed, loglik_estimated - loglik_sigma_perturbed));
        end
    end
end