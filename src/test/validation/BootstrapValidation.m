classdef BootstrapValidation < BaseTest
    % BOOTSTRAPVALIDATION Class for comprehensive validation of bootstrap methods in the MFE Toolbox
    %
    % This class validates the statistical properties, numerical stability, and correctness 
    % of bootstrap methods implemented in the MFE Toolbox, including block bootstrap, 
    % stationary bootstrap, bootstrap variance estimation, and bootstrap confidence interval 
    % procedures.
    %
    % The validation suite tests these methods against known reference implementations and 
    % theoretical properties to ensure accurate resampling, proper preservation of time series
    % dependence structure, and correct statistical inference.
    %
    % BootstrapValidation inherits from BaseTest and provides specialized methods for
    % validating each bootstrap implementation through rigorous statistical testing.
    %
    % Example:
    %   validator = BootstrapValidation();
    %   results = validator.validateAllBootstrapMethods();
    %   validator.displayValidationSummary();
    %
    % See also: BaseTest, block_bootstrap, stationary_bootstrap, bootstrap_variance,
    % bootstrap_confidence_intervals
    
    properties
        % Test data
        financialReturnsData
        simulatedTimeSeries
        dependentTimeSeries
        
        % Reference results
        referenceResults
        
        % Validation results
        validationResults
        
        % Numerical comparator for floating-point comparisons
        comparator
        
        % Default tolerance for numerical comparisons
        defaultTolerance
        
        % Verbosity flag
        verbose
    end
    
    methods
        function obj = BootstrapValidation(options)
            % Construct a BootstrapValidation object for bootstrap method testing
            %
            % INPUTS:
            %   options - Optional structure with fields:
            %             .tolerance - Custom tolerance for numerical comparisons
            %             .verbose - Boolean flag for verbose output
            %
            % OUTPUTS:
            %   obj - Initialized BootstrapValidation instance
            
            % Call the superclass constructor
            obj@BaseTest('BootstrapValidation');
            
            % Initialize the numerical comparator
            obj.comparator = NumericalComparator();
            
            % Set default parameters
            if nargin < 1
                options = struct();
            end
            
            % Set tolerance for numerical comparisons
            if isfield(options, 'tolerance')
                obj.defaultTolerance = options.tolerance;
            else
                obj.defaultTolerance = 1e-8;
            end
            
            % Set verbosity
            if isfield(options, 'verbose')
                obj.verbose = options.verbose;
            else
                obj.verbose = false;
            end
            
            % Initialize validation results structure
            obj.validationResults = struct();
        end
        
        function setUp(obj)
            % Prepares the test environment before validation tests
            %
            % This method sets up the necessary test data and environment
            % for bootstrap validation, including loading test data,
            % generating simulated time series, and setting RNG seed.
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random seed for reproducibility
            rng(1234, 'twister');
            
            % Load financial returns test data
            try
                data = obj.loadTestData('financial_returns.mat');
                if isfield(data, 'returns')
                    obj.financialReturnsData = data.returns;
                else
                    % If the exact field isn't available, use the first field
                    fields = fieldnames(data);
                    obj.financialReturnsData = data.(fields{1});
                end
            catch ME
                % Generate synthetic financial returns if data file not found
                fprintf('Test data file not found. Generating synthetic financial returns.\n');
                obj.financialReturnsData = 0.01 * randn(1000, 1) + 0.0005;  % Daily returns
            end
            
            % Generate simulated time series with known properties
            obj.simulatedTimeSeries = obj.generateTestTimeSeries(1000, 0.5, 'AR');
            
            % Generate dependent time series with controlled autocorrelation
            obj.dependentTimeSeries = obj.generateTestTimeSeries(1000, 0.7, 'ARMA');
            
            % Try to load reference bootstrap results if available
            try
                refData = obj.loadTestData('bootstrap_reference.mat');
                obj.referenceResults = refData;
            catch ME
                % Initialize empty reference results if file not found
                obj.referenceResults = struct();
            end
            
            % Configure numerical comparator with appropriate tolerances
            obj.comparator.setDefaultTolerances(obj.defaultTolerance, obj.defaultTolerance * 10);
            
            % Initialize validation result structure for each bootstrap method
            obj.validationResults.block_bootstrap = struct();
            obj.validationResults.stationary_bootstrap = struct();
            obj.validationResults.bootstrap_variance = struct();
            obj.validationResults.bootstrap_confidence_intervals = struct();
        end
        
        function tearDown(obj)
            % Cleans up resources after validation tests
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Generate summary report of validation results
            if obj.verbose
                obj.displayValidationSummary();
            end
            
            % Clear temporary test data to free memory
            obj.simulatedTimeSeries = [];
            obj.dependentTimeSeries = [];
        end
        
        function results = validateBlockBootstrap(obj)
            % Validates block_bootstrap implementation against reference results and theoretical properties
            %
            % OUTPUTS:
            %   results - Validation results for block_bootstrap function
            
            if obj.verbose
                disp('Validating block_bootstrap function...');
            end
            
            % Initialize results structure
            results = struct('tests', struct(), 'passed', 0, 'failed', 0, 'total', 0);
            
            % Set test parameters
            blockSizes = [5, 10, 20];
            numReplications = 500;
            
            % Test 1: Verify dimensions of bootstrap samples match original data
            try
                % Apply block bootstrap to financial returns data
                bsData = block_bootstrap(obj.financialReturnsData, 10, numReplications);
                
                % Check dimensions
                [origRows, origCols] = size(obj.financialReturnsData);
                [bsRows, bsCols, bsReps] = size(bsData);
                
                obj.assertTrue(origRows == bsRows, 'Bootstrap sample rows should match original data');
                obj.assertTrue(origCols == bsCols, 'Bootstrap sample columns should match original data');
                obj.assertTrue(bsReps == numReplications, 'Number of bootstrap samples should match replications');
                
                results.tests.dimensionTest = struct('passed', true, 'message', 'Bootstrap sample dimensions are correct');
                results.passed = results.passed + 1;
            catch ME
                results.tests.dimensionTest = struct('passed', false, 'message', ['Dimension test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 2: Verify block structure preservation
            try
                % Generate a series with strong autocorrelation
                ar_series = obj.dependentTimeSeries;
                
                % Apply block bootstrap
                block_size = 10;
                bs_ar_data = block_bootstrap(ar_series, block_size, numReplications);
                
                % Check autocorrelation preservation within blocks
                acf_orig = autocorr(ar_series, block_size - 1);
                
                % Calculate mean autocorrelation across bootstrap samples
                acf_bs = zeros(block_size, numReplications);
                for i = 1:numReplications
                    acf_bs(:, i) = autocorr(bs_ar_data(:, 1, i), block_size - 1);
                end
                mean_acf_bs = mean(acf_bs, 2);
                
                % Check that autocorrelation is preserved for small lags (within blocks)
                acf_error = abs(acf_orig(1:3) - mean_acf_bs(1:3));
                obj.assertTrue(all(acf_error < 0.2), 'Block bootstrap should preserve short-lag autocorrelation');
                
                results.tests.autocorrelationTest = struct('passed', true, 'message', 'Block structure preserves local dependence');
                results.passed = results.passed + 1;
            catch ME
                results.tests.autocorrelationTest = struct('passed', false, 'message', ['Autocorrelation test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 3: Test different block sizes
            try
                stats_orig = struct();
                stats_orig.mean = mean(obj.financialReturnsData);
                stats_orig.var = var(obj.financialReturnsData);
                stats_orig.median = median(obj.financialReturnsData);
                
                block_size_results = struct();
                
                for i = 1:length(blockSizes)
                    bs = blockSizes(i);
                    bs_data = block_bootstrap(obj.financialReturnsData, bs, numReplications);
                    
                    bs_means = squeeze(mean(bs_data));
                    bs_vars = squeeze(var(bs_data));
                    bs_medians = squeeze(median(bs_data));
                    
                    block_size_results.(sprintf('size_%d', bs)) = struct(...
                        'mean_error', abs(mean(bs_means) - stats_orig.mean) / abs(stats_orig.mean), ...
                        'var_error', abs(mean(bs_vars) - stats_orig.var) / abs(stats_orig.var), ...
                        'median_error', abs(mean(bs_medians) - stats_orig.median) / abs(stats_orig.median) ...
                    );
                end
                
                % Check that errors are within acceptable limits
                max_mean_error = max([block_size_results.size_5.mean_error, ...
                                    block_size_results.size_10.mean_error, ...
                                    block_size_results.size_20.mean_error]);
                                
                max_var_error = max([block_size_results.size_5.var_error, ...
                                  block_size_results.size_10.var_error, ...
                                  block_size_results.size_20.var_error]);
                              
                obj.assertTrue(max_mean_error < 0.1, 'Mean statistics should be preserved');
                obj.assertTrue(max_var_error < 0.2, 'Variance statistics should be approximately preserved');
                
                results.tests.blockSizeTest = struct('passed', true, 'message', 'Different block sizes produce consistent results');
                results.block_size_results = block_size_results;
                results.passed = results.passed + 1;
            catch ME
                results.tests.blockSizeTest = struct('passed', false, 'message', ['Block size test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 4: Test circular vs. non-circular options
            try
                bs_circ = block_bootstrap(obj.financialReturnsData, 10, numReplications, true);
                bs_non_circ = block_bootstrap(obj.financialReturnsData, 10, numReplications, false);
                
                % Compute statistics for both
                means_circ = squeeze(mean(bs_circ));
                means_non_circ = squeeze(mean(bs_non_circ));
                
                vars_circ = squeeze(var(bs_circ));
                vars_non_circ = squeeze(var(bs_non_circ));
                
                % Check that both methods produce valid results
                obj.assertTrue(~any(isnan(means_circ)), 'Circular bootstrap should not produce NaN means');
                obj.assertTrue(~any(isnan(means_non_circ)), 'Non-circular bootstrap should not produce NaN means');
                
                obj.assertTrue(~any(isnan(vars_circ)), 'Circular bootstrap should not produce NaN variances');
                obj.assertTrue(~any(isnan(vars_non_circ)), 'Non-circular bootstrap should not produce NaN variances');
                
                % The two methods should produce similar but not identical results
                mean_diff = abs(mean(means_circ) - mean(means_non_circ));
                var_diff = abs(mean(vars_circ) - mean(vars_non_circ));
                
                obj.assertTrue(mean_diff < 0.01, 'Circular and non-circular bootstrap means should be similar');
                obj.assertTrue(var_diff < 0.01, 'Circular and non-circular bootstrap variances should be similar');
                
                results.tests.circularTest = struct('passed', true, 'message', 'Circular and non-circular options work correctly');
                results.passed = results.passed + 1;
            catch ME
                results.tests.circularTest = struct('passed', false, 'message', ['Circular option test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 5: Validate error handling for invalid inputs
            try
                % Test with invalid block size
                obj.assertThrows(@() block_bootstrap(obj.financialReturnsData, 1001, 10), ...
                    'MATLAB:error', 'Should reject block size >= data length');
                
                % Test with invalid number of replications
                obj.assertThrows(@() block_bootstrap(obj.financialReturnsData, 10, -1), ...
                    'MATLAB:error', 'Should reject negative number of replications');
                
                % Test with invalid circular parameter
                obj.assertThrows(@() block_bootstrap(obj.financialReturnsData, 10, 10, 'invalid'), ...
                    'MATLAB:error', 'Should reject non-boolean circular parameter');
                
                results.tests.errorHandlingTest = struct('passed', true, 'message', 'Error handling for invalid inputs works correctly');
                results.passed = results.passed + 1;
            catch ME
                results.tests.errorHandlingTest = struct('passed', false, 'message', ['Error handling test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Store validation results
            obj.validationResults.block_bootstrap = results;
            
            % Display summary
            if obj.verbose
                disp(['Block bootstrap validation: ' num2str(results.passed) '/' num2str(results.total) ' tests passed.']);
            end
            
            return;
        end
        
        function results = validateStationaryBootstrap(obj)
            % Validates stationary_bootstrap implementation against reference results and theoretical properties
            %
            % OUTPUTS:
            %   results - Validation results for stationary_bootstrap function
            
            if obj.verbose
                disp('Validating stationary_bootstrap function...');
            end
            
            % Initialize results structure
            results = struct('tests', struct(), 'passed', 0, 'failed', 0, 'total', 0);
            
            % Set test parameters
            pValues = [0.1, 0.2, 0.5];  % Probability parameters (expected block lengths: 10, 5, 2)
            numReplications = 500;
            
            % Test 1: Verify dimensions of bootstrap samples match original data
            try
                % Apply stationary bootstrap to financial returns data
                bsData = stationary_bootstrap(obj.financialReturnsData, 0.1, numReplications);
                
                % Check dimensions
                [origRows, origCols] = size(obj.financialReturnsData);
                [bsRows, bsCols, bsReps] = size(bsData);
                
                obj.assertTrue(origRows == bsRows, 'Bootstrap sample rows should match original data');
                obj.assertTrue(origCols == bsCols, 'Bootstrap sample columns should match original data');
                obj.assertTrue(bsReps == numReplications, 'Number of bootstrap samples should match replications');
                
                results.tests.dimensionTest = struct('passed', true, 'message', 'Bootstrap sample dimensions are correct');
                results.passed = results.passed + 1;
            catch ME
                results.tests.dimensionTest = struct('passed', false, 'message', ['Dimension test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 2: Verify stationarity preservation
            try
                % Generate a series with strong autocorrelation
                ar_series = obj.dependentTimeSeries;
                
                % Apply stationary bootstrap
                p_value = 0.1;  % expected block length = 10
                bs_ar_data = stationary_bootstrap(ar_series, p_value, numReplications);
                
                % Check autocorrelation preservation
                acf_orig = autocorr(ar_series, 10);
                
                % Calculate mean autocorrelation across bootstrap samples
                acf_bs = zeros(11, numReplications);
                for i = 1:numReplications
                    acf_bs(:, i) = autocorr(bs_ar_data(:, 1, i), 10);
                end
                mean_acf_bs = mean(acf_bs, 2);
                
                % Check that autocorrelation is preserved for small lags
                acf_error = abs(acf_orig(1:3) - mean_acf_bs(1:3));
                obj.assertTrue(all(acf_error < 0.2), 'Stationary bootstrap should preserve short-lag autocorrelation');
                
                results.tests.autocorrelationTest = struct('passed', true, 'message', 'Stationary bootstrap preserves dependence structure');
                results.passed = results.passed + 1;
            catch ME
                results.tests.autocorrelationTest = struct('passed', false, 'message', ['Autocorrelation test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 3: Test different probability parameters
            try
                stats_orig = struct();
                stats_orig.mean = mean(obj.financialReturnsData);
                stats_orig.var = var(obj.financialReturnsData);
                stats_orig.median = median(obj.financialReturnsData);
                
                p_value_results = struct();
                
                for i = 1:length(pValues)
                    p = pValues(i);
                    bs_data = stationary_bootstrap(obj.financialReturnsData, p, numReplications);
                    
                    bs_means = squeeze(mean(bs_data));
                    bs_vars = squeeze(var(bs_data));
                    bs_medians = squeeze(median(bs_data));
                    
                    p_value_results.(sprintf('p_%g', p)) = struct(...
                        'mean_error', abs(mean(bs_means) - stats_orig.mean) / abs(stats_orig.mean), ...
                        'var_error', abs(mean(bs_vars) - stats_orig.var) / abs(stats_orig.var), ...
                        'median_error', abs(mean(bs_medians) - stats_orig.median) / abs(stats_orig.median) ...
                    );
                end
                
                % Check that errors are within acceptable limits
                max_mean_error = max([p_value_results.p_0_1.mean_error, ...
                                     p_value_results.p_0_2.mean_error, ...
                                     p_value_results.p_0_5.mean_error]);
                                 
                max_var_error = max([p_value_results.p_0_1.var_error, ...
                                   p_value_results.p_0_2.var_error, ...
                                   p_value_results.p_0_5.var_error]);
                
                obj.assertTrue(max_mean_error < 0.1, 'Mean statistics should be preserved');
                obj.assertTrue(max_var_error < 0.2, 'Variance statistics should be approximately preserved');
                
                results.tests.pValueTest = struct('passed', true, 'message', 'Different p values produce consistent results');
                results.p_value_results = p_value_results;
                results.passed = results.passed + 1;
            catch ME
                results.tests.pValueTest = struct('passed', false, 'message', ['p value test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 4: Test dependence preservation with different p values
            try
                % The p parameter controls expected block length (1/p)
                % Smaller p means longer blocks, better preservation of dependence
                
                ar_series = obj.dependentTimeSeries;
                acf_orig = autocorr(ar_series, 5);
                
                dependence_results = struct();
                
                for i = 1:length(pValues)
                    p = pValues(i);
                    bs_data = stationary_bootstrap(ar_series, p, numReplications);
                    
                    acf_errors = zeros(6, numReplications);
                    for j = 1:numReplications
                        sample_acf = autocorr(bs_data(:, 1, j), 5);
                        acf_errors(:, j) = abs(sample_acf - acf_orig);
                    end
                    
                    mean_acf_error = mean(acf_errors, 2);
                    dependence_results.(sprintf('p_%g', p)) = struct(...
                        'mean_acf_error', mean_acf_error, ...
                        'max_acf_error', max(acf_errors, [], 2) ...
                    );
                end
                
                % Smaller p (longer blocks) should preserve autocorrelation better
                mean_error_p01 = mean(dependence_results.p_0_1.mean_acf_error(1:3));
                mean_error_p05 = mean(dependence_results.p_0_5.mean_acf_error(1:3));
                
                obj.assertTrue(mean_error_p01 < mean_error_p05, 'Smaller p values should better preserve dependence');
                
                results.tests.dependencePreservationTest = struct('passed', true, 'message', 'Dependence preservation behaves as expected with different p values');
                results.dependence_results = dependence_results;
                results.passed = results.passed + 1;
            catch ME
                results.tests.dependencePreservationTest = struct('passed', false, 'message', ['Dependence preservation test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 5: Validate error handling for invalid inputs
            try
                % Test with invalid p value (negative)
                obj.assertThrows(@() stationary_bootstrap(obj.financialReturnsData, -0.1, 10), ...
                    'MATLAB:error', 'Should reject negative p value');
                
                % Test with invalid p value (greater than 1)
                obj.assertThrows(@() stationary_bootstrap(obj.financialReturnsData, 1.1, 10), ...
                    'MATLAB:error', 'Should reject p value > 1');
                
                % Test with invalid number of replications
                obj.assertThrows(@() stationary_bootstrap(obj.financialReturnsData, 0.1, -5), ...
                    'MATLAB:error', 'Should reject negative number of replications');
                
                results.tests.errorHandlingTest = struct('passed', true, 'message', 'Error handling for invalid inputs works correctly');
                results.passed = results.passed + 1;
            catch ME
                results.tests.errorHandlingTest = struct('passed', false, 'message', ['Error handling test failed: ' ME.message]);
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Store validation results
            obj.validationResults.stationary_bootstrap = results;
            
            % Display summary
            if obj.verbose
                disp(['Stationary bootstrap validation: ' num2str(results.passed) '/' num2str(results.total) ' tests passed.']);
            end
            
            return;
        end
        
        function results = validateBootstrapVariance(obj)
            % Validates bootstrap_variance implementation for accuracy in variance estimation
            %
            % OUTPUTS:
            %   results - Validation results for bootstrap_variance function
            
            if obj.verbose
                disp('Validating bootstrap_variance function...');
            end
            
            % Initialize results structure
            results = struct('tests', struct(), 'passed', 0, 'failed', 0, 'total', 0);
            
            % Test 1: Variance estimation for mean statistic with known variance
            try
                % Generate normal data with known variance
                n = 500;
                sigma = 0.5;
                data = sigma * randn(n, 1);
                
                % Calculate theoretical standard error of the mean
                theoretical_se = sigma / sqrt(n);
                
                % Define mean function
                mean_fn = @mean;
                
                % Set bootstrap options for block bootstrap
                options_block = struct();
                options_block.bootstrap_type = 'block';
                options_block.block_size = 10;
                options_block.replications = 1000;
                
                % Estimate variance using bootstrap_variance
                results_block = bootstrap_variance(data, mean_fn, options_block);
                
                % Compare with theoretical value
                estimated_se = sqrt(results_block.variance);
                se_error = abs(estimated_se - theoretical_se) / theoretical_se;
                
                obj.assertTrue(se_error < 0.2, 'Estimated standard error should be close to theoretical value');
                
                results.tests.meanVarianceTest = struct(...
                    'passed', true, ...
                    'message', 'Bootstrap variance estimation for mean is accurate', ...
                    'theoretical_se', theoretical_se, ...
                    'estimated_se', estimated_se, ...
                    'relative_error', se_error ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.meanVarianceTest = struct(...
                    'passed', false, ...
                    'message', ['Mean variance test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 2: Consistency between block and stationary bootstrap
            try
                % Use financial returns data
                data = obj.financialReturnsData;
                
                % Define median function
                median_fn = @median;
                
                % Set bootstrap options
                options_block = struct();
                options_block.bootstrap_type = 'block';
                options_block.block_size = 10;
                options_block.replications = 500;
                
                options_stationary = struct();
                options_stationary.bootstrap_type = 'stationary';
                options_stationary.p = 0.1;  % Expected block length of 10
                options_stationary.replications = 500;
                
                % Estimate variance using both methods
                results_block = bootstrap_variance(data, median_fn, options_block);
                results_stationary = bootstrap_variance(data, median_fn, options_stationary);
                
                % Compare results
                var_ratio = results_block.variance / results_stationary.variance;
                
                obj.assertTrue(var_ratio > 0.5 && var_ratio < 2.0, ...
                    'Block and stationary bootstrap should give similar variance estimates');
                
                results.tests.bootstrapConsistencyTest = struct(...
                    'passed', true, ...
                    'message', 'Block and stationary bootstrap variance estimates are consistent', ...
                    'block_variance', results_block.variance, ...
                    'stationary_variance', results_stationary.variance, ...
                    'variance_ratio', var_ratio ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.bootstrapConsistencyTest = struct(...
                    'passed', false, ...
                    'message', ['Bootstrap consistency test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 3: Variance estimation for different statistics
            try
                % Use simulated time series
                data = obj.simulatedTimeSeries;
                
                % Define different statistics
                stats = struct();
                stats.mean = @mean;
                stats.std = @std;
                stats.median = @median;
                stats.q90 = @(x) prctile(x, 90);
                
                % Set bootstrap options
                options = struct();
                options.bootstrap_type = 'block';
                options.block_size = 10;
                options.replications = 500;
                
                % Estimate variance for each statistic
                stat_results = struct();
                for field = fieldnames(stats)'
                    fn_name = field{1};
                    stat_fn = stats.(fn_name);
                    result = bootstrap_variance(data, stat_fn, options);
                    stat_results.(fn_name) = result;
                end
                
                % Verify all variance estimates are positive
                all_positive = true;
                for field = fieldnames(stat_results)'
                    fn_name = field{1};
                    if stat_results.(fn_name).variance <= 0
                        all_positive = false;
                        break;
                    end
                end
                
                obj.assertTrue(all_positive, 'All variance estimates should be positive');
                
                % Verify confidence intervals contain the statistic
                all_contained = true;
                for field = fieldnames(stat_results)'
                    fn_name = field{1};
                    result = stat_results.(fn_name);
                    statistic = stats.(fn_name)(data);
                    if statistic < result.conf_lower || statistic > result.conf_upper
                        all_contained = false;
                        break;
                    end
                end
                
                obj.assertTrue(all_contained, 'Confidence intervals should contain the original statistic');
                
                results.tests.multipleStatisticsTest = struct(...
                    'passed', true, ...
                    'message', 'Bootstrap variance estimation works for multiple statistics', ...
                    'statistic_results', stat_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.multipleStatisticsTest = struct(...
                    'passed', false, ...
                    'message', ['Multiple statistics test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 4: Effect of replication count on estimate precision
            try
                % Use financial returns data
                data = obj.financialReturnsData;
                
                % Define mean function
                mean_fn = @mean;
                
                % Set bootstrap options with different replication counts
                rep_counts = [100, 500, 1000];
                rep_results = cell(length(rep_counts), 1);
                
                for i = 1:length(rep_counts)
                    options = struct();
                    options.bootstrap_type = 'block';
                    options.block_size = 10;
                    options.replications = rep_counts(i);
                    
                    % Estimate variance using bootstrap_variance
                    result = bootstrap_variance(data, mean_fn, options);
                    rep_results{i} = result;
                end
                
                % Verify that results converge with increasing replications
                % Compare variance of the variance estimates
                bootstrap_vars = zeros(rep_counts(end), 1);
                bs_data = block_bootstrap(data, 10, rep_counts(end));
                for i = 1:rep_counts(end)
                    bootstrap_vars(i) = mean_fn(bs_data(:, :, i));
                end
                
                % Calculate actual variance (reference)
                ref_variance = var(bootstrap_vars);
                
                % Calculate error for each replication count
                var_errors = zeros(length(rep_counts), 1);
                for i = 1:length(rep_counts)
                    var_errors(i) = abs(rep_results{i}.variance - ref_variance) / ref_variance;
                end
                
                % Verify errors decrease with increasing replications
                is_decreasing = (var_errors(1) >= var_errors(2)) && (var_errors(2) >= var_errors(3));
                obj.assertTrue(is_decreasing, 'Estimation error should decrease with increasing replications');
                
                results.tests.replicationCountTest = struct(...
                    'passed', true, ...
                    'message', 'Variance estimates converge with increasing replication counts', ...
                    'replication_counts', rep_counts, ...
                    'variance_errors', var_errors ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.replicationCountTest = struct(...
                    'passed', false, ...
                    'message', ['Replication count test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 5: Validate error handling for invalid inputs
            try
                % Test with invalid statistic function
                obj.assertThrows(@() bootstrap_variance(obj.financialReturnsData, 'not_a_function', struct()), ...
                    'MATLAB:error', 'Should reject non-function handle statistic');
                
                % Test with invalid bootstrap type
                options = struct('bootstrap_type', 'invalid');
                obj.assertThrows(@() bootstrap_variance(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject invalid bootstrap type');
                
                % Test with negative block size
                options = struct('bootstrap_type', 'block', 'block_size', -5);
                obj.assertThrows(@() bootstrap_variance(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject negative block size');
                
                % Test with invalid p value
                options = struct('bootstrap_type', 'stationary', 'p', 2);
                obj.assertThrows(@() bootstrap_variance(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject p value > 1');
                
                results.tests.errorHandlingTest = struct(...
                    'passed', true, ...
                    'message', 'Error handling for invalid inputs works correctly' ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.errorHandlingTest = struct(...
                    'passed', false, ...
                    'message', ['Error handling test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Store validation results
            obj.validationResults.bootstrap_variance = results;
            
            % Display summary
            if obj.verbose
                disp(['Bootstrap variance validation: ' num2str(results.passed) '/' num2str(results.total) ' tests passed.']);
            end
            
            return;
        end
        
        function results = validateBootstrapConfidenceIntervals(obj)
            % Validates bootstrap_confidence_intervals implementation for accuracy in interval estimation
            %
            % OUTPUTS:
            %   results - Validation results for bootstrap_confidence_intervals function
            
            if obj.verbose
                disp('Validating bootstrap_confidence_intervals function...');
            end
            
            % Initialize results structure
            results = struct('tests', struct(), 'passed', 0, 'failed', 0, 'total', 0);
            
            % Test 1: Coverage properties for normal data with known parameters
            try
                % Generate normal data with known mean and variance
                n = 500;
                mu = 2;
                sigma = 0.5;
                data = mu + sigma * randn(n, 1);
                
                % Calculate theoretical confidence interval for the mean
                alpha = 0.05;  % 95% confidence
                z_score = norminv(1 - alpha/2);
                theoretical_lower = mu - z_score * sigma / sqrt(n);
                theoretical_upper = mu + z_score * sigma / sqrt(n);
                
                % Define mean function
                mean_fn = @mean;
                
                % Set bootstrap options for all methods
                methods = {'percentile', 'basic', 'bc', 'bca'};
                ci_results = struct();
                
                for i = 1:length(methods)
                    method = methods{i};
                    options = struct();
                    options.bootstrap_type = 'block';
                    options.block_size = 10;
                    options.replications = 1000;
                    options.conf_level = 1 - alpha;
                    options.method = method;
                    
                    % Estimate confidence interval
                    ci_result = bootstrap_confidence_intervals(data, mean_fn, options);
                    ci_results.(method) = ci_result;
                    
                    % Check that confidence interval contains the true mean
                    contains_mean = (ci_result.lower <= mu && ci_result.upper >= mu);
                    
                    % Calculate width ratio compared to theoretical
                    theoretical_width = theoretical_upper - theoretical_lower;
                    ci_width = ci_result.upper - ci_result.lower;
                    width_ratio = ci_width / theoretical_width;
                    
                    ci_results.(method).contains_mean = contains_mean;
                    ci_results.(method).width_ratio = width_ratio;
                end
                
                % Verify at least one method contains the true mean
                any_contains = false;
                for i = 1:length(methods)
                    method = methods{i};
                    if ci_results.(method).contains_mean
                        any_contains = true;
                        break;
                    end
                end
                
                obj.assertTrue(any_contains, 'At least one CI method should contain the true mean');
                
                % Verify all width ratios are reasonable
                all_reasonable = true;
                for i = 1:length(methods)
                    method = methods{i};
                    if ci_results.(method).width_ratio < 0.5 || ci_results.(method).width_ratio > 2.0
                        all_reasonable = false;
                        break;
                    end
                end
                
                obj.assertTrue(all_reasonable, 'All CI methods should produce reasonable width ratios');
                
                results.tests.normalCoverageTest = struct(...
                    'passed', true, ...
                    'message', 'Bootstrap confidence intervals show proper coverage for normal data', ...
                    'theoretical_interval', [theoretical_lower, theoretical_upper], ...
                    'ci_results', ci_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.normalCoverageTest = struct(...
                    'passed', false, ...
                    'message', ['Normal coverage test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 2: Different confidence levels
            try
                % Use financial returns data
                data = obj.financialReturnsData;
                
                % Define median function
                median_fn = @median;
                
                % Set bootstrap options with different confidence levels
                conf_levels = [0.90, 0.95, 0.99];
                conf_results = struct();
                
                for i = 1:length(conf_levels)
                    conf_level = conf_levels(i);
                    options = struct();
                    options.bootstrap_type = 'block';
                    options.block_size = 10;
                    options.replications = 1000;
                    options.conf_level = conf_level;
                    options.method = 'percentile';
                    
                    % Estimate confidence interval
                    ci_result = bootstrap_confidence_intervals(data, median_fn, options);
                    conf_results.(sprintf('conf_%g', conf_level*100)) = ci_result;
                    
                    % Calculate interval width
                    ci_width = ci_result.upper - ci_result.lower;
                    conf_results.(sprintf('conf_%g', conf_level*100)).width = ci_width;
                end
                
                % Verify interval widths increase with confidence level
                width_90 = conf_results.conf_90.width;
                width_95 = conf_results.conf_95.width;
                width_99 = conf_results.conf_99.width;
                
                is_increasing = (width_90 < width_95) && (width_95 < width_99);
                obj.assertTrue(is_increasing, 'Interval width should increase with confidence level');
                
                results.tests.confidenceLevelTest = struct(...
                    'passed', true, ...
                    'message', 'Confidence interval widths increase with confidence level', ...
                    'conf_results', conf_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.confidenceLevelTest = struct(...
                    'passed', false, ...
                    'message', ['Confidence level test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 3: Comparison of different CI methods
            try
                % Use simulated time series
                data = obj.simulatedTimeSeries;
                
                % Define Sharpe ratio function (mean/std)
                sharpe_fn = @(x) mean(x) / std(x);
                
                % Set bootstrap options
                methods = {'percentile', 'basic', 'bc', 'bca'};
                method_results = struct();
                
                for i = 1:length(methods)
                    method = methods{i};
                    options = struct();
                    options.bootstrap_type = 'block';
                    options.block_size = 10;
                    options.replications = 1000;
                    options.conf_level = 0.95;
                    options.method = method;
                    
                    % Estimate confidence interval
                    ci_result = bootstrap_confidence_intervals(data, sharpe_fn, options);
                    method_results.(method) = ci_result;
                    
                    % Calculate interval width
                    ci_width = ci_result.upper - ci_result.lower;
                    method_results.(method).width = ci_width;
                    
                    % Check if interval contains original statistic
                    original_stat = sharpe_fn(data);
                    contains_original = (ci_result.lower <= original_stat && ci_result.upper >= original_stat);
                    method_results.(method).contains_original = contains_original;
                end
                
                % Verify all methods contain the original statistic
                all_contain = true;
                for i = 1:length(methods)
                    method = methods{i};
                    if ~method_results.(method).contains_original
                        all_contain = false;
                        break;
                    end
                end
                
                obj.assertTrue(all_contain, 'All CI methods should contain the original statistic');
                
                % Verify BCa interval width is reasonable compared to percentile
                width_percentile = method_results.percentile.width;
                width_bca = method_results.bca.width;
                width_ratio = width_bca / width_percentile;
                
                obj.assertTrue(width_ratio > 0.5 && width_ratio < 2.0, 'BCa width should be comparable to percentile width');
                
                results.tests.methodComparisonTest = struct(...
                    'passed', true, ...
                    'message', 'Different CI methods produce reasonable and consistent results', ...
                    'method_results', method_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.methodComparisonTest = struct(...
                    'passed', false, ...
                    'message', ['Method comparison test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 4: Consistency between block and stationary bootstrap
            try
                % Use financial returns data
                data = obj.financialReturnsData;
                
                % Define median function
                median_fn = @median;
                
                % Set bootstrap options
                options_block = struct();
                options_block.bootstrap_type = 'block';
                options_block.block_size = 10;
                options_block.replications = 1000;
                options_block.conf_level = 0.95;
                options_block.method = 'percentile';
                
                options_stationary = struct();
                options_stationary.bootstrap_type = 'stationary';
                options_stationary.p = 0.1;  % Expected block length of 10
                options_stationary.replications = 1000;
                options_stationary.conf_level = 0.95;
                options_stationary.method = 'percentile';
                
                % Estimate confidence intervals using both methods
                ci_block = bootstrap_confidence_intervals(data, median_fn, options_block);
                ci_stationary = bootstrap_confidence_intervals(data, median_fn, options_stationary);
                
                % Calculate interval widths
                width_block = ci_block.upper - ci_block.lower;
                width_stationary = ci_stationary.upper - ci_stationary.lower;
                width_ratio = width_block / width_stationary;
                
                % Check overlap between intervals
                overlap_lower = max(ci_block.lower, ci_stationary.lower);
                overlap_upper = min(ci_block.upper, ci_stationary.upper);
                has_overlap = overlap_lower <= overlap_upper;
                
                obj.assertTrue(has_overlap, 'Block and stationary bootstrap CIs should overlap');
                obj.assertTrue(width_ratio > 0.7 && width_ratio < 1.3, 'Block and stationary bootstrap CI widths should be similar');
                
                results.tests.bootstrapConsistencyTest = struct(...
                    'passed', true, ...
                    'message', 'Block and stationary bootstrap produce consistent confidence intervals', ...
                    'ci_block', ci_block, ...
                    'ci_stationary', ci_stationary, ...
                    'width_ratio', width_ratio, ...
                    'has_overlap', has_overlap ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.bootstrapConsistencyTest = struct(...
                    'passed', false, ...
                    'message', ['Bootstrap consistency test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 5: Validate error handling for invalid inputs
            try
                % Test with invalid statistic function
                obj.assertThrows(@() bootstrap_confidence_intervals(obj.financialReturnsData, 'not_a_function', struct()), ...
                    'MATLAB:error', 'Should reject non-function handle statistic');
                
                % Test with invalid bootstrap type
                options = struct('bootstrap_type', 'invalid');
                obj.assertThrows(@() bootstrap_confidence_intervals(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject invalid bootstrap type');
                
                % Test with invalid confidence level
                options = struct('conf_level', 1.5);
                obj.assertThrows(@() bootstrap_confidence_intervals(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject confidence level > 1');
                
                % Test with invalid CI method
                options = struct('method', 'invalid_method');
                obj.assertThrows(@() bootstrap_confidence_intervals(obj.financialReturnsData, @mean, options), ...
                    'MATLAB:error', 'Should reject invalid CI method');
                
                results.tests.errorHandlingTest = struct(...
                    'passed', true, ...
                    'message', 'Error handling for invalid inputs works correctly' ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.errorHandlingTest = struct(...
                    'passed', false, ...
                    'message', ['Error handling test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Store validation results
            obj.validationResults.bootstrap_confidence_intervals = results;
            
            % Display summary
            if obj.verbose
                disp(['Bootstrap confidence intervals validation: ' num2str(results.passed) '/' num2str(results.total) ' tests passed.']);
            end
            
            return;
        end
        
        function results = validateDependencePreservation(obj)
            % Validates that bootstrap methods correctly preserve time series dependence structure
            %
            % OUTPUTS:
            %   results - Validation results for dependence preservation
            
            if obj.verbose
                disp('Validating dependence preservation in bootstrap methods...');
            end
            
            % Initialize results structure
            results = struct('tests', struct(), 'passed', 0, 'failed', 0, 'total', 0);
            
            % Test 1: Block bootstrap dependence preservation
            try
                % Generate AR(1) series with known autocorrelation
                ar_coeff = 0.7;
                ar_series = obj.generateTestTimeSeries(1000, ar_coeff, 'AR');
                
                % Calculate theoretical autocorrelation function for AR(1)
                lags = 0:10;
                theoretical_acf = ar_coeff .^ lags;
                
                % Calculate sample ACF
                sample_acf = autocorr(ar_series, 10);
                
                % Apply block bootstrap with different block sizes
                block_sizes = [5, 10, 20];
                num_replications = 500;
                
                block_results = struct();
                for i = 1:length(block_sizes)
                    bs = block_sizes(i);
                    bs_data = block_bootstrap(ar_series, bs, num_replications);
                    
                    % Calculate ACF for each bootstrap sample
                    bs_acf = zeros(11, num_replications);
                    for j = 1:num_replications
                        bs_acf(:, j) = autocorr(bs_data(:, 1, j), 10);
                    end
                    
                    % Calculate mean and standard deviation of bootstrap ACFs
                    mean_bs_acf = mean(bs_acf, 2);
                    std_bs_acf = std(bs_acf, 0, 2);
                    
                    % Calculate error in ACF preservation
                    acf_error = abs(mean_bs_acf - sample_acf);
                    
                    block_results.(sprintf('block_%d', bs)) = struct(...
                        'mean_acf', mean_bs_acf, ...
                        'std_acf', std_bs_acf, ...
                        'acf_error', acf_error ...
                    );
                end
                
                % Verify that larger block sizes better preserve long-lag autocorrelation
                error_lag5_bs5 = block_results.block_5.acf_error(6);  % error at lag 5 for block size 5
                error_lag5_bs10 = block_results.block_10.acf_error(6); % error at lag 5 for block size 10
                error_lag5_bs20 = block_results.block_20.acf_error(6); % error at lag 5 for block size 20
                
                is_improving = (error_lag5_bs5 >= error_lag5_bs10) && (error_lag5_bs10 >= error_lag5_bs20);
                obj.assertTrue(is_improving, 'Larger block sizes should better preserve long-lag autocorrelation');
                
                results.tests.blockDependenceTest = struct(...
                    'passed', true, ...
                    'message', 'Block bootstrap preserves dependence structure according to block size', ...
                    'theoretical_acf', theoretical_acf, ...
                    'sample_acf', sample_acf, ...
                    'block_results', block_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.blockDependenceTest = struct(...
                    'passed', false, ...
                    'message', ['Block dependence test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 2: Stationary bootstrap dependence preservation
            try
                % Use the same AR(1) series
                ar_coeff = 0.7;
                ar_series = obj.generateTestTimeSeries(1000, ar_coeff, 'AR');
                
                % Calculate sample ACF
                sample_acf = autocorr(ar_series, 10);
                
                % Apply stationary bootstrap with different p values
                p_values = [0.05, 0.1, 0.2];  % Expected block lengths: 20, 10, 5
                num_replications = 500;
                
                stationary_results = struct();
                for i = 1:length(p_values)
                    p = p_values(i);
                    bs_data = stationary_bootstrap(ar_series, p, num_replications);
                    
                    % Calculate ACF for each bootstrap sample
                    bs_acf = zeros(11, num_replications);
                    for j = 1:num_replications
                        bs_acf(:, j) = autocorr(bs_data(:, 1, j), 10);
                    end
                    
                    % Calculate mean and standard deviation of bootstrap ACFs
                    mean_bs_acf = mean(bs_acf, 2);
                    std_bs_acf = std(bs_acf, 0, 2);
                    
                    % Calculate error in ACF preservation
                    acf_error = abs(mean_bs_acf - sample_acf);
                    
                    stationary_results.(sprintf('p_%g', p)) = struct(...
                        'mean_acf', mean_bs_acf, ...
                        'std_acf', std_bs_acf, ...
                        'acf_error', acf_error ...
                    );
                end
                
                % Verify that smaller p values (longer expected blocks) better preserve autocorrelation
                error_lag5_p20 = stationary_results.p_0_2.acf_error(6);  % error at lag 5 for p=0.2
                error_lag5_p10 = stationary_results.p_0_1.acf_error(6);  % error at lag 5 for p=0.1
                error_lag5_p05 = stationary_results.p_0_05.acf_error(6); % error at lag 5 for p=0.05
                
                is_improving = (error_lag5_p20 >= error_lag5_p10) && (error_lag5_p10 >= error_lag5_p05);
                obj.assertTrue(is_improving, 'Smaller p values should better preserve long-lag autocorrelation');
                
                results.tests.stationaryDependenceTest = struct(...
                    'passed', true, ...
                    'message', 'Stationary bootstrap preserves dependence structure according to p value', ...
                    'sample_acf', sample_acf, ...
                    'stationary_results', stationary_results ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.stationaryDependenceTest = struct(...
                    'passed', false, ...
                    'message', ['Stationary dependence test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Test 3: Comparison between block and stationary bootstrap
            try
                % Use AR(1) series from previous tests
                ar_coeff = 0.7;
                ar_series = obj.generateTestTimeSeries(1000, ar_coeff, 'AR');
                
                % Calculate sample ACF
                sample_acf = autocorr(ar_series, 10);
                
                % Apply both bootstrap methods with comparable parameters
                block_size = 10;
                p_value = 0.1;  % Expected block length of 10
                num_replications = 500;
                
                bs_block = block_bootstrap(ar_series, block_size, num_replications);
                bs_stationary = stationary_bootstrap(ar_series, p_value, num_replications);
                
                % Calculate ACF for each bootstrap sample
                block_acf = zeros(11, num_replications);
                stationary_acf = zeros(11, num_replications);
                
                for j = 1:num_replications
                    block_acf(:, j) = autocorr(bs_block(:, 1, j), 10);
                    stationary_acf(:, j) = autocorr(bs_stationary(:, 1, j), 10);
                end
                
                % Calculate mean ACFs
                mean_block_acf = mean(block_acf, 2);
                mean_stationary_acf = mean(stationary_acf, 2);
                
                % Calculate errors
                block_error = abs(mean_block_acf - sample_acf);
                stationary_error = abs(mean_stationary_acf - sample_acf);
                
                % Calculate relative performance for each lag
                relative_performance = block_error ./ stationary_error;
                
                % Calculate overall error (mean absolute error across lags)
                block_mae = mean(block_error);
                stationary_mae = mean(stationary_error);
                
                % Verify that errors are comparable
                error_ratio = block_mae / stationary_mae;
                comparable = (error_ratio > 0.7 && error_ratio < 1.3);
                
                obj.assertTrue(comparable, 'Block and stationary bootstrap should have comparable dependence preservation');
                
                results.tests.comparisonTest = struct(...
                    'passed', true, ...
                    'message', 'Block and stationary bootstrap show comparable dependence preservation', ...
                    'block_error', block_error, ...
                    'stationary_error', stationary_error, ...
                    'relative_performance', relative_performance, ...
                    'block_mae', block_mae, ...
                    'stationary_mae', stationary_mae, ...
                    'error_ratio', error_ratio ...
                );
                results.passed = results.passed + 1;
            catch ME
                results.tests.comparisonTest = struct(...
                    'passed', false, ...
                    'message', ['Comparison test failed: ' ME.message] ...
                );
                results.failed = results.failed + 1;
            end
            results.total = results.total + 1;
            
            % Store validation results
            if ~isfield(obj.validationResults, 'dependence_preservation')
                obj.validationResults.dependence_preservation = results;
            end
            
            % Display summary
            if obj.verbose
                disp(['Dependence preservation validation: ' num2str(results.passed) '/' num2str(results.total) ' tests passed.']);
            end
            
            return;
        end
        
        function results = validateAllBootstrapMethods(obj)
            % Runs comprehensive validation for all bootstrap method implementations
            %
            % OUTPUTS:
            %   results - Complete validation results for all bootstrap methods
            
            if obj.verbose
                disp('Running comprehensive validation for all bootstrap methods...');
            end
            
            % Validate each bootstrap method
            block_results = obj.validateBlockBootstrap();
            stationary_results = obj.validateStationaryBootstrap();
            variance_results = obj.validateBootstrapVariance();
            ci_results = obj.validateBootstrapConfidenceIntervals();
            dependence_results = obj.validateDependencePreservation();
            
            % Combine all results
            results = struct(...
                'block_bootstrap', block_results, ...
                'stationary_bootstrap', stationary_results, ...
                'bootstrap_variance', variance_results, ...
                'bootstrap_confidence_intervals', ci_results, ...
                'dependence_preservation', dependence_results ...
            );
            
            % Calculate overall pass rate
            total_tests = block_results.total + stationary_results.total + ...
                         variance_results.total + ci_results.total + ...
                         dependence_results.total;
                     
            total_passed = block_results.passed + stationary_results.passed + ...
                          variance_results.passed + ci_results.passed + ...
                          dependence_results.passed;
                      
            pass_rate = total_passed / total_tests;
            
            results.summary = struct(...
                'total_tests', total_tests, ...
                'total_passed', total_passed, ...
                'pass_rate', pass_rate ...
            );
            
            % Store complete validation results
            obj.validationResults = results;
            
            % Display overall summary
            if obj.verbose
                obj.displayValidationSummary();
            end
            
            return;
        end
        
        function displayValidationSummary(obj)
            % Displays summary of bootstrap validation results
            
            % Ensure validation results exist
            if ~isstruct(obj.validationResults) || isempty(fieldnames(obj.validationResults))
                disp('No validation results available. Run validateAllBootstrapMethods first.');
                return;
            end
            
            % Check if summary field exists (after running validateAllBootstrapMethods)
            if isfield(obj.validationResults, 'summary')
                summary = obj.validationResults.summary;
                
                % Display overall summary
                disp('===============================================');
                disp('BOOTSTRAP METHODS VALIDATION SUMMARY');
                disp('===============================================');
                disp(['Total Tests: ' num2str(summary.total_tests)]);
                disp(['Tests Passed: ' num2str(summary.total_passed)]);
                disp(['Pass Rate: ' num2str(summary.pass_rate * 100, '%.1f') '%']);
                disp('===============================================');
                
                % Display results for each method
                methods = {'block_bootstrap', 'stationary_bootstrap', 'bootstrap_variance', ...
                           'bootstrap_confidence_intervals', 'dependence_preservation'};
                
                for i = 1:length(methods)
                    method = methods{i};
                    if isfield(obj.validationResults, method)
                        method_result = obj.validationResults.(method);
                        method_pass_rate = method_result.passed / method_result.total * 100;
                        
                        disp(sprintf('\n%s:', strrep(method, '_', ' ')));
                        disp(['  Tests: ' num2str(method_result.total)]);
                        disp(['  Passed: ' num2str(method_result.passed)]);
                        disp(['  Pass Rate: ' num2str(method_pass_rate, '%.1f') '%']);
                        
                        % Display individual test results if verbose is high
                        if obj.verbose
                            disp('  Test details:');
                            test_fields = fieldnames(method_result.tests);
                            for j = 1:length(test_fields)
                                test_name = test_fields{j};
                                test = method_result.tests.(test_name);
                                status = 'PASS';
                                if ~test.passed
                                    status = 'FAIL';
                                end
                                disp(['    - ' test_name ': ' status ' - ' test.message]);
                            end
                        end
                    end
                end
                
                disp('===============================================');
            else
                % Calculate summary from individual method results
                methods = fieldnames(obj.validationResults);
                total_tests = 0;
                total_passed = 0;
                
                disp('===============================================');
                disp('BOOTSTRAP METHODS VALIDATION SUMMARY');
                disp('===============================================');
                
                for i = 1:length(methods)
                    method = methods{i};
                    method_result = obj.validationResults.(method);
                    
                    if isfield(method_result, 'passed') && isfield(method_result, 'total')
                        total_tests = total_tests + method_result.total;
                        total_passed = total_passed + method_result.passed;
                        
                        method_pass_rate = method_result.passed / method_result.total * 100;
                        disp(sprintf('%s: %d/%d (%.1f%%)', ...
                            strrep(method, '_', ' '), method_result.passed, method_result.total, method_pass_rate));
                    end
                end
                
                if total_tests > 0
                    overall_pass_rate = total_passed / total_tests * 100;
                    disp('-----------------------------------------------');
                    disp(sprintf('Overall: %d/%d (%.1f%%)', total_passed, total_tests, overall_pass_rate));
                end
                
                disp('===============================================');
            end
        end
        
        function report = generateValidationReport(obj)
            % Generates structured validation report for bootstrap methods
            %
            % OUTPUTS:
            %   report - Comprehensive validation report structure
            
            % Ensure validation results exist
            if ~isstruct(obj.validationResults) || isempty(fieldnames(obj.validationResults))
                error('No validation results available. Run validateAllBootstrapMethods first.');
            end
            
            % Initialize report structure
            report = struct();
            
            % Add validation timestamp
            report.timestamp = datestr(now);
            report.toolbox_version = '4.0';  % MFE Toolbox version
            
            % Add summary statistics
            if isfield(obj.validationResults, 'summary')
                report.summary = obj.validationResults.summary;
            else
                % Calculate summary from individual method results
                methods = fieldnames(obj.validationResults);
                total_tests = 0;
                total_passed = 0;
                
                for i = 1:length(methods)
                    method = methods{i};
                    method_result = obj.validationResults.(method);
                    
                    if isfield(method_result, 'passed') && isfield(method_result, 'total')
                        total_tests = total_tests + method_result.total;
                        total_passed = total_passed + method_result.passed;
                    end
                end
                
                if total_tests > 0
                    pass_rate = total_passed / total_tests;
                else
                    pass_rate = 0;
                end
                
                report.summary = struct(...
                    'total_tests', total_tests, ...
                    'total_passed', total_passed, ...
                    'pass_rate', pass_rate ...
                );
            end
            
            % Add detailed results for each method
            report.methods = struct();
            
            if isfield(obj.validationResults, 'block_bootstrap')
                report.methods.block_bootstrap = obj.summarizeMethodValidation(obj.validationResults.block_bootstrap);
            end
            
            if isfield(obj.validationResults, 'stationary_bootstrap')
                report.methods.stationary_bootstrap = obj.summarizeMethodValidation(obj.validationResults.stationary_bootstrap);
            end
            
            if isfield(obj.validationResults, 'bootstrap_variance')
                report.methods.bootstrap_variance = obj.summarizeMethodValidation(obj.validationResults.bootstrap_variance);
            end
            
            if isfield(obj.validationResults, 'bootstrap_confidence_intervals')
                report.methods.bootstrap_confidence_intervals = obj.summarizeMethodValidation(obj.validationResults.bootstrap_confidence_intervals);
            end
            
            if isfield(obj.validationResults, 'dependence_preservation')
                report.methods.dependence_preservation = obj.summarizeMethodValidation(obj.validationResults.dependence_preservation);
            end
            
            return;
        end
        
        function method_summary = summarizeMethodValidation(obj, method_results)
            % Helper function to summarize validation results for a single method
            %
            % INPUTS:
            %   method_results - Validation results for a specific bootstrap method
            %
            % OUTPUTS:
            %   method_summary - Summarized validation results
            
            method_summary = struct();
            
            % Add basic summary statistics
            if isfield(method_results, 'passed') && isfield(method_results, 'total')
                method_summary.tests_total = method_results.total;
                method_summary.tests_passed = method_results.passed;
                method_summary.tests_failed = method_results.failed;
                
                if method_results.total > 0
                    method_summary.pass_rate = method_results.passed / method_results.total;
                else
                    method_summary.pass_rate = 0;
                end
            end
            
            % Add individual test results
            if isfield(method_results, 'tests')
                test_fields = fieldnames(method_results.tests);
                
                method_summary.tests = struct();
                for i = 1:length(test_fields)
                    test_name = test_fields{i};
                    test = method_results.tests.(test_name);
                    
                    % Add basic test information
                    method_summary.tests.(test_name) = struct(...
                        'passed', test.passed, ...
                        'message', test.message ...
                    );
                    
                    % Add additional test-specific fields while excluding large data
                    test_fields_to_include = setdiff(fieldnames(test), {'passed', 'message'});
                    for j = 1:length(test_fields_to_include)
                        field = test_fields_to_include{j};
                        % Skip very large fields to keep report manageable
                        field_value = test.(field);
                        if isnumeric(field_value) && numel(field_value) > 1000
                            % Summarize large fields instead of including them directly
                            if isvector(field_value)
                                method_summary.tests.(test_name).(field) = struct(...
                                    'size', size(field_value), ...
                                    'min', min(field_value), ...
                                    'max', max(field_value), ...
                                    'mean', mean(field_value), ...
                                    'std', std(field_value) ...
                                );
                            else
                                method_summary.tests.(test_name).(field) = struct(...
                                    'size', size(field_value), ...
                                    'summary', 'Large matrix summarized' ...
                                );
                            end
                        else
                            method_summary.tests.(test_name).(field) = field_value;
                        end
                    end
                end
            end
            
            % Add method-specific fields while excluding very large fields
            additional_fields = setdiff(fieldnames(method_results), {'tests', 'passed', 'failed', 'total'});
            for i = 1:length(additional_fields)
                field = additional_fields{i};
                field_value = method_results.(field);
                
                % Skip very large fields
                if isstruct(field_value) && isfield(field_value, 'size_5') && isfield(field_value, 'size_10')
                    % This is likely block_size_results or similar, keep it
                    method_summary.(field) = field_value;
                elseif isstruct(field_value) && isfield(field_value, 'p_0_1') && isfield(field_value, 'p_0_2')
                    % This is likely p_value_results or similar, keep it
                    method_summary.(field) = field_value;
                elseif isnumeric(field_value) && numel(field_value) > 1000
                    % Summarize large numeric fields
                    method_summary.(field) = struct(...
                        'size', size(field_value), ...
                        'min', min(field_value(:)), ...
                        'max', max(field_value(:)), ...
                        'mean', mean(field_value(:)), ...
                        'std', std(field_value(:)) ...
                    );
                else
                    % Include other fields directly
                    method_summary.(field) = field_value;
                end
            end
            
            return;
        end
        
        function series = generateTestTimeSeries(obj, numObservations, autocorrelation, type)
            % Helper method to generate time series with known dependence structure for testing
            %
            % INPUTS:
            %   numObservations - Length of time series to generate
            %   autocorrelation - Target autocorrelation at lag 1
            %   type - Type of series: 'AR', 'MA', or 'ARMA'
            %
            % OUTPUTS:
            %   series - Generated time series with controlled properties
            
            % Initialize with normal innovations
            innovations = randn(numObservations + 100, 1);
            
            % Generate series based on specified type
            switch upper(type)
                case 'AR'
                    % AR(1) process: y(t) = phi * y(t-1) + e(t)
                    % Ensure |phi| < 1 for stationarity
                    phi = sign(autocorrelation) * min(abs(autocorrelation), 0.99);
                    
                    % Generate AR(1) series
                    series = zeros(numObservations + 100, 1);
                    for t = 2:length(series)
                        series(t) = phi * series(t-1) + innovations(t);
                    end
                    
                case 'MA'
                    % MA(1) process: y(t) = e(t) + theta * e(t-1)
                    % For MA(1), acf(1) = theta / (1 + theta^2)
                    % Solve for theta given target autocorrelation
                    if abs(autocorrelation) >= 0.5
                        % Maximum autocorrelation for MA(1) is 0.5, adjust if exceeded
                        autocorrelation = sign(autocorrelation) * 0.499;
                    end
                    
                    % Solve quadratic equation for theta
                    % theta / (1 + theta^2) = acf(1)
                    % theta - acf(1) * theta^2 - acf(1) = 0
                    a = -autocorrelation;
                    b = 1;
                    c = -autocorrelation;
                    
                    % Use quadratic formula, choose solution with |theta| < 1 for invertibility
                    theta1 = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
                    theta2 = (-b - sqrt(b^2 - 4*a*c)) / (2*a);
                    
                    if abs(theta1) < 1
                        theta = theta1;
                    else
                        theta = theta2;
                    end
                    
                    % Generate MA(1) series
                    series = innovations(1:end-1) + theta * innovations(2:end);
                    
                case 'ARMA'
                    % ARMA(1,1) process: y(t) = phi * y(t-1) + e(t) + theta * e(t-1)
                    % Choose phi and theta to achieve target autocorrelation
                    % Set theta to be related to phi to create a specific autocorrelation pattern
                    phi = sign(autocorrelation) * min(abs(autocorrelation), 0.9);
                    theta = -0.3 * phi;  % Arbitrary relation to create realistic ARMA dynamics
                    
                    % Generate ARMA(1,1) series
                    series = zeros(numObservations + 100, 1);
                    for t = 2:length(series)
                        series(t) = phi * series(t-1) + innovations(t) + theta * innovations(t-1);
                    end
                    
                otherwise
                    error('Unknown time series type. Use ''AR'', ''MA'', or ''ARMA''.');
            end
            
            % Discard burn-in samples
            series = series(101:end);
            
            % Standardize to unit variance
            series = series / std(series);
            
            % Check that we achieved approximately the desired autocorrelation
            actual_acf = autocorr(series, 1);
            actual_acf = actual_acf(2);  % acf at lag 1
            
            if abs(actual_acf - autocorrelation) > 0.1
                warning('Achieved autocorrelation (%.2f) differs from target (%.2f).', actual_acf, autocorrelation);
            end
            
            return;
        end
    end
end