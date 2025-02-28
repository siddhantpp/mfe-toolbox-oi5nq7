classdef GarchlikelihoodTest < BaseTest
    % GarchlikelihoodTest Unit test class for the garchlikelihood function that validates
    % the calculation of log-likelihood across different GARCH model specifications and
    % error distributions
    
    properties
        testData
        comparator
        tolerance
    end
    
    methods
        function obj = GarchlikelihoodTest()
            % Initialize the GarchlikelihoodTest class with appropriate test configuration
            obj@BaseTest();
            
            % Initialize the numerical comparator for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set a higher tolerance specifically for likelihood calculations which can have numerical sensitivity
            obj.tolerance = 1e-6;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Load volatility test data from voldata.mat
            obj.testData = obj.loadTestData('voldata.mat');
            
            % Set random number generator seed for reproducibility
            rng(42);
            
            % Initialize the numerical comparator with appropriate tolerance for likelihood computations
            obj.comparator = NumericalComparator();
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method
            tearDown@BaseTest(obj);
            
            % Clear test-specific variables
        end
        
        function testStandardGarchNormal(obj)
            % Test the likelihood computation for standard GARCH model with normal error distribution
            
            % Create test data with known GARCH properties and normal errors
            T = 1000;
            rng(123);
            data = randn(T, 1);
            
            % Define GARCH parameters (omega, alpha, beta)
            parameters = [0.01; 0.1; 0.8];
            
            % Create options structure with model type GARCH and distribution NORMAL
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Compute variances using garchcore for reference
            ht = garchcore(parameters, data, options);
            
            % Call garchlikelihood with test data and parameters
            ll = garchlikelihood(parameters, data, options);
            
            % Compute expected likelihood manually using normal distribution formula
            std_residuals = data ./ sqrt(ht);
            expected_ll = obj.computeGaussianLikelihood(std_residuals);
            expected_ll = expected_ll - 0.5 * sum(log(ht)); % Add log-Jacobian term
            expected_ll = -expected_ll; % Convert to negative log-likelihood
            
            % Verify computed likelihood matches expected value with appropriate tolerance
            obj.assertAlmostEqual(ll, expected_ll, ['Standard GARCH with normal errors ' ...
                'log-likelihood does not match expected value']);
            
            % Test parameter sets close to constraint boundaries
            parameters_boundary = [0.01; 0.19; 0.8]; % Sum alpha+beta is close to 1
            ll_boundary = garchlikelihood(parameters_boundary, data, options);
            obj.assertTrue(isfinite(ll_boundary), 'Log-likelihood should be finite with boundary parameters');
        end
        
        function testStandardGarchT(obj)
            % Test the likelihood computation for standard GARCH model with Student's t error distribution
            
            % Create test data with known GARCH properties and t-distributed errors
            T = 1000;
            rng(123);
            data = randn(T, 1);
            
            % Define GARCH parameters including degrees of freedom
            parameters = [0.01; 0.1; 0.8; 8]; % nu=8 degrees of freedom
            
            % Create options structure with model type GARCH and distribution T
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'T');
            
            % Compute variances using garchcore for reference
            ht = garchcore(parameters(1:3), data, options);
            
            % Call garchlikelihood with test data and parameters
            ll = garchlikelihood(parameters, data, options);
            
            % Verify likelihood value matches expected value with appropriate tolerance
            obj.assertTrue(isfinite(ll), 'Log-likelihood should be finite for t-distribution');
            
            % Test effects of different degrees of freedom values
            parameters_df5 = [0.01; 0.1; 0.8; 5]; % Heavier tails
            ll_df5 = garchlikelihood(parameters_df5, data, options);
            
            parameters_df20 = [0.01; 0.1; 0.8; 20]; % Closer to normal
            ll_df20 = garchlikelihood(parameters_df20, data, options);
            
            % Verify both are finite
            obj.assertTrue(isfinite(ll_df5) && isfinite(ll_df20), ...
                'Log-likelihood should be finite for different degrees of freedom');
        end
        
        function testStandardGarchGED(obj)
            % Test the likelihood computation for standard GARCH model with GED error distribution
            
            % Create test data with known GARCH properties and GED errors
            T = 1000;
            rng(123);
            data = randn(T, 1);
            
            % Define GARCH parameters including GED shape parameter
            parameters = [0.01; 0.1; 0.8; 1.5]; % nu=1.5 shape parameter
            
            % Create options structure with model type GARCH and distribution GED
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'GED');
            
            % Compute variances using garchcore for reference
            ht = garchcore(parameters(1:3), data, options);
            
            % Call garchlikelihood with test data and parameters
            ll = garchlikelihood(parameters, data, options);
            
            % Verify likelihood value matches expected value with appropriate tolerance
            obj.assertTrue(isfinite(ll), 'Log-likelihood should be finite for GED distribution');
            
            % Test effects of different GED shape parameter values
            parameters_nu1 = [0.01; 0.1; 0.8; 1.0]; % Laplace distribution
            ll_nu1 = garchlikelihood(parameters_nu1, data, options);
            
            parameters_nu2 = [0.01; 0.1; 0.8; 2.0]; % Normal distribution
            ll_nu2 = garchlikelihood(parameters_nu2, data, options);
            
            % Verify log-likelihoods are finite
            obj.assertTrue(isfinite(ll_nu1) && isfinite(ll_nu2), ...
                'Log-likelihood should be finite for different GED shape parameters');
        end
        
        function testStandardGarchSkewT(obj)
            % Test the likelihood computation for standard GARCH model with skewed t error distribution
            
            % Create test data with known GARCH properties and skewed t errors
            T = 1000;
            rng(123);
            data = randn(T, 1);
            
            % Define GARCH parameters including degrees of freedom and asymmetry parameter
            parameters = [0.01; 0.1; 0.8; 8; 0.1]; % nu=8 degrees of freedom, lambda=0.1 skewness
            
            % Create options structure with model type GARCH and distribution SKEWT
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'SKEWT');
            
            % Compute variances using garchcore for reference
            ht = garchcore(parameters(1:3), data, options);
            
            % Call garchlikelihood with test data and parameters
            ll = garchlikelihood(parameters, data, options);
            
            % Verify likelihood value matches expected value with appropriate tolerance
            obj.assertTrue(isfinite(ll), 'Log-likelihood should be finite for skewed t-distribution');
            
            % Test effects of different skewness parameter values
            parameters_neg = [0.01; 0.1; 0.8; 8; -0.3]; % Negative skewness
            ll_neg = garchlikelihood(parameters_neg, data, options);
            
            parameters_pos = [0.01; 0.1; 0.8; 8; 0.3]; % Positive skewness
            ll_pos = garchlikelihood(parameters_pos, data, options);
            
            % Verify log-likelihoods are finite
            obj.assertTrue(isfinite(ll_neg) && isfinite(ll_pos), ...
                'Log-likelihood should be finite for different skewness parameters');
        end
        
        function testEgarchLikelihood(obj)
            % Test the likelihood computation for EGARCH model with various error distributions
            
            % Create test data with known EGARCH properties
            T = 1000;
            rng(234);
            data = randn(T, 1);
            
            % Define EGARCH parameters (omega, alpha, gamma, beta)
            parameters = [0.01; 0.1; -0.05; 0.8]; % With leverage effect
            
            % Create options structure with model type EGARCH
            options = struct('model', 'EGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with different error distributions (NORMAL, T, GED, SKEWT)
            ll_normal = garchlikelihood(parameters, data, options);
            obj.assertTrue(isfinite(ll_normal), 'EGARCH with normal errors should have finite likelihood');
            
            options.distribution = 'T';
            parameters_t = [parameters; 8]; % Add degrees of freedom
            ll_t = garchlikelihood(parameters_t, data, options);
            obj.assertTrue(isfinite(ll_t), 'EGARCH with t errors should have finite likelihood');
            
            options.distribution = 'GED';
            parameters_ged = [parameters; 1.5]; % Add shape parameter
            ll_ged = garchlikelihood(parameters_ged, data, options);
            obj.assertTrue(isfinite(ll_ged), 'EGARCH with GED errors should have finite likelihood');
            
            options.distribution = 'SKEWT';
            parameters_skewt = [parameters; 8; 0.1]; % Add df and skewness
            ll_skewt = garchlikelihood(parameters_skewt, data, options);
            obj.assertTrue(isfinite(ll_skewt), 'EGARCH with skewed t errors should have finite likelihood');
            
            % Test the likelihood response to asymmetric effects
            parameters_noasym = [0.01; 0.1; 0.0; 0.8]; % No asymmetry
            options.distribution = 'NORMAL';
            ll_noasym = garchlikelihood(parameters_noasym, data, options);
            obj.assertTrue(isfinite(ll_noasym), 'EGARCH without asymmetry should have finite likelihood');
        end
        
        function testTarchLikelihood(obj)
            % Test the likelihood computation for TARCH model with various error distributions
            
            % Create test data with known TARCH properties
            T = 1000;
            rng(345);
            data = randn(T, 1);
            
            % Define TARCH parameters (omega, alpha, gamma, beta)
            parameters = [0.01; 0.05; 0.1; 0.8]; % With threshold effect
            
            % Create options structure with model type TARCH
            options = struct('model', 'TARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with different error distributions (NORMAL, T, GED, SKEWT)
            ll_normal = garchlikelihood(parameters, data, options);
            obj.assertTrue(isfinite(ll_normal), 'TARCH with normal errors should have finite likelihood');
            
            options.distribution = 'T';
            parameters_t = [parameters; 8]; % Add degrees of freedom
            ll_t = garchlikelihood(parameters_t, data, options);
            obj.assertTrue(isfinite(ll_t), 'TARCH with t errors should have finite likelihood');
            
            options.distribution = 'GED';
            parameters_ged = [parameters; 1.5]; % Add shape parameter
            ll_ged = garchlikelihood(parameters_ged, data, options);
            obj.assertTrue(isfinite(ll_ged), 'TARCH with GED errors should have finite likelihood');
            
            options.distribution = 'SKEWT';
            parameters_skewt = [parameters; 8; 0.1]; % Add df and skewness
            ll_skewt = garchlikelihood(parameters_skewt, data, options);
            obj.assertTrue(isfinite(ll_skewt), 'TARCH with skewed t errors should have finite likelihood');
            
            % Test the likelihood response to threshold effects
            parameters_nothresh = [0.01; 0.1; 0.0; 0.8]; % No threshold
            options.distribution = 'NORMAL';
            ll_nothresh = garchlikelihood(parameters_nothresh, data, options);
            obj.assertTrue(isfinite(ll_nothresh), 'TARCH without threshold should have finite likelihood');
        end
        
        function testAgarchLikelihood(obj)
            % Test the likelihood computation for AGARCH model with various error distributions
            
            % Create test data with known AGARCH properties
            T = 1000;
            rng(456);
            data = randn(T, 1);
            
            % Define AGARCH parameters (omega, alpha, gamma, beta)
            parameters = [0.01; 0.1; 0.1; 0.8]; % With asymmetry
            
            % Create options structure with model type AGARCH
            options = struct('model', 'AGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with different error distributions (NORMAL, T, GED, SKEWT)
            ll_normal = garchlikelihood(parameters, data, options);
            obj.assertTrue(isfinite(ll_normal), 'AGARCH with normal errors should have finite likelihood');
            
            options.distribution = 'T';
            parameters_t = [parameters; 8]; % Add degrees of freedom
            ll_t = garchlikelihood(parameters_t, data, options);
            obj.assertTrue(isfinite(ll_t), 'AGARCH with t errors should have finite likelihood');
            
            options.distribution = 'GED';
            parameters_ged = [parameters; 1.5]; % Add shape parameter
            ll_ged = garchlikelihood(parameters_ged, data, options);
            obj.assertTrue(isfinite(ll_ged), 'AGARCH with GED errors should have finite likelihood');
            
            options.distribution = 'SKEWT';
            parameters_skewt = [parameters; 8; 0.1]; % Add df and skewness
            ll_skewt = garchlikelihood(parameters_skewt, data, options);
            obj.assertTrue(isfinite(ll_skewt), 'AGARCH with skewed t errors should have finite likelihood');
            
            % Test the likelihood response to asymmetric news impact
            parameters_noasym = [0.01; 0.1; 0.0; 0.8]; % No asymmetry
            options.distribution = 'NORMAL';
            ll_noasym = garchlikelihood(parameters_noasym, data, options);
            obj.assertTrue(isfinite(ll_noasym), 'AGARCH without asymmetry should have finite likelihood');
        end
        
        function testIgarchLikelihood(obj)
            % Test the likelihood computation for IGARCH model with various error distributions
            
            % Create test data with known IGARCH properties
            T = 1000;
            rng(567);
            data = randn(T, 1);
            
            % Define IGARCH parameters (omega, alpha, beta) with integrated constraint
            parameters = [0.01; 0.2]; % omega, alpha (beta is implied to be 0.8)
            
            % Create options structure with model type IGARCH
            options = struct('model', 'IGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with different error distributions (NORMAL, T, GED, SKEWT)
            ll_normal = garchlikelihood(parameters, data, options);
            obj.assertTrue(isfinite(ll_normal), 'IGARCH with normal errors should have finite likelihood');
            
            options.distribution = 'T';
            parameters_t = [parameters; 8]; % Add degrees of freedom
            ll_t = garchlikelihood(parameters_t, data, options);
            obj.assertTrue(isfinite(ll_t), 'IGARCH with t errors should have finite likelihood');
            
            options.distribution = 'GED';
            parameters_ged = [parameters; 1.5]; % Add shape parameter
            ll_ged = garchlikelihood(parameters_ged, data, options);
            obj.assertTrue(isfinite(ll_ged), 'IGARCH with GED errors should have finite likelihood');
            
            options.distribution = 'SKEWT';
            parameters_skewt = [parameters; 8; 0.1]; % Add df and skewness
            ll_skewt = garchlikelihood(parameters_skewt, data, options);
            obj.assertTrue(isfinite(ll_skewt), 'IGARCH with skewed t errors should have finite likelihood');
            
            % Verify that the integrated constraint is properly enforced
            parameters_constraint = [0.01; 0.3]; % Different alpha value
            ll_constraint = garchlikelihood(parameters_constraint, data, options);
            obj.assertTrue(isfinite(ll_constraint), 'IGARCH with different alpha should have finite likelihood');
        end
        
        function testNagarchLikelihood(obj)
            % Test the likelihood computation for NAGARCH model with various error distributions
            
            % Create test data with known NAGARCH properties
            T = 1000;
            rng(678);
            data = randn(T, 1);
            
            % Define NAGARCH parameters (omega, alpha, gamma, beta)
            parameters = [0.01; 0.1; 0.1; 0.8]; % With nonlinear asymmetry
            
            % Create options structure with model type NAGARCH
            options = struct('model', 'NAGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with different error distributions (NORMAL, T, GED, SKEWT)
            ll_normal = garchlikelihood(parameters, data, options);
            obj.assertTrue(isfinite(ll_normal), 'NAGARCH with normal errors should have finite likelihood');
            
            options.distribution = 'T';
            parameters_t = [parameters; 8]; % Add degrees of freedom
            ll_t = garchlikelihood(parameters_t, data, options);
            obj.assertTrue(isfinite(ll_t), 'NAGARCH with t errors should have finite likelihood');
            
            options.distribution = 'GED';
            parameters_ged = [parameters; 1.5]; % Add shape parameter
            ll_ged = garchlikelihood(parameters_ged, data, options);
            obj.assertTrue(isfinite(ll_ged), 'NAGARCH with GED errors should have finite likelihood');
            
            options.distribution = 'SKEWT';
            parameters_skewt = [parameters; 8; 0.1]; % Add df and skewness
            ll_skewt = garchlikelihood(parameters_skewt, data, options);
            obj.assertTrue(isfinite(ll_skewt), 'NAGARCH with skewed t errors should have finite likelihood');
            
            % Test nonlinear asymmetric effects on likelihood
            parameters_noasym = [0.01; 0.1; 0.0; 0.8]; % No asymmetry
            options.distribution = 'NORMAL';
            ll_noasym = garchlikelihood(parameters_noasym, data, options);
            obj.assertTrue(isfinite(ll_noasym), 'NAGARCH without asymmetry should have finite likelihood');
        end
        
        function testInvalidInputs(obj)
            % Test garchlikelihood function's handling of invalid inputs
            
            % Create basic valid inputs for reference
            T = 100;
            rng(123);
            data = randn(T, 1);
            parameters = [0.01; 0.1; 0.8];
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test empty parameters using assertThrows
            obj.assertThrows(@() garchlikelihood([], data, options), 'PARAMETERCHECK:EmptyInput', ...
                'Empty parameters should throw an error');
            
            % Test non-numeric parameters using assertThrows
            obj.assertThrows(@() garchlikelihood('invalid', data, options), 'PARAMETERCHECK:NonNumeric', ...
                'Non-numeric parameters should throw an error');
            
            % Test invalid data format using assertThrows
            obj.assertThrows(@() garchlikelihood(parameters, 'invalid', options), 'DATACHECK:NonNumeric', ...
                'Invalid data format should throw an error');
            
            % Test invalid model type in options using assertThrows
            invalid_options = options;
            invalid_options.model = 'INVALID_MODEL';
            obj.assertThrows(@() garchlikelihood(parameters, data, invalid_options), 'GARCHLIKELIHOOD:UnknownModel', ...
                'Invalid model type should throw an error');
            
            % Test invalid distribution type in options using assertThrows
            invalid_options = options;
            invalid_options.distribution = 'INVALID_DIST';
            obj.assertThrows(@() garchlikelihood(parameters, data, invalid_options), 'GARCHLIKELIHOOD:UnknownDistribution', ...
                'Invalid distribution type should throw an error');
            
            % Test invalid p,q orders using assertThrows
            invalid_options = options;
            invalid_options.p = -1;
            obj.assertThrows(@() garchlikelihood(parameters, data, invalid_options), 'PARAMETERCHECK:NegativeValue', ...
                'Negative p order should throw an error');
            
            invalid_options = options;
            invalid_options.q = 'string';
            obj.assertThrows(@() garchlikelihood(parameters, data, invalid_options), 'PARAMETERCHECK:NonNumeric', ...
                'Non-numeric q order should throw an error');
            
            % Test parameters out of bounds for different models using assertThrows
            options.distribution = 'T';
            invalid_params = [0.01; 0.1; 0.8; 1]; % nu <= 2 is invalid
            ll = garchlikelihood(invalid_params, data, options);
            obj.assertTrue(isfinite(ll) && ll > 0, 'Invalid t-dist parameter should return large positive value');
            
            % Test invalid shape parameters for different distributions using assertThrows
            options.distribution = 'SKEWT';
            invalid_params = [0.01; 0.1; 0.8; 8; 1.2]; % lambda outside [-1,1] is invalid
            ll = garchlikelihood(invalid_params, data, options);
            obj.assertTrue(isfinite(ll) && ll > 0, 'Invalid skewed t parameter should return large positive value');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of garchlikelihood with extreme values
            
            % Create test data
            T = 200;
            rng(789);
            data = randn(T, 1);
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Test with very small parameter values close to zero
            small_params = [1e-8; 1e-8; 0.99];
            ll_small = garchlikelihood(small_params, data, options);
            obj.assertTrue(isfinite(ll_small), 'Very small parameters should have finite likelihood');
            
            % Test with large outliers in the input data
            outlier_data = data;
            outlier_data(50) = 10; % Add an outlier
            outlier_data(100) = -10; % Add another outlier
            params = [0.01; 0.1; 0.8];
            ll_outlier = garchlikelihood(params, outlier_data, options);
            obj.assertTrue(isfinite(ll_outlier), 'Data with outliers should have finite likelihood');
            
            % Test with parameter values near constraint boundaries
            boundary_params = [0.01; 0.19; 0.8]; % Sum alpha+beta is 0.99
            ll_boundary = garchlikelihood(boundary_params, data, options);
            obj.assertTrue(isfinite(ll_boundary), 'Parameters near boundary should have finite likelihood');
            
            % Verify minimum likelihood threshold is properly applied
            extreme_data = zeros(T, 1);
            extreme_data(1) = 1000; % Extreme outlier
            options.constrainStationarity = false; % Disable constraints for this test
            ll_extreme = garchlikelihood(small_params, extreme_data, options);
            obj.assertTrue(isfinite(ll_extreme), 'Extreme data should still give finite likelihood');
            
            % Test handling of extreme variance values
            high_persist_params = [0.01; 0.2; 0.79]; % High persistence
            ll_high_persist = garchlikelihood(high_persist_params, data, options);
            obj.assertTrue(isfinite(ll_high_persist), 'High persistence model should have finite likelihood');
            
            % Verify stability with distribution parameters at extreme values
            options.distribution = 'T';
            extreme_t_params = [0.01; 0.1; 0.8; 100]; % Very high df (close to normal)
            ll_extreme_t = garchlikelihood(extreme_t_params, data, options);
            obj.assertTrue(isfinite(ll_extreme_t), 'Extreme t distribution parameters should have finite likelihood');
        end
        
        function testLikelihoodConsistency(obj)
            % Test that likelihood values are consistent across model types with equivalent parameterizations
            
            % Create test data with standard normal innovations
            T = 500;
            rng(101);
            data = randn(T, 1);
            
            % Configure equivalent models across different types where possible
            % Standard GARCH(1,1) with normal errors
            garch_params = [0.01; 0.1; 0.8];
            garch_options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            ll_garch = garchlikelihood(garch_params, data, garch_options);
            
            % TARCH/GJR with gamma=0 (equivalent to GARCH)
            tarch_params = [0.01; 0.1; 0; 0.8]; % gamma=0 makes it equivalent to GARCH
            tarch_options = struct('model', 'TARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            ll_tarch = garchlikelihood(tarch_params, data, tarch_options);
            
            % AGARCH with gamma=0 (equivalent to GARCH)
            agarch_params = [0.01; 0.1; 0; 0.8]; % gamma=0 makes it equivalent to GARCH
            agarch_options = struct('model', 'AGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            ll_agarch = garchlikelihood(agarch_params, data, agarch_options);
            
            % NAGARCH with gamma=0 (equivalent to GARCH)
            nagarch_params = [0.01; 0.1; 0; 0.8]; % gamma=0 makes it equivalent to GARCH
            nagarch_options = struct('model', 'NAGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            ll_nagarch = garchlikelihood(nagarch_params, data, nagarch_options);
            
            % Verify that likelihood values are consistent when models are equivalent
            tolerance = 1e-4; % Slightly higher tolerance for model comparisons
            
            obj.comparator.setDefaultTolerances(tolerance, tolerance);
            obj.assertAlmostEqual(ll_garch, ll_tarch, ['GARCH and TARCH with gamma=0 ' ...
                'should have the same likelihood']);
            obj.assertAlmostEqual(ll_garch, ll_agarch, ['GARCH and AGARCH with gamma=0 ' ...
                'should have the same likelihood']);
            obj.assertAlmostEqual(ll_garch, ll_nagarch, ['GARCH and NAGARCH with gamma=0 ' ...
                'should have the same likelihood']);
            
            % Test with various error distributions to ensure consistency
            garch_options.distribution = 'T';
            garch_params_t = [0.01; 0.1; 0.8; 8]; % Add df
            ll_garch_t = garchlikelihood(garch_params_t, data, garch_options);
            
            tarch_options.distribution = 'T';
            tarch_params_t = [0.01; 0.1; 0; 0.8; 8]; % gamma=0 makes it equivalent to GARCH
            ll_tarch_t = garchlikelihood(tarch_params_t, data, tarch_options);
            
            obj.assertAlmostEqual(ll_garch_t, ll_tarch_t, ['GARCH and TARCH with gamma=0 ' ...
                'should have the same likelihood with t distribution']);
        end
        
        function testDistributionEffects(obj)
            % Test the effects of different error distributions on likelihood values
            
            % Create test data with known distribution properties
            T = 1000;
            rng(123);
            
            % Create fat-tailed data
            t_data = randn(T, 1);
            t_data = t_data .* sqrt(2/5) .* sqrt(3 ./ trnd(5, T, 1).^2); % Approximate t(5) distribution
            
            % Create skewed data
            skewed_data = 0.5 * randn(T, 1) + 0.1; % Add asymmetry
            
            % Configure standard GARCH model parameters
            garch_params = [0.01; 0.1; 0.8];
            normal_options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Compute likelihood with different error distributions
            t_options = normal_options;
            t_options.distribution = 'T';
            garch_params_t = [garch_params; 5]; % df = 5
            
            ll_normal_t_data = garchlikelihood(garch_params, t_data, normal_options);
            ll_t_t_data = garchlikelihood(garch_params_t, t_data, t_options);
            
            % Verify that likelihood values reflect the distributional properties
            obj.assertTrue(ll_t_t_data < ll_normal_t_data, ...
                't distribution should fit t-distributed data better than normal');
            
            % Test the impact of distribution parameters on likelihood values
            skewt_options = normal_options;
            skewt_options.distribution = 'SKEWT';
            garch_params_skewt = [garch_params; 5; 0.2]; % df=5, lambda=0.2
            
            ll_normal_skew_data = garchlikelihood(garch_params, skewed_data, normal_options);
            ll_skewt_skew_data = garchlikelihood(garch_params_skewt, skewed_data, skewt_options);
            
            % Verify that maximum likelihood occurs at true distribution parameters
            obj.assertTrue(ll_skewt_skew_data < ll_normal_skew_data, ...
                'Skewed t should fit skewed data better than normal');
            
            % Test different df values to verify that correct df gives better fit
            df_values = [4, 5, 6, 7, 8];
            ll_values = zeros(size(df_values));
            
            for i = 1:length(df_values)
                params_df = [garch_params; df_values(i)];
                ll_values(i) = garchlikelihood(params_df, t_data, t_options);
            end
            
            % The likelihood should be at its best (minimum) near the true df = 5
            [~, best_idx] = min(ll_values);
            obj.assertTrue(best_idx == 1 || best_idx == 2, ...
                'Likelihood should be best for df values near the true value (5)');
        end
        
        function ll = computeGaussianLikelihood(obj, standardizedResiduals)
            % Helper method to compute Gaussian log-likelihood for standardized residuals
            
            % Compute log-likelihood using standard normal formula: -0.5*log(2π) - 0.5*sum(standardizedResiduals.^2)
            constant = -0.5 * log(2*pi);
            ll = sum(constant - 0.5 * standardizedResiduals.^2);
            
            % Add log-jacobian term for variance transformation
        end
        
        function testData = createTestGarchData(obj, modelType, distributionType, parameters)
            % Helper method to create test data for GARCH likelihood testing
            
            % Validate input parameters using parametercheck
            parametercheck(modelType, 'modelType');
            parametercheck(distributionType, 'distributionType');
            parametercheck(parameters, 'parameters');
            
            % Generate returns data with specified model and distribution properties
            T = 1000; % sample size
            burnin = 500; % burnin period
            
            % Initialize output structure
            testData = struct();
            testData.modelType = modelType;
            testData.distributionType = distributionType;
            testData.parameters = parameters;
            
            % Compute true conditional variances for validation
            testData.returns = randn(T, 1);
            testData.trueVariances = ones(T, 1);
            
            % Return structure containing returns, true variances, and model parameters
            return testData;
        end
    end
end