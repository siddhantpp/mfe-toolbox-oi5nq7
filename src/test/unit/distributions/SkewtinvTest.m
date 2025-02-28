classdef SkewtinvTest < BaseTest
    % SkewtinvTest validates the functionality of the skewtinv function,
    % which computes the inverse cumulative distribution function (quantile function)
    % of Hansen's skewed t-distribution.
    
    properties
        comparator          % NumericalComparator instance for floating-point comparisons
        defaultTolerance    % Default tolerance for numerical comparisons
        nu                  % Array of degrees of freedom values for testing
        lambda              % Array of skewness parameter values for testing
        testProbabilities   % Standard probability values for testing
        knownQuantiles      % Matrix of known probability/quantile pairs for validation
    end
    
    methods
        function obj = SkewtinvTest()
            % Initialize a new SkewtinvTest instance with default settings
            obj@BaseTest(); % Call superclass constructor
            obj.comparator = NumericalComparator();
            obj.defaultTolerance = 1e-10;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Reset numerical comparator configuration
            obj.comparator = NumericalComparator();
            
            % Initialize test parameters
            obj.nu = [5, 10, 15, 30];
            obj.lambda = [-0.9, -0.5, 0, 0.5, 0.9];
            obj.testProbabilities = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
            
            % Initialize known quantiles for validation - these are approximate values
            % for specific parameter combinations
            obj.knownQuantiles = [
                0.5, 5, 0, 0;                  % Median of symmetric t(5)
                0.025, 5, 0, -2.57058;         % 2.5% quantile of symmetric t(5)
                0.975, 5, 0, 2.57058;          % 97.5% quantile of symmetric t(5)
                0.5, 10, 0, 0;                 % Median of symmetric t(10)
                0.025, 10, 0, -2.22814;        % 2.5% quantile of symmetric t(10)
                0.975, 10, 0, 2.22814;         % 97.5% quantile of symmetric t(10)
                0.5, 5, 0.5, -0.3266;          % Median of skewed t(5) with lambda=0.5
                0.5, 5, -0.5, 0.3266;          % Median of skewed t(5) with lambda=-0.5
                0.05, 5, 0.5, -3.3884;         % 5% quantile of skewed t(5) with lambda=0.5
                0.95, 5, 0.5, 1.7099;          % 95% quantile of skewed t(5) with lambda=0.5
                0.05, 5, -0.5, -1.7099;        % 5% quantile of skewed t(5) with lambda=-0.5
                0.95, 5, -0.5, 3.3884;         % 95% quantile of skewed t(5) with lambda=-0.5
            ];
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Tests the basic functionality of skewtinv with valid inputs
            p = linspace(0.01, 0.99, 11);
            nu = 5;
            lambda = 0.5;
            
            % Test scalar inputs
            q = skewtinv(0.5, nu, lambda);
            obj.assertTrue(isscalar(q), 'Output should be scalar for scalar inputs');
            obj.assertTrue(isfinite(q), 'Output should be finite for valid inputs');
            
            % Test vector inputs
            q = skewtinv(p, nu, lambda);
            obj.assertEqual(size(q), size(p), 'Output size should match input size');
            obj.assertTrue(all(isfinite(q)), 'All outputs should be finite for valid inputs');
            
            % Test with different valid parameter combinations
            for i = 1:length(obj.nu)
                for j = 1:length(obj.lambda)
                    q = skewtinv(p, obj.nu(i), obj.lambda(j));
                    obj.assertEqual(size(q), size(p), 'Output size mismatch');
                    obj.assertTrue(all(isfinite(q)), 'Non-finite output detected');
                end
            end
        end
        
        function testSymmetry(obj)
            % Tests symmetry properties of skewtinv with lambda=0
            nu = 5;
            lambda = 0;
            
            % For lambda = 0, the distribution should be symmetric around 0
            % This means skewtinv(p, nu, 0) = -skewtinv(1-p, nu, 0)
            
            % Test symmetry around median
            obj.assertAlmostEqual(skewtinv(0.5, nu, lambda), 0, 'Median should be 0 for symmetric case');
            
            % Test symmetry of quantiles
            p_values = [0.01, 0.05, 0.1, 0.25, 0.4];
            for p = p_values
                q_left = skewtinv(p, nu, lambda);
                q_right = skewtinv(1-p, nu, lambda);
                obj.assertAlmostEqual(q_left, -q_right, 'Quantiles should be symmetric around 0');
            end
            
            % Test with multiple degrees of freedom
            for i = 1:length(obj.nu)
                curr_nu = obj.nu(i);
                q_left = skewtinv(0.25, curr_nu, 0);
                q_right = skewtinv(0.75, curr_nu, 0);
                obj.assertAlmostEqual(q_left, -q_right, 'Quantiles should be symmetric around 0');
            end
        end
        
        function testSkewness(obj)
            % Tests the effect of skewness parameter lambda on the quantile function
            nu = 5;
            p = 0.5; % median
            
            % For p = 0.5, positive lambda should give negative median
            q_pos_lambda = skewtinv(p, nu, 0.5);
            obj.assertTrue(q_pos_lambda < 0, 'Median should be negative for positive lambda');
            
            % For p = 0.5, negative lambda should give positive median
            q_neg_lambda = skewtinv(p, nu, -0.5);
            obj.assertTrue(q_neg_lambda > 0, 'Median should be positive for negative lambda');
            
            % Test relationship: skewtinv(p, nu, -lambda) = -skewtinv(1-p, nu, lambda)
            lambda = 0.7;
            p_test = 0.3;
            q1 = skewtinv(p_test, nu, -lambda);
            q2 = -skewtinv(1-p_test, nu, lambda);
            obj.assertAlmostEqual(q1, q2, 'Skewness relationship fails');
            
            % Test with various lambda values
            lambda_values = [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9];
            for lambda = lambda_values
                % As lambda increases (more right-skewed), left tail quantiles should become more extreme
                if lambda > 0
                    obj.assertTrue(skewtinv(0.1, nu, lambda) < skewtinv(0.1, nu, 0), ...
                        'Left tail should be more extreme with positive lambda');
                    obj.assertTrue(skewtinv(0.9, nu, lambda) < skewtinv(0.9, nu, 0), ...
                        'Right tail should be less extreme with positive lambda');
                elseif lambda < 0
                    obj.assertTrue(skewtinv(0.1, nu, lambda) > skewtinv(0.1, nu, 0), ...
                        'Left tail should be less extreme with negative lambda');
                    obj.assertTrue(skewtinv(0.9, nu, lambda) > skewtinv(0.9, nu, 0), ...
                        'Right tail should be more extreme with negative lambda');
                end
            end
        end
        
        function testBoundaryConditions(obj)
            % Tests behavior at boundary conditions (p=0, p=1) and near boundaries
            nu = 5;
            lambda = 0.5;
            
            % Test p = 0 and p = 1
            obj.assertEqual(skewtinv(0, nu, lambda), -Inf, 'p=0 should return -Inf');
            obj.assertEqual(skewtinv(1, nu, lambda), Inf, 'p=1 should return Inf');
            
            % Test values very close to boundaries
            small_p = 1e-10;
            large_p = 1 - small_p;
            obj.assertTrue(isfinite(skewtinv(small_p, nu, lambda)), 'Near-zero p should give finite result');
            obj.assertTrue(isfinite(skewtinv(large_p, nu, lambda)), 'Near-one p should give finite result');
            
            % Test with different lambda values
            for i = 1:length(obj.lambda)
                curr_lambda = obj.lambda(i);
                obj.assertEqual(skewtinv(0, nu, curr_lambda), -Inf, 'p=0 should always return -Inf');
                obj.assertEqual(skewtinv(1, nu, curr_lambda), Inf, 'p=1 should always return Inf');
            end
            
            % Test with different nu values
            for i = 1:length(obj.nu)
                curr_nu = obj.nu(i);
                obj.assertEqual(skewtinv(0, curr_nu, lambda), -Inf, 'p=0 should always return -Inf');
                obj.assertEqual(skewtinv(1, curr_nu, lambda), Inf, 'p=1 should always return Inf');
            end
        end
        
        function testInverseCDFProperty(obj)
            % Tests that skewtinv is the proper inverse of skewtcdf
            nu = 5;
            lambda = 0.5;
            
            % Test composition: skewtcdf(skewtinv(p)) = p
            p_values = linspace(0.01, 0.99, 11);
            for p = p_values
                q = skewtinv(p, nu, lambda);
                p_computed = skewtcdf(q, nu, lambda);
                obj.assertAlmostEqual(p_computed, p, ['Inverse relationship fails at p=', num2str(p)]);
            end
            
            % Test composition in the other direction: skewtinv(skewtcdf(x)) = x
            x_values = linspace(-5, 5, 11);
            for x = x_values
                p = skewtcdf(x, nu, lambda);
                x_computed = skewtinv(p, nu, lambda);
                obj.assertAlmostEqual(x_computed, x, ['Inverse relationship fails at x=', num2str(x)]);
            end
            
            % Test with different parameter combinations
            for i = 1:length(obj.nu)
                for j = 1:length(obj.lambda)
                    curr_nu = obj.nu(i);
                    curr_lambda = obj.lambda(j);
                    
                    % Sample random probabilities
                    p_random = rand(5, 1);
                    for k = 1:length(p_random)
                        p = p_random(k);
                        q = skewtinv(p, curr_nu, curr_lambda);
                        p_computed = skewtcdf(q, curr_nu, curr_lambda);
                        obj.assertAlmostEqual(p_computed, p, 'Inverse CDF relationship fails for random p');
                    end
                end
            end
        end
        
        function testErrorHandling(obj)
            % Tests error handling for invalid inputs
            nu = 5;
            lambda = 0.5;
            
            % Test invalid probabilities
            obj.assertThrows(@() skewtinv(-0.1, nu, lambda), 'p must contain values between 0 and 1');
            obj.assertThrows(@() skewtinv(1.1, nu, lambda), 'p must contain values between 0 and 1');
            
            % Test invalid degrees of freedom
            obj.assertThrows(@() skewtinv(0.5, 1, lambda), 'nu must be greater than or equal to 2');
            obj.assertThrows(@() skewtinv(0.5, 0, lambda), 'nu must be greater than or equal to 2');
            obj.assertThrows(@() skewtinv(0.5, -5, lambda), 'nu must be greater than or equal to 2');
            
            % Test invalid skewness parameter
            obj.assertThrows(@() skewtinv(0.5, nu, -1.1), 'lambda must be less than or equal to 1');
            obj.assertThrows(@() skewtinv(0.5, nu, 1.1), 'lambda must be less than or equal to 1');
            
            % Test non-numeric inputs
            obj.assertThrows(@() skewtinv('0.5', nu, lambda), 'p must be numeric');
            obj.assertThrows(@() skewtinv(0.5, 'nu', lambda), 'nu must be numeric');
            obj.assertThrows(@() skewtinv(0.5, nu, 'lambda'), 'lambda must be numeric');
            
            % Test NaN and Inf inputs
            obj.assertThrows(@() skewtinv(NaN, nu, lambda), 'p cannot contain NaN');
            obj.assertThrows(@() skewtinv(0.5, NaN, lambda), 'nu cannot contain NaN');
            obj.assertThrows(@() skewtinv(0.5, nu, NaN), 'lambda cannot contain NaN');
            obj.assertThrows(@() skewtinv(Inf, nu, lambda), 'p cannot contain Inf');
            obj.assertThrows(@() skewtinv(0.5, Inf, lambda), 'nu cannot contain Inf');
            obj.assertThrows(@() skewtinv(0.5, nu, Inf), 'lambda cannot contain Inf');
        end
        
        function testVectorization(obj)
            % Tests vectorized operation and broadcasting behavior
            
            % Test with vector p, scalar nu and lambda
            p = linspace(0.1, 0.9, 5);
            nu = 5;
            lambda = 0.5;
            q = skewtinv(p, nu, lambda);
            obj.assertEqual(size(q), size(p), 'Output size should match input size');
            
            % Test with vector nu, scalar p and lambda
            p = 0.5;
            nu = [3, 5, 10, 20];
            lambda = 0.5;
            q = skewtinv(p, nu, lambda);
            obj.assertEqual(size(q), size(nu), 'Output size should match nu size');
            
            % Test with vector lambda, scalar p and nu
            p = 0.5;
            nu = 5;
            lambda = [-0.9, -0.5, 0, 0.5, 0.9];
            q = skewtinv(p, nu, lambda);
            obj.assertEqual(size(q), size(lambda), 'Output size should match lambda size');
            
            % Test with compatible vector inputs for all parameters
            p = [0.25, 0.5, 0.75];
            nu = [5, 10, 15];
            lambda = [0.1, 0.3, 0.5];
            q = skewtinv(p, nu, lambda);
            obj.assertEqual(size(q), size(p), 'Output size should match input sizes');
            
            % Test with matrix inputs
            p_matrix = reshape(linspace(0.1, 0.9, 9), [3, 3]);
            q = skewtinv(p_matrix, nu, lambda(1));
            obj.assertEqual(size(q), size(p_matrix), 'Output size should match matrix input size');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision and stability across a wide range of inputs
            
            % Test with very large degrees of freedom (approaching normal distribution)
            nu_large = 1000;
            p = 0.5;
            lambda = 0;
            q_large_nu = skewtinv(p, nu_large, lambda);
            obj.assertTrue(isfinite(q_large_nu), 'Result should be finite for large nu');
            obj.assertAlmostEqual(q_large_nu, 0, 1e-6, 'Median should approach 0 for large nu');
            
            % Test with degrees of freedom close to lower bound
            nu_small = 2.001;
            q_small_nu = skewtinv(p, nu_small, lambda);
            obj.assertTrue(isfinite(q_small_nu), 'Result should be finite for nu close to 2');
            
            % Test with extreme lambda values
            lambda_extreme = 0.999;
            q_extreme_lambda = skewtinv(p, 5, lambda_extreme);
            obj.assertTrue(isfinite(q_extreme_lambda), 'Result should be finite for extreme lambda');
            
            % Test with probabilities very close to 0 and 1
            p_low = 1e-8;
            p_high = 1 - p_low;
            q_low = skewtinv(p_low, 5, 0);
            q_high = skewtinv(p_high, 5, 0);
            obj.assertTrue(isfinite(q_low), 'Result should be finite for extremely low p');
            obj.assertTrue(isfinite(q_high), 'Result should be finite for extremely high p');
            obj.assertTrue(q_low < -10, 'Result for extremely low p should be strongly negative');
            obj.assertTrue(q_high > 10, 'Result for extremely high p should be strongly positive');
        end
        
        function testKnownValues(obj)
            % Tests skewtinv against pre-computed known values for validation
            
            % Set tolerance for these comparisons
            tol = 1e-6;
            
            % Test each case from the knownQuantiles property
            for i = 1:size(obj.knownQuantiles, 1)
                p = obj.knownQuantiles(i, 1);
                nu = obj.knownQuantiles(i, 2);
                lambda = obj.knownQuantiles(i, 3);
                expected_q = obj.knownQuantiles(i, 4);
                
                q = skewtinv(p, nu, lambda);
                obj.assertAlmostEqual(q, expected_q, tol, ...
                    sprintf('Known value test failed for case: p=%g, nu=%g, lambda=%g', p, nu, lambda));
            end
        end
    end
end