classdef SkewtpdfTest < BaseTest
    % SkewtpdfTest Test class for validating the skewtpdf function implementation
    % which computes the PDF of Hansen's skewed t-distribution
    %
    % This test suite validates the correctness, numerical stability, and error
    % handling of the skewtpdf function, which computes the probability density
    % function of Hansen's skewed t-distribution. Tests include verification
    % against known values, parameter validation, vectorization capabilities,
    % edge cases, special cases (such as symmetry when lambda=0), and performance
    % characteristics with large datasets.
    %
    % Hansen's skewed t-distribution extends the standard t-distribution with
    % a skewness parameter (lambda), making it particularly useful for modeling
    % financial time series that exhibit asymmetric behavior.
    %
    % References:
    %   Hansen, B. E. (1994). "Autoregressive Conditional Density Estimation".
    %   International Economic Review, 35(3), 705-730.
    
    properties
        testData           % Structure containing test data
        numericalPrecision % Precision for numerical comparisons
        defaultTolerance   % Default tolerance for PDF comparisons
    end
    
    methods
        function obj = SkewtpdfTest()
            % Initialize a new SkewtpdfTest instance
            obj@BaseTest(); % Call superclass constructor
            
            % Set default tolerance specific to PDF computations
            % Using higher precision for distribution calculations
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj); % Call superclass setUp method
            
            % Load test data from file
            data = obj.loadTestData('known_distributions.mat');
            
            % Extract relevant test data
            obj.testData = struct('skewt_pdf_values', data.skewt_pdf_values, ...
                                 'skewt_parameters', data.skewt_parameters);
            
            % Set numerical precision parameters
            obj.numericalPrecision = data.numerical_precision;
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj); % Call superclass tearDown method
        end
        
        function testKnownValues(obj)
            % Test that skewtpdf computes correct values for known cases
            
            % Extract test data
            params = obj.testData.skewt_parameters;
            expected_values = obj.testData.skewt_pdf_values;
            
            % Test points to evaluate
            x = linspace(-4, 4, 100);
            
            % For different parameter combinations
            for i = 1:size(params, 1)
                nu = params(i, 1);     % Degrees of freedom
                lambda = params(i, 2); % Skewness parameter
                
                % Compute PDF values
                actual_values = skewtpdf(x, nu, lambda);
                
                % Compare with expected values using NumericalComparator
                result = obj.numericalComparator.compareMatrices(expected_values(i, :)', actual_values, obj.defaultTolerance);
                
                obj.assertTrue(result.isEqual, ...
                    sprintf('PDF values differ for nu=%g, lambda=%g. Max difference: %g', ...
                    nu, lambda, result.maxAbsoluteDifference));
            end
            
            % Test specific known values from literature
            % For lambda=0, skewed t at x=0 should equal standard t at x=0
            x_test = 0;
            nu_test = 5;
            lambda_test = 0;
            
            % For lambda=0, skewed t at x=0 should equal standard t at x=0
            expected_pdf_at_zero = gamma((nu_test+1)/2) / (sqrt(pi*(nu_test-2)) * gamma(nu_test/2));
            actual_pdf_at_zero = skewtpdf(x_test, nu_test, lambda_test);
            
            obj.assertAlmostEqual(actual_pdf_at_zero, expected_pdf_at_zero, ...
                'PDF at x=0 with lambda=0 should match standard t-distribution');
        end
        
        function testParameterValidation(obj)
            % Test that parameter validation correctly identifies invalid inputs
            
            % Valid inputs for reference
            x = linspace(-4, 4, 10);
            valid_nu = 5;
            valid_lambda = 0.5;
            
            % Test invalid nu (< 2)
            try
                skewtpdf(x, 1.5, valid_lambda);
                obj.assertTrue(false, 'Function should throw error for nu < 2');
            catch
                % Expected error - test passes
            end
            
            % Test invalid lambda (< -1)
            try
                skewtpdf(x, valid_nu, -1.5);
                obj.assertTrue(false, 'Function should throw error for lambda < -1');
            catch
                % Expected error - test passes
            end
            
            % Test invalid lambda (> 1)
            try
                skewtpdf(x, valid_nu, 1.5);
                obj.assertTrue(false, 'Function should throw error for lambda > 1');
            catch
                % Expected error - test passes
            end
            
            % Test non-numeric inputs
            try
                skewtpdf('string', valid_nu, valid_lambda);
                obj.assertTrue(false, 'Function should throw error for non-numeric x');
            catch
                % Expected error - test passes
            end
            
            % Test empty inputs
            try
                skewtpdf([], valid_nu, valid_lambda);
                obj.assertTrue(false, 'Function should throw error for empty x');
            catch
                % Expected error - test passes
            end
            
            % Test NaN inputs
            try
                skewtpdf(x, NaN, valid_lambda);
                obj.assertTrue(false, 'Function should throw error for NaN nu');
            catch
                % Expected error - test passes
            end
            
            % Test Inf inputs
            try
                skewtpdf(x, valid_nu, Inf);
                obj.assertTrue(false, 'Function should throw error for Inf lambda');
            catch
                % Expected error - test passes
            end
            
            % Test dimension mismatch
            try
                nu_vector = repmat(valid_nu, [10, 2]);  % 10x2 matrix
                skewtpdf(x, nu_vector, valid_lambda);
                obj.assertTrue(false, 'Function should throw error for dimension mismatch');
            catch
                % Expected error - test passes
            end
        end
        
        function testVectorization(obj)
            % Test that function handles vector inputs and broadcasts parameters correctly
            
            % Generate test data
            x = linspace(-4, 4, 20)';  % Column vector
            
            % Test case 1: Scalar parameters, vector x
            nu_scalar = 5;
            lambda_scalar = 0.3;
            result1 = skewtpdf(x, nu_scalar, lambda_scalar);
            
            % Verify dimensions
            obj.assertEqual(size(result1), [20, 1], ...
                'Output size incorrect for vector x with scalar parameters');
            
            % Test case 2: Vector nu, scalar lambda, vector x
            nu_vector = 5 * ones(size(x));
            result2 = skewtpdf(x, nu_vector, lambda_scalar);
            
            % Compare with scalar case
            obj.assertMatrixEqualsWithTolerance(result1, result2, obj.defaultTolerance, ...
                'Vectorized nu gives different results');
            
            % Test case 3: Scalar nu, vector lambda, vector x
            lambda_vector = 0.3 * ones(size(x));
            result3 = skewtpdf(x, nu_scalar, lambda_vector);
            
            % Compare with scalar case
            obj.assertMatrixEqualsWithTolerance(result1, result3, obj.defaultTolerance, ...
                'Vectorized lambda gives different results');
            
            % Test case 4: Vector nu, vector lambda, vector x (all of matching size)
            result4 = skewtpdf(x, nu_vector, lambda_vector);
            
            % Compare with scalar case
            obj.assertMatrixEqualsWithTolerance(result1, result4, obj.defaultTolerance, ...
                'Fully vectorized inputs give different results');
            
            % Test case 5: Row vector input
            x_row = x';  % Row vector
            result5 = skewtpdf(x_row, nu_scalar, lambda_scalar);
            
            % Verify correct handling of row vector (should be transposed)
            obj.assertEqual(size(result5), [20, 1], ...
                'Row vector input should be transposed to column vector');
        end
        
        function testEdgeCases(obj)
            % Test behavior at distribution edges and extreme parameter values
            
            % Test points
            x = linspace(-5, 5, 50)';
            
            % Test with nu close to lower bound (2+eps)
            nu_min = 2 + eps;
            lambda_zero = 0;
            pdf_min_nu = skewtpdf(x, nu_min, lambda_zero);
            
            % Verify non-negative values
            obj.assertTrue(all(pdf_min_nu >= 0), ...
                'PDF values should be non-negative with nu close to 2');
            
            % Verify PDF has highest value at mode (should be at x=0 for lambda=0)
            [~, max_idx] = max(pdf_min_nu);
            obj.assertTrue(abs(x(max_idx)) < 0.1, ...
                'Mode of symmetric PDF should be near x=0');
            
            % Test with lambda close to bounds
            lambda_neg_edge = -1 + eps*10;
            lambda_pos_edge = 1 - eps*10;
            nu_typical = 5;
            
            pdf_neg_edge = skewtpdf(x, nu_typical, lambda_neg_edge);
            pdf_pos_edge = skewtpdf(x, nu_typical, lambda_pos_edge);
            
            % Verify non-negative values
            obj.assertTrue(all(pdf_neg_edge >= 0), ...
                'PDF values should be non-negative with lambda close to -1');
            obj.assertTrue(all(pdf_pos_edge >= 0), ...
                'PDF values should be non-negative with lambda close to 1');
            
            % Verify extreme skewness shifts the distribution mode appropriately
            [~, mode_neg_idx] = max(pdf_neg_edge);
            [~, mode_pos_idx] = max(pdf_pos_edge);
            
            % For negative lambda, mode should be right of zero
            % For positive lambda, mode should be left of zero
            obj.assertTrue(x(mode_neg_idx) > 0, ...
                'Mode with lambda near -1 should be right of zero');
            obj.assertTrue(x(mode_pos_idx) < 0, ...
                'Mode with lambda near 1 should be left of zero');
            
            % Test with extreme x values
            x_extreme = [-100; 100];
            pdf_extreme = skewtpdf(x_extreme, nu_typical, lambda_zero);
            
            % Very large x values should give very small PDF values
            obj.assertTrue(all(pdf_extreme < 1e-10), ...
                'PDF for extreme x values should approach zero');
            
            % Test at junction point between left and right tails
            nu_test = 5;
            lambda_test = 0.5;
            
            % Calculate the threshold parameter (-a/b)
            c = gamma((nu_test + 1) / 2) / (sqrt(pi * (nu_test - 2)) * gamma(nu_test / 2));
            a = 4 * lambda_test * c * ((nu_test - 2) / nu_test);
            b = sqrt(1 + 3 * lambda_test^2 - a^2);
            threshold = -a / b;
            
            % Evaluate PDF slightly to the left and right of threshold
            delta = eps * 100;
            pdf_left = skewtpdf(threshold - delta, nu_test, lambda_test);
            pdf_right = skewtpdf(threshold + delta, nu_test, lambda_test);
            
            % Verify continuity at the junction point
            obj.assertAlmostEqual(pdf_left, pdf_right, ...
                'PDF should be continuous at the junction point');
        end
        
        function testSpecialCases(obj)
            % Test special cases including symmetry and known limiting behaviors
            
            % Test points
            x = linspace(-5, 5, 100)';
            
            % Test case 1: lambda = 0 should produce a symmetric distribution
            nu_test = 5;
            lambda_zero = 0;
            
            pdf_symmetric = skewtpdf(x, nu_test, lambda_zero);
            
            % Find indices of points equidistant from zero
            mid_idx = ceil(length(x) / 2);
            for i = 1:(mid_idx-1)
                left_idx = mid_idx - i;
                right_idx = mid_idx + i;
                if left_idx > 0 && right_idx <= length(x)
                    % Compare PDF values at -x and +x
                    obj.assertAlmostEqual(pdf_symmetric(left_idx), pdf_symmetric(right_idx), ...
                        sprintf('PDF with lambda=0 should be symmetric: f(%g) = f(%g)', ...
                        x(left_idx), x(right_idx)));
                end
            end
            
            % Test case 2: As nu approaches infinity, should approach normal-like behavior
            nu_large = 1000;
            lambda_test = 0.5;
            
            pdf_large_nu = skewtpdf(x, nu_large, lambda_test);
            
            % Verify the PDF is smooth and non-negative
            obj.assertTrue(all(pdf_large_nu >= 0), ...
                'PDF with large nu should be non-negative');
            
            % Verify the tails decay faster with large nu
            pdf_mid_nu = skewtpdf(x, 5, lambda_test);
            left_tail = x < -3;
            right_tail = x > 3;
            
            % Tail ratios should be smaller for large nu
            left_ratio_large = pdf_large_nu(left_tail) ./ max(pdf_large_nu);
            left_ratio_mid = pdf_mid_nu(left_tail) ./ max(pdf_mid_nu);
            right_ratio_large = pdf_large_nu(right_tail) ./ max(pdf_large_nu);
            right_ratio_mid = pdf_mid_nu(right_tail) ./ max(pdf_mid_nu);
            
            obj.assertTrue(mean(left_ratio_large) < mean(left_ratio_mid), ...
                'Left tail should decay faster with larger nu');
            obj.assertTrue(mean(right_ratio_large) < mean(right_ratio_mid), ...
                'Right tail should decay faster with larger nu');
            
            % Test case 3: Verify PDF integrates to approximately 1
            area_approx = trapz(x, pdf_symmetric);
            obj.assertAlmostEqual(area_approx, 1, 0.01, ...
                'PDF should integrate to approximately 1');
            
            % Test case 4: Mode shifts with lambda
            % Generate PDFs with varying lambda
            lambda_values = [-0.8, -0.4, 0, 0.4, 0.8];
            modes = zeros(size(lambda_values));
            
            for i = 1:length(lambda_values)
                pdf_i = skewtpdf(x, nu_test, lambda_values(i));
                [~, max_idx] = max(pdf_i);
                modes(i) = x(max_idx);
            end
            
            % Mode should move from right to left as lambda increases
            obj.assertTrue(all(diff(modes) < 0), ...
                'Mode should shift monotonically from right to left as lambda increases');
        end
        
        function testPerformance(obj)
            % Test performance with large-scale inputs
            
            % Generate large vector of x values
            x_large = linspace(-10, 10, 10000)';
            nu_test = 5;
            lambda_test = 0.3;
            
            % Measure execution time
            executionTime = obj.measureExecutionTime(@() skewtpdf(x_large, nu_test, lambda_test));
            
            % Check if execution time is acceptable (adjust threshold as needed)
            maxAcceptableTime = 1.0; % seconds
            obj.assertTrue(executionTime < maxAcceptableTime, ...
                sprintf('Performance issue: execution took %.3f seconds (threshold: %.3f)', ...
                executionTime, maxAcceptableTime));
            
            % Test memory usage
            memoryInfo = obj.checkMemoryUsage(@() skewtpdf(x_large, nu_test, lambda_test));
            
            % Check memory usage is reasonable
            maxMemoryMB = 10; % MB
            obj.assertTrue(memoryInfo.memoryDifferenceMB < maxMemoryMB, ...
                sprintf('Memory usage issue: %.2f MB used (threshold: %.2f MB)', ...
                memoryInfo.memoryDifferenceMB, maxMemoryMB));
        end
    end
end