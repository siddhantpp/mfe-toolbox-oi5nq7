classdef GedpdfTest < BaseTest
    % Test class for the gedpdf function, providing comprehensive test coverage
    % for Generalized Error Distribution PDF calculations
    
    properties
        comparator         % NumericalComparator instance for precise floating-point comparisons
        defaultTolerance   % Default tolerance for numerical comparisons
        testData           % Structure to store test data
        testValues         % Array of test x values for distribution evaluation
        nuValues           % Array of test shape parameters
        expectedResults    % Structure to store expected PDF results
    end
    
    methods
        function obj = GedpdfTest()
            % Initializes a new GedpdfTest instance with numerical comparator
            
            % Call the superclass (BaseTest) constructor with 'GedpdfTest' name
            obj@BaseTest('GedpdfTest');
            
            % Initialize the testData structure for storing test vectors
            obj.testData = struct();
            
            % Create a NumericalComparator instance for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set defaultTolerance to 1e-12 for high-precision numeric comparisons
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            
            % Call superclass setUp method
            obj.setUp@BaseTest();
            
            % Initialize array of test x values
            obj.testValues = [-3, -2, -1, 0, 1, 2, 3]';
            
            % Initialize array of test nu values
            obj.nuValues = [0.5, 1, 1.5, 2, 2.5, 5]';
            
            % Prepare expected PDF values for reference cases
            obj.expectedResults = struct();
            
            % Configure numerical comparator with appropriate tolerance
            obj.comparator.setDefaultTolerances(obj.defaultTolerance, obj.defaultTolerance);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            
            % Call superclass tearDown method
            obj.tearDown@BaseTest();
            
            % Clear any temporary test data to free memory
            obj.testData = struct();
        end
        
        function testBasicPdf(obj)
            % Tests gedpdf with basic parameter values
            
            % Call gedpdf(0, 2) which corresponds to standard normal PDF at 0
            result = gedpdf(0, 2);
            
            % Expected value is 1/sqrt(2*pi) ≈ 0.3989
            expected = 1/sqrt(2*pi);
            
            % Verify result is approximately 0.3989 (1/sqrt(2π))
            obj.assertAlmostEqual(result, expected, 'Standard normal PDF at x=0 should be 1/sqrt(2*pi)');
            
            % Test that the result is a double with correct dimensions
            obj.assertTrue(isa(result, 'double'), 'Result should be of type double');
            obj.assertEqual(size(result), [1, 1], 'Result should be a scalar');
            
            % Verify PDF value is positive
            obj.assertTrue(result > 0, 'PDF value should be positive');
        end
        
        function testVectorInput(obj)
            % Tests gedpdf with vectorized inputs
            
            % Create vector of x values [-3, -2, -1, 0, 1, 2, 3]
            x = obj.testValues;
            
            % Call gedpdf with vector input and nu=2
            result = gedpdf(x, 2);
            
            % Verify output is a vector with same dimensions as input
            obj.assertEqual(size(result), size(x), 'Output should have same dimensions as input');
            
            % Compare results with expected normal distribution PDF values
            expected = (1/sqrt(2*pi)) * exp(-x.^2/2);
            obj.assertMatrixEqualsWithTolerance(result, expected, obj.defaultTolerance, 'PDF values should match normal distribution for nu=2');
            
            % Test with multiple nu values to verify correct vectorization
            for i = 1:length(obj.nuValues)
                nu = obj.nuValues(i);
                result = gedpdf(x, nu);
                expected = obj.calculateManualGedPdf(x, nu);
                obj.assertMatrixEqualsWithTolerance(result, expected, obj.defaultTolerance*10, sprintf('PDF should match expected for nu=%g', nu));
            end
        end
        
        function testLaplaceCase(obj)
            % Tests gedpdf with nu=1 (Laplace distribution case)
            
            % Create array of test points
            x = obj.testValues;
            
            % Calculate expected Laplace PDF values: 0.5*exp(-abs(x))
            expected = 0.5 * exp(-abs(x));
            
            % Call gedpdf with test points and nu=1
            result = gedpdf(x, 1);
            
            % Compare calculated values with expected values
            obj.assertMatrixEqualsWithTolerance(result, expected, obj.defaultTolerance, 'PDF should match Laplace distribution for nu=1');
            
            % Verify correct implementation of special case
            x_test = 1.5;
            expected_single = 0.5 * exp(-abs(x_test));
            result_single = gedpdf(x_test, 1);
            obj.assertAlmostEqual(result_single, expected_single, 'Single point Laplace PDF calculation should be correct');
        end
        
        function testNormalCase(obj)
            % Tests gedpdf with nu=2 (Normal distribution case)
            
            % Create array of test points
            x = obj.testValues;
            
            % Calculate expected Normal PDF values: (1/sqrt(2π))*exp(-x^2/2)
            expected = (1/sqrt(2*pi)) * exp(-x.^2/2);
            
            % Call gedpdf with test points and nu=2
            result = gedpdf(x, 2);
            
            % Compare calculated values with expected values
            obj.assertMatrixEqualsWithTolerance(result, expected, obj.defaultTolerance, 'PDF should match Normal distribution for nu=2');
            
            % Verify correct implementation of special case
            x_test = 1.5;
            expected_single = (1/sqrt(2*pi)) * exp(-x_test^2/2);
            result_single = gedpdf(x_test, 2);
            obj.assertAlmostEqual(result_single, expected_single, 'Single point Normal PDF calculation should be correct');
        end
        
        function testTailBehavior(obj)
            % Tests gedpdf tail behavior with different shape parameters
            
            % Test with fat tails (nu < 2) at extreme x values
            x_extreme = [-10, -5, 5, 10]';
            nu_fat = 0.8;    % Fat tails
            nu_normal = 2;   % Normal tails
            nu_thin = 5;     % Thin tails
            
            % Test with normal tails (nu = 2) at extreme x values
            pdf_fat = gedpdf(x_extreme, nu_fat);
            pdf_normal = gedpdf(x_extreme, nu_normal);
            pdf_thin = gedpdf(x_extreme, nu_thin);
            
            % Test with thin tails (nu > 2) at extreme x values
            for i = 1:length(x_extreme)
                if abs(x_extreme(i)) > 3
                    % Verify tail behavior matches theoretical expectations
                    obj.assertTrue(pdf_fat(i) > pdf_normal(i), ...
                        sprintf('Fat tails (nu=%g) should have higher density than normal at x=%g', nu_fat, x_extreme(i)));
                    obj.assertTrue(pdf_normal(i) > pdf_thin(i), ...
                        sprintf('Normal tails should have higher density than thin tails (nu=%g) at x=%g', nu_thin, x_extreme(i)));
                end
            end
            
            % Compare decay rates in the tails across different nu values
            x_very_extreme = 20;
            pdf_fat_extreme = gedpdf(x_very_extreme, nu_fat);
            pdf_normal_extreme = gedpdf(x_very_extreme, nu_normal);
            pdf_thin_extreme = gedpdf(x_very_extreme, nu_thin);
            
            obj.assertTrue(pdf_fat_extreme > pdf_normal_extreme, 'Fat tails should decay slower than normal tails');
            obj.assertTrue(pdf_normal_extreme > pdf_thin_extreme, 'Normal tails should decay slower than thin tails');
        end
        
        function testParameterValidation(obj)
            % Tests gedpdf error handling for invalid parameters
            
            % Test with invalid shape parameter (nu ≤ 0)
            obj.assertThrows(@() gedpdf(0, 0), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for nu=0');
            obj.assertThrows(@() gedpdf(0, -1), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for negative nu');
            
            % Test with NaN and Inf values in parameters
            obj.assertThrows(@() gedpdf(NaN, 2), 'MATLAB:datacheck:invalidData', ...
                'Should throw error for NaN in x');
            obj.assertThrows(@() gedpdf(0, NaN), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for NaN in nu');
            obj.assertThrows(@() gedpdf(Inf, 2), 'MATLAB:datacheck:invalidData', ...
                'Should throw error for Inf in x');
            obj.assertThrows(@() gedpdf(0, Inf), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for Inf in nu');
            
            % Test with mismatched dimensions in inputs
            obj.assertThrows(@() gedpdf(0, [1, 2]), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for vector nu');
            
            % Verify appropriate error messages are generated
            obj.assertThrows(@() gedpdf([], 2), 'MATLAB:datacheck:invalidData', ...
                'Should throw error for empty x');
            obj.assertThrows(@() gedpdf(0, []), 'MATLAB:parametercheck:invalidParameter', ...
                'Should throw error for empty nu');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of gedpdf for extreme values
            
            % Test with very small nu values (approaching 0)
            nu_small = 0.1;
            x_test = [-1, 0, 1]';
            result_small = gedpdf(x_test, nu_small);
            
            % Test with very large nu values (approaching infinity)
            nu_large = 50;
            result_large = gedpdf(x_test, nu_large);
            
            % Test with extreme x values (far from mean)
            x_extreme = [-100, 100]';
            result_extreme = gedpdf(x_extreme, 2);
            
            % Verify numerical stability under these conditions
            obj.assertTrue(all(isfinite(result_small)), 'Should handle small nu values without numerical issues');
            obj.assertTrue(all(result_small >= 0), 'PDF values should be non-negative for small nu');
            obj.assertTrue(all(isfinite(result_large)), 'Should handle large nu values without numerical issues');
            obj.assertTrue(all(result_large >= 0), 'PDF values should be non-negative for large nu');
            obj.assertTrue(all(result_extreme > 0), 'PDF values for extreme x should be positive');
            obj.assertTrue(all(result_extreme < 1e-10), 'PDF values for extreme x should be close to 0');
            
            % Compare against high-precision reference calculations
            expected_small = obj.calculateManualGedPdf(x_test, nu_small);
            obj.assertMatrixEqualsWithTolerance(result_small, expected_small, 1e-10, 'Should be numerically accurate for small nu');
        end
        
        function testPdfProperties(obj)
            % Tests that the PDF satisfies mathematical properties of a valid PDF
            
            % Verify PDF is non-negative for all test points
            x_grid = linspace(-10, 10, 1000)';
            
            % Verify PDF is symmetric around x=0 (gedpdf(x) = gedpdf(-x))
            for nu = [0.8, 1, 2, 5]
                pdf_values = gedpdf(x_grid, nu);
                
                % Verify PDF is non-negative for all test points
                obj.assertTrue(all(pdf_values >= 0), sprintf('PDF should be non-negative for nu=%g', nu));
                
                % Verify PDF is symmetric around x=0
                mid_idx = ceil(length(x_grid)/2);
                left_half = pdf_values(1:mid_idx-1);
                right_half = flipud(pdf_values(mid_idx+1:end));
                min_len = min(length(left_half), length(right_half));
                
                if min_len > 0
                    obj.assertMatrixEqualsWithTolerance(left_half(1:min_len), right_half(1:min_len), 1e-10, ...
                        sprintf('PDF should be symmetric for nu=%g', nu));
                end
                
                % Verify PDF mode is at x=0 for all nu values
                [max_pdf, max_idx] = max(pdf_values);
                idx_range = abs(x_grid(max_idx)) < 0.05;
                obj.assertTrue(idx_range, sprintf('PDF mode should be at x=0 for nu=%g', nu));
                
                % Verify approximate integration of PDF equals 1 using numerical integration
                dx = x_grid(2) - x_grid(1);
                integral_approx = sum(pdf_values) * dx;
                obj.assertTrue(abs(integral_approx - 1) < 0.01, sprintf('Integral of PDF should be approximately 1 for nu=%g', nu));
            end
        end
        
        function pdf = calculateManualGedPdf(obj, x, nu)
            % Helper method to manually calculate GED PDF for validation
            
            % Calculate lambda = sqrt(gamma(3/nu)/gamma(1/nu))
            lambda = sqrt(gamma(3/nu)/gamma(1/nu));
            
            % Calculate normalization constant c = nu / (2 * gamma(1/nu) * lambda)
            c = nu / (2 * lambda * gamma(1/nu));
            
            % Compute PDF: p = c * exp(-0.5 * abs(x/lambda).^nu)
            pdf = c * exp(-0.5 * abs(x/lambda).^nu);
            
            % Return calculated PDF values for comparison with gedpdf function output
        end
    end
end