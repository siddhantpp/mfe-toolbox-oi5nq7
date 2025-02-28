classdef StdtrndTest < BaseTest
    % Test class for stdtrnd function that generates random numbers from standardized Student's t-distribution
    
    properties
        NumericalComparator comparator       % Instance for floating-point comparisons
        double defaultTolerance              % Default tolerance for numerical comparisons
        struct testData                      % Structure to store test data
        array nuValues                       % Array of test nu values
        double sampleSize                    % Size of samples for statistical tests
        double seedValue                     % Seed value for reproducible tests
    end
    
    methods
        function obj = StdtrndTest()
            % Initializes a new StdtrndTest instance with numerical comparator
            obj = obj@BaseTest('StdtrndTest');
            
            % Initialize the testData structure for storing test vectors
            obj.testData = struct();
            
            % Create a NumericalComparator instance for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set defaultTolerance to 1e-10 for high-precision numeric comparisons
            obj.defaultTolerance = 1e-10;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            setUp@BaseTest(obj);
            
            % Initialize array of test nu values [3, 4, 5, 8, 10, 20, 30]
            obj.nuValues = [3, 4, 5, 8, 10, 20, 30];
            
            % Set sampleSize to 10000 for statistical validation
            obj.sampleSize = 10000;
            
            % Set seedValue to 20090101 for reproducible random numbers
            obj.seedValue = 20090101;
            
            % Load reference data from known_distributions.mat, specifically the stdt_rnd_seeds variable
            try
                testData = obj.loadTestData('known_distributions.mat');
                obj.testData = testData.stdt_rnd_seeds;
            catch
                warning('Reference data not available. Some tests may be limited.');
            end
            
            % Configure numerical comparator with appropriate tolerance
            obj.comparator.setDefaultTolerances(1e-8, 1e-6);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            tearDown@BaseTest(obj);
            
            % Clear any temporary test variables to free memory
            clear temp_*;
        end
        
        function testBasicRandomGeneration(obj)
            % Tests basic random number generation with default parameters
            
            % Set random seed for reproducibility
            rng(obj.seedValue);
            
            % Generate a single random number using stdtrnd(1, 5)
            r = stdtrnd(1, 5);
            
            % Verify output is a scalar double
            obj.assertTrue(isscalar(r), 'Output should be a scalar');
            obj.assertTrue(isa(r, 'double'), 'Output should be of type double');
            
            % Generate a small sample and verify dimensions
            r_vec = stdtrnd(10, 5);
            obj.assertEqual(size(r_vec), [10, 1], 'Output should be a 10x1 vector');
            
            % Verify each call produces different random numbers
            r1 = stdtrnd(1, 5);
            r2 = stdtrnd(1, 5);
            obj.assertFalse(r1 == r2, 'Sequential calls should produce different random values');
        end
        
        function testVectorizedGeneration(obj)
            % Tests vectorized random number generation with size parameters
            
            % Generate vector of random numbers using stdtrnd([10,1], 5)
            r_vec = stdtrnd([10, 1], 5);
            obj.assertEqual(size(r_vec), [10, 1], 'Output should be a 10x1 vector');
            
            % Generate matrix of random numbers using stdtrnd([5,5], 5)
            r_mat = stdtrnd([5, 5], 5);
            obj.assertEqual(size(r_mat), [5, 5], 'Output should be a 5x5 matrix');
            
            % Generate 3D array of random numbers using stdtrnd([2,3,4], 5)
            r_3d = stdtrnd([2, 3, 4], 5);
            obj.assertEqual(size(r_3d), [2, 3, 4], 'Output should be a 2x3x4 array');
            
            % Check that all elements are unique (true randomness)
            r_large = stdtrnd(100, 5);
            obj.assertEqual(numel(r_large), numel(unique(r_large)), 'All elements should be unique');
        end
        
        function testStatisticalProperties(obj)
            % Tests statistical properties of generated random numbers
            
            % Set random seed for reproducibility
            rng(obj.seedValue);
            
            % Generate large sample of random numbers for nu=5
            nu = 5;
            samples = stdtrnd(obj.sampleSize, nu);
            
            % Verify sample mean is close to 0 (expected value)
            sample_mean = mean(samples);
            obj.assertAlmostEqual(sample_mean, 0, 'Sample mean should be close to 0');
            
            % Verify sample variance is close to 1 (standardized distribution)
            sample_var = var(samples);
            obj.assertAlmostEqual(sample_var, 1, 'Sample variance should be close to 1');
            
            % Check skewness is approximately 0 for symmetric distribution
            sample_skew = skewness(samples);
            obj.assertAlmostEqual(sample_skew, 0, 'Sample skewness should be close to 0');
            
            % Verify kurtosis matches theoretical value for t-distribution with nu=5
            sample_kurt = kurtosis(samples) - 3; % Convert to excess kurtosis
            theoretical_kurt = 6/(nu-4);
            obj.assertAlmostEqual(sample_kurt, theoretical_kurt, 'Sample kurtosis should match theoretical value');
            
            % Repeat tests for multiple degrees of freedom
            for i = 1:length(obj.nuValues)
                nu_i = obj.nuValues(i);
                if nu_i > 4 % Kurtosis only defined for nu > 4
                    samples_i = stdtrnd(obj.sampleSize, nu_i);
                    obj.assertTrue(obj.verifyDistributionMoments(samples_i, nu_i, 0.1), ...
                        sprintf('Statistical properties failed for nu=%d', nu_i));
                end
            end
        end
        
        function testDistributionFit(obj)
            % Tests that generated samples fit the expected theoretical distribution
            
            % Generate large sample from standardized t-distribution
            rng(obj.seedValue);
            nu = 5;
            samples = stdtrnd(obj.sampleSize, nu);
            
            % Calculate empirical CDF from the sample
            [~, ~, ksstat] = kstest(samples, 'CDF', {@stdtcdf, nu});
            
            % Compare with theoretical CDF using Kolmogorov-Smirnov test
            obj.assertTrue(ksstat < 0.02, 'KS test should indicate good distribution fit');
            
            % Generate histogram and compare with theoretical PDF
            x = linspace(-4, 4, 100);
            pdf_values = stdtpdf(x, nu);
            hist(samples, 50);
            
            % Test for multiple degrees of freedom to verify consistent behavior
            for i = 1:length(obj.nuValues)
                nu_i = obj.nuValues(i);
                samples_i = stdtrnd(obj.sampleSize, nu_i);
                [~, ~, ksstat_i] = kstest(samples_i, 'CDF', {@stdtcdf, nu_i});
                obj.assertTrue(ksstat_i < 0.02, ...
                    sprintf('Distribution fit failed for nu=%d', nu_i));
            end
        end
        
        function testParameterValidation(obj)
            % Tests error handling for invalid input parameters
            
            % Test with invalid degrees of freedom (nu ≤ 2)
            obj.assertThrows(@() stdtrnd(10, 2), 'parametercheck:lowerBound', ...
                'Should throw error for nu ≤ 2');
            
            % Test with negative nu values
            obj.assertThrows(@() stdtrnd(10, -5), 'parametercheck:lowerBound', ...
                'Should throw error for negative nu');
            
            % Test with non-numeric nu values
            obj.assertThrows(@() stdtrnd(10, 'invalid'), 'parametercheck:numericType', ...
                'Should throw error for non-numeric nu');
            
            % Test with NaN and Inf in parameters
            obj.assertThrows(@() stdtrnd(10, NaN), 'parametercheck:nanValue', ...
                'Should throw error for NaN nu');
            
            % Test with invalid size parameters
            obj.assertThrows(@() stdtrnd(10, Inf), 'parametercheck:infValue', ...
                'Should throw error for Inf nu');
            
            % Verify appropriate error messages are generated using assertThrows
            obj.assertThrows(@() stdtrnd('invalid', 5), 'datacheck:numericType', ...
                'Should throw error for non-numeric size');
        end
        
        function testReproducibility(obj)
            % Tests reproducibility of random number generation with fixed seeds
            
            % Set MATLAB's random number generator to a specific seed
            rng(obj.seedValue);
            
            % Generate sequence of random numbers
            samples1 = stdtrnd(100, 5);
            
            % Reset the random seed to the same value
            rng(obj.seedValue);
            
            % Generate second sequence of random numbers
            samples2 = stdtrnd(100, 5);
            
            % Verify both sequences are identical
            obj.assertTrue(all(samples1 == samples2), 'Random sequences should be identical with same seed');
            
            % Compare with pre-computed reference values from test data
            if isfield(obj.testData, 'nu5') && ~isempty(obj.testData.nu5)
                rng(obj.seedValue);
                samples = stdtrnd(length(obj.testData.nu5), 5);
                obj.assertMatrixEqualsWithTolerance(samples, obj.testData.nu5, obj.defaultTolerance, ...
                    'Generated values should match reference data');
            end
        end
        
        function testEdgeCases(obj)
            % Tests behavior with edge case parameter values
            
            % Test with nu very close to 2 (boundary case)
            nu_near_2 = 2.001;
            sample_near_2 = stdtrnd(100, nu_near_2);
            obj.assertTrue(all(isfinite(sample_near_2)), 'Should handle nu close to 2');
            
            % Test with very large nu values (approaching normal distribution)
            nu_large = 1000;
            sample_large_nu = stdtrnd(100, nu_large);
            obj.assertTrue(all(isfinite(sample_large_nu)), 'Should handle large nu values');
            
            % Test with empty size parameter
            sample_empty = stdtrnd([], 5);
            obj.assertTrue(isscalar(sample_empty), 'Empty size should return a scalar');
            
            % Test generation of a single random number
            sample_single = stdtrnd(1, 5);
            obj.assertTrue(isscalar(sample_single), 'Should generate a single number');
            
            % Verify correct handling of all edge cases
            obj.assertTrue(all(isfinite([sample_near_2; sample_large_nu; sample_empty; sample_single])), ...
                'All edge cases should produce finite values');
        end
        
        function testAgainstReferenceData(obj)
            % Tests against pre-computed reference values
            
            % Skip test if reference data isn't available
            if ~isfield(obj.testData, 'seeds') || isempty(obj.testData.seeds)
                warning('Reference data not available. Skipping test.');
                return;
            end
            
            % Load reference data from known_distributions.mat
            nu_values = fieldnames(obj.testData.seeds);
            for i = 1:length(nu_values)
                nu_field = nu_values{i};
                nu = str2double(nu_field(3:end)); % Extract nu from field name "nu5" -> 5
                
                % Set random seed to match reference data generation
                rng(obj.testData.seeds.(nu_field).seed);
                
                % Generate random numbers using same parameters
                samples = stdtrnd(length(obj.testData.seeds.(nu_field).values), nu);
                
                % Compare generated values with reference values
                obj.assertMatrixEqualsWithTolerance(samples, obj.testData.seeds.(nu_field).values, ...
                    obj.defaultTolerance, sprintf('Mismatch in reference data for nu=%d', nu));
            end
        end
        
        function testNormalApproximation(obj)
            % Tests convergence to normal distribution as degrees of freedom increase
            
            % Generate samples with very large degrees of freedom (nu=100)
            rng(obj.seedValue);
            nu_large = 100;
            samples_t = stdtrnd(obj.sampleSize, nu_large);
            
            % Generate samples from normal distribution with same parameters
            samples_normal = randn(obj.sampleSize, 1);
            
            % Compare empirical distributions using statistical tests
            mean_t = mean(samples_t);
            var_t = var(samples_t);
            skew_t = skewness(samples_t);
            kurt_t = kurtosis(samples_t);
            
            mean_n = mean(samples_normal);
            var_n = var(samples_normal);
            skew_n = skewness(samples_normal);
            kurt_n = kurtosis(samples_normal);
            
            % Verify convergence rate is appropriate (proportional to 1/nu)
            obj.assertAlmostEqual(mean_t, mean_n, 0.1, 'Mean should be similar to normal');
            obj.assertAlmostEqual(var_t, var_n, 0.1, 'Variance should be similar to normal');
            obj.assertAlmostEqual(skew_t, skew_n, 0.2, 'Skewness should be similar to normal');
            obj.assertAlmostEqual(kurt_t, kurt_n, 0.3, 'Kurtosis should be similar to normal');
            
            % Check moments of high-nu t-distribution approach normal distribution
            [h, ~, ksstat] = kstest2(samples_t, samples_normal);
            obj.assertFalse(h, 'KS test should not reject null hypothesis of same distribution');
            obj.assertTrue(ksstat < 0.03, 'KS statistic should be small for high nu');
        end
        
        function testPerformance(obj)
            % Tests performance of random number generation for large sample sizes
            
            % Measure execution time for generating different sample sizes
            sample_sizes = [1000, 10000, 100000];
            nu = 5;
            times = zeros(length(sample_sizes), 1);
            
            for i = 1:length(sample_sizes)
                size_i = sample_sizes(i);
                tic;
                stdtrnd(size_i, nu);
                times(i) = toc;
            end
            
            % Verify linear scaling with sample size
            ratios = times(2:end) ./ times(1:end-1);
            expected_ratios = sample_sizes(2:end) ./ sample_sizes(1:end-1);
            
            for i = 1:length(ratios)
                obj.assertTrue(ratios(i) < expected_ratios(i) * 2, ...
                    'Performance should scale roughly linearly with sample size');
            end
            
            % Compare performance with MATLAB's built-in rng functions
            tic;
            stdtrnd(10000, nu);
            time_stdtrnd = toc;
            
            tic;
            randn(10000, 1);
            time_randn = toc;
            
            % Test memory usage for large sample generation
            if time_stdtrnd > time_randn * 10
                warning(['stdtrnd is more than 10x slower than randn. ', ...
                    'Consider performance optimization.']);
            end
            
            % Verify efficient vectorization
            obj.assertTrue(time_stdtrnd < 1.0, 'Large sample generation should be efficient');
        end
        
        function isValid = verifyDistributionMoments(obj, samples, nu, tolerance)
            % Helper method to verify statistical moments of generated samples
            %
            % INPUTS:
            %   samples - Matrix of samples to analyze
            %   nu - Degrees of freedom parameter
            %   tolerance - Tolerance for comparisons
            %
            % OUTPUTS:
            %   logical - Boolean indicating if all moments are within tolerance
            
            % Calculate sample mean and verify close to 0
            sample_mean = mean(samples);
            mean_valid = abs(sample_mean) < tolerance;
            
            % Calculate sample variance and verify close to 1
            sample_var = var(samples);
            var_valid = abs(sample_var - 1) < tolerance;
            
            % Calculate sample skewness and verify close to 0
            sample_skew = skewness(samples);
            skew_valid = abs(sample_skew) < tolerance;
            
            % Calculate theoretical excess kurtosis: 6/(nu-4) for nu > 4
            if nu > 4
                sample_kurt = kurtosis(samples) - 3; % Convert to excess kurtosis
                theoretical_kurt = 6/(nu-4);
                kurt_valid = abs(sample_kurt - theoretical_kurt) < (tolerance * 2); % Higher tolerance for kurtosis
            else
                kurt_valid = true; % Skip kurtosis check for nu <= 4
            end
            
            % Return true if all moments are within tolerance, false otherwise
            isValid = mean_valid && var_valid && skew_valid && kurt_valid;
        end
    end
end