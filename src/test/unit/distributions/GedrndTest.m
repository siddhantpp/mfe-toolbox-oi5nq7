classdef GedrndTest < BaseTest
    % Test class for the gedrnd function that generates random numbers from Generalized Error Distribution
    
    properties
        testData       % Structure containing test data
        testTolerance  % Tolerance for numerical comparisons
    end
    
    methods
        function obj = GedrndTest()
            % Initialize the GedrndTest class with test data
            obj = obj@BaseTest();
            obj.testTolerance = 1e-10;
            obj.testData = obj.loadTestData('known_distributions.mat');
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            % Set fixed random seed for reproducible tests
            rng(1);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj);
            % Reset random number generator state
            rng('default');
        end
        
        function testBasicFunctionality(obj)
            % Test that gedrnd generates matrix of correct size
            nu = 2;  % Normal distribution
            rows = 5;
            cols = 10;
            output = gedrnd(nu, rows, cols);
            
            % Verify dimensions
            obj.assertEqual(size(output), [rows, cols], 'Output matrix dimensions incorrect');
            
            % Verify all values are finite
            obj.assertTrue(all(isfinite(output(:))), 'Output contains non-finite values');
        end
        
        function testOutputDistribution(obj)
            % Test that output values follow the GED distribution
            % Test various shape parameters
            rng(123); % Fixed seed for reproducibility
            
            nuValues = [0.5, 1, 2, 5];
            n = 10000; % Large sample for distribution testing
            
            for i = 1:length(nuValues)
                nu = nuValues(i);
                
                % Generate random samples
                samples = gedrnd(nu, n, 1);
                
                % Calculate sample statistics
                sampleMean = mean(samples);
                sampleVar = var(samples);
                sampleSkew = skewness(samples);
                
                % Expected statistics for standardized GED
                expectedMean = 0;
                expectedVar = 1;
                expectedSkew = 0; % GED is symmetric
                
                % Test mean is close to expected
                obj.assertEqualsWithTolerance(sampleMean, expectedMean, 0.05, ...
                    sprintf('Mean for nu=%g differs from expected', nu));
                
                % Test variance is close to 1 (unit variance)
                obj.assertEqualsWithTolerance(sampleVar, expectedVar, 0.1, ...
                    sprintf('Variance for nu=%g differs from expected', nu));
                
                % Test skewness is close to 0 (symmetric)
                obj.assertEqualsWithTolerance(sampleSkew, expectedSkew, 0.15, ...
                    sprintf('Skewness for nu=%g differs from expected', nu));
                
                % Chi-square goodness-of-fit test with relaxed criteria
                [~, p] = chi2gof(samples, 'CDF', @(x) gedcdf(x, nu), 'NBins', 50);
                
                % For random tests, we'll use a lower significance level
                obj.assertTrue(p > 0.01, sprintf('Chi-square test failed for nu=%g, p=%g', nu, p));
            end
        end
        
        function testShapeParameterEffects(obj)
            % Test that different shape parameters produce appropriate distributions
            rng(234); % Fixed seed
            
            nuValues = [0.5, 1, 2, 5];
            n = 5000;
            
            kurtosisValues = zeros(size(nuValues));
            
            for i = 1:length(nuValues)
                nu = nuValues(i);
                samples = gedrnd(nu, n, 1);
                
                % Calculate sample kurtosis
                kurtosisValues(i) = kurtosis(samples);
                
                % Check symmetry around zero
                leftMean = mean(samples(samples < 0));
                rightMean = mean(samples(samples > 0));
                
                if ~isempty(leftMean) && ~isempty(rightMean)
                    obj.assertEqualsWithTolerance(abs(leftMean), abs(rightMean), 0.2, ...
                        sprintf('Distribution for nu=%g is not symmetric', nu));
                end
                
                % Verify variance is approximately 1
                obj.assertEqualsWithTolerance(var(samples), 1, 0.2, ...
                    sprintf('Variance for nu=%g is not approximately 1', nu));
            end
            
            % Verify that kurtosis decreases as nu increases
            % (higher nu = thinner tails = lower kurtosis)
            for i = 1:length(nuValues)-1
                if nuValues(i) < nuValues(i+1)
                    obj.assertTrue(kurtosisValues(i) >= kurtosisValues(i+1) - 0.5, ...
                        sprintf('Kurtosis should decrease with increasing nu: %g to %g', ...
                        nuValues(i), nuValues(i+1)));
                end
            end
        end
        
        function testLocationScaleTransformation(obj)
            % Test location and scale parameters correctly transform the distribution
            rng(345); % Fixed seed
            
            nu = 1.5;
            n = 1000;
            mu = 3;
            sigma = 2;
            
            % Generate samples with default parameters (mu=0, sigma=1)
            defaultSamples = gedrnd(nu, n, 1);
            
            % Generate samples with specific location and scale
            transformedSamples = gedrnd(nu, n, 1, mu, sigma);
            
            % Check that the mean approximates mu
            obj.assertEqualsWithTolerance(mean(transformedSamples), mu, 0.2, ...
                'Mean of transformed distribution does not match location parameter');
            
            % Check that the variance scales by sigma^2
            obj.assertEqualsWithTolerance(var(transformedSamples), var(defaultSamples) * sigma^2, 0.5, ...
                'Variance of transformed distribution does not scale correctly');
            
            % Check that standardizing the transformed samples gives similar distribution
            standardized = (transformedSamples - mu) / sigma;
            
            % Compare standardized samples to default samples via quantiles
            defaultQuantiles = quantile(defaultSamples, [0.1, 0.25, 0.5, 0.75, 0.9]);
            standardizedQuantiles = quantile(standardized, [0.1, 0.25, 0.5, 0.75, 0.9]);
            
            % The quantiles should be approximately equal
            obj.assertMatrixEqualsWithTolerance(defaultQuantiles, standardizedQuantiles, 0.2, ...
                'Standardized samples differ from default distribution');
        end
        
        function testInvalidInputs(obj)
            % Test error handling for invalid input parameters
            
            % Test negative shape parameter
            obj.assertThrows(@() gedrnd(-1, 10, 1), '', ...
                'Negative nu should raise an error');
            
            % Test zero shape parameter
            obj.assertThrows(@() gedrnd(0, 10, 1), '', ...
                'Zero nu should raise an error');
            
            % Test negative scale parameter
            obj.assertThrows(@() gedrnd(1.5, 10, 1, 0, -1), '', ...
                'Negative sigma should raise an error');
            
            % Test negative size parameters
            obj.assertThrows(@() gedrnd(1.5, -10, 1), '', ...
                'Negative row count should raise an error');
            
            obj.assertThrows(@() gedrnd(1.5, 10, -1), '', ...
                'Negative column count should raise an error');
            
            % Test non-numeric inputs
            obj.assertThrows(@() gedrnd('string', 10, 1), '', ...
                'Non-numeric nu should raise an error');
            
            obj.assertThrows(@() gedrnd(1.5, 'string', 1), '', ...
                'Non-numeric row count should raise an error');
        end
        
        function testReproducibility(obj)
            % Test that results are reproducible with fixed random seed
            
            % Set specific seed
            rng(789);
            
            % Generate first sample
            sample1 = gedrnd(1.5, 100, 1);
            
            % Reset same seed
            rng(789);
            
            % Generate second sample
            sample2 = gedrnd(1.5, 100, 1);
            
            % Samples should be identical
            obj.assertTrue(all(sample1 == sample2), 'Samples are not reproducible with fixed seed');
        end
        
        function testKnownValues(obj)
            % Test against pre-computed known values from test data
            if isfield(obj.testData, 'ged_rnd_seeds')
                knownData = obj.testData.ged_rnd_seeds;
                
                % Check format of test data and adapt accordingly
                if isstruct(knownData)
                    % Process structured test data
                    fields = fieldnames(knownData);
                    if all(ismember({'nu', 'seed', 'values'}, fields))
                        for i = 1:length(knownData.nu)
                            nu = knownData.nu(i);
                            seed = knownData.seed(i);
                            expectedValues = knownData.values(i, :);
                            
                            rng(seed);
                            actualValues = gedrnd(nu, 1, length(expectedValues));
                            
                            obj.assertMatrixEqualsWithTolerance(actualValues, expectedValues, obj.testTolerance, ...
                                sprintf('Random values do not match known values for nu=%g', nu));
                        end
                    end
                elseif ismatrix(knownData.params) && ismatrix(knownData.values)
                    % Process matrix-based test data
                    for i = 1:size(knownData.params, 1)
                        % Extract parameters and expected values
                        nu = knownData.params(i, 1);
                        seed = knownData.seeds(i);
                        expectedValues = knownData.values(i, :);
                        
                        % Set the specified seed
                        rng(seed);
                        
                        % Generate values
                        actualValues = gedrnd(nu, 1, length(expectedValues));
                        
                        % Compare with expected values
                        obj.assertMatrixEqualsWithTolerance(actualValues, expectedValues, obj.testTolerance, ...
                            sprintf('Random values do not match known values for nu=%g', nu));
                    end
                end
            else
                warning('Test data for gedrnd not found in test data file');
            end
        end
        
        function testOptionalParameters(obj)
            % Test behavior with optional parameters (mu, sigma)
            
            nu = 1.5;
            n = 100;
            m = 1;
            
            % Test with default parameters
            rng(101);
            sample1 = gedrnd(nu, n, m);
            
            % Test with explicit default parameters
            rng(101);
            sample2 = gedrnd(nu, n, m, 0, 1);
            
            % Results should be identical
            obj.assertTrue(all(sample1 == sample2), 'Default parameters do not match explicit defaults');
            
            % Test with different mu and sigma
            mu = 2;
            sigma = 3;
            
            % Generate samples with explicit parameters
            rng(101);
            sample3 = gedrnd(nu, n, m, mu, sigma);
            
            % Results should be different from default
            obj.assertFalse(all(sample1 == sample3), 'Custom parameters should produce different results');
            
            % Check that sample statistics reflect the parameters
            obj.assertEqualsWithTolerance(mean(sample3), mu, 0.5, 'Mean does not match location parameter');
            
            % Check that standard deviation scales appropriately
            lambda = sqrt(gamma(3/nu)/gamma(1/nu));
            expectedStd = lambda * sigma;
            obj.assertEqualsWithTolerance(std(sample3), expectedStd, 0.5, ...
                'Standard deviation does not match scale parameter');
        end
    end
end