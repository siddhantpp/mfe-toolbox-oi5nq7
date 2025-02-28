classdef BdsTestTest < BaseTest
    % BdsTestTest Unit test for the bds_test function, which implements the 
    % Brock-Dechert-Scheinkman test for independence and non-linear structure 
    % in financial time series data.
    
    properties
        testData              % Test data structure
        independentSeries     % Purely random series (white noise)
        dependentSeries       % Series with linear dependence (AR process)
        nonlinearSeries       % Series with non-linear structure (GARCH-type)
    end
    
    methods
        function obj = BdsTestTest()
            % Constructor - initializes the BdsTestTest class, inheriting from BaseTest
            obj = obj@BaseTest();
            obj.testData = struct();
            obj.independentSeries = [];
            obj.dependentSeries = [];
            obj.nonlinearSeries = [];
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Set random number generator seed for reproducibility
            rng(1234);
            
            % Load or generate test data sets
            
            % Create independent series using randn
            obj.independentSeries = randn(1000, 1);
            
            % Create dependent series (AR(1) process) with known autocorrelation
            arCoef = 0.6;
            arSeries = zeros(1000, 1);
            arSeries(1) = randn();
            for t = 2:length(arSeries)
                arSeries(t) = arCoef * arSeries(t-1) + randn();
            end
            obj.dependentSeries = arSeries;
            
            % Create non-linear series (GARCH-type) with volatility clustering
            nonlinSeries = zeros(1000, 1);
            volatility = ones(1000, 1);
            omega = 0.05;
            alpha = 0.2;
            beta = 0.75;
            
            nonlinSeries(1) = volatility(1) * randn();
            for t = 2:length(nonlinSeries)
                volatility(t) = omega + alpha * nonlinSeries(t-1)^2 + beta * volatility(t-1);
                nonlinSeries(t) = sqrt(volatility(t)) * randn();
            end
            obj.nonlinearSeries = nonlinSeries;
        end
        
        function tearDown(obj)
            % Clean up after each test method execution
            tearDown@BaseTest(obj);
            % Clear any temporary variables or resources
        end
        
        function testIndependentSeries(obj)
            % Test that BDS test correctly fails to reject independence for truly independent series
            
            % Apply BDS test to independent series
            results = bds_test(obj.independentSeries);
            
            % Verify p-values are above standard significance levels (0.05, 0.01)
            obj.assertTrue(all(results.pval > 0.05), 'P-values should be above 0.05 for independent series');
            
            % Check that test statistics are within expected range for random data
            obj.assertTrue(all(abs(results.stat) < 3), 'Test statistics should be within reasonable range for random data');
            
            % Validate that the test correctly doesn't reject independence
            obj.assertFalse(any(results.H), 'Should not reject independence for random series');
        end
        
        function testDependentSeries(obj)
            % Test that BDS test correctly detects linear dependence in AR(1) series
            
            % Apply BDS test to dependent series (AR process)
            results = bds_test(obj.dependentSeries);
            
            % Verify p-values are below significance level (0.05)
            obj.assertTrue(any(results.pval < 0.05), 'At least some p-values should be below 0.05 for dependent series');
            
            % Check that test statistics are significant
            obj.assertTrue(any(abs(results.stat) > 2), 'Test statistics should be significant for AR process');
            
            % Validate that the test correctly rejects independence
            obj.assertTrue(any(results.H), 'Should reject independence for AR process');
        end
        
        function testNonlinearSeries(obj)
            % Test that BDS test correctly detects non-linear structure in GARCH-type series
            
            % Apply BDS test to non-linear series (GARCH process)
            results = bds_test(obj.nonlinearSeries);
            
            % Verify p-values are below significance level (0.05)
            obj.assertTrue(any(results.pval < 0.05), 'Some p-values should be below 0.05 for non-linear series');
            
            % Check that test statistics are significant
            obj.assertTrue(any(abs(results.stat) > 3), 'Test statistics should be significant for non-linear structure');
            
            % Validate that the test correctly rejects independence and detects non-linear structure
            obj.assertTrue(any(results.H), 'Should reject independence for non-linear process');
        end
        
        function testEmbeddingDimensions(obj)
            % Test BDS test behavior with different embedding dimensions
            
            % Run BDS test with various embedding dimensions (2, 3, 4, 5)
            dimensions = [2, 3, 4, 5];
            results = bds_test(obj.dependentSeries, dimensions);
            
            % Verify results structure contains correct dimensions
            obj.assertEqual(length(results.dim), length(dimensions), 'Results should have correct number of dimensions');
            obj.assertEqual(results.dim, dimensions, 'Dimensions in results should match input dimensions');
            
            % Check that statistics are calculated correctly for each dimension
            obj.assertEqual(length(results.stat), length(dimensions), 'Should have one statistic per dimension');
            obj.assertEqual(length(results.pval), length(dimensions), 'Should have one p-value per dimension');
            
            % Validate that power of the test increases with appropriate dimensions
            obj.assertTrue(any(results.H), 'Test should have power to reject independence with multiple dimensions');
        end
        
        function testEpsilonValues(obj)
            % Test BDS test sensitivity to different epsilon (radius) values
            
            % Run BDS test with various epsilon values (0.5, 0.7, 1.0, 1.5 times std)
            stdVal = std(obj.nonlinearSeries);
            epsilons = [0.5, 0.7, 1.0, 1.5] * stdVal;
            
            % Test with first epsilon value
            results1 = bds_test(obj.nonlinearSeries, [], epsilons(1));
            
            % Test with last epsilon value
            results2 = bds_test(obj.nonlinearSeries, [], epsilons(4));
            
            % Verify results structure contains correct epsilon values
            obj.assertAlmostEqual(results1.epsilon, epsilons(1), 'Results should store correct epsilon value');
            obj.assertAlmostEqual(results2.epsilon, epsilons(4), 'Results should store correct epsilon value');
            
            % Check that statistics change appropriately with epsilon size
            obj.assertFalse(isequal(results1.stat, results2.stat), 'Statistics should differ with different epsilon values');
            
            % Validate that the test maintains power across reasonable epsilon values
            obj.assertTrue(any(results1.H) || any(results2.H), 'Test should detect structure with at least one epsilon value');
        end
        
        function testInputValidation(obj)
            % Test error handling and input validation in BDS test function
            
            % Test with invalid data types (non-numeric, complex)
            obj.assertThrows(@() bds_test('string'), '*', 'Should reject non-numeric input');
            
            complexData = obj.independentSeries + 1i * obj.independentSeries;
            obj.assertThrows(@() bds_test(complexData), '*', 'Should reject complex input');
            
            % Test with improper dimensions (empty matrix, matrix with NaN/Inf)
            obj.assertThrows(@() bds_test([]), '*', 'Should reject empty matrix');
            
            nanData = obj.independentSeries;
            nanData(10) = NaN;
            obj.assertThrows(@() bds_test(nanData), '*', 'Should reject NaN values');
            
            infData = obj.independentSeries;
            infData(10) = Inf;
            obj.assertThrows(@() bds_test(infData), '*', 'Should reject Inf values');
            
            % Test with invalid embedding dimensions (negative, zero, too large)
            obj.assertThrows(@() bds_test(obj.independentSeries, 0), '*', 'Should reject zero embedding dimension');
            obj.assertThrows(@() bds_test(obj.independentSeries, -1), '*', 'Should reject negative embedding dimension');
            obj.assertThrows(@() bds_test(obj.independentSeries, length(obj.independentSeries)+1), '*', 'Should reject too large embedding dimension');
            
            % Test with invalid epsilon values (negative, zero, NaN)
            obj.assertThrows(@() bds_test(obj.independentSeries, [], -1), '*', 'Should reject negative epsilon');
            obj.assertThrows(@() bds_test(obj.independentSeries, [], 0), '*', 'Should reject zero epsilon');
            obj.assertThrows(@() bds_test(obj.independentSeries, [], NaN), '*', 'Should reject NaN epsilon');
        end
        
        function testOutputStructure(obj)
            % Test that the BDS test returns the correct output structure
            
            % Run BDS test with standard parameters
            results = bds_test(obj.dependentSeries);
            
            % Verify all expected fields exist in the result structure
            expectedFields = {'stat', 'pval', 'cv', 'H', 'dim', 'epsilon', 'nobs', 'alpha', 'message'};
            for i = 1:length(expectedFields)
                obj.assertTrue(isfield(results, expectedFields{i}), ['Results should include field: ', expectedFields{i}]);
            end
            
            % Check that dimensions of output arrays match the input embedding dimensions
            dims = results.dim;
            obj.assertEqual(length(results.stat), length(dims), 'Test statistics array should match dimensions');
            obj.assertEqual(length(results.pval), length(dims), 'P-values array should match dimensions');
            obj.assertEqual(length(results.cv), length(dims), 'Critical values array should match dimensions');
            
            % Validate that the output includes test statistics, p-values, and critical values
            obj.assertEqual(results.nobs, length(obj.dependentSeries), 'Number of observations should be correct');
            obj.assertTrue(all(results.pval >= 0 & results.pval <= 1), 'P-values should be between 0 and 1');
            obj.assertTrue(all(isfinite(results.stat)), 'Test statistics should be finite');
        end
        
        function testPerformanceMetrics(obj)
            % Test the computational performance of the BDS test implementation
            
            % Measure execution time for different input sizes
            sizes = [200, 500, 1000];
            times = zeros(length(sizes), 1);
            
            for i = 1:length(sizes)
                data = randn(sizes(i), 1);
                tic;
                bds_test(data);
                times(i) = toc;
            end
            
            % Check memory usage for large datasets
            % Ensure performance scales reasonably with data size and embedding dimensions
            if times(1) > 0  % Avoid division by zero
                scaling = (times(2)/times(1)) / (sizes(2)/sizes(1));
                obj.assertTrue(scaling < 3, 'Execution time should scale reasonably with data size');
            end
            
            % Ensure computational efficiency for typical financial time series applications
            obj.assertTrue(times(3) < 10, 'Execution time should be reasonable for 1000 observations');
        end
        
        function testReproducibility(obj)
            % Test that BDS test results are reproducible with the same inputs
            
            % Run BDS test multiple times with identical inputs
            results1 = bds_test(obj.dependentSeries);
            results2 = bds_test(obj.dependentSeries);
            
            % Verify that results are identical across runs
            obj.assertEqual(results1.stat, results2.stat, 'Test statistics should be identical across runs');
            obj.assertEqual(results1.pval, results2.pval, 'P-values should be identical across runs');
            obj.assertEqual(results1.H, results2.H, 'Test results should be identical across runs');
            
            % Check that random number generator state doesn't affect deterministic calculations
            dims = [2, 3, 4];
            epsilon = 0.7 * std(obj.dependentSeries);
            
            results3 = bds_test(obj.dependentSeries, dims, epsilon);
            results4 = bds_test(obj.dependentSeries, dims, epsilon);
            
            obj.assertEqual(results3.stat, results4.stat, 'Results should be consistent with specific parameters');
            obj.assertEqual(results3.pval, results4.pval, 'Results should be consistent with specific parameters');
        end
    end
end