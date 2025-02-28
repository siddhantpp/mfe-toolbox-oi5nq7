classdef JumpTestTest < BaseTest
    properties
        testData          % Structure to store loaded test data
        syntheticData     % Structure to store synthetic test data
        testTolerance     % Numerical tolerance for comparisons
    end
    
    methods
        function obj = JumpTestTest()
            % Initialize the JumpTestTest class
            obj@BaseTest();
            obj.testTolerance = 1e-8;  % Set numerical tolerance for jump tests
            obj.testData = [];
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            obj.setUp@BaseTest();
            
            % Load high-frequency test data from MAT file
            try
                obj.testData = obj.loadTestData('high_frequency_data.mat');
            catch ME
                warning('Could not load high-frequency test data: %s. Using synthetic data only.', ME.message);
            end
            
            % Generate synthetic high-frequency data with known properties
            params = struct('jumpIntensity', 2, 'jumpSize', 0.01);
            obj.syntheticData = TestDataGenerator.generateHighFrequencyData(5, 78, params);
            
            % Set fixed random seed for reproducibility
            rng(42);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            obj.tearDown@BaseTest();
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of jump_test with standard inputs
            
            % Use a sample of high-frequency returns from testData
            if ~isempty(obj.testData) && isfield(obj.testData, 'returns')
                returns = obj.testData.returns;
            else
                % Fall back to synthetic data if test data unavailable
                returns = obj.syntheticData.returns;
            end
            
            % Call jump_test with default parameters
            results = jump_test(returns);
            
            % Verify that results structure has expected fields
            obj.assertTrue(isstruct(results), 'Results should be a structure');
            expectedFields = {'zStatistic', 'pValue', 'criticalValues', 'significanceLevels', ...
                'jumpDetected', 'jumpComponent', 'contComponent', 'rv', 'bv', 'ratio'};
            for i = 1:length(expectedFields)
                obj.assertTrue(isfield(results, expectedFields{i}), ...
                    ['Results should contain field: ' expectedFields{i}]);
            end
            
            % Check that test statistic is of expected type and dimension
            obj.assertEqual(size(results.zStatistic, 2), size(returns, 2), ...
                'Z-statistic should have same number of columns as returns');
            
            % Validate that p-values are in the range [0,1]
            obj.assertTrue(all(results.pValue >= 0 & results.pValue <= 1), ...
                'P-values should be in the range [0,1]');
            
            % Verify critical values are properly computed
            obj.assertEqual(size(results.criticalValues, 1), length(results.significanceLevels), ...
                'Number of critical values should match number of significance levels');
            
            % Check that jumps are detected correctly based on test statistic and critical values
            obj.assertEqual(size(results.jumpDetected), [length(results.significanceLevels), size(returns, 2)], ...
                'jumpDetected should have dimensions [numLevels x numSeries]');
        end
        
        function testWithOptionsInput(obj)
            % Test jump_test with various option configurations
            
            % Get test returns
            if ~isempty(obj.testData) && isfield(obj.testData, 'returns')
                returns = obj.testData.returns;
            else
                returns = obj.syntheticData.returns;
            end
            
            % Create options structure with custom significance levels
            options = struct('alpha', 0.01);
            results1 = jump_test(returns, options);
            
            % Verify that custom significance levels are correctly applied
            obj.assertTrue(any(results1.significanceLevels == 0.01), ...
                'Custom significance level should be used');
            
            % Test with options for alternative test statistics
            options = struct('bvOptions', struct('scaleFactor', 0.8));
            results2 = jump_test(returns, options);
            
            % Verify that BV is different when using custom options
            obj.assertFalse(isequal(results1.bv, results2.bv), ...
                'BV should differ when using custom BV options');
            
            % Validate that all option combinations produce valid results
            options = struct('alpha', 0.10, ...
                'bvOptions', struct('scaleFactor', 0.9), ...
                'rvOptions', struct('scale', 1));
            results3 = jump_test(returns, options);
            
            obj.assertTrue(all(results3.pValue >= 0 & results3.pValue <= 1), ...
                'P-values should remain valid with combined options');
        end
        
        function testConsistencyWithRvBv(obj)
            % Test that jump_test results are consistent with direct rv_compute and bv_compute calls
            
            % Compute RV directly using rv_compute
            returns = obj.syntheticData.returns;
            rv_direct = rv_compute(returns);
            
            % Compute BV directly using bv_compute
            bv_direct = bv_compute(returns);
            
            % Calculate test statistic manually using RV/BV - 1
            ratio_direct = rv_direct ./ bv_direct;
            stat_direct = ratio_direct - 1;
            
            % Get jump_test results
            results = jump_test(returns);
            
            % Compare manual calculation with jump_test results
            obj.assertAlmostEqual(stat_direct, results.ratio - 1, ...
                'Manual ratio calculation should match jump_test calculation');
            
            % Verify that ratio statistic matches within numerical tolerance
            obj.assertAlmostEqual(rv_direct, results.rv, ...
                'RV from direct calculation should match jump_test RV');
            obj.assertAlmostEqual(bv_direct, results.bv, ...
                'BV from direct calculation should match jump_test BV');
        end
        
        function testWithSyntheticData(obj)
            % Test jump_test with synthetic data containing known jumps
            
            % Use synthetic data with known jump properties
            returns = obj.syntheticData.returns;
            jumpTimes = obj.syntheticData.jumpTimes;
            jumpSizes = obj.syntheticData.jumpSizes;
            
            % Run jump_test on this data
            results = jump_test(returns);
            
            % Verify that jumps are detected at appropriate locations
            jumpIndices = jumpTimes(jumpTimes <= length(returns));
            largeJumpIndices = jumpIndices(abs(jumpSizes(jumpTimes <= length(returns))) > median(abs(jumpSizes)));
            
            if ~isempty(largeJumpIndices)
                % Check that jump component estimation is accurate
                jumpDetectionRate = sum(results.jumpDetected(2, largeJumpIndices)) / length(largeJumpIndices);
                obj.assertTrue(jumpDetectionRate >= 0.5, ...
                    'At least 50% of large synthetic jumps should be detected');
            end
            
            % Validate consistency between test decisions and true jump occurrences
            obj.assertTrue(all(results.jumpComponent >= 0 & results.jumpComponent <= results.rv), ...
                'Jump component should be between 0 and RV');
            
            % Check that continuous + jump components sum to total RV
            obj.assertAlmostEqual(results.contComponent + results.jumpComponent, results.rv, ...
                'Continuous and jump components should sum to total RV');
        end
        
        function testErrorHandling(obj)
            % Test error handling for invalid inputs
            
            % Test with empty input, should throw error
            obj.assertThrows(@() jump_test([]), 'MATLAB:minrhs', ...
                'Empty input should throw an error');
            
            % Test with non-numeric input, should throw error
            obj.assertThrows(@() jump_test('invalid'), 'MATLAB:invalidInput', ...
                'Non-numeric input should throw an error');
            
            % Test with NaN/Inf values, should throw error
            returns = [1; 2; NaN; 4; 5];
            obj.assertThrows(@() jump_test(returns), 'MATLAB:invalidInput', ...
                'Input with NaN values should throw an error');
            
            % Test with invalid options structure, should throw error
            returns = randn(10, 1);
            obj.assertThrows(@() jump_test(returns, 'invalid'), 'MATLAB:invalidInput', ...
                'Invalid options parameter should throw an error');
            
            % Verify appropriate error messages are generated
            obj.assertThrows(@() jump_test(returns, struct('alpha', 2)), 'MATLAB:invalidParam', ...
                'Alpha value outside [0,1] should throw an error');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability with extreme values and edge cases
            
            % Test with very small return values
            smallReturns = 1e-8 * randn(100, 1);
            results = jump_test(smallReturns);
            
            % Verify results remain numerically stable
            obj.assertFalse(any(isnan([results.zStatistic, results.pValue, results.ratio])), ...
                'Results should not contain NaN values with small returns');
            
            % Test with very large return values
            largeReturns = 1e6 * randn(100, 1);
            results = jump_test(largeReturns);
            
            % Check that no NaN or Inf values appear in results
            obj.assertFalse(any(isnan([results.zStatistic, results.pValue, results.ratio])), ...
                'Results should not contain NaN values with large returns');
            obj.assertFalse(any(isinf([results.zStatistic, results.pValue, results.ratio])), ...
                'Results should not contain Inf values with large returns');
            
            % Test with returns near zero but non-zero
            nearZeroReturns = 1e-14 * (rand(100, 1) - 0.5);
            results = jump_test(nearZeroReturns);
            
            obj.assertFalse(any(isnan([results.zStatistic, results.pValue, results.ratio])), ...
                'Results should not contain NaN values with near-zero returns');
        end
        
        function testJumpComponentEstimation(obj)
            % Test the jump component estimation feature
            
            % Use returns with known jump properties
            returns = obj.syntheticData.returns;
            
            % Call jump_test with component estimation enabled
            results = jump_test(returns);
            
            % Verify that continuous and jump components sum to total RV
            obj.assertAlmostEqual(results.contComponent + results.jumpComponent, results.rv, ...
                'Continuous and jump components should sum to total RV');
            
            % Check bounds of jump component (should be between 0 and RV)
            obj.assertTrue(all(results.jumpComponent >= 0 & results.jumpComponent <= results.rv), ...
                'Jump component should be between 0 and RV');
            
            % Validate that jump component is zero when no jumps are detected
            noJumpIndices = find(~results.jumpDetected(2, :));
            if ~isempty(noJumpIndices)
                obj.assertTrue(all(abs(results.jumpComponent(noJumpIndices)) < obj.testTolerance), ...
                    'Jump component should be effectively zero where no jumps are detected');
                
                % Continuous component should equal RV where no jumps are detected
                obj.assertAlmostEqual(results.contComponent(noJumpIndices), results.rv(noJumpIndices), ...
                    'Continuous component should equal RV where no jumps are detected');
            end
        end
        
        function testPerformance(obj)
            % Test performance with large datasets
            
            % Generate large high-frequency dataset
            params = struct('jumpIntensity', 1);
            largeData = TestDataGenerator.generateHighFrequencyData(20, 78, params);
            returns = largeData.returns;
            
            % Measure execution time using measureExecutionTime method
            executionTime = obj.measureExecutionTime(@() jump_test(returns));
            
            % Generate even larger dataset to test scalability
            veryLargeData = TestDataGenerator.generateHighFrequencyData(40, 78, params);
            veryLargeReturns = veryLargeData.returns;
            largerExecutionTime = obj.measureExecutionTime(@() jump_test(veryLargeReturns));
            
            % Verify that execution time scales reasonably with data size
            expectedRatio = length(veryLargeReturns) / length(returns);
            actualRatio = largerExecutionTime / executionTime;
            
            obj.assertTrue(actualRatio < expectedRatio * 3, ...
                'Execution time should scale reasonably with data size');
            
            % Check memory usage patterns
            fprintf('jump_test execution time: %.4f s (small), %.4f s (large), ratio: %.2f\n', ...
                executionTime, largerExecutionTime, actualRatio);
        end
    end
end