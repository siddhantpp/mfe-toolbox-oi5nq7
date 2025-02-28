classdef AicsbicTest < BaseTest
    % Test case for aicsbic.m function that validates the calculation of 
    % information criteria (AIC, SBIC) for time series model selection.
    
    properties
        % Test data
        testLogLikelihood % Log-likelihood value for testing
        testNumParams     % Number of parameters for testing
        testNumObs        % Number of observations for testing
        expectedAIC       % Expected AIC value
        expectedSBIC      % Expected SBIC value
        comparator        % NumericalComparator instance
        testDataMultiple  % Multiple test cases data
        tolerance         % Tolerance for numerical comparisons
    end
    
    methods
        function obj = AicsbicTest()
            % Initializes the AicsbicTest with default test properties
            obj = obj@BaseTest();
            obj.tolerance = 1e-10; % Set tolerance for numerical comparisons
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            setUp@BaseTest(obj);
            
            % Initialize comparator for numerical comparisons
            obj.comparator = NumericalComparator();
            
            % Set up standard test case with known values
            obj.testLogLikelihood = -100.5;
            obj.testNumParams = 3;
            obj.testNumObs = 100;
            
            % Calculate expected AIC manually using formula: -2*logL + 2*k
            obj.expectedAIC = -2 * obj.testLogLikelihood + 2 * obj.testNumParams;
            
            % Calculate expected SBIC manually using formula: -2*logL + k*log(T)
            obj.expectedSBIC = -2 * obj.testLogLikelihood + obj.testNumParams * log(obj.testNumObs);
            
            % Generate multiple test cases with different parameter values
            obj.testDataMultiple = struct();
            obj.testDataMultiple.logL = [-95.5; -98.2; -101.5];
            obj.testDataMultiple.k = [2; 3; 4];
            obj.testDataMultiple.T = [100; 120; 150];
            
            % Pre-calculate expected values
            obj.testDataMultiple.expectedAIC = zeros(3, 1);
            obj.testDataMultiple.expectedSBIC = zeros(3, 1);
            
            for i = 1:3
                obj.testDataMultiple.expectedAIC(i) = -2 * obj.testDataMultiple.logL(i) + 2 * obj.testDataMultiple.k(i);
                obj.testDataMultiple.expectedSBIC(i) = -2 * obj.testDataMultiple.logL(i) + obj.testDataMultiple.k(i) * log(obj.testDataMultiple.T(i));
            end
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method runs
            tearDown@BaseTest(obj);
            
            % Clear test properties
            obj.testLogLikelihood = [];
            obj.testNumParams = [];
            obj.testNumObs = [];
            obj.expectedAIC = [];
            obj.expectedSBIC = [];
            obj.testDataMultiple = [];
            
            % Reset the comparator
            obj.comparator = [];
        end
        
        function testBasicCalculation(obj)
            % Tests that aicsbic.m correctly calculates both AIC and SBIC for a standard case
            
            % Call aicsbic with testLogLikelihood, testNumParams, and testNumObs
            result = aicsbic(obj.testLogLikelihood, obj.testNumParams, obj.testNumObs);
            
            % Extract result structure with AIC and SBIC fields
            % Assert that calculated AIC matches expectedAIC within tolerance
            obj.assertEqualsWithTolerance(obj.expectedAIC, result.aic, obj.tolerance, ...
                'AIC calculation is incorrect');
            
            % Assert that calculated SBIC matches expectedSBIC within tolerance
            obj.assertEqualsWithTolerance(obj.expectedSBIC, result.sbic, obj.tolerance, ...
                'SBIC calculation is incorrect');
        end
        
        function testMultipleModels(obj)
            % Tests that aicsbic.m correctly calculates information criteria for multiple models
            
            % Create array of log-likelihood values for multiple models
            logL_array = obj.testDataMultiple.logL;
            
            % Create array of parameter counts for multiple models
            k_array = obj.testDataMultiple.k;
            
            % Call aicsbic with arrays as inputs and same observation count
            result = aicsbic(logL_array, k_array, obj.testDataMultiple.T);
            
            % Verify that output contains arrays of AIC and SBIC values
            % Assert that each calculated value matches expected value
            obj.assertMatrixEqualsWithTolerance(obj.testDataMultiple.expectedAIC, result.aic, obj.tolerance, ...
                'AIC calculation for multiple models is incorrect');
            
            obj.assertMatrixEqualsWithTolerance(obj.testDataMultiple.expectedSBIC, result.sbic, obj.tolerance, ...
                'SBIC calculation for multiple models is incorrect');
        end
        
        function testEdgeCases(obj)
            % Tests that aicsbic.m correctly handles edge cases
            
            % Test with minimum viable parameters (k=1, T=2)
            logL_min = -10;
            k_min = 1;
            T_min = 2;
            result_min = aicsbic(logL_min, k_min, T_min);
            
            expected_aic_min = -2 * logL_min + 2 * k_min;
            expected_sbic_min = -2 * logL_min + k_min * log(T_min);
            
            obj.assertEqualsWithTolerance(expected_aic_min, result_min.aic, obj.tolerance, ...
                'AIC calculation for minimum case is incorrect');
            obj.assertEqualsWithTolerance(expected_sbic_min, result_min.sbic, obj.tolerance, ...
                'SBIC calculation for minimum case is incorrect');
            
            % Test with large sample size (T=1000000)
            T_large = 1000000;
            result_large = aicsbic(obj.testLogLikelihood, obj.testNumParams, T_large);
            
            expected_aic_large = -2 * obj.testLogLikelihood + 2 * obj.testNumParams;
            expected_sbic_large = -2 * obj.testLogLikelihood + obj.testNumParams * log(T_large);
            
            obj.assertEqualsWithTolerance(expected_aic_large, result_large.aic, obj.tolerance, ...
                'AIC calculation for large sample is incorrect');
            obj.assertEqualsWithTolerance(expected_sbic_large, result_large.sbic, obj.tolerance, ...
                'SBIC calculation for large sample is incorrect');
            
            % Test with large parameter count relative to sample (k=T-1)
            T_case = 10;
            k_case = T_case - 1;
            logL_case = -15;
            result_case = aicsbic(logL_case, k_case, T_case);
            
            expected_aic_case = -2 * logL_case + 2 * k_case;
            expected_sbic_case = -2 * logL_case + k_case * log(T_case);
            
            obj.assertEqualsWithTolerance(expected_aic_case, result_case.aic, obj.tolerance, ...
                'AIC calculation for k close to T is incorrect');
            obj.assertEqualsWithTolerance(expected_sbic_case, result_case.sbic, obj.tolerance, ...
                'SBIC calculation for k close to T is incorrect');
            
            % Test with extreme log-likelihood values
            logL_extreme = -1e6;
            result_extreme = aicsbic(logL_extreme, obj.testNumParams, obj.testNumObs);
            
            expected_aic_extreme = -2 * logL_extreme + 2 * obj.testNumParams;
            expected_sbic_extreme = -2 * logL_extreme + obj.testNumParams * log(obj.testNumObs);
            
            obj.assertEqualsWithTolerance(expected_aic_extreme, result_extreme.aic, obj.tolerance, ...
                'AIC calculation for extreme log-likelihood is incorrect');
            obj.assertEqualsWithTolerance(expected_sbic_extreme, result_extreme.sbic, obj.tolerance, ...
                'SBIC calculation for extreme log-likelihood is incorrect');
        end
        
        function testInputValidation(obj)
            % Tests that aicsbic.m correctly validates inputs and throws appropriate errors
            
            % Test with invalid logL (non-numeric, NaN, Inf)
            obj.assertThrows(@() aicsbic('invalid', obj.testNumParams, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for non-numeric logL');
            
            obj.assertThrows(@() aicsbic(NaN, obj.testNumParams, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for NaN logL');
            
            obj.assertThrows(@() aicsbic(Inf, obj.testNumParams, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for Inf logL');
            
            % Test with invalid k (non-integer, negative, zero)
            obj.assertThrows(@() aicsbic(obj.testLogLikelihood, 2.5, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for non-integer k');
            
            obj.assertThrows(@() aicsbic(obj.testLogLikelihood, -1, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for negative k');
            
            obj.assertThrows(@() aicsbic(obj.testLogLikelihood, 0, obj.testNumObs), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for zero k');
            
            % Test with invalid T (non-integer, too small, not > k)
            obj.assertThrows(@() aicsbic(obj.testLogLikelihood, obj.testNumParams, 10.5), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for non-integer T');
            
            obj.assertThrows(@() aicsbic(obj.testLogLikelihood, obj.testNumParams, 0), ...
                'PARAMETERCHECK:InvalidInput', 'aicsbic should throw error for T <= 0');
            
            % Test with T < k (not explicitly checked in aicsbic.m, but important for model validity)
            T_small = 2;
            k_large = 3;
            result_invalid = aicsbic(obj.testLogLikelihood, k_large, T_small);
            
            expected_aic_invalid = -2 * obj.testLogLikelihood + 2 * k_large;
            expected_sbic_invalid = -2 * obj.testLogLikelihood + k_large * log(T_small);
            
            obj.assertEqualsWithTolerance(expected_aic_invalid, result_invalid.aic, obj.tolerance, ...
                'AIC calculation for T < k case is incorrect');
            obj.assertEqualsWithTolerance(expected_sbic_invalid, result_invalid.sbic, obj.tolerance, ...
                'SBIC calculation for T < k case is incorrect');
        end
        
        function testModelComparison(obj)
            % Tests that aicsbic.m produces values that correctly rank competing models
            
            % Generate several models with known properties using generateTimeSeriesData
            % Generate data for models with different complexities
            model1_params = struct('ar', 0.5, 'numObs', 100);
            model2_params = struct('ar', [0.5, 0.2], 'numObs', 100);
            model3_params = struct('ar', [0.5, 0.2, 0.1], 'numObs', 100);
            
            % Generate test data with known properties for time series tests
            model1 = generateTimeSeriesData(model1_params);
            model2 = generateTimeSeriesData(model2_params);
            model3 = generateTimeSeriesData(model3_params);
            
            % Calculate log-likelihoods and parameter counts
            % (For test purposes, we'll use simulated values)
            model1_logL = -95.5;
            model1_k = length(model1_params.ar) + 1; % AR parameters + constant
            
            model2_logL = -94.0;
            model2_k = length(model2_params.ar) + 1;
            
            model3_logL = -93.8;
            model3_k = length(model3_params.ar) + 1;
            
            % Common sample size
            T = 100;
            
            % Calculate information criteria for each model
            ic1 = aicsbic(model1_logL, model1_k, T);
            ic2 = aicsbic(model2_logL, model2_k, T);
            ic3 = aicsbic(model3_logL, model3_k, T);
            
            % Verify that models with better fit but same complexity have lower criteria values
            % Create two models with same complexity but different fit
            same_k = 2;
            better_logL = -90;
            worse_logL = -100;
            
            ic_better = aicsbic(better_logL, same_k, T);
            ic_worse = aicsbic(worse_logL, same_k, T);
            
            obj.assertTrue(ic_better.aic < ic_worse.aic, ...
                'Model with better fit should have lower AIC');
            obj.assertTrue(ic_better.sbic < ic_worse.sbic, ...
                'Model with better fit should have lower SBIC');
            
            % Verify models with same fit but less complexity have lower criteria values
            same_logL = -95;
            simpler_k = 2;
            complex_k = 3;
            
            ic_simpler = aicsbic(same_logL, simpler_k, T);
            ic_complex = aicsbic(same_logL, complex_k, T);
            
            obj.assertTrue(ic_simpler.aic < ic_complex.aic, ...
                'Simpler model with same fit should have lower AIC');
            obj.assertTrue(ic_simpler.sbic < ic_complex.sbic, ...
                'Simpler model with same fit should have lower SBIC');
            
            % Verify that SBIC penalizes complexity more heavily than AIC
            % Compare differences between models with different complexity but similar fit
            aic_diff = ic3.aic - ic2.aic;
            sbic_diff = ic3.sbic - ic2.sbic;
            
            obj.assertTrue(sbic_diff > aic_diff, ...
                'SBIC should penalize additional complexity more heavily than AIC');
        end
        
        function testFormulaVerification(obj)
            % Tests that aicsbic.m implements the correct mathematical formulas
            
            % Manually calculate AIC using formula: -2*logL + 2*k
            logL_test = -75.3;
            k_test = 4;
            T_test = 120;
            
            expected_aic = -2 * logL_test + 2 * k_test;
            
            % Manually calculate SBIC using formula: -2*logL + k*log(T)
            expected_sbic = -2 * logL_test + k_test * log(T_test);
            
            % Compare manual calculations with function output
            result = aicsbic(logL_test, k_test, T_test);
            
            % Verify exact match for mathematical correctness
            obj.assertEqual(expected_aic, result.aic, ...
                'AIC formula implementation is incorrect');
            obj.assertEqual(expected_sbic, result.sbic, ...
                'SBIC formula implementation is incorrect');
        end
    end
end