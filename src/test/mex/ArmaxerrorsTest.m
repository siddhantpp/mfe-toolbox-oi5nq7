classdef ArmaxerrorsTest < BaseTest
    % ARMAXERRORSTEST Test class for validating the armaxerrors MEX implementation
    %
    % This test class validates the functionality, performance, and numerical
    % accuracy of the armaxerrors MEX implementation that computes residuals/innovations
    % for ARMAX (AutoRegressive Moving Average with eXogenous inputs) time series models.
    %
    % The tests ensure the MEX implementation:
    % 1. Correctly exists and can be loaded
    % 2. Produces accurate results for various ARMA/ARMAX model specifications
    % 3. Handles edge cases and invalid inputs appropriately
    % 4. Achieves significant performance improvements over MATLAB implementation
    % 5. Works correctly with large datasets
    % 6. Integrates properly with the armaxfilter MATLAB function
    
    properties
        mexValidator       % MEX validation utility
        testData           % Test data structure
        benchmarkResults   % Performance benchmark results
        mexTolerance       % Numerical tolerance for comparing results
    end
    
    methods
        function obj = ArmaxerrorsTest()
            % Initialize the ArmaxerrorsTest class with test data and validator
            
            % Call parent constructor with test name
            obj@BaseTest('ArmaxerrorsTest');
            
            % Create MEX validator for armaxerrors validation
            obj.mexValidator = MEXValidator();
            
            % Set numerical tolerance for comparing results
            obj.mexTolerance = 1e-12;
            
            % Initialize empty benchmark results structure
            obj.benchmarkResults = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            
            % Call parent setUp method
            obj.setUp@BaseTest();
            
            % Load financial test data
            try
                financialData = obj.loadTestData('financial_data.mat');
                obj.testData.financial = financialData;
            catch
                % Create synthetic data if test data file doesn't exist
                obj.testData.financial.returns = randn(1000, 1);
            end
            
            % Prepare simulated ARMAX test data of various sizes
            obj.testData.small = struct('T', 100, 'data', randn(100, 1));
            obj.testData.medium = struct('T', 1000, 'data', randn(1000, 1));
            obj.testData.large = struct('T', 10000, 'data', randn(10000, 1));
            
            % Initialize benchmark results
            obj.benchmarkResults = struct('mexTime', [], 'matlabTime', [], 'improvement', []);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call parent tearDown method
            obj.tearDown@BaseTest();
            
            % Store benchmark results if applicable
            if isfield(obj.testResults, 'benchmarkResults')
                obj.benchmarkResults = obj.testResults.benchmarkResults;
            end
        end
        
        function testMEXFileExists(obj)
            % Test that the armaxerrors MEX file exists in the expected location
            
            % Verify MEX file exists using the validator
            exists = obj.mexValidator.validateMEXExists('armaxerrors');
            
            % Assert that the MEX file exists
            obj.assertTrue(exists, 'armaxerrors MEX file does not exist');
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of armaxerrors MEX implementation
            
            % Create simple test case
            T = 100;
            data = randn(T, 1);
            ar_params = [0.5]; % AR(1) model
            ma_params = [0.3]; % MA(1) model
            
            % Call armaxerrors MEX function
            errors = armaxerrors(data, ar_params, ma_params);
            
            % Generate expected errors using reference implementation
            expected_errors = obj.implementReferenceARMAXErrors(data, ar_params, ma_params, [], [], false, 0);
            
            % Verify MEX output has correct dimensions
            obj.assertEqual(size(errors), [T, 1], 'MEX output has incorrect dimensions');
            
            % Compare results with reference implementation
            obj.assertMatrixEqualsWithTolerance(errors, expected_errors, obj.mexTolerance, ...
                'MEX function output differs from reference implementation');
        end
        
        function testARModel(obj)
            % Test armaxerrors with pure AR model specifications
            
            % Create pure AR(2) model test case
            T = 200;
            data = randn(T, 1);
            ar_params = [0.5; -0.2]; % AR(2) model
            ma_params = []; % No MA component
            
            % Call armaxerrors MEX function
            errors = armaxerrors(data, ar_params, ma_params);
            
            % Generate expected errors using reference implementation
            expected_errors = obj.implementReferenceARMAXErrors(data, ar_params, [], [], [], false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors, expected_errors, obj.mexTolerance, ...
                'AR model output differs from reference implementation');
        end
        
        function testMAModel(obj)
            % Test armaxerrors with pure MA model specifications
            
            % Create pure MA(2) model test case
            T = 200;
            data = randn(T, 1);
            ar_params = []; % No AR component
            ma_params = [0.3; 0.1]; % MA(2) model
            
            % Call armaxerrors MEX function
            errors = armaxerrors(data, ar_params, ma_params);
            
            % Generate expected errors using reference implementation
            expected_errors = obj.implementReferenceARMAXErrors(data, [], ma_params, [], [], false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors, expected_errors, obj.mexTolerance, ...
                'MA model output differs from reference implementation');
        end
        
        function testExogenousVariables(obj)
            % Test armaxerrors with exogenous variables included
            
            % Create test case with exogenous variables
            T = 200;
            data = randn(T, 1);
            ar_params = [0.5];
            ma_params = [0.3];
            exog_data = randn(T, 2); % Two exogenous variables
            exog_params = [0.7; -0.2]; % Exogenous parameters
            
            % Call armaxerrors MEX function
            errors = armaxerrors(data, ar_params, ma_params, exog_data, exog_params);
            
            % Generate expected errors using reference implementation
            expected_errors = obj.implementReferenceARMAXErrors(data, ar_params, ma_params, exog_params, exog_data, false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors, expected_errors, obj.mexTolerance, ...
                'ARMAX model with exogenous variables output differs from reference implementation');
        end
        
        function testConstantTerm(obj)
            % Test armaxerrors with constant term included
            
            % Create test cases with and without constant term
            T = 200;
            data = randn(T, 1) + 2; % Adding a constant mean
            ar_params = [0.5];
            ma_params = [0.3];
            constant_value = 2;
            
            % Call armaxerrors MEX function with constant
            errors_with_constant = armaxerrors(data, ar_params, ma_params, [], [], constant_value);
            
            % Call armaxerrors MEX function without constant
            errors_without_constant = armaxerrors(data, ar_params, ma_params);
            
            % Generate expected errors using reference implementation
            expected_with_constant = obj.implementReferenceARMAXErrors(data, ar_params, ma_params, [], [], true, constant_value);
            expected_without_constant = obj.implementReferenceARMAXErrors(data, ar_params, ma_params, [], [], false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors_with_constant, expected_with_constant, obj.mexTolerance, ...
                'ARMAX model with constant term output differs from reference implementation');
            
            % Verify constant makes a difference
            obj.assertFalse(all(abs(errors_with_constant - errors_without_constant) < obj.mexTolerance), ...
                'Constant term does not affect the results as expected');
            
            % Verify without constant matches reference
            obj.assertMatrixEqualsWithTolerance(errors_without_constant, expected_without_constant, obj.mexTolerance, ...
                'ARMAX model without constant term output differs from reference implementation');
        end
        
        function testEdgeCases(obj)
            % Test armaxerrors with edge cases and boundary conditions
            
            % Test case 1: Minimal time series length
            min_T = 3;
            min_data = randn(min_T, 1);
            ar_params = [0.5];
            ma_params = [0.3];
            
            % Call armaxerrors MEX function with minimal data
            errors_min = armaxerrors(min_data, ar_params, ma_params);
            
            % Verify output dimensions
            obj.assertEqual(size(errors_min), [min_T, 1], 'MEX output has incorrect dimensions for minimal data');
            
            % Test case 2: High-order AR/MA specifications
            high_T = 100;
            high_data = randn(high_T, 1);
            high_ar_params = rand(10, 1) * 0.05; % Divide by 20 to ensure stationarity
            high_ma_params = rand(10, 1) * 0.05; % High-order MA(10)
            
            % Call armaxerrors MEX function with high-order model
            errors_high = armaxerrors(high_data, high_ar_params, high_ma_params);
            
            % Generate expected errors using reference implementation
            expected_high = obj.implementReferenceARMAXErrors(high_data, high_ar_params, high_ma_params, [], [], false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors_high, expected_high, obj.mexTolerance, ...
                'High-order ARMA model output differs from reference implementation');
            
            % Test case 3: Near unit-root AR process
            unit_T = 100;
            unit_data = randn(unit_T, 1);
            unit_ar_params = [0.999]; % Very close to unit root
            unit_ma_params = [0.3];
            
            % Call armaxerrors MEX function with near unit-root model
            errors_unit = armaxerrors(unit_data, unit_ar_params, unit_ma_params);
            
            % Generate expected errors using reference implementation
            expected_unit = obj.implementReferenceARMAXErrors(unit_data, unit_ar_params, unit_ma_params, [], [], false, 0);
            
            % Verify results match reference implementation
            obj.assertMatrixEqualsWithTolerance(errors_unit, expected_unit, obj.mexTolerance, ...
                'Near unit-root ARMA model output differs from reference implementation');
        end
        
        function testErrorHandling(obj)
            % Test that armaxerrors correctly handles invalid inputs
            
            % Test case 1: Invalid data dimensions
            invalid_data = randn(10, 2); % Not a column vector
            ar_params = [0.5];
            ma_params = [0.3];
            
            % Verify that appropriate error is raised
            obj.assertThrows(@() armaxerrors(invalid_data, ar_params, ma_params), ...
                'MATLAB:badsubscript', 'MEX function should raise error for invalid data dimensions');
            
            % Test case 2: Inconsistent parameter specifications
            inconsistent_data = randn(100, 1);
            inconsistent_exog_data = randn(100, 2);
            inconsistent_exog_params = [0.7]; % Only one parameter for two exogenous variables
            
            % Verify that appropriate error is raised
            obj.assertThrows(@() armaxerrors(inconsistent_data, ar_params, ma_params, inconsistent_exog_data, inconsistent_exog_params), ...
                'MATLAB:mex:error', 'MEX function should raise error for inconsistent exogenous parameters');
            
            % Test case 3: NaN in input data
            nan_data = randn(100, 1);
            nan_data(10) = NaN;
            
            % Verify that appropriate error is raised
            obj.assertThrows(@() armaxerrors(nan_data, ar_params, ma_params), ...
                'MATLAB:mex:error', 'MEX function should raise error for NaN in input data');
            
            % Test case 4: Inf in input data
            inf_data = randn(100, 1);
            inf_data(10) = Inf;
            
            % Verify that appropriate error is raised
            obj.assertThrows(@() armaxerrors(inf_data, ar_params, ma_params), ...
                'MATLAB:mex:error', 'MEX function should raise error for Inf in input data');
        end
        
        function testNumericalAccuracy(obj)
            % Test numerical accuracy of armaxerrors against reference implementation
            
            % Generate test cases with known numerical characteristics
            test_cases = obj.generateTestCases();
            
            % Test each case
            for i = 1:length(test_cases)
                case_name = test_cases(i).name;
                data = test_cases(i).data;
                ar_params = test_cases(i).ar_params;
                ma_params = test_cases(i).ma_params;
                exog_params = test_cases(i).exog_params;
                exog_data = test_cases(i).exog_data;
                include_constant = test_cases(i).include_constant;
                constant_value = test_cases(i).constant_value;
                
                % Call armaxerrors MEX function
                if isempty(exog_data) || isempty(exog_params)
                    if include_constant
                        mex_errors = armaxerrors(data, ar_params, ma_params, [], [], constant_value);
                    else
                        mex_errors = armaxerrors(data, ar_params, ma_params);
                    end
                else
                    if include_constant
                        mex_errors = armaxerrors(data, ar_params, ma_params, exog_data, exog_params, constant_value);
                    else
                        mex_errors = armaxerrors(data, ar_params, ma_params, exog_data, exog_params);
                    end
                end
                
                % Generate expected errors using reference implementation
                expected_errors = obj.implementReferenceARMAXErrors(data, ar_params, ma_params, ...
                    exog_params, exog_data, include_constant, constant_value);
                
                % Verify results match reference implementation
                obj.assertMatrixEqualsWithTolerance(mex_errors, expected_errors, obj.mexTolerance, ...
                    ['Numerical accuracy test failed for case: ' case_name]);
            end
        end
        
        function testPerformance(obj)
            % Test performance of armaxerrors MEX implementation against MATLAB implementation
            
            % Generate large test dataset for meaningful performance comparison
            T = 10000;
            data = randn(T, 1);
            ar_params = [0.5; 0.2; -0.1]; % AR(3) model
            ma_params = [0.3; 0.1; -0.05]; % MA(3) model
            
            % Define MATLAB reference implementation as function handle
            matlab_impl = @(d, ar, ma) obj.implementReferenceARMAXErrors(d, ar, ma, [], [], false, 0);
            
            % Number of repetitions for stable timing measurements
            num_repetitions = 10;
            
            % Measure MEX function execution time
            mex_times = zeros(num_repetitions, 1);
            for i = 1:num_repetitions
                tic;
                armaxerrors(data, ar_params, ma_params);
                mex_times(i) = toc;
            end
            avg_mex_time = mean(mex_times);
            
            % Measure MATLAB implementation execution time
            matlab_times = zeros(num_repetitions, 1);
            for i = 1:num_repetitions
                tic;
                matlab_impl(data, ar_params, ma_params);
                matlab_times(i) = toc;
            end
            avg_matlab_time = mean(matlab_times);
            
            % Calculate performance improvement
            performance_improvement = (avg_matlab_time - avg_mex_time) / avg_matlab_time * 100;
            
            % Store benchmark results
            obj.benchmarkResults.mexTime = avg_mex_time;
            obj.benchmarkResults.matlabTime = avg_matlab_time;
            obj.benchmarkResults.improvement = performance_improvement;
            
            % Print performance results
            fprintf('Performance comparison:\n');
            fprintf('  MEX time: %.6f seconds\n', avg_mex_time);
            fprintf('  MATLAB time: %.6f seconds\n', avg_matlab_time);
            fprintf('  Improvement: %.2f%%\n', performance_improvement);
            
            % Assert that performance improvement meets requirement (>50%)
            obj.assertTrue(performance_improvement > 50, ...
                sprintf('MEX performance improvement is only %.2f%%, but >50%% is required', performance_improvement));
        end
        
        function testLargeDatasets(obj)
            % Test armaxerrors with large datasets to verify memory efficiency
            
            % Generate large time series dataset (10,000+ observations)
            T = 10000;
            data = randn(T, 1);
            ar_params = [0.5; 0.2]; % AR(2) model
            ma_params = [0.3; 0.1]; % MA(2) model
            
            % Measure memory usage before
            before_whos = whos();
            before_mem = sum([before_whos.bytes]);
            
            % Call armaxerrors MEX function with large dataset
            tic;
            errors = armaxerrors(data, ar_params, ma_params);
            execution_time = toc;
            
            % Measure memory usage after
            after_whos = whos();
            after_mem = sum([after_whos.bytes]);
            
            % Calculate memory difference
            mem_diff = (after_mem - before_mem) / (1024*1024); % In MB
            
            % Verify output dimensions
            obj.assertEqual(size(errors), [T, 1], 'MEX output has incorrect dimensions for large dataset');
            
            % Print memory and time statistics
            fprintf('Large dataset performance:\n');
            fprintf('  Execution time: %.6f seconds\n', execution_time);
            fprintf('  Memory difference: %.2f MB\n', mem_diff);
            
            % No specific assertion on memory usage, just logging for inspection
        end
        
        function testIntegration(obj)
            % Test integration with armaxfilter MATLAB function
            
            % Create test case suitable for armaxfilter
            T = 500;
            data = randn(T, 1) * 0.1 + 0.01; % Some small returns with slight drift
            
            % Configure armaxfilter to use armaxerrors MEX function
            options = struct();
            options.p = 1;
            options.q = 1;
            options.constant = true;
            
            % Call armaxfilter which uses armaxerrors internally
            results = armaxfilter(data, [], options);
            
            % Verify armaxfilter executed successfully
            obj.assertTrue(isfield(results, 'residuals'), 'armaxfilter did not produce residuals');
            obj.assertEqual(size(results.residuals), [T, 1], 'armaxfilter residuals have incorrect dimensions');
            
            % Verify parameters were properly estimated
            obj.assertTrue(isfield(results, 'parameters'), 'armaxfilter did not estimate parameters');
            
            % Manually compute residuals using the estimated parameters
            params = results.parameters;
            
            % Reconstruct the parameters based on the model options
            ar_params = params(2:1+options.p);
            ma_params = params(2+options.p:1+options.p+options.q);
            constant_value = params(1);
            
            % Compute residuals using armaxerrors directly
            direct_residuals = armaxerrors(data, ar_params, ma_params, [], [], constant_value);
            
            % Verify integration by checking if residuals match
            obj.assertMatrixEqualsWithTolerance(results.residuals, direct_residuals, obj.mexTolerance, ...
                'armaxfilter residuals do not match those computed directly with armaxerrors');
        end
        
        function errors = implementReferenceARMAXErrors(obj, data, ar_params, ma_params, exog_params, exog_data, include_constant, constant_value)
            % Implements a reference MATLAB-only version of ARMAX error computation for comparison
            
            % Extract dimensions from input arrays
            T = length(data);
            
            % Initialize AR and MA orders
            p = length(ar_params);
            q = length(ma_params);
            
            % Initialize error array
            errors = zeros(T, 1);
            
            % Determine if exogenous variables are present
            has_exog = ~isempty(exog_params) && ~isempty(exog_data);
            r = 0;
            if has_exog
                [~, r] = size(exog_data);
                if length(exog_params) ~= r
                    error('Number of exogenous parameters must match number of exogenous variables');
                end
            end
            
            % Implement time-series iteration with proper initialization
            for t = 1:T
                % Start with constant term if included
                predicted = 0;
                if include_constant
                    predicted = constant_value;
                end
                
                % Apply AR components to predictive model
                for i = 1:min(t-1, p)
                    predicted = predicted + ar_params(i) * data(t-i);
                end
                
                % Apply MA components using previous errors
                for i = 1:min(t-1, q)
                    predicted = predicted + ma_params(i) * errors(t-i);
                end
                
                % Add exogenous variable effects if present
                if has_exog
                    for j = 1:r
                        predicted = predicted + exog_params(j) * exog_data(t, j);
                    end
                end
                
                % Compute errors as difference between actual and predicted values
                errors(t) = data(t) - predicted;
            end
        end
        
        function test_cases = generateTestCases(obj)
            % Generates diverse test cases for ARMAX model validation
            
            % Initialize test case array
            test_cases = struct();
            case_count = 0;
            
            % Generate basic AR(1) test case
            case_count = case_count + 1;
            test_cases(case_count).name = 'AR1';
            test_cases(case_count).data = 0.5 * randn(100, 1);
            test_cases(case_count).ar_params = [0.7];
            test_cases(case_count).ma_params = [];
            test_cases(case_count).exog_params = [];
            test_cases(case_count).exog_data = [];
            test_cases(case_count).include_constant = false;
            test_cases(case_count).constant_value = 0;
            
            % Generate basic MA(1) test case
            case_count = case_count + 1;
            test_cases(case_count).name = 'MA1';
            test_cases(case_count).data = 0.5 * randn(100, 1);
            test_cases(case_count).ar_params = [];
            test_cases(case_count).ma_params = [0.5];
            test_cases(case_count).exog_params = [];
            test_cases(case_count).exog_data = [];
            test_cases(case_count).include_constant = false;
            test_cases(case_count).constant_value = 0;
            
            % Generate ARMA(1,1) test case
            case_count = case_count + 1;
            test_cases(case_count).name = 'ARMA11';
            test_cases(case_count).data = 0.5 * randn(100, 1);
            test_cases(case_count).ar_params = [0.7];
            test_cases(case_count).ma_params = [0.5];
            test_cases(case_count).exog_params = [];
            test_cases(case_count).exog_data = [];
            test_cases(case_count).include_constant = false;
            test_cases(case_count).constant_value = 0;
            
            % Generate ARMAX test case with exogenous variables
            case_count = case_count + 1;
            test_cases(case_count).name = 'ARMAX_Exog';
            test_cases(case_count).data = 0.5 * randn(100, 1) + 1.0; % Data with mean 1.0
            test_cases(case_count).ar_params = [0.7];
            test_cases(case_count).ma_params = [0.5];
            test_cases(case_count).exog_data = randn(100, 2);
            test_cases(case_count).exog_params = [0.3; -0.2];
            test_cases(case_count).include_constant = true;
            test_cases(case_count).constant_value = 1.0;
            
            % Generate edge case with minimal data
            case_count = case_count + 1;
            test_cases(case_count).name = 'Minimal';
            test_cases(case_count).data = 0.5 * randn(5, 1);
            test_cases(case_count).ar_params = [0.5];
            test_cases(case_count).ma_params = [0.3];
            test_cases(case_count).exog_params = [];
            test_cases(case_count).exog_data = [];
            test_cases(case_count).include_constant = false;
            test_cases(case_count).constant_value = 0;
            
            % Generate large-scale test case
            case_count = case_count + 1;
            test_cases(case_count).name = 'LargeScale';
            test_cases(case_count).data = 0.5 * randn(1000, 1);
            test_cases(case_count).ar_params = [0.5; 0.2; -0.1];
            test_cases(case_count).ma_params = [0.3; 0.1; -0.05];
            test_cases(case_count).exog_params = [];
            test_cases(case_count).exog_data = [];
            test_cases(case_count).include_constant = false;
            test_cases(case_count).constant_value = 0;
        end
    end
end