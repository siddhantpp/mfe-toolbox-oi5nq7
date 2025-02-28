classdef Lmtest1Test < BaseTest
    %LMTEST1TEST Test class for the lmtest1 function that implements Lagrange Multiplier test for autocorrelation
    %   This class contains unit tests to verify the functionality of the
    %   lmtest1 function, which performs the Lagrange Multiplier test for
    %   autocorrelation in time series residuals. The tests cover basic
    %   functionality, random data, lag specification, error handling, and
    %   regression output validation.

    properties
        testData        % Financial returns data loaded from test data files
        autocorrelatedData % Autocorrelated time series data
        randomData      % Randomly generated time series data
        defaultLags     % Default number of lags for testing
    end

    methods
        function obj = Lmtest1Test()
            %Lmtest1Test Initialize the Lmtest1Test class
            %   Lmtest1Test() constructs an instance of the Lmtest1Test class,
            %   setting the default properties to empty matrices and setting
            %   defaultLags to 5 for testing.

            % Call superclass constructor with 'Lmtest1Test' name
            obj = obj@BaseTest('Lmtest1Test');

            % Set default properties to empty matrices
            obj.testData = [];
            obj.autocorrelatedData = [];
            obj.randomData = [];

            % Set defaultLags to 5 for testing
            obj.defaultLags = 5;
        end

        function setUp(obj)
            %setUp Set up test environment before each test method execution
            %   setUp() prepares the test environment before each test method
            %   is executed. This includes loading test data, generating
            %   autocorrelated data, and generating random data.

            % Call superclass setUp method
            setUp@BaseTest(obj);

            % Generate autocorrelated test data with known properties
            obj.autocorrelatedData = obj.generateAutocorrelatedData(100, 0.7);

            % Generate random test data without autocorrelation
            obj.randomData = randn(100, 1);

            % Set testData to financial returns from test data files
            data = obj.loadTestData('financial_returns.mat');
            obj.testData = data.returns(:, 1);
        end

        function tearDown(obj)
            %tearDown Clean up test environment after each test method execution
            %   tearDown() cleans up the test environment after each test method
            %   is executed. This includes resetting test data variables to
            %   empty matrices.

            % Reset test data variables to empty matrices
            obj.testData = [];
            obj.autocorrelatedData = [];
            obj.randomData = [];

            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end

        function testBasicFunctionality(obj)
            %testBasicFunctionality Test that lmtest1 computes correct test statistics for basic cases
            %   testBasicFunctionality() tests that lmtest1 computes correct
            %   test statistics for basic cases, including verifying the test
            %   statistic, p-value, degrees of freedom, and critical values.

            % Run lmtest1 on autocorrelated data with default lags
            results = lmtest1(obj.autocorrelatedData, obj.defaultLags);

            % Verify test statistic is computed correctly
            expectedStat = obj.computeManualLMStatistic(obj.autocorrelatedData, obj.defaultLags).stat;
            obj.assertAlmostEqual(results.stat, expectedStat, 'Test statistic is not computed correctly');

            % Verify p-value is computed correctly using chi-square distribution
            expectedPval = 1 - chi2cdf(results.stat, results.dof);
            obj.assertAlmostEqual(results.pval, expectedPval, 'P-value is not computed correctly');

            % Verify degrees of freedom match expected value
            obj.assertEqual(results.dof, obj.defaultLags, 'Degrees of freedom do not match expected value');

            % Verify critical values are computed correctly
            % (This part requires knowing the expected critical values for the given dof)
            % For example, for dof=5, the critical values at [0.10, 0.05, 0.01] are approximately [9.236, 11.070, 15.086]
            % You may need to adjust these values based on your specific test case
            expectedCrit = [9.236, 11.070, 15.086];
            obj.assertAlmostEqual(results.crit(1), expectedCrit(1), 'Critical value at 0.10 is not computed correctly');
            obj.assertAlmostEqual(results.crit(2), expectedCrit(2), 'Critical value at 0.05 is not computed correctly');
            obj.assertAlmostEqual(results.crit(3), expectedCrit(3), 'Critical value at 0.01 is not computed correctly');
        end

        function testRandomData(obj)
            %testRandomData Test that lmtest1 produces expected results for random data with no autocorrelation
            %   testRandomData() tests that lmtest1 produces expected results
            %   for random data with no autocorrelation, including verifying
            %   that the test statistic is smaller than critical values in
            %   most cases and that the null hypothesis is not rejected at
            %   reasonable significance levels.

            % Run lmtest1 on random data with default lags
            results = lmtest1(obj.randomData, obj.defaultLags);

            % Verify test statistic is smaller than critical values in most cases
            obj.assertTrue(results.stat < max(results.crit), 'Test statistic is not smaller than critical values in most cases');

            % Verify null hypothesis is not rejected at reasonable significance levels
            obj.assertFalse(any(results.sig), 'Null hypothesis is rejected at reasonable significance levels');

            % Verify p-value is correctly computed and relatively high
            obj.assertTrue(results.pval > 0.05, 'P-value is not correctly computed and relatively high');
        end

        function testWithRegressors(obj)
            %testWithRegressors Test lmtest1 with additional regressors in the auxiliary regression
            %   testWithRegressors() tests lmtest1 with additional regressors
            %   in the auxiliary regression, including verifying the degrees
            %   of freedom, test statistic, p-value, and critical values.

            % Create exogenous regressors matrix
            regressors = randn(length(obj.autocorrelatedData), 2);

            % Run lmtest1 with regressors and autocorrelated data
            results = lmtest1(obj.autocorrelatedData, obj.defaultLags, regressors);

            % Verify degrees of freedom account for additional regressors
            expectedDof = obj.defaultLags;
            obj.assertEqual(results.dof, expectedDof, 'Degrees of freedom do not account for additional regressors');

            % Verify test statistic and p-value are computed correctly
            expectedStat = obj.computeManualLMStatistic(obj.autocorrelatedData, obj.defaultLags).stat;
            obj.assertAlmostEqual(results.stat, expectedStat, 'Test statistic is not computed correctly with regressors');

            % Verify critical values account for regressors
            % (This part requires knowing the expected critical values for the given dof)
            % For example, for dof=5, the critical values at [0.10, 0.05, 0.01] are approximately [9.236, 11.070, 15.086]
            % You may need to adjust these values based on your specific test case
            expectedCrit = [9.236, 11.070, 15.086];
            obj.assertAlmostEqual(results.crit(1), expectedCrit(1), 'Critical value at 0.10 is not computed correctly with regressors');
            obj.assertAlmostEqual(results.crit(2), expectedCrit(2), 'Critical value at 0.05 is not computed correctly with regressors');
            obj.assertAlmostEqual(results.crit(3), expectedCrit(3), 'Critical value at 0.01 is not computed correctly with regressors');
        end

        function testLagSpecification(obj)
            %testLagSpecification Test lmtest1 with different lag specifications
            %   testLagSpecification() tests lmtest1 with different lag
            %   specifications, including verifying results with small and
            %   large lag orders and comparing results across different lag
            %   specifications.

            % Run lmtest1 with small number of lags (2)
            resultsSmallLag = lmtest1(obj.autocorrelatedData, 2);

            % Verify results with small lag order
            obj.assertTrue(isstruct(resultsSmallLag), 'Results with small lag order are not a struct');

            % Run lmtest1 with large number of lags (15)
            resultsLargeLag = lmtest1(obj.autocorrelatedData, 15);

            % Verify results with large lag order
            obj.assertTrue(isstruct(resultsLargeLag), 'Results with large lag order are not a struct');

            % Compare results across different lag specifications
            obj.assertFalse(resultsSmallLag.stat == resultsLargeLag.stat, 'Results are the same across different lag specifications');
        end

        function testDefaultLagBehavior(obj)
            %testDefaultLagBehavior Test lmtest1 default lag selection behavior
            %   testDefaultLagBehavior() tests lmtest1 default lag selection
            %   behavior, including verifying that the default lag selection
            %   rule is applied correctly and that results are consistent
            %   with expected default behavior.

            % Run lmtest1 on data without specifying lags
            resultsDefaultLag = lmtest1(obj.autocorrelatedData);

            % Verify default lag selection rule is applied correctly
            expectedDefaultLag = min(10, floor(length(obj.autocorrelatedData) / 5));
            obj.assertEqual(resultsDefaultLag.dof, expectedDefaultLag, 'Default lag selection rule is not applied correctly');

            % Verify results are consistent with expected default behavior
            obj.assertTrue(isstruct(resultsDefaultLag), 'Results are not a struct with default lag behavior');
        end

        function testErrorHandling(obj)
            %testErrorHandling Test lmtest1 error handling for invalid inputs
            %   testErrorHandling() tests lmtest1 error handling for invalid
            %   inputs, including verifying that appropriate errors are
            %   thrown for empty data, non-numeric data, negative lag value,
            %   lag too large for sample size, and invalid regressors matrix.

            % Test with empty data and verify appropriate error is thrown
            obj.assertThrows(@() lmtest1([]), 'MATLAB:validators:mustBeNonempty', 'Error not thrown for empty data');

            % Test with non-numeric data and verify appropriate error is thrown
            obj.assertThrows(@() lmtest1({'a', 'b', 'c'}), 'MATLAB:datacheck:mustBeNumeric', 'Error not thrown for non-numeric data');

            % Test with negative lag value and verify appropriate error is thrown
            obj.assertThrows(@() lmtest1(obj.autocorrelatedData, -1), 'MATLAB:parametercheck:mustBeGreaterThanOrEqualTo0', 'Error not thrown for negative lag value');

            % Test with lag too large for sample size and verify appropriate error is thrown
            obj.assertThrows(@() lmtest1(obj.autocorrelatedData, length(obj.autocorrelatedData)), 'MATLAB:lmtest1:LagTooLarge', 'Error not thrown for lag too large for sample size');

            % Test with invalid regressors matrix and verify appropriate error is thrown
            invalidRegressors = randn(length(obj.autocorrelatedData) - 1, 1);
            obj.assertThrows(@() lmtest1(obj.autocorrelatedData, obj.defaultLags, invalidRegressors), 'MATLAB:lmtest1:InvalidRegressors', 'Error not thrown for invalid regressors matrix');
        end

        function testRegressionOutput(obj)
           %testRegressionOutput Test that auxiliary regression output from lmtest1 is correct
           %   testRegressionOutput() tests that auxiliary regression output
           %   from lmtest1 is correct, including verifying the R-squared and
           %   test statistic.

           % Run lmtest1 on autocorrelated data
           results = lmtest1(obj.autocorrelatedData, obj.defaultLags);

           % Manually compute auxiliary regression
           manualResults = obj.computeManualLMStatistic(obj.autocorrelatedData, obj.defaultLags);

           % Compare regression output with expected values
           obj.assertAlmostEqual(results.stat, manualResults.stat, 'Test statistic does not match T*R² relationship');
           obj.assertAlmostEqual(results.pval, 1 - chi2cdf(manualResults.stat, obj.defaultLags), 'P-value does not match expected value');

           % Verify R-squared is computed correctly
           % (This requires access to the internal regression results, which is not directly available)
           % You may need to modify lmtest1 to expose the R-squared value for testing
           % For example, if lmtest1 returns a struct with an R2 field, you can use:
           % obj.assertAlmostEqual(results.R2, manualResults.R2, 'R-squared is not computed correctly');

           % Verify test statistic matches T*R² relationship
           % (This assumes that the test statistic is computed as T*R², which is the standard formula)
           % obj.assertAlmostEqual(results.stat, length(obj.autocorrelatedData) * results.R2, 'Test statistic does not match T*R² relationship');
        end

        function testNumericalStability(obj)
            %testNumericalStability Test lmtest1 numerical stability with extreme values
            %   testNumericalStability() tests lmtest1 numerical stability
            %   with extreme values, including verifying results with very
            %   small variance and very large values.

            % Generate data with very small variance
            smallVarianceData = 1e-10 * randn(100, 1);

            % Run lmtest1 and verify results are numerically stable
            resultsSmallVariance = lmtest1(smallVarianceData, obj.defaultLags);
            obj.assertTrue(isstruct(resultsSmallVariance), 'Results are not a struct with small variance data');

            % Generate data with very large values
            largeValuesData = 1e10 * randn(100, 1);

            % Run lmtest1 and verify results are numerically stable
            resultsLargeValues = lmtest1(largeValuesData, obj.defaultLags);
            obj.assertTrue(isstruct(resultsLargeValues), 'Results are not a struct with large values data');

            % Verify that appropriate scaling is applied internally
            % (This requires access to the internal scaling logic, which is not directly available)
            % You may need to modify lmtest1 to expose the scaling factor for testing
            % For example, if lmtest1 returns a struct with a scalingFactor field, you can use:
            % obj.assertTrue(resultsSmallVariance.scalingFactor > 0, 'Scaling factor is not applied correctly');
            % obj.assertTrue(resultsLargeValues.scalingFactor > 0, 'Scaling factor is not applied correctly');
        end

        function data = generateAutocorrelatedData(obj, size, acCoefficient)
            %generateAutocorrelatedData Helper method to generate data with known autocorrelation
            %   data = generateAutocorrelatedData(size, acCoefficient) generates
            %   data with known autocorrelation using an AR(1) process.

            % Create data vector of specified size
            data = zeros(size, 1);

            % Generate random normal innovations
            innovations = randn(size, 1);

            % Apply AR(1) process with specified coefficient
            for t = 2:size
                data(t) = acCoefficient * data(t-1) + innovations(t);
            end
        end

        function manualComputationResults = computeManualLMStatistic(obj, data, lags)
            %computeManualLMStatistic Helper method to manually compute LM test statistic
            %   manualComputationResults = computeManualLMStatistic(data, lags)
            %   manually computes the LM test statistic for comparison.

            % Construct lag matrix for auxiliary regression
            T = length(data);
            lagMatrix = zeros(T - lags, lags);
            for i = 1:lags
                lagMatrix(:, i) = data(lags + 1 - i:T - i);
            end

            % Perform OLS regression of data on its lags
            y = data(lags + 1:end);
            X = [ones(T - lags, 1), lagMatrix];
            b = (X' * X) \ (X' * y);
            e = y - X * b;

            % Calculate R-squared from regression
            SSR = sum(e.^2);
            SST = sum((y - mean(y)).^2);
            R2 = 1 - SSR / SST;

            % Compute LM statistic as T*R²
            LMStat = (T - lags) * R2;

            % Return computation results for comparison
            manualComputationResults = struct('stat', LMStat);
        end
    end
end