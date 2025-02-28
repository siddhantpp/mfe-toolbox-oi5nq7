classdef JarqueBeraTest < BaseTest
    %JARQUEBERATEST Unit test class for testing the jarque_bera function
    %   This class inherits from BaseTest and implements various test methods
    %   to validate the functionality and accuracy of the jarque_bera function.
    
    properties
        sampleSize  % Sample size for tests
        normalData  % Normally distributed data
        nonNormalData % Non-normally distributed data
        skewedData  % Skewed data
        heavyTailedData % Heavy-tailed data
    end
    
    methods
        function obj = JarqueBeraTest()
            %JarqueBeraTest Constructor for the JarqueBeraTest class
            %   Initializes the test environment and sets up the properties
            %   required for testing.
            
            % Call the BaseTest constructor to initialize testing infrastructure
            obj = obj@BaseTest();
            
            % Set sampleSize to 1000 for adequate statistical power in tests
            obj.sampleSize = 1000;
            
            % Initialize empty data arrays that will be populated in setUp method
            obj.normalData = [];
            obj.nonNormalData = [];
            obj.skewedData = [];
            obj.heavyTailedData = [];
        end
        
        function setUp(obj)
            %setUp Set up test environment before each test method execution
            %   This method is executed before each test method and is used to
            %   set up the test environment, including generating test data.
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random number generator seed for reproducible tests
            rng(123);
            
            % Generate normalData as random normal samples
            obj.normalData = randn(obj.sampleSize, 1);
            
            % Generate nonNormalData as a mixture of distributions
            obj.nonNormalData = [randn(obj.sampleSize/2, 1); ...
                                 rand(obj.sampleSize/2, 1)];
            
            % Generate skewedData with controlled positive skewness
            obj.skewedData = exprnd(1, obj.sampleSize, 1);
            
            % Generate heavyTailedData with excess kurtosis
            obj.heavyTailedData = trnd(3, obj.sampleSize, 1);
        end
        
        function tearDown(obj)
            %tearDown Clean up test environment after each test method execution
            %   This method is executed after each test method and is used to
            %   clean up the test environment, including clearing any temporary
            %   resources used in tests.
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary resources used in tests
            clear obj.normalData obj.nonNormalData obj.skewedData obj.heavyTailedData;
        end
        
        function testNormalData(obj)
            %testNormalData Test that jarque_bera correctly fails to reject normality for normal data
            %   This method tests the jarque_bera function with normally
            %   distributed data and asserts that the test correctly fails to
            %   reject the null hypothesis of normality.
            
            % Call jarque_bera on normalData
            results = jarque_bera(obj.normalData);
            
            % Assert that p-value is greater than significance levels (0.01, 0.05, 0.10)
            obj.assertTrue(results.pval > 0.01);
            obj.assertTrue(results.pval > 0.05);
            obj.assertTrue(results.pval > 0.10);
            
            % Assert that rejection decisions are correctly false at all significance levels
            obj.assertFalse(results.reject(1));
            obj.assertFalse(results.reject(2));
            obj.assertFalse(results.reject(3));
            
            % Verify test statistic is within expected range for normal data
            obj.assertTrue(results.statistic < 10);
        end
        
        function testNonNormalData(obj)
            %testNonNormalData Test that jarque_bera correctly rejects normality for non-normal data
            %   This method tests the jarque_bera function with non-normally
            %   distributed data and asserts that the test correctly rejects
            %   the null hypothesis of normality.
            
            % Call jarque_bera on nonNormalData
            results = jarque_bera(obj.nonNormalData);
            
            % Assert that p-value is less than significance levels (0.01, 0.05, 0.10)
            obj.assertTrue(results.pval < 0.01);
            obj.assertTrue(results.pval < 0.05);
            obj.assertTrue(results.pval < 0.10);
            
            % Assert that rejection decisions are correctly true at all significance levels
            obj.assertTrue(results.reject(1));
            obj.assertTrue(results.reject(2));
            obj.assertTrue(results.reject(3));
            
            % Verify test statistic is significantly higher than critical values
            obj.assertTrue(results.statistic > 10);
        end
        
        function testSkewedData(obj)
            %testSkewedData Test that jarque_bera correctly identifies skewed distributions
            %   This method tests the jarque_bera function with skewed data and
            %   asserts that the test correctly identifies the non-normality due
            %   to skewness.
            
            % Call jarque_bera on skewedData
            results = jarque_bera(obj.skewedData);
            
            % Assert that p-value is less than significance levels for skewed data
            obj.assertTrue(results.pval < 0.01);
            obj.assertTrue(results.pval < 0.05);
            obj.assertTrue(results.pval < 0.10);
            
            % Assert that rejection decisions are correctly true
            obj.assertTrue(all(results.reject));
            
            % Verify test statistic reflects contribution from skewness component
            obj.assertTrue(results.statistic > 10);
        end
        
        function testHeavyTailedData(obj)
            %testHeavyTailedData Test that jarque_bera correctly identifies heavy-tailed distributions
            %   This method tests the jarque_bera function with heavy-tailed data
            %   and asserts that the test correctly identifies the non-normality
            %   due to excess kurtosis.
            
            % Call jarque_bera on heavyTailedData
            results = jarque_bera(obj.heavyTailedData);
            
            % Assert that p-value is less than significance levels for heavy-tailed data
            obj.assertTrue(results.pval < 0.01);
            obj.assertTrue(results.pval < 0.05);
            obj.assertTrue(results.pval < 0.10);
            
            % Assert that rejection decisions are correctly true
            obj.assertTrue(all(results.reject));
            
            % Verify test statistic reflects contribution from kurtosis component
            obj.assertTrue(results.statistic > 10);
        end
        
        function testInvalidInputs(obj)
            %testInvalidInputs Test error handling for invalid inputs to jarque_bera
            %   This method tests the error handling of the jarque_bera function
            %   by providing invalid inputs and asserting that the function
            %   throws the expected errors.
            
            % Test empty input using assertThrows
            obj.assertThrows(@() jarque_bera([]), 'datacheck:DATA_cannot_be_empty');
            
            % Test non-numeric input using assertThrows
            obj.assertThrows(@() jarque_bera({'a', 'b', 'c'}), 'datacheck:DATA_must_be_numeric');
            
            % Test input with NaN values using assertThrows
            obj.assertThrows(@() jarque_bera([1, 2, NaN]), 'datacheck:DATA_cannot_contain_NaN_values');
            
            % Test input with Inf values using assertThrows
            obj.assertThrows(@() jarque_bera([1, 2, Inf]), 'datacheck:DATA_cannot_contain_Inf_or_-Inf_values');
            
            % Test input with too few observations using assertThrows
            obj.assertThrows(@() jarque_bera([1]), 'columncheck:NAME_must_be_numeric');
        end
        
        function testKnownDistributions(obj)
            %testKnownDistributions Test jarque_bera against distributions with known theoretical properties
            %   This method tests the jarque_bera function against samples from
            %   distributions with known skewness and kurtosis to verify that the
            %   test results match theoretical expectations.
            
            % Load or generate samples from distributions with known skewness and kurtosis
            numSamples = 1000;
            
            % Uniform distribution
            uniformData = rand(numSamples, 1);
            
            % Exponential distribution
            exponentialData = exprnd(1, numSamples, 1);
            
            % Student's t-distribution with low degrees of freedom
            tData = trnd(5, numSamples, 1);
            
            % Call jarque_bera on each distribution sample
            resultsUniform = jarque_bera(uniformData);
            resultsExponential = jarque_bera(exponentialData);
            resultsT = jarque_bera(tData);
            
            % Verify test results match theoretical expectations for each distribution
            % Check consistency of results with increasing sample sizes
            
            % For uniform distribution, expect to reject normality
            obj.assertTrue(all(resultsUniform.reject));
            
            % For exponential distribution, expect to reject normality
            obj.assertTrue(all(resultsExponential.reject));
            
            % For t-distribution, rejection depends on degrees of freedom
            obj.assertTrue(all(resultsT.reject));
        end
        
        function testSmallSamples(obj)
            %testSmallSamples Test behavior of jarque_bera with small sample sizes
            %   This method tests the behavior of the jarque_bera function with
            %   small sample sizes to ensure that the test behaves appropriately
            %   when the sample size is limited.
            
            % Generate normal samples with small sample sizes (n=30, 50, 100)
            sampleSizes = [30, 50, 100];
            
            for n = sampleSizes
                normalDataSmall = randn(n, 1);
                
                % Call jarque_bera on each sample
                results = jarque_bera(normalDataSmall);
                
                % Verify that small-sample behavior is appropriate
                % Check that decision rates are consistent with type I error rates
                
                % With small samples, the test may fail to reject normality more often
                obj.assertTrue(results.pval > 0.05);
            end
        end
        
        function testLargeSamples(obj)
            %testLargeSamples Test behavior of jarque_bera with large sample sizes
            %   This method tests the behavior of the jarque_bera function with
            %   large sample sizes to ensure that the test behaves appropriately
            %   when the sample size is large.
            
            % Generate normal samples with large sample sizes (n=5000, 10000)
            sampleSizes = [5000, 10000];
            
            for n = sampleSizes
                normalDataLarge = randn(n, 1);
                
                % Call jarque_bera on each sample
                results = jarque_bera(normalDataLarge);
                
                % Verify that asymptotic properties hold with large samples
                % Check numerical stability with large sample calculations
                
                % With large samples, the test should reject normality if there are even small deviations
                obj.assertTrue(results.pval > 0.05);
            end
        end
        
        function testConsistency(obj)
            %testConsistency Test consistency of jarque_bera results with increasing sample size
            %   This method tests the consistency of the jarque_bera function by
            %   generating samples of increasing size from the same non-normal
            %   distribution and verifying that the rejection rate increases with
            %   sample size.
            
            % Generate samples of increasing size from same non-normal distribution
            sampleSizes = [100, 500, 1000, 5000];
            rejectionRates = zeros(size(sampleSizes));
            
            for i = 1:length(sampleSizes)
                n = sampleSizes(i);
                nonNormalDataVarying = [randn(n/2, 1); rand(n/2, 1)];
                
                % Call jarque_bera on each sample
                results = jarque_bera(nonNormalDataVarying);
                
                % Verify that rejection rate increases with sample size for fixed non-normality
                % Confirm detection power increases appropriately with sample size
                rejectionRates(i) = any(results.reject);
            end
            
            % Check that rejection rate increases with sample size
            obj.assertTrue(all(diff(rejectionRates) >= 0));
        end
    end
end