classdef ArchTestTest < BaseTest
    % Test class for validating the functionality of the ARCH test implementation for detecting volatility clustering
    
    properties
        dataGenerator
        garchData
        normalData
        defaultLags
        significanceLevel
        tolerance
    end
    
    methods
        function obj = ArchTestTest()
            % Initialize the ArchTestTest class with default configuration
            obj = obj@BaseTest();
            obj.tolerance = 1e-10;  % Set numerical tolerance for floating-point comparisons
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            % Initialize dataGenerator with fixed seed for reproducibility
            rng(42);
            
            % Generate GARCH time series data with known volatility clustering
            garchParams = struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
            volData = TestDataGenerator('generateVolatilitySeries', 1000, 'GARCH', garchParams);
            obj.garchData = volData.returns;
            
            % Generate normal random data without volatility clustering
            obj.normalData = randn(1000, 1);
            
            % Set default test parameters
            obj.defaultLags = 10;
            obj.significanceLevel = 0.05;
        end
        
        function tearDown(obj)
            % Cleans up resources after each test method execution
            obj.garchData = [];
            obj.normalData = [];
        end
        
        function testGARCHDataDetection(obj)
            % Tests that the ARCH test correctly identifies volatility clustering in GARCH data
            results = arch_test(obj.garchData, obj.defaultLags);
            
            % Verify p-values are below significance level (strong rejection of no-ARCH null hypothesis)
            obj.assertTrue(results.pval < obj.significanceLevel, ...
                'ARCH test should detect volatility clustering in GARCH data');
                
            % Verify test statistics are positive and statistically significant
            obj.assertTrue(results.statistic > 0, 'Test statistic should be positive');
            obj.assertTrue(results.H0rejected.five, 'Should reject H0 at 5% level');
            
            % Confirm detection of ARCH effects across various lag specifications
            results5 = arch_test(obj.garchData, 5);
            results20 = arch_test(obj.garchData, 20);
            
            obj.assertTrue(results5.pval < obj.significanceLevel, 'Should detect ARCH with 5 lags');
            obj.assertTrue(results20.pval < obj.significanceLevel, 'Should detect ARCH with 20 lags');
        end
        
        function testNormalDataNoFalsePositives(obj)
            % Tests that the ARCH test correctly fails to reject null hypothesis in normal data without volatility clustering
            results = arch_test(obj.normalData, obj.defaultLags);
            
            % Verify p-values are generally above significance level
            obj.assertTrue(results.pval > obj.significanceLevel, ...
                'ARCH test should not detect volatility clustering in normal data');
            
            % Verify test fails to reject null hypothesis of no ARCH effects
            obj.assertFalse(results.H0rejected.five, 'Should not reject H0 at 5% level');
        end
        
        function testLagSpecifications(obj)
            % Tests the ARCH test with different lag specifications
            lags = [1, 5, 10, 20];
            
            for i = 1:length(lags)
                results = arch_test(obj.garchData, lags(i));
                
                % Verify results have correct dimensions for each lag specification
                obj.assertEqual(results.lags, lags(i), 'Lag specification incorrect');
                
                % Verify test power increases with appropriate lags for known GARCH process
                if i > 1
                    prevResults = arch_test(obj.garchData, lags(i-1));
                    obj.assertTrue(results.critical.five > prevResults.critical.five, ...
                        'Critical values should increase with more lags');
                end
            end
        end
        
        function testOutputFormat(obj)
            % Tests the output format and structure of the ARCH test results
            results = arch_test(obj.garchData, obj.defaultLags);
            
            % Confirm presence of test statistic, p-value, critical values, and rejection fields
            obj.assertTrue(isfield(results, 'statistic'), 'Missing statistic field');
            obj.assertTrue(isfield(results, 'pval'), 'Missing pval field');
            obj.assertTrue(isfield(results, 'critical'), 'Missing critical field');
            obj.assertTrue(isfield(results, 'lags'), 'Missing lags field');
            obj.assertTrue(isfield(results, 'H0rejected'), 'Missing H0rejected field');
            
            % Verify field names match expectations
            obj.assertTrue(isfield(results.critical, 'ten'), 'Missing 10% critical value');
            obj.assertTrue(isfield(results.critical, 'five'), 'Missing 5% critical value');
            obj.assertTrue(isfield(results.critical, 'one'), 'Missing 1% critical value');
            
            obj.assertTrue(isfield(results.H0rejected, 'ten'), 'Missing 10% rejection decision');
            obj.assertTrue(isfield(results.H0rejected, 'five'), 'Missing 5% rejection decision');
            obj.assertTrue(isfield(results.H0rejected, 'one'), 'Missing 1% rejection decision');
            
            % Verify dimensions of output fields are correct
            obj.assertTrue(results.statistic >= 0, 'Statistic should be non-negative');
            obj.assertTrue(results.pval >= 0 && results.pval <= 1, 'P-value should be between 0 and 1');
            obj.assertTrue(results.critical.five > results.critical.ten, '5% critical value should exceed 10%');
            obj.assertTrue(results.critical.one > results.critical.five, '1% critical value should exceed 5%');
        end
        
        function testInvalidInputs(obj)
            % Tests error handling with invalid inputs to the ARCH test function
            
            % Test with empty data, verifying appropriate error
            try
                arch_test([], 5);
                obj.assertTrue(false, 'Empty data should throw an error');
            catch
                % Success - error was thrown
            end
            
            % Test with non-numeric data, verifying appropriate error
            try
                arch_test('not numeric', 5);
                obj.assertTrue(false, 'Non-numeric data should throw an error');
            catch
                % Success - error was thrown
            end
            
            % Test with NaN/Inf values, verifying appropriate error
            try
                invalidData = obj.garchData;
                invalidData(10) = NaN;
                arch_test(invalidData, 5);
                obj.assertTrue(false, 'Data with NaN should throw an error');
            catch
                % Success - error was thrown
            end
            
            try
                invalidData = obj.garchData;
                invalidData(10) = Inf;
                arch_test(invalidData, 5);
                obj.assertTrue(false, 'Data with Inf should throw an error');
            catch
                % Success - error was thrown
            end
            
            % Test with negative lags, verifying appropriate error
            try
                arch_test(obj.garchData, -1);
                obj.assertTrue(false, 'Negative lags should throw an error');
            catch
                % Success - error was thrown
            end
            
            % Test with lag values exceeding data length, verifying appropriate error
            try
                testData = obj.garchData;
                maxLags = length(testData) - 9; % Ensure we're over the limit
                arch_test(testData, maxLags+1);
                obj.assertTrue(false, 'Excessive lags should throw an error');
            catch
                % Success - error was thrown
            end
        end
        
        function testManualARCHCalculation(obj)
            % Validates ARCH test implementation against manually calculated values
            
            % Generate fixed test data for verification
            rng(123); % Set seed for reproducibility
            testData = randn(200, 1);
            lags = 5;
            
            % Get results from the implementation being tested
            results = arch_test(testData, lags);
            
            % Calculate test statistic using manual implementation
            manualResults = obj.calculateManualARCHStatistic(testData, lags);
            
            % Compare manual calculation with arch_test results
            obj.assertEqualsWithTolerance(results.statistic, manualResults.statistic, obj.tolerance, ...
                'LM statistic calculation mismatch');
            obj.assertEqualsWithTolerance(results.pval, manualResults.pval, obj.tolerance, ...
                'P-value calculation mismatch');
        end
        
        function results = calculateManualARCHStatistic(obj, data, lags)
            % Helper method that manually calculates ARCH LM test statistic for verification
            
            % Square the data to get squared residuals
            squaredData = data.^2;
            
            % Compute sample mean of squared residuals
            T = length(squaredData);
            effective_T = T - lags;
            
            % Set up lagged matrices for auxiliary regression
            y = squaredData(lags+1:T);
            X = zeros(effective_T, lags+1);
            X(:,1) = 1; % Constant term
            
            % Perform regression of squared residuals on lagged squared residuals
            for i = 1:lags
                X(:,i+1) = squaredData(lags+1-i:T-i);
            end
            
            % Calculate R² from regression
            beta = X \ y;
            fitted = X * beta;
            residuals = y - fitted;
            SSR = residuals'*residuals;
            SST = (y - mean(y))'*(y - mean(y));
            Rsquared = 1 - SSR/SST;
            
            % Compute LM statistic as T*R²
            stat = effective_T * Rsquared;
            
            % Calculate p-value using chi-square distribution
            pval = 1 - chi2cdf(stat, lags);
            
            % Return structure with test statistic and p-value
            results.statistic = stat;
            results.pval = pval;
        end
    end
end