classdef StdtpdfTest < BaseTest
    % STDTPDFTEST Test class for the stdtpdf function, providing comprehensive
    % test coverage for standardized Student's t-distribution PDF calculations
    %
    % This test class validates the stdtpdf function, which computes the PDF of
    % the standardized Student's t-distribution with mean 0 and variance 1.
    % Tests include basic PDF validation, vectorized inputs, normal
    % approximation for large degrees of freedom, tail behavior,
    % parameter validation, numerical precision in extreme cases, and
    % verification of mathematical properties of a valid PDF.
    %
    % See also BASETEST, STDTPDF, NUMERICALCOMPARATOR
    
    properties
        comparator          % NumericalComparator for floating-point comparisons
        defaultTolerance    % Default tolerance for numerical comparisons
        testData            % Structure for storing test data
        testValues          % Array of test x values
        nuValues            % Array of test degrees of freedom values
        expectedResults     % Array of expected PDF values
    end
    
    methods
        function obj = StdtpdfTest()
            % Initialize a new StdtpdfTest instance with numerical comparator
            %
            % Call the superclass (BaseTest) constructor with 'StdtpdfTest' name,
            % initialize testData structure, create a NumericalComparator instance,
            % and set defaultTolerance for high-precision numeric comparisons.
            
            % Call superclass constructor
            obj = obj@BaseTest('StdtpdfTest');
            
            % Initialize test data structure
            obj.testData = struct();
            
            % Create NumericalComparator for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for high-precision comparisons
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            %
            % Sets up common test values, reference data, and expected results
            % that will be used across multiple test methods.
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Initialize array of test x values
            obj.testValues = [-3, -2, -1, 0, 1, 2, 3];
            
            % Initialize array of test nu values
            obj.nuValues = [3, 4, 5, 6, 8, 10, 20, 30];
            
            % Try to load reference data from known_distributions.mat
            try
                testDataFile = obj.findTestDataFile('known_distributions.mat');
                data = load(testDataFile);
                if isfield(data, 'stdt_pdf_values')
                    obj.testData.referenceValues = data.stdt_pdf_values;
                end
            catch
                % If file not found, we'll compute expected values for reference cases
                warning('Reference data file not found. Using computed values instead.');
            end
            
            % Prepare expected PDF values for reference cases
            % For nu = 5 at x = [-3, -2, -1, 0, 1, 2, 3]
            nu = 5;
            scale = sqrt((nu-2)/nu);
            const = gamma((nu+1)/2) / (sqrt(pi*nu) * gamma(nu/2));
            
            x = obj.testValues;
            z = x / scale;
            obj.expectedResults = const * (1 + (z.^2)/nu).^(-((nu+1)/2)) / scale;
            
            % Configure numerical comparator with appropriate tolerance
            compOpts = struct('absoluteTolerance', obj.defaultTolerance);
            obj.comparator = NumericalComparator(compOpts);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary test data to free memory
            obj.testData = struct();
        end
        
        function testBasicPdf(obj)
            % Tests stdtpdf with basic parameter values
            
            % Test stdtpdf at x=0 with nu=5
            result = stdtpdf(0, 5);
            
            % The value at x=0 for t-distribution with df=5 should be approximately 0.3796
            expected = 0.3796;
            
            % Verify result with specified tolerance
            obj.assertAlmostEqual(expected, result, ...
                'PDF value at x=0, nu=5 should be approximately 0.3796');
            
            % Test that the result is a double with correct dimensions
            obj.assertEqual(1, size(result, 1), 'Result should be a scalar');
            obj.assertEqual(1, size(result, 2), 'Result should be a scalar');
            
            % Verify PDF value is positive
            obj.assertTrue(result > 0, 'PDF value should be positive');
        end
        
        function testVectorInput(obj)
            % Tests stdtpdf with vectorized inputs
            
            % Create vector of x values
            x = obj.testValues;
            
            % Test with vector input and nu=5
            nu = 5;
            results = stdtpdf(x, nu);
            
            % Verify output is a vector with same dimensions as input
            obj.assertEqual(size(x), size(results), 'Output should have same dimensions as input');
            
            % Compare results with expected t-distribution PDF values
            obj.assertAlmostEqual(obj.expectedResults, results, ...
                'Vector results should match expected values');
            
            % Test with multiple nu values to verify correct vectorization
            for i = 1:length(obj.nuValues)
                nu = obj.nuValues(i);
                results = stdtpdf(x, nu);
                
                % Calculate expected values for this nu
                expected = obj.calculateManualStdtPdf(x, nu);
                
                % Verify results match expected values
                obj.assertAlmostEqual(expected, results, ...
                    sprintf('Results for nu=%d should match expected values', nu));
            end
        end
        
        function testNormalApproximation(obj)
            % Tests that stdtpdf approaches normal PDF as degrees of freedom increase
            
            % Create array of test points
            x = linspace(-3, 3, 101);
            
            % Calculate standardized t-distribution with large degrees of freedom
            largeNu = 100;
            tPdf = stdtpdf(x, largeNu);
            
            % Calculate normal PDF values for comparison
            % Normal PDF: (1/sqrt(2π)) * exp(-x^2/2)
            normalPdf = exp(-x.^2/2) / sqrt(2*pi);
            
            % Verify that t-distribution approaches normal distribution
            % As nu increases, the difference should become smaller
            diff = abs(tPdf - normalPdf);
            maxDiff = max(diff);
            
            % For nu=100, the maximum difference should be small (< 0.01)
            obj.assertTrue(maxDiff < 0.01, ...
                'T-distribution with nu=100 should be close to normal distribution');
            
            % Test with even larger nu to verify convergence
            veryLargeNu = 1000;
            tPdfLarge = stdtpdf(x, veryLargeNu);
            diffLarge = abs(tPdfLarge - normalPdf);
            maxDiffLarge = max(diffLarge);
            
            % Verify that larger nu leads to smaller difference
            obj.assertTrue(maxDiffLarge < maxDiff, ...
                'Increasing nu should make t-distribution closer to normal');
            
            % Check convergence rate is appropriate (proportional to 1/nu)
            % The ratio of differences should be approximately the inverse ratio of nu values
            expectedRatio = largeNu / veryLargeNu;
            actualRatio = maxDiff / maxDiffLarge;
            
            % Allow some tolerance for the convergence rate
            obj.assertTrue(abs(actualRatio - expectedRatio) < 0.5, ...
                'Convergence rate to normal distribution should be approximately proportional to 1/nu');
        end
        
        function testTailBehavior(obj)
            % Tests stdtpdf tail behavior with different degrees of freedom
            
            % Test with small degrees of freedom (nu = 3) at extreme x values
            smallNu = 3;
            extremeX = [-10, -5, 5, 10];
            smallNuTail = stdtpdf(extremeX, smallNu);
            
            % Test with moderate degrees of freedom (nu = 10) at extreme x values
            moderateNu = 10;
            moderateNuTail = stdtpdf(extremeX, moderateNu);
            
            % Test with large degrees of freedom (nu = 30) at extreme x values
            largeNu = 30;
            largeNuTail = stdtpdf(extremeX, largeNu);
            
            % Verify that small nu has heavier tails
            % For the same |x|, smaller nu should result in larger PDF values in the tails
            obj.assertTrue(all(smallNuTail > moderateNuTail), ...
                'Small nu should have heavier tails than moderate nu');
            obj.assertTrue(all(moderateNuTail > largeNuTail), ...
                'Moderate nu should have heavier tails than large nu');
            
            % Verify that tail probabilities decrease as |x| increases
            for i = 1:length(obj.nuValues)
                nu = obj.nuValues(i);
                xExtended = [obj.testValues, 5, 10, 15, 20];
                pdfValues = stdtpdf(xExtended, nu);
                
                % Check that PDF values at extreme points are small but positive
                obj.assertTrue(all(pdfValues > 0), ...
                    sprintf('All PDF values for nu=%d should be positive', nu));
                
                % Check monotonic decrease in tails
                posX = xExtended(xExtended > 0);
                posPdf = pdfValues(xExtended > 0);
                obj.assertTrue(issorted(posPdf, 'descend'), ...
                    sprintf('PDF should monotonically decrease in positive tail for nu=%d', nu));
                
                negX = xExtended(xExtended < 0);
                negPdf = pdfValues(xExtended < 0);
                obj.assertTrue(issorted(negPdf, 'ascend'), ...
                    sprintf('PDF should monotonically increase in negative tail for nu=%d', nu));
            end
        end
        
        function testParameterValidation(obj)
            % Tests stdtpdf error handling for invalid parameters
            
            % Test with invalid degrees of freedom (nu ≤ 2)
            invalidNu = 2;
            obj.assertThrows(@() stdtpdf(0, invalidNu), ...
                'parametercheck:lowerBound', ...
                'Should throw error for nu ≤ 2');
            
            invalidNu = 1.5;
            obj.assertThrows(@() stdtpdf(0, invalidNu), ...
                'parametercheck:lowerBound', ...
                'Should throw error for nu ≤ 2');
            
            % Test with NaN values in parameters
            obj.assertThrows(@() stdtpdf(NaN, 5), ...
                'datacheck:NaN', ...
                'Should throw error for NaN in x');
            
            obj.assertThrows(@() stdtpdf(0, NaN), ...
                'parametercheck:NaN', ...
                'Should throw error for NaN in nu');
            
            % Test with Inf values in parameters
            obj.assertThrows(@() stdtpdf(Inf, 5), ...
                'datacheck:Inf', ...
                'Should throw error for Inf in x');
            
            obj.assertThrows(@() stdtpdf(0, Inf), ...
                'parametercheck:Inf', ...
                'Should throw error for Inf in nu');
            
            % Test with mismatched dimensions in inputs
            x = [1, 2, 3];
            nu = [3, 4];
            obj.assertThrows(@() stdtpdf(x, nu), ...
                'columncheck:vector', ...
                'Should throw error for mismatched dimensions');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of stdtpdf for extreme values
            
            % Test with degrees of freedom very close to 2 (boundary case)
            nuNearBoundary = 2.00001;
            x = 0;
            result = stdtpdf(x, nuNearBoundary);
            
            % The PDF should be finite and positive
            obj.assertTrue(isfinite(result) && result > 0, ...
                'PDF value should be finite and positive for nu very close to 2');
            
            % Test with very large degrees of freedom (approaching infinity)
            veryLargeNu = 1e6;
            result = stdtpdf(x, veryLargeNu);
            
            % Should approximate standard normal PDF at x=0, which is 1/sqrt(2*pi)
            expected = 1/sqrt(2*pi);
            obj.assertAlmostEqual(expected, result, 1e-4, ...
                'With very large nu, PDF at x=0 should approach normal PDF value');
            
            % Test with extreme x values (far from mean)
            extremeX = 100;
            result = stdtpdf(extremeX, 5);
            
            % The PDF should be very small but finite and positive
            obj.assertTrue(result > 0 && result < 1e-10, ...
                'PDF at extreme x should be very small but positive');
            
            % Verify that manual calculation agrees with function output
            expected = obj.calculateManualStdtPdf(extremeX, 5);
            obj.assertAlmostEqual(expected, result, ...
                'Function output should match manual calculation for extreme values');
        end
        
        function testPdfProperties(obj)
            % Tests that the PDF satisfies mathematical properties of a valid PDF
            
            % Verify PDF is non-negative for all test points
            x = linspace(-10, 10, 101);
            
            for i = 1:length(obj.nuValues)
                nu = obj.nuValues(i);
                pdfValues = stdtpdf(x, nu);
                
                obj.assertTrue(all(pdfValues >= 0), ...
                    sprintf('PDF should be non-negative for all x with nu=%d', nu));
                
                % Verify PDF is symmetric around x=0 (stdtpdf(x) = stdtpdf(-x))
                negX = -x(x > 0);
                negPdf = stdtpdf(negX, nu);
                posPdf = stdtpdf(-negX, nu);
                
                obj.assertAlmostEqual(negPdf, posPdf, ...
                    sprintf('PDF should be symmetric around x=0 for nu=%d', nu));
                
                % Verify PDF mode is at x=0
                zeroResult = stdtpdf(0, nu);
                smallX = linspace(0.1, 1, 10);
                smallResults = stdtpdf(smallX, nu);
                
                obj.assertTrue(all(zeroResult > smallResults), ...
                    sprintf('PDF should have its maximum at x=0 for nu=%d', nu));
                
                % Verify approximate integration of PDF equals 1
                % Use simple numerical integration (trapezoidal rule)
                % over a sufficiently wide range
                wideX = linspace(-20, 20, 1001);
                widePdf = stdtpdf(wideX, nu);
                
                dx = wideX(2) - wideX(1);
                integralApprox = dx * (sum(widePdf) - 0.5*widePdf(1) - 0.5*widePdf(end));
                
                obj.assertAlmostEqual(1, integralApprox, 1e-4, ...
                    sprintf('Integral of PDF should be approximately 1 for nu=%d', nu));
            end
        end
        
        function testAgainstReferenceData(obj)
            % Tests stdtpdf against pre-computed reference values from known_distributions.mat
            
            % Skip test if reference data is not available
            if ~isfield(obj.testData, 'referenceValues')
                warning('Reference data not available. Skipping reference data comparison.');
                return;
            end
            
            % Extract reference data
            refValues = obj.testData.referenceValues;
            
            % Verify structure of reference data
            if ~isstruct(refValues) || ~isfield(refValues, 'x') || ...
                    ~isfield(refValues, 'nu') || ~isfield(refValues, 'pdf')
                warning('Reference data structure is invalid. Skipping reference data comparison.');
                return;
            end
            
            % Compute PDF values using stdtpdf function with same inputs
            computedValues = stdtpdf(refValues.x, refValues.nu);
            
            % Compare calculated values with reference values
            obj.assertAlmostEqual(refValues.pdf, computedValues, ...
                'Computed PDF values should match reference values');
        end
        
        function manualPdf = calculateManualStdtPdf(obj, x, nu)
            % Helper method to manually calculate standardized Student's t-distribution PDF
            %
            % INPUTS:
            %   x - Points to evaluate the PDF at
            %   nu - Degrees of freedom parameter
            %
            % OUTPUTS:
            %   manualPdf - Manually calculated PDF values
            
            % Validate inputs
            assert(nu > 2, 'Degrees of freedom must be greater than 2');
            
            % Scaling factor for standardized t-distribution
            scale = sqrt((nu-2)/nu);
            
            % Points at which to evaluate standard t-distribution
            z = x / scale;
            
            % Calculate PDF using gamma function formula
            const = gamma((nu+1)/2) / (sqrt(pi*nu) * gamma(nu/2));
            
            % Compute the PDF
            std_pdf = const * (1 + (z.^2)/nu).^(-((nu+1)/2));
            
            % Scale the PDF
            manualPdf = std_pdf / scale;
        end
    end
end