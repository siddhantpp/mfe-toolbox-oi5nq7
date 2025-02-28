classdef NumericalComparator
    % NUMERICALCOMPARATOR Provides precise numerical comparison methods for financial econometrics
    %
    % A specialized utility class that provides precise numerical comparison methods
    % for floating-point values in the MFE Toolbox testing framework. It handles the
    % complexities of floating-point arithmetic by implementing tolerance-based
    % comparison algorithms for scalars and matrices, essential for financial 
    % econometric applications where numerical stability is critical.
    %
    % Financial econometric calculations often involve iterative algorithms that
    % can accumulate floating-point errors. This comparator accounts for these
    % numerical instabilities when validating test results across different
    % platforms and computational environments.
    %
    % Examples:
    %   % Create comparator with default settings
    %   nc = NumericalComparator();
    %
    %   % Compare scalar values
    %   result = nc.compareScalars(0.1 + 0.2, 0.3);
    %   disp(result.isEqual); % Should be true despite floating-point imprecision
    %
    %   % Compare matrices
    %   A = [1.1, 2.2; 3.3, 4.4];
    %   B = [1.1000001, 2.2; 3.3, 4.4000002];
    %   result = nc.compareMatrices(A, B);
    %   disp(result.isEqual); % Should be true with default tolerances
    %
    % See also matrixdiagnostics, parametercheck
    
    properties
        % Default absolute tolerance for numerical comparisons
        defaultAbsoluteTolerance
        
        % Default relative tolerance for numerical comparisons
        defaultRelativeTolerance
        
        % Machine epsilon (floating-point precision)
        machineEpsilon
        
        % Flag to enable/disable adaptive tolerance calculation
        useAdaptiveTolerance
    end
    
    methods
        function obj = NumericalComparator(options)
            % Constructor for NumericalComparator
            %
            % INPUTS:
            %   options - Structure with configuration options:
            %       .absoluteTolerance - Custom absolute tolerance [default: 1e-10]
            %       .relativeTolerance - Custom relative tolerance [default: 1e-8]
            %       .useAdaptiveTolerance - Enable adaptive tolerance [default: true]
            %
            % OUTPUTS:
            %   obj - Initialized NumericalComparator object
            %
            % EXAMPLES:
            %   % Default initialization
            %   nc = NumericalComparator();
            %
            %   % Custom tolerances
            %   options = struct('absoluteTolerance', 1e-12, 'relativeTolerance', 1e-10);
            %   nc = NumericalComparator(options);
            
            % Set default machine epsilon (floating-point precision)
            obj.machineEpsilon = eps;
            
            % Set default tolerances
            if nargin < 1
                options = struct();
            end
            
            % Initialize default values
            obj.defaultAbsoluteTolerance = 1e-10;
            obj.defaultRelativeTolerance = 1e-8;
            obj.useAdaptiveTolerance = true;
            
            % Override with custom values if provided
            if isfield(options, 'absoluteTolerance')
                obj.defaultAbsoluteTolerance = options.absoluteTolerance;
            end
            
            if isfield(options, 'relativeTolerance')
                obj.defaultRelativeTolerance = options.relativeTolerance;
            end
            
            if isfield(options, 'useAdaptiveTolerance')
                obj.useAdaptiveTolerance = options.useAdaptiveTolerance;
            end
            
            % Validate tolerance values
            if obj.defaultAbsoluteTolerance <= 0
                error('NUMERICALCOMPARATOR:InvalidTolerance', ...
                      'Absolute tolerance must be positive');
            end
            
            if obj.defaultRelativeTolerance <= 0
                error('NUMERICALCOMPARATOR:InvalidTolerance', ...
                      'Relative tolerance must be positive');
            end
            
            if ~islogical(obj.useAdaptiveTolerance) || ~isscalar(obj.useAdaptiveTolerance)
                error('NUMERICALCOMPARATOR:InvalidOption', ...
                      'useAdaptiveTolerance must be a logical scalar');
            end
        end
        
        function result = compareScalars(obj, expected, actual, tolerance)
            % COMPARESCALARS Compares two scalar values with appropriate tolerance
            %
            % INPUTS:
            %   expected - Expected scalar value
            %   actual - Actual scalar value to compare
            %   tolerance - Optional custom tolerance value
            %
            % OUTPUTS:
            %   result - Struct with fields:
            %       .isEqual - Boolean indicating equality within tolerance
            %       .absoluteDifference - Absolute difference between values
            %       .relativeDifference - Relative difference between values
            %       .toleranceUsed - Tolerance value used for comparison
            %       .expected - Expected value
            %       .actual - Actual value
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   result = nc.compareScalars(0.1 + 0.2, 0.3);
            %   disp(['Equal: ', num2str(result.isEqual)]);
            %   disp(['Absolute difference: ', num2str(result.absoluteDifference)]);
            
            % Validate inputs
            parameterOptions = struct('isscalar', true);
            parametercheck(expected, 'expected', parameterOptions);
            parametercheck(actual, 'actual', parameterOptions);
            
            % Initialize result structure
            result = struct(...
                'isEqual', false, ...
                'absoluteDifference', NaN, ...
                'relativeDifference', NaN, ...
                'toleranceUsed', NaN, ...
                'expected', expected, ...
                'actual', actual ...
            );
            
            % Handle special cases: NaN and Inf
            if isnan(expected) && isnan(actual)
                result.isEqual = true;
                result.absoluteDifference = 0;
                result.relativeDifference = 0;
                result.toleranceUsed = 0;
                return;
            end
            
            if isinf(expected) && isinf(actual) && sign(expected) == sign(actual)
                result.isEqual = true;
                result.absoluteDifference = 0;
                result.relativeDifference = 0;
                result.toleranceUsed = 0;
                return;
            end
            
            % Calculate absolute difference for regular values
            result.absoluteDifference = abs(expected - actual);
            
            % Calculate relative difference
            result.relativeDifference = obj.getRelativeDifference(expected, actual);
            
            % Determine tolerance to use
            if nargin < 4 || isempty(tolerance)
                result.toleranceUsed = obj.calculateTolerance(expected, actual);
            else
                % Validate custom tolerance
                toleranceOpts = struct('isscalar', true, 'isPositive', true);
                parametercheck(tolerance, 'tolerance', toleranceOpts);
                result.toleranceUsed = tolerance;
            end
            
            % Check if values are equal within tolerance
            result.isEqual = result.absoluteDifference <= result.toleranceUsed;
        end
        
        function result = compareMatrices(obj, expected, actual, tolerance)
            % COMPAREMATRICES Compares two matrices element-wise with tolerance
            %
            % INPUTS:
            %   expected - Expected matrix
            %   actual - Actual matrix to compare
            %   tolerance - Optional custom tolerance value
            %
            % OUTPUTS:
            %   result - Struct with fields:
            %       .isEqual - Boolean indicating all elements equal within tolerance
            %       .mismatchCount - Number of mismatched elements
            %       .mismatchIndices - Indices of mismatched elements [row, col]
            %       .maxAbsoluteDifference - Maximum absolute difference
            %       .maxRelativeDifference - Maximum relative difference
            %       .toleranceUsed - Tolerance value used for comparison
            %       .expected - Expected matrix
            %       .actual - Actual matrix
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   A = [1.1, 2.2; 3.3, 4.4];
            %   B = [1.1000001, 2.2; 3.3, 4.4000002];
            %   result = nc.compareMatrices(A, B);
            %   disp(['Equal: ', num2str(result.isEqual)]);
            %   disp(['Mismatches: ', num2str(result.mismatchCount)]);
            
            % Validate inputs
            parametercheck(expected, 'expected');
            parametercheck(actual, 'actual');
            
            % Check dimension compatibility
            if ~isequal(size(expected), size(actual))
                error('NUMERICALCOMPARATOR:DimensionMismatch', ...
                      'Matrices must have the same dimensions');
            end
            
            % Initialize result structure
            result = struct(...
                'isEqual', false, ...
                'mismatchCount', 0, ...
                'mismatchIndices', [], ...
                'maxAbsoluteDifference', 0, ...
                'maxRelativeDifference', 0, ...
                'toleranceUsed', NaN, ...
                'expected', expected, ...
                'actual', actual ...
            );
            
            % Handle empty matrices
            if isempty(expected) && isempty(actual)
                result.isEqual = true;
                return;
            end
            
            % Determine tolerance to use
            if nargin < 4 || isempty(tolerance)
                % Use adaptive tolerance based on matrix values
                result.toleranceUsed = obj.calculateTolerance(expected, actual);
            else
                % Validate custom tolerance
                toleranceOpts = struct('isscalar', true, 'isPositive', true);
                parametercheck(tolerance, 'tolerance', toleranceOpts);
                result.toleranceUsed = tolerance;
            end
            
            % Handle NaN values - consider NaN == NaN
            nanInExpected = isnan(expected);
            nanInActual = isnan(actual);
            nanMismatch = xor(nanInExpected, nanInActual);
            
            % Handle Inf values - consider matching signs of Inf equal
            infInExpected = isinf(expected);
            infInActual = isinf(actual);
            sameSignInf = (infInExpected & infInActual & (sign(expected) == sign(actual)));
            infMismatch = (infInExpected | infInActual) & ~sameSignInf;
            
            % Calculate absolute difference for regular values
            regularValues = ~(nanInExpected | nanInActual | infInExpected | infInActual);
            absoluteDifference = zeros(size(expected));
            
            if any(regularValues(:))
                absoluteDifference(regularValues) = abs(expected(regularValues) - actual(regularValues));
            end
            
            % Find mismatches
            valueMismatch = absoluteDifference > result.toleranceUsed;
            mismatch = nanMismatch | infMismatch | valueMismatch;
            
            % Update result structure
            result.mismatchCount = sum(mismatch(:));
            
            if result.mismatchCount > 0
                [misRow, misCol] = find(mismatch);
                result.mismatchIndices = [misRow, misCol];
            end
            
            % Calculate max differences
            if any(regularValues(:))
                result.maxAbsoluteDifference = max(absoluteDifference(:));
                result.maxRelativeDifference = obj.getMaxRelativeDifference(expected, actual);
            end
            
            % Set final equality flag
            result.isEqual = (result.mismatchCount == 0);
        end
        
        function tolerance = calculateTolerance(obj, expected, actual)
            % CALCULATETOLERANCE Determines appropriate tolerance for comparison
            %
            % Calculates tolerance based on the magnitude of the compared values,
            % ensuring that both absolute and relative tolerance components are
            % considered. This is essential for financial calculations where the
            % scale of values can vary significantly.
            %
            % INPUTS:
            %   expected - Expected value(s)
            %   actual - Actual value(s)
            %
            % OUTPUTS:
            %   tolerance - Calculated tolerance value appropriate for comparison
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   tol = nc.calculateTolerance(1000, 1000.01);
            %   disp(['Tolerance: ', num2str(tol)]);
            
            % Validate inputs
            parametercheck(expected, 'expected');
            parametercheck(actual, 'actual');
            
            if obj.useAdaptiveTolerance
                % Get maximum magnitude of finite values
                finiteValues = [expected(isfinite(expected(:))); actual(isfinite(actual(:)))];
                
                if isempty(finiteValues)
                    % All values are non-finite (NaN/Inf)
                    tolerance = obj.defaultAbsoluteTolerance;
                    return;
                end
                
                maxMagnitude = max(abs(finiteValues));
                
                % Avoid division by zero for very small values
                if maxMagnitude < obj.machineEpsilon
                    tolerance = obj.defaultAbsoluteTolerance;
                else
                    % Combine absolute and relative components
                    relativePart = maxMagnitude * obj.defaultRelativeTolerance;
                    tolerance = max(obj.defaultAbsoluteTolerance, relativePart);
                end
            else
                % Use fixed absolute tolerance
                tolerance = obj.defaultAbsoluteTolerance;
            end
        end
        
        function isSignificant = areMismatchesSignificant(obj, differences, significanceLevel)
            % AREMISMATCHESSIGNIFICANT Analyzes pattern of mismatches for statistical significance
            %
            % Determines if the pattern of numerical differences represents
            % statistically significant mismatches rather than expected
            % floating-point noise. This is particularly useful for financial
            % econometric testing where distinguishing between numerical noise
            % and actual algorithmic differences is critical.
            %
            % INPUTS:
            %   differences - Matrix of differences between expected and actual
            %   significanceLevel - Optional threshold for significance [default: 0.05]
            %
            % OUTPUTS:
            %   isSignificant - True if mismatches are statistically significant
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   A = rand(10);
            %   B = A + 1e-15*randn(10); % Small random noise
            %   diffs = A - B;
            %   significant = nc.areMismatchesSignificant(diffs);
            %   disp(['Significant: ', num2str(significant)]);
            
            % Default significance level
            if nargin < 3 || isempty(significanceLevel)
                significanceLevel = 0.05;
            end
            
            % Validate inputs
            parametercheck(differences, 'differences');
            significanceOpts = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
            parametercheck(significanceLevel, 'significanceLevel', significanceOpts);
            
            % Use matrixdiagnostics to analyze the differences
            diagOptions = struct('ComputeEigenvalues', false);
            diffDiag = matrixdiagnostics(differences, diagOptions);
            
            % If matrix contains NaN or Inf, focus on finite values
            if diffDiag.ContainsNaN || diffDiag.ContainsInf
                % Get only finite values
                differences = differences(isfinite(differences));
                if isempty(differences)
                    isSignificant = false;
                    return;
                end
            end
            
            % Get basic statistics on the differences
            differences = differences(:);
            meanDiff = mean(differences);
            stdDiff = std(differences);
            maxAbsDiff = max(abs(differences));
            
            % Determine significance based on multiple criteria
            
            % 1. If max difference exceeds machine precision by orders of magnitude
            if maxAbsDiff > obj.machineEpsilon * 1e4
                isSignificant = true;
                return;
            end
            
            % 2. If mean difference is significantly non-zero
            zScore = abs(meanDiff) / (stdDiff / sqrt(length(differences)));
            if zScore > 3 % 3-sigma rule, approximately 99.7% confidence
                isSignificant = true;
                return;
            end
            
            % 3. If proportion of "large" differences exceeds significance threshold
            largeErrorCount = sum(abs(differences) > obj.defaultAbsoluteTolerance);
            if largeErrorCount / length(differences) > significanceLevel
                isSignificant = true;
                return;
            end
            
            % 4. If pattern suggests systematic bias rather than random noise
            positiveCount = sum(differences > 0);
            negativeCount = sum(differences < 0);
            totalNonZero = positiveCount + negativeCount;
            
            if totalNonZero > 0
                % Check for strong bias in one direction
                expectedRatio = 0.5; % Expect roughly equal distribution in random noise
                observedRatio = max(positiveCount, negativeCount) / totalNonZero;
                
                % If heavy bias toward positive or negative, it's significant
                if observedRatio > 0.75 && totalNonZero > 10
                    isSignificant = true;
                    return;
                end
            end
            
            % Default: not significant
            isSignificant = false;
        end
        
        function relDiff = getRelativeDifference(obj, expected, actual)
            % GETRELATIVEDIFFERENCE Calculates relative difference between values
            %
            % Computes the relative difference between expected and actual values,
            % handling special cases appropriately. This is especially important
            % for financial calculations where the magnitude of relative error
            % may be more meaningful than absolute error.
            %
            % INPUTS:
            %   expected - Expected value(s)
            %   actual - Actual value(s)
            %
            % OUTPUTS:
            %   relDiff - Relative difference as a proportion
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   relDiff = nc.getRelativeDifference(100, 101);
            %   disp(['Relative difference: ', num2str(relDiff)]);
            
            % Validate inputs are numeric
            if ~isnumeric(expected) || ~isnumeric(actual)
                error('NUMERICALCOMPARATOR:InvalidInput', ...
                      'Expected and actual values must be numeric');
            end
            
            % Handle exact equality
            if isequal(expected, actual)
                relDiff = 0;
                return;
            end
            
            % Handle special cases
            if isnan(expected) && isnan(actual)
                relDiff = 0;
                return;
            end
            
            if isinf(expected) && isinf(actual) && sign(expected) == sign(actual)
                relDiff = 0;
                return;
            end
            
            % Handle non-scalar inputs
            if ~isscalar(expected) || ~isscalar(actual)
                error('NUMERICALCOMPARATOR:NonScalarInput', ...
                      'getRelativeDifference requires scalar inputs');
            end
            
            % Handle cases where one value is zero
            if expected == 0 && actual == 0
                relDiff = 0;
                return;
            elseif expected == 0
                % When expected is zero, use absolute difference scaled by tolerance
                relDiff = abs(actual) / max(obj.defaultAbsoluteTolerance, obj.machineEpsilon);
                return;
            elseif actual == 0
                % When actual is zero, use absolute difference scaled by tolerance
                relDiff = abs(expected) / max(obj.defaultAbsoluteTolerance, obj.machineEpsilon);
                return;
            end
            
            % Handle very small values (near machine precision)
            if abs(expected) < obj.machineEpsilon && abs(actual) < obj.machineEpsilon
                relDiff = 0;
                return;
            end
            
            % Normal case: calculate relative difference
            % Using formula: |expected - actual| / max(|expected|, |actual|)
            denominator = max(abs(expected), abs(actual));
            relDiff = abs(expected - actual) / denominator;
        end
        
        function maxRelDiff = getMaxRelativeDifference(obj, expected, actual)
            % GETMAXRELATIVEDIFFERENCE Calculates maximum relative difference across matrices
            %
            % Finds the maximum relative difference between corresponding elements
            % in two matrices. This is useful for identifying the worst-case
            % relative error in financial model outputs.
            %
            % INPUTS:
            %   expected - Matrix of expected values
            %   actual - Matrix of actual values
            %
            % OUTPUTS:
            %   maxRelDiff - Maximum relative difference found between matrices
            %
            % EXAMPLES:
            %   nc = NumericalComparator();
            %   A = [1.1, 2.2; 3.3, 4.4];
            %   B = [1.1001, 2.2; 3.3, 4.401];
            %   maxDiff = nc.getMaxRelativeDifference(A, B);
            %   disp(['Maximum relative difference: ', num2str(maxDiff)]);
            
            % Check dimensions match
            if ~isequal(size(expected), size(actual))
                error('NUMERICALCOMPARATOR:DimensionMismatch', ...
                      'Matrices must have the same dimensions');
            end
            
            % Handle empty matrices
            if isempty(expected) && isempty(actual)
                maxRelDiff = 0;
                return;
            end
            
            % For efficiency with large matrices, convert to vectors
            expected = expected(:);
            actual = actual(:);
            
            % Initialize maximum relative difference
            maxRelDiff = 0;
            
            % Find all finite pairs to compare
            validPairs = isfinite(expected) & isfinite(actual);
            
            if any(validPairs)
                % Calculate relative differences for valid pairs
                validExp = expected(validPairs);
                validAct = actual(validPairs);
                
                % Handle exact matches efficiently
                exactMatch = (validExp == validAct);
                
                % Process non-exact matches
                if any(~exactMatch)
                    nonMatchExp = validExp(~exactMatch);
                    nonMatchAct = validAct(~exactMatch);
                    
                    % Check for zero values
                    zeroExp = (nonMatchExp == 0);
                    zeroAct = (nonMatchAct == 0);
                    
                    % Handle non-zero pairs
                    nonZeroPairs = ~zeroExp & ~zeroAct;
                    if any(nonZeroPairs)
                        % Calculate relative difference for non-zero pairs
                        e = nonMatchExp(nonZeroPairs);
                        a = nonMatchAct(nonZeroPairs);
                        absE = abs(e);
                        absA = abs(a);
                        
                        % Get the maximum value for each pair
                        maxVals = max(absE, absA);
                        
                        % Calculate relative differences
                        relDiffs = abs(e - a) ./ maxVals;
                        maxPairDiff = max(relDiffs);
                        
                        maxRelDiff = max(maxRelDiff, maxPairDiff);
                    end
                    
                    % Handle pairs with zeros
                    if any(zeroExp | zeroAct)
                        % For pairs with zeros, use absolute value scaled by tolerance
                        % as a proxy for relative difference
                        zeroInPair = (zeroExp | zeroAct);
                        if any(zeroInPair)
                            zE = nonMatchExp(zeroInPair);
                            zA = nonMatchAct(zeroInPair);
                            zeroDiffs = abs(zE - zA) / max(obj.defaultAbsoluteTolerance, obj.machineEpsilon);
                            maxZeroDiff = max(zeroDiffs);
                            
                            maxRelDiff = max(maxRelDiff, maxZeroDiff);
                        end
                    end
                end
            end
            
            % Handle special cases (NaN and Inf)
            % Only count mismatched NaN/Inf as having relative difference
            nanExpOnly = isnan(expected) & ~isnan(actual);
            nanActOnly = ~isnan(expected) & isnan(actual);
            
            infExpOnly = isinf(expected) & ~isinf(actual);
            infActOnly = ~isinf(expected) & isinf(actual);
            
            diffInfSigns = isinf(expected) & isinf(actual) & (sign(expected) ~= sign(actual));
            
            if any(nanExpOnly | nanActOnly | infExpOnly | infActOnly | diffInfSigns)
                % Assign maximum possible relative difference (effectively infinity)
                % but cap it at a large value for practical use
                maxRelDiff = max(maxRelDiff, 1e6);
            end
        end
        
        function setDefaultTolerances(obj, absoluteTolerance, relativeTolerance)
            % SETDEFAULTTOLERANCES Sets the default tolerance values used for comparisons
            %
            % Updates the default tolerance values used for numerical comparisons
            % when specific tolerances aren't provided.
            %
            % INPUTS:
            %   absoluteTolerance - New absolute tolerance value
            %   relativeTolerance - New relative tolerance value
            %
            % EXAMPLE:
            %   nc = NumericalComparator();
            %   nc.setDefaultTolerances(1e-12, 1e-10);
            
            % Validate inputs
            absOptions = struct('isscalar', true, 'isPositive', true);
            relOptions = struct('isscalar', true, 'isPositive', true);
            
            parametercheck(absoluteTolerance, 'absoluteTolerance', absOptions);
            parametercheck(relativeTolerance, 'relativeTolerance', relOptions);
            
            % Set new tolerance values
            obj.defaultAbsoluteTolerance = absoluteTolerance;
            obj.defaultRelativeTolerance = relativeTolerance;
        end
        
        function setAdaptiveToleranceMode(obj, useAdaptive)
            % SETADAPTIVETOLERANCEMODE Enables or disables adaptive tolerance calculation
            %
            % Toggles whether tolerance is adaptively calculated based on the
            % magnitude of the values being compared.
            %
            % INPUTS:
            %   useAdaptive - Boolean to enable/disable adaptive tolerance
            %
            % EXAMPLE:
            %   nc = NumericalComparator();
            %   nc.setAdaptiveToleranceMode(false); % Use fixed tolerance
            
            % Validate input
            if ~islogical(useAdaptive) || ~isscalar(useAdaptive)
                error('NUMERICALCOMPARATOR:InvalidInput', ...
                      'useAdaptive must be a logical scalar value');
            end
            
            % Set adaptive tolerance mode
            obj.useAdaptiveTolerance = useAdaptive;
        end
    end
end