classdef CrossPlatformValidator < BaseTest
    % CROSSPLATFORMVALIDATOR A specialized utility class for cross-platform validation and comparison of MEX implementations
    %
    % This validator ensures numerical consistency, performance equivalence, and error handling uniformity 
    % between platform-specific MEX binaries across Windows and Unix platforms in the MFE Toolbox.
    %
    % The class provides methods for:
    %   - Cross-platform validation of computational components
    %   - Generating and comparing reference results across platforms
    %   - Analyzing platform-specific numerical differences
    %   - Verifying consistent behavior across operating systems
    %
    % Example:
    %   % Create a validator
    %   validator = CrossPlatformValidator();
    %
    %   % Validate a component across platforms
    %   testInputs = {randn(1000,1), [0.01; 0.1; 0.85], var(randn(1000,1))};
    %   result = validator.validateComponent('agarch_core', testInputs);
    %
    %   % Check if a MEX implementation is compatible across platforms
    %   isCompatible = validator.isCompatibleAcrossPlatforms('egarch_core', testInputs);
    %
    %   % Generate a comprehensive compatibility report
    %   report = validator.generateCompatibilityReport(validationResults);
    %
    % See also: BaseTest, NumericalComparator, MEXValidator, parametercheck
    
    properties
        referenceResultsPath   % Path to reference results storage
        currentPlatform        % Current platform identifier
        supportedPlatforms     % Cell array of supported platform identifiers
        platformResults        % Structure to store validation results by platform
        comparator             % NumericalComparator instance for result validation
        mexValidator           % MEXValidator instance for MEX-specific validation
        platformTolerance      % Numerical tolerance for cross-platform comparisons
        verbose                % Flag to control output verbosity
    end
    
    methods
        function obj = CrossPlatformValidator(options)
            % Initialize a new CrossPlatformValidator instance with appropriate configuration
            %
            % INPUTS:
            %   options - Optional structure with configuration options:
            %       .referenceResultsPath - Custom path to reference results [default: 'src/test/data/cross_platform/']
            %       .platformTolerance - Custom tolerance for platform comparisons [default: 1e-9]
            %       .verbose - Flag to enable verbose output [default: false]
            %
            % OUTPUTS:
            %   obj - Initialized CrossPlatformValidator instance
            
            % Call parent constructor with class name
            obj@BaseTest('CrossPlatformValidator');
            
            % Set default options if not provided
            if nargin < 1
                options = struct();
            end
            
            % Set reference results path
            if isfield(options, 'referenceResultsPath')
                obj.referenceResultsPath = options.referenceResultsPath;
            else
                obj.referenceResultsPath = 'src/test/data/cross_platform/';
            end
            
            % Determine current platform
            obj.currentPlatform = computer();
            
            % Set supported platforms
            obj.supportedPlatforms = {'PCWIN64', 'GLNXA64'};
            
            % Initialize MEXValidator for MEX-specific validation
            mexOptions = struct();
            if isfield(options, 'mexBasePath')
                mexOptions.mexBasePath = options.mexBasePath;
            end
            obj.mexValidator = MEXValidator(mexOptions);
            
            % Initialize NumericalComparator with appropriate tolerance
            compOptions = struct();
            if isfield(options, 'comparatorTolerance')
                compOptions.absoluteTolerance = options.comparatorTolerance;
                compOptions.relativeTolerance = options.comparatorTolerance * 10;
            end
            obj.comparator = NumericalComparator(compOptions);
            
            % Set platform comparison tolerance
            if isfield(options, 'platformTolerance')
                obj.platformTolerance = options.platformTolerance;
            else
                obj.platformTolerance = 1e-9;
            end
            
            % Initialize empty platformResults structure
            obj.platformResults = struct();
            
            % Set verbosity
            if isfield(options, 'verbose')
                obj.verbose = options.verbose;
            else
                obj.verbose = false;
            end
            
            % Create reference results directory if it doesn't exist
            if ~exist(obj.referenceResultsPath, 'dir')
                mkdir(obj.referenceResultsPath);
                if obj.verbose
                    fprintf('Created reference results directory: %s\n', obj.referenceResultsPath);
                end
            end
        end
        
        function result = validateComponent(obj, componentName, testInputs, tolerance)
            % Validates a specific component across platforms by comparing numerical results
            %
            % This method checks if a component produces consistent results across different
            % platforms by generating reference results and comparing them with results from
            % other platforms.
            %
            % INPUTS:
            %   componentName - Name of the component to validate
            %   testInputs - Cell array of inputs for testing the component
            %   tolerance - Optional numerical tolerance for comparison [default: obj.platformTolerance]
            %
            % OUTPUTS:
            %   result - Validation result structure with:
            %       .componentName - Name of the validated component
            %       .isCompatible - Whether component is compatible across platforms
            %       .platforms - Platforms where component was tested
            %       .differences - Detailed differences between platforms
            %       .maxDifference - Maximum difference across platforms
            %       .timestamp - Validation timestamp
            
            % Validate inputs
            parametercheck(componentName, 'componentName');
            if ~iscell(testInputs)
                error('CrossPlatformValidator:InvalidInput', 'testInputs must be a cell array');
            end
            
            % Set default tolerance if not provided
            if nargin < 4 || isempty(tolerance)
                tolerance = obj.platformTolerance;
            end
            
            % Initialize result structure
            result = struct(...
                'componentName', componentName, ...
                'isCompatible', true, ...
                'platforms', {{}}, ...
                'differences', struct(), ...
                'maxDifference', 0, ...
                'timestamp', datestr(now) ...
            );
            
            % Generate reference results on current platform
            currentResults = obj.generateReferenceResults(componentName, testInputs, []);
            
            % Record current platform
            result.platforms{end+1} = obj.currentPlatform;
            
            % Try to load reference results from other platforms
            for i = 1:length(obj.supportedPlatforms)
                platform = obj.supportedPlatforms{i};
                
                % Skip current platform (already generated)
                if strcmp(platform, obj.currentPlatform)
                    continue;
                end
                
                % Load reference results from this platform
                platformResults = obj.loadReferenceResults(componentName, platform);
                
                % If reference results exist for this platform, compare them
                if ~isempty(platformResults)
                    % Record platform in results
                    result.platforms{end+1} = platform;
                    
                    % Compare results
                    comparisonResult = obj.compareResults(platformResults, currentResults, tolerance);
                    
                    % Update overall compatibility flag
                    result.isCompatible = result.isCompatible && comparisonResult.isEqual;
                    
                    % Store detailed comparison results
                    fieldName = [obj.currentPlatform, '_vs_', platform];
                    result.differences.(fieldName) = comparisonResult;
                    
                    % Update maximum difference
                    result.maxDifference = max(result.maxDifference, comparisonResult.maxAbsoluteDifference);
                    
                    % Output information if verbose
                    if obj.verbose
                        if comparisonResult.isEqual
                            fprintf('Component %s is compatible between %s and %s\n', ...
                                componentName, obj.currentPlatform, platform);
                        else
                            fprintf('Component %s shows differences between %s and %s (max diff: %g)\n', ...
                                componentName, obj.currentPlatform, platform, comparisonResult.maxAbsoluteDifference);
                        end
                    end
                end
            end
            
            % If no other platform results were found, can't determine compatibility
            if length(result.platforms) <= 1
                result.isCompatible = false;
                warning('CrossPlatformValidator:NoOtherPlatformResults', ...
                    'No reference results found for other platforms. Cannot determine cross-platform compatibility.');
            end
            
            % Store validation result in platformResults structure
            obj.platformResults.(componentName) = result;
        end
        
        function result = validateMEXCompatibility(obj, mexBaseName, testInputs, tolerance)
            % Validates the cross-platform compatibility of a specific MEX implementation
            %
            % This method checks if a MEX implementation produces consistent results across
            % different platforms by comparing numerical outputs and error handling.
            %
            % INPUTS:
            %   mexBaseName - Base name of the MEX file without extension
            %   testInputs - Cell array of inputs for testing the MEX function
            %   tolerance - Optional numerical tolerance for comparison [default: obj.platformTolerance]
            %
            % OUTPUTS:
            %   result - Compatibility assessment with:
            %       .mexBaseName - Name of the MEX function
            %       .isCompatible - Whether MEX is compatible across platforms
            %       .platforms - Platforms where MEX was tested
            %       .differences - Detailed differences between platforms
            %       .errorConsistency - Whether error handling is consistent
            %       .timestamp - Validation timestamp
            
            % Validate inputs
            parametercheck(mexBaseName, 'mexBaseName');
            if ~iscell(testInputs)
                error('CrossPlatformValidator:InvalidInput', 'testInputs must be a cell array');
            end
            
            % Set default tolerance if not provided
            if nargin < 4 || isempty(tolerance)
                tolerance = obj.platformTolerance;
            end
            
            % Initialize result structure
            result = struct(...
                'mexBaseName', mexBaseName, ...
                'isCompatible', false, ...
                'platforms', {{}}, ...
                'differences', struct(), ...
                'errorConsistency', true, ...
                'timestamp', datestr(now) ...
            );
            
            % Check if MEX file exists for current platform
            mexExists = obj.mexValidator.validateMEXExists(mexBaseName);
            if ~mexExists
                error('CrossPlatformValidator:MEXNotFound', ...
                    'MEX file %s not found for current platform %s', mexBaseName, obj.currentPlatform);
            end
            
            % Generate MEX reference results on current platform
            currentResults = struct();
            currentResults.platform = obj.currentPlatform;
            currentResults.timestamp = datestr(now);
            
            % Execute MEX function and capture output
            try
                mexFunc = str2func(mexBaseName);
                currentResults.output = mexFunc(testInputs{:});
                currentResults.executionSuccess = true;
                currentResults.errorMessage = '';
            catch ME
                currentResults.executionSuccess = false;
                currentResults.errorMessage = ME.message;
                currentResults.output = [];
            end
            
            % Save reference results for current platform
            obj.generateReferenceResults(mexBaseName, testInputs, currentResults);
            
            % Record current platform
            result.platforms{end+1} = obj.currentPlatform;
            
            % Try to load reference results from other platforms
            for i = 1:length(obj.supportedPlatforms)
                platform = obj.supportedPlatforms{i};
                
                % Skip current platform (already tested)
                if strcmp(platform, obj.currentPlatform)
                    continue;
                end
                
                % Load reference results from this platform
                platformResults = obj.loadReferenceResults(mexBaseName, platform);
                
                % If reference results exist for this platform, compare them
                if ~isempty(platformResults)
                    % Record platform in results
                    result.platforms{end+1} = platform;
                    
                    % Check for consistent error handling
                    if currentResults.executionSuccess ~= platformResults.executionSuccess
                        result.errorConsistency = false;
                        fieldName = [obj.currentPlatform, '_vs_', platform];
                        result.differences.(fieldName) = struct(...
                            'isEqual', false, ...
                            'errorMessage', 'Inconsistent error handling between platforms', ...
                            'currentPlatformSuccess', currentResults.executionSuccess, ...
                            'otherPlatformSuccess', platformResults.executionSuccess ...
                        );
                        continue;
                    end
                    
                    % If both failed, check error message similarity
                    if ~currentResults.executionSuccess && ~platformResults.executionSuccess
                        % Compare error messages (not expecting exact match, but similar content)
                        errorSimilarity = levenshteinDistance(currentResults.errorMessage, platformResults.errorMessage) / ...
                            max(length(currentResults.errorMessage), length(platformResults.errorMessage));
                        
                        fieldName = [obj.currentPlatform, '_vs_', platform];
                        result.differences.(fieldName) = struct(...
                            'isEqual', errorSimilarity < 0.5, ... % Arbitrary threshold for similarity
                            'errorMessage', 'Error message comparison', ...
                            'errorSimilarity', 1 - errorSimilarity, ... % Convert to similarity (0-1)
                            'currentPlatformError', currentResults.errorMessage, ...
                            'otherPlatformError', platformResults.errorMessage ...
                        );
                        
                        continue;
                    end
                    
                    % Both succeeded, compare outputs
                    if currentResults.executionSuccess && platformResults.executionSuccess
                        % Compare results using compareResults method
                        compResult = struct();
                        compResult.results = platformResults.output;
                        
                        currResult = struct();
                        currResult.results = currentResults.output;
                        
                        comparisonResult = obj.compareResults(compResult, currResult, tolerance);
                        
                        % Store detailed comparison results
                        fieldName = [obj.currentPlatform, '_vs_', platform];
                        result.differences.(fieldName) = comparisonResult;
                    end
                end
            end
            
            % Determine overall compatibility
            result.isCompatible = true;
            
            if ~result.errorConsistency
                result.isCompatible = false;
            else
                % Check result differences
                diffFields = fieldnames(result.differences);
                for i = 1:length(diffFields)
                    if isfield(result.differences.(diffFields{i}), 'isEqual') && ...
                            ~result.differences.(diffFields{i}).isEqual
                        result.isCompatible = false;
                        break;
                    end
                end
            end
            
            % If no other platform results were found, can't determine compatibility
            if length(result.platforms) <= 1
                result.isCompatible = false;
                warning('CrossPlatformValidator:NoOtherPlatformResults', ...
                    'No reference results found for other platforms. Cannot determine cross-platform compatibility.');
            end
            
            % Store validation result in platformResults structure
            obj.platformResults.(mexBaseName) = result;
        end
        
        function refResults = generateReferenceResults(obj, componentName, testInputs, results)
            % Generates reference results for cross-platform testing using the current platform
            %
            % This method executes a component on the current platform and stores the
            % results for later comparison with other platforms.
            %
            % INPUTS:
            %   componentName - Name of the component
            %   testInputs - Cell array of inputs for the component
            %   results - Optional pre-computed results to store
            %
            % OUTPUTS:
            %   refResults - Reference result structure with platform metadata
            
            % Validate inputs
            parametercheck(componentName, 'componentName');
            if ~iscell(testInputs) && ~isempty(testInputs)
                error('CrossPlatformValidator:InvalidInput', 'testInputs must be a cell array or empty');
            end
            
            % Create reference results structure
            refResults = struct(...
                'componentName', componentName, ...
                'platform', obj.currentPlatform, ...
                'timestamp', datestr(now), ...
                'matlabVersion', version, ...
                'inputs', {testInputs} ...
            );
            
            % If results are provided, store them directly
            if nargin >= 4 && ~isempty(results)
                refResults.results = results;
            else
                % Otherwise, execute the component with the provided inputs
                try
                    % Check if component is a function
                    if exist(componentName, 'file') == 2 || exist(componentName, 'file') == 3  % M-file or MEX-file
                        % Get function handle and execute
                        func = str2func(componentName);
                        refResults.results = func(testInputs{:});
                    else
                        error('CrossPlatformValidator:ComponentNotFound', ...
                            'Component %s not found or is not a function', componentName);
                    end
                catch ME
                    warning('CrossPlatformValidator:ExecutionFailed', ...
                        'Failed to execute component %s: %s', componentName, ME.message);
                    refResults.results = [];
                    refResults.error = ME;
                end
            end
            
            % Create directory for reference results if it doesn't exist
            refDir = fileparts(obj.getReferenceResultsPath(componentName, obj.currentPlatform));
            if ~exist(refDir, 'dir')
                mkdir(refDir);
            end
            
            % Save reference results
            refPath = obj.getReferenceResultsPath(componentName, obj.currentPlatform);
            save(refPath, 'refResults');
            
            if obj.verbose
                fprintf('Reference results for %s generated and saved to %s\n', componentName, refPath);
            end
        end
        
        function refResults = loadReferenceResults(obj, componentName, platform)
            % Loads reference results for a component from a specific platform
            %
            % This method loads previously generated reference results for comparison.
            %
            % INPUTS:
            %   componentName - Name of the component
            %   platform - Platform identifier to load results from
            %
            % OUTPUTS:
            %   refResults - Loaded reference result structure or empty if not found
            
            % Validate inputs
            parametercheck(componentName, 'componentName');
            parametercheck(platform, 'platform');
            
            % Construct path to reference results
            refPath = obj.getReferenceResultsPath(componentName, platform);
            
            % Check if reference results exist
            if exist(refPath, 'file')
                % Load reference results
                try
                    loaded = load(refPath);
                    refResults = loaded.refResults;
                    
                    % Validate structure
                    if ~isfield(refResults, 'componentName') || ~isfield(refResults, 'platform') || ...
                            ~isfield(refResults, 'results')
                        warning('CrossPlatformValidator:InvalidReferenceResults', ...
                            'Reference results for %s on platform %s have invalid format', componentName, platform);
                        refResults = [];
                    end
                catch ME
                    warning('CrossPlatformValidator:LoadFailed', ...
                        'Failed to load reference results for %s on platform %s: %s', ...
                        componentName, platform, ME.message);
                    refResults = [];
                end
            else
                % No reference results found
                if obj.verbose
                    fprintf('No reference results found for %s on platform %s\n', componentName, platform);
                end
                refResults = [];
            end
        end
        
        function result = compareResults(obj, referenceResults, currentResults, tolerance)
            % Compares results between platforms with specialized handling for floating-point differences
            %
            % This method performs detailed comparison of component outputs across platforms,
            % accounting for expected platform-specific floating-point differences.
            %
            % INPUTS:
            %   referenceResults - Reference results from one platform
            %   currentResults - Current results to compare
            %   tolerance - Numerical tolerance for comparison
            %
            % OUTPUTS:
            %   result - Comparison result with detailed analysis
            
            % Validate inputs
            if ~isstruct(referenceResults)
                error('CrossPlatformValidator:InvalidInput', 'referenceResults must be a structure');
            end
            
            if ~isstruct(currentResults)
                error('CrossPlatformValidator:InvalidInput', 'currentResults must be a structure');
            end
            
            % Set default tolerance if not provided
            if nargin < 4 || isempty(tolerance)
                tolerance = obj.platformTolerance;
            end
            
            % Initialize result structure
            result = struct(...
                'isEqual', false, ...
                'maxAbsoluteDifference', 0, ...
                'maxRelativeDifference', 0, ...
                'toleranceUsed', tolerance, ...
                'platform1', '', ...
                'platform2', '', ...
                'timestamp', datestr(now) ...
            );
            
            % Get platform information
            if isfield(referenceResults, 'platform')
                result.platform1 = referenceResults.platform;
            else
                result.platform1 = 'unknown';
            end
            
            if isfield(currentResults, 'platform')
                result.platform2 = currentResults.platform;
            else
                result.platform2 = obj.currentPlatform;
            end
            
            % Extract result data
            if isfield(referenceResults, 'results')
                ref = referenceResults.results;
            else
                ref = referenceResults;
            end
            
            if isfield(currentResults, 'results')
                curr = currentResults.results;
            else
                curr = currentResults;
            end
            
            % Handle different result types
            if isstruct(ref) && isstruct(curr)
                % Compare struct fields
                refFields = fieldnames(ref);
                currFields = fieldnames(curr);
                
                % Check if fields match
                if ~isequal(sort(refFields), sort(currFields))
                    result.isEqual = false;
                    result.errorMessage = 'Structures have different fields';
                    return;
                end
                
                % Compare each field
                result.isEqual = true;
                result.fieldResults = struct();
                
                for i = 1:length(refFields)
                    field = refFields{i};
                    fieldResult = obj.compareResults(struct('results', ref.(field)), ...
                        struct('results', curr.(field)), tolerance);
                    
                    result.fieldResults.(field) = fieldResult;
                    
                    % Update overall result
                    result.isEqual = result.isEqual && fieldResult.isEqual;
                    result.maxAbsoluteDifference = max(result.maxAbsoluteDifference, fieldResult.maxAbsoluteDifference);
                    result.maxRelativeDifference = max(result.maxRelativeDifference, fieldResult.maxRelativeDifference);
                end
                
            elseif isnumeric(ref) && isnumeric(curr)
                % Compare numeric data
                if ~isequal(size(ref), size(curr))
                    result.isEqual = false;
                    result.errorMessage = 'Numeric arrays have different dimensions';
                    return;
                end
                
                % Use NumericalComparator for robust comparison
                compResult = obj.comparator.compareMatrices(ref, curr, tolerance);
                
                % Copy comparison results
                result.isEqual = compResult.isEqual;
                result.maxAbsoluteDifference = compResult.maxAbsoluteDifference;
                result.maxRelativeDifference = compResult.maxRelativeDifference;
                
                % Include additional information for debugging
                if ~result.isEqual
                    result.mismatchCount = compResult.mismatchCount;
                    result.mismatchIndices = compResult.mismatchIndices;
                    
                    % Analyze if mismatches are significant using NumericalComparator
                    result.mismatchesSignificant = obj.comparator.areMismatchesSignificant(ref - curr);
                    
                    % Include detailed floating-point analysis
                    result.floatingPointAnalysis = obj.analyzeFloatingPointDifferences(ref, curr, ...
                        result.platform1, result.platform2);
                end
                
            elseif iscell(ref) && iscell(curr)
                % Compare cell arrays
                if ~isequal(size(ref), size(curr))
                    result.isEqual = false;
                    result.errorMessage = 'Cell arrays have different dimensions';
                    return;
                end
                
                % Compare each cell element
                result.isEqual = true;
                result.cellResults = cell(size(ref));
                
                for i = 1:numel(ref)
                    cellResult = obj.compareResults(struct('results', ref{i}), ...
                        struct('results', curr{i}), tolerance);
                    
                    result.cellResults{i} = cellResult;
                    
                    % Update overall result
                    result.isEqual = result.isEqual && cellResult.isEqual;
                    result.maxAbsoluteDifference = max(result.maxAbsoluteDifference, cellResult.maxAbsoluteDifference);
                    result.maxRelativeDifference = max(result.maxRelativeDifference, cellResult.maxRelativeDifference);
                end
                
            else
                % For other types, use direct comparison
                result.isEqual = isequal(ref, curr);
                if ~result.isEqual
                    result.errorMessage = 'Non-numeric values are not equal';
                end
            end
        end
        
        function result = isCompatibleAcrossPlatforms(obj, componentName, testInputs, tolerance)
            % Determines if a component's results are compatible across all supported platforms
            %
            % This method validates a component across all supported platforms and
            % determines if its behavior is consistent regardless of platform.
            %
            % INPUTS:
            %   componentName - Name of the component to validate
            %   testInputs - Cell array of inputs for testing the component
            %   tolerance - Optional numerical tolerance for comparison
            %
            % OUTPUTS:
            %   result - True if component is compatible across platforms
            
            % Validate component
            validationResult = obj.validateComponent(componentName, testInputs, tolerance);
            
            % Component is compatible if validation passed and it was tested on multiple platforms
            result = validationResult.isCompatible && length(validationResult.platforms) > 1;
        end
        
        function analysis = analyzeFloatingPointDifferences(obj, platformA_result, platformB_result, platformA, platformB)
            % Analyzes floating-point differences between platforms to determine if they are within acceptable limits
            %
            % This method performs detailed analysis of numerical differences between
            % platform results to identify patterns and classify the differences.
            %
            % INPUTS:
            %   platformA_result - Result from first platform
            %   platformB_result - Result from second platform
            %   platformA - First platform identifier
            %   platformB - Second platform identifier
            %
            % OUTPUTS:
            %   analysis - Detailed analysis of numerical differences
            
            % Initialize analysis structure
            analysis = struct(...
                'patternType', 'unknown', ...
                'acceptableDifference', true, ...
                'meanDifference', 0, ...
                'stdDifference', 0, ...
                'maxAbsoluteDifference', 0, ...
                'maxRelativeDifference', 0, ...
                'differenceClusters', 0, ...
                'characterization', '', ...
                'platforms', struct('A', platformA, 'B', platformB) ...
            );
            
            % Validate inputs are numeric
            if ~isnumeric(platformA_result) || ~isnumeric(platformB_result)
                analysis.patternType = 'non-numeric';
                analysis.characterization = 'Cannot analyze non-numeric values';
                return;
            end
            
            % Check dimensions
            if ~isequal(size(platformA_result), size(platformB_result))
                analysis.patternType = 'dimension-mismatch';
                analysis.acceptableDifference = false;
                analysis.characterization = 'Results have different dimensions';
                return;
            end
            
            % Calculate differences
            differences = platformA_result - platformB_result;
            
            % Find finite values for statistical analysis
            finiteIdx = isfinite(differences);
            
            if ~any(finiteIdx)
                analysis.patternType = 'non-finite';
                analysis.characterization = 'All differences are non-finite';
                return;
            end
            
            finiteDiffs = differences(finiteIdx);
            
            % Basic statistics
            analysis.meanDifference = mean(finiteDiffs);
            analysis.stdDifference = std(finiteDiffs);
            analysis.maxAbsoluteDifference = max(abs(finiteDiffs));
            
            % Calculate relative differences for non-zero values
            nonZeroIdx = (platformA_result ~= 0 | platformB_result ~= 0) & finiteIdx;
            if any(nonZeroIdx)
                nonZeroA = platformA_result(nonZeroIdx);
                nonZeroB = platformB_result(nonZeroIdx);
                
                % Use maximum absolute value as denominator for relative difference
                denominators = max(abs(nonZeroA), abs(nonZeroB));
                relativeDiffs = abs(nonZeroA - nonZeroB) ./ denominators;
                
                analysis.maxRelativeDifference = max(relativeDiffs);
            else
                analysis.maxRelativeDifference = 0;
            end
            
            % Analyze pattern of differences
            % 1. Check if differences are concentrated around specific values
            [uniqueDiffs, ~, diffIdx] = unique(round(finiteDiffs * 1e12) / 1e12);
            diffCounts = histcounts(diffIdx, 1:length(uniqueDiffs)+1);
            
            % Count significant clusters
            significantDiffs = uniqueDiffs(diffCounts > length(finiteDiffs) * 0.01);
            analysis.differenceClusters = length(significantDiffs);
            
            % 2. Determine pattern type
            if analysis.maxAbsoluteDifference < 1e-12
                analysis.patternType = 'negligible';
                analysis.characterization = 'Differences are negligible (< 1e-12)';
            elseif analysis.differenceClusters <= 3 && abs(analysis.meanDifference) < 1e-10
                analysis.patternType = 'rounding';
                analysis.characterization = 'Differences appear to be due to floating-point rounding';
            elseif abs(analysis.meanDifference) > analysis.stdDifference
                analysis.patternType = 'systematic-bias';
                analysis.acceptableDifference = analysis.maxAbsoluteDifference < 1e-6;
                analysis.characterization = 'Systematic bias between platforms';
            elseif analysis.differenceClusters > 10
                analysis.patternType = 'random-noise';
                analysis.acceptableDifference = analysis.maxAbsoluteDifference < 1e-6;
                analysis.characterization = 'Differences appear random, likely due to platform-specific optimizations';
            else
                analysis.patternType = 'mixed';
                analysis.acceptableDifference = analysis.maxAbsoluteDifference < 1e-8;
                analysis.characterization = 'Mixed pattern of differences';
            end
            
            % Final classification
            if analysis.maxAbsoluteDifference > 1e-4
                analysis.acceptableDifference = false;
                analysis.characterization = [analysis.characterization, ' - Differences exceed acceptable threshold'];
            end
        end
        
        function results = validateAllComponents(obj, componentList, options)
            % Validates all registered components across supported platforms
            %
            % This method runs cross-platform validation on a list of components
            % and generates a comprehensive compatibility report.
            %
            % INPUTS:
            %   componentList - Cell array of component names to validate
            %   options - Optional structure with validation options
            %
            % OUTPUTS:
            %   results - Comprehensive validation results
            
            % Set default options
            if nargin < 3
                options = struct();
            end
            
            % Initialize results structure
            results = struct(...
                'componentResults', struct(), ...
                'summary', struct(...
                    'totalComponents', length(componentList), ...
                    'compatibleComponents', 0, ...
                    'incompatibleComponents', 0, ...
                    'untestableComponents', 0, ...
                    'platforms', {{}}, ...
                    'maxDifference', 0 ...
                ), ...
                'timestamp', datestr(now) ...
            );
            
            % Record current platform
            results.summary.platforms{end+1} = obj.currentPlatform;
            
            % Create test inputs generator function if provided
            if isfield(options, 'inputGenerator')
                inputGenerator = options.inputGenerator;
            else
                % Default input generator creates random inputs
                inputGenerator = @(componentName) obj.generateTestInputs(componentName);
            end
            
            % Set tolerance
            if isfield(options, 'tolerance')
                tolerance = options.tolerance;
            else
                tolerance = obj.platformTolerance;
            end
            
            % Validate each component
            for i = 1:length(componentList)
                componentName = componentList{i};
                
                % Generate test inputs for this component
                testInputs = inputGenerator(componentName);
                
                % Validate component
                try
                    componentResult = obj.validateComponent(componentName, testInputs, tolerance);
                    results.componentResults.(componentName) = componentResult;
                    
                    % Update summary statistics
                    if componentResult.isCompatible
                        results.summary.compatibleComponents = results.summary.compatibleComponents + 1;
                    elseif length(componentResult.platforms) <= 1
                        results.summary.untestableComponents = results.summary.untestableComponents + 1;
                    else
                        results.summary.incompatibleComponents = results.summary.incompatibleComponents + 1;
                    end
                    
                    % Update max difference
                    results.summary.maxDifference = max(results.summary.maxDifference, componentResult.maxDifference);
                    
                    % Update platform list
                    for j = 1:length(componentResult.platforms)
                        platform = componentResult.platforms{j};
                        if ~ismember(platform, results.summary.platforms)
                            results.summary.platforms{end+1} = platform;
                        end
                    end
                    
                catch ME
                    warning('CrossPlatformValidator:ValidationFailed', ...
                        'Validation failed for component %s: %s', componentName, ME.message);
                    
                    results.componentResults.(componentName) = struct(...
                        'componentName', componentName, ...
                        'isCompatible', false, ...
                        'error', ME.message, ...
                        'platforms', {obj.currentPlatform}, ...
                        'differences', struct(), ...
                        'maxDifference', 0, ...
                        'timestamp', datestr(now) ...
                    );
                    
                    results.summary.untestableComponents = results.summary.untestableComponents + 1;
                end
                
                % Progress update
                if obj.verbose
                    fprintf('Validated component %d/%d: %s\n', i, length(componentList), componentName);
                end
            end
            
            % Calculate summary statistics
            results.summary.compatibilityRate = results.summary.compatibleComponents / results.summary.totalComponents * 100;
            
            % Generate compatibility report
            results.report = obj.generateCompatibilityReport(results);
        end
        
        function report = generateCompatibilityReport(obj, validationResults)
            % Generates a comprehensive cross-platform compatibility report
            %
            % This method analyzes validation results and generates a structured
            % report with findings and recommendations.
            %
            % INPUTS:
            %   validationResults - Structure with validation results
            %
            % OUTPUTS:
            %   report - Structured compatibility report
            
            % Initialize report structure
            report = struct(...
                'title', 'Cross-Platform Compatibility Report', ...
                'timestamp', datestr(now), ...
                'platforms', {{}}, ...
                'summary', struct(), ...
                'componentCategories', struct(), ...
                'recommendations', {{}}, ...
                'detailedFindings', struct() ...
            );
            
            % Extract platforms
            if isfield(validationResults, 'summary') && isfield(validationResults.summary, 'platforms')
                report.platforms = validationResults.summary.platforms;
            else
                report.platforms = {obj.currentPlatform};
            end
            
            % Copy summary statistics
            if isfield(validationResults, 'summary')
                report.summary = validationResults.summary;
            else
                report.summary = struct(...
                    'totalComponents', 0, ...
                    'compatibleComponents', 0, ...
                    'incompatibleComponents', 0, ...
                    'untestableComponents', 0, ...
                    'compatibilityRate', 0, ...
                    'platforms', report.platforms, ...
                    'maxDifference', 0 ...
                );
            end
            
            % Categorize components
            report.componentCategories = struct(...
                'compatible', {{}}, ...
                'minorDifferences', {{}}, ...
                'majorDifferences', {{}}, ...
                'untestable', {{}} ...
            );
            
            % Analyze each component
            if isfield(validationResults, 'componentResults')
                componentNames = fieldnames(validationResults.componentResults);
                
                for i = 1:length(componentNames)
                    componentName = componentNames{i};
                    result = validationResults.componentResults.(componentName);
                    
                    % Categorize based on compatibility
                    if ~isfield(result, 'isCompatible')
                        % Skip invalid results
                        continue;
                    end
                    
                    if result.isCompatible
                        % Compatible component
                        report.componentCategories.compatible{end+1} = componentName;
                    elseif length(result.platforms) <= 1
                        % Untestable component
                        report.componentCategories.untestable{end+1} = componentName;
                    elseif isfield(result, 'maxDifference') && result.maxDifference < 1e-6
                        % Minor differences
                        report.componentCategories.minorDifferences{end+1} = componentName;
                    else
                        % Major differences
                        report.componentCategories.majorDifferences{end+1} = componentName;
                    end
                    
                    % Store detailed findings
                    report.detailedFindings.(componentName) = result;
                end
            end
            
            % Generate recommendations
            recommendations = {};
            
            % Recommendation for untestable components
            if ~isempty(report.componentCategories.untestable)
                if length(report.platforms) <= 1
                    recommendations{end+1} = 'Generate reference results on additional platforms to enable cross-platform compatibility testing.';
                else
                    recommendations{end+1} = sprintf('Generate reference results for untestable components (%d components) to enable cross-platform validation.', ...
                        length(report.componentCategories.untestable));
                end
            end
            
            % Recommendation for major differences
            if ~isempty(report.componentCategories.majorDifferences)
                recommendations{end+1} = sprintf('Investigate and resolve major cross-platform differences in %d components: %s', ...
                    length(report.componentCategories.majorDifferences), ...
                    strjoin(report.componentCategories.majorDifferences(1:min(5, length(report.componentCategories.majorDifferences))), ', '));
                
                if length(report.componentCategories.majorDifferences) > 5
                    recommendations{end} = [recommendations{end}, ', etc.'];
                end
            end
            
            % Recommendation for minor differences
            if ~isempty(report.componentCategories.minorDifferences)
                recommendations{end+1} = sprintf('Consider reviewing %d components with minor cross-platform differences for potential improvements.', ...
                    length(report.componentCategories.minorDifferences));
            end
            
            % Overall recommendation
            if report.summary.compatibilityRate < 90
                recommendations{end+1} = sprintf('Overall cross-platform compatibility rate (%.1f%%) is below target. Focus on improving platform-specific numerical consistency.', ...
                    report.summary.compatibilityRate);
            else
                recommendations{end+1} = sprintf('Overall cross-platform compatibility rate (%.1f%%) is good. Continue monitoring for platform-specific differences in new components.', ...
                    report.summary.compatibilityRate);
            end
            
            % Store recommendations
            report.recommendations = recommendations;
        end
        
        function refPath = getReferenceResultsPath(obj, componentName, platform)
            % Gets the path to reference results for a specific component and platform
            %
            % INPUTS:
            %   componentName - Name of the component
            %   platform - Optional platform identifier [default: currentPlatform]
            %
            % OUTPUTS:
            %   refPath - Full path to reference results file
            
            % Validate inputs
            parametercheck(componentName, 'componentName');
            
            % Use current platform if not specified
            if nargin < 3 || isempty(platform)
                platform = obj.currentPlatform;
            end
            
            % Construct directory path - organize by component then platform
            componentDir = fullfile(obj.referenceResultsPath, componentName);
            
            % Create directory if it doesn't exist
            if ~exist(componentDir, 'dir')
                mkdir(componentDir);
            end
            
            % Construct full file path
            refPath = fullfile(componentDir, [platform, '.mat']);
        end
        
        function platform = getCurrentPlatform(obj)
            % Gets the current platform identifier for reference result generation
            %
            % OUTPUTS:
            %   platform - Platform identifier (PCWIN64 or GLNXA64)
            
            % Return current platform
            platform = obj.currentPlatform;
        end
        
        function setSupportedPlatforms(obj, platforms)
            % Sets the list of supported platforms for validation
            %
            % INPUTS:
            %   platforms - Cell array of platform identifiers
            
            % Validate input is a cell array
            if ~iscell(platforms)
                error('CrossPlatformValidator:InvalidInput', 'platforms must be a cell array of platform identifiers');
            end
            
            % Validate each platform is a valid identifier
            for i = 1:length(platforms)
                if ~ischar(platforms{i})
                    error('CrossPlatformValidator:InvalidPlatform', 'Platform identifiers must be character arrays');
                end
            end
            
            % Set supported platforms
            obj.supportedPlatforms = platforms;
        end
        
        function setPlatformTolerance(obj, tolerance)
            % Sets the numerical tolerance for cross-platform comparisons
            %
            % INPUTS:
            %   tolerance - Numerical tolerance value
            
            % Validate tolerance is a positive double
            toleranceOpts = struct('isscalar', true, 'isPositive', true);
            parametercheck(tolerance, 'tolerance', toleranceOpts);
            
            % Set platform tolerance
            obj.platformTolerance = tolerance;
        end
        
        function testInputs = generateTestInputs(obj, componentName)
            % Generates appropriate test inputs for a specific component
            %
            % INPUTS:
            %   componentName - Name of the component
            %
            % OUTPUTS:
            %   testInputs - Cell array of test inputs
            
            % Use MEXValidator's input generator if the component is a MEX file
            if strncmpi(componentName, 'agarch', 6) || strncmpi(componentName, 'agarch_core', 11)
                category = 'agarch';
            elseif strncmpi(componentName, 'tarch', 5) || strncmpi(componentName, 'tarch_core', 10)
                category = 'tarch';
            elseif strncmpi(componentName, 'egarch', 6) || strncmpi(componentName, 'egarch_core', 11)
                category = 'egarch';
            elseif strncmpi(componentName, 'igarch', 6) || strncmpi(componentName, 'igarch_core', 11)
                category = 'igarch';
            elseif strncmpi(componentName, 'armax', 5) || strncmpi(componentName, 'armaxerrors', 11)
                category = 'armax';
            elseif strncmpi(componentName, 'composite', 9) || strncmpi(componentName, 'composite_likelihood', 19)
                category = 'composite_likelihood';
            else
                category = 'other';
            end
            
            % Generate inputs
            testInputs = obj.mexValidator.generateTestInputs(category);
        end
    end
end

function distance = levenshteinDistance(s1, s2)
    % Calculates the Levenshtein distance between two strings
    %
    % INPUTS:
    %   s1 - First string
    %   s2 - Second string
    %
    % OUTPUTS:
    %   distance - Levenshtein distance

    % Convert to character arrays if needed
    if ~ischar(s1)
        s1 = char(s1);
    end
    
    if ~ischar(s2)
        s2 = char(s2);
    end
    
    % Get string lengths
    m = length(s1);
    n = length(s2);
    
    % Create distance matrix
    d = zeros(m+1, n+1);
    
    % Initialize first row and column
    for i = 0:m
        d(i+1, 1) = i;
    end
    
    for j = 0:n
        d(1, j+1) = j;
    end
    
    % Fill in the rest of the matrix
    for j = 1:n
        for i = 1:m
            if s1(i) == s2(j)
                cost = 0;
            else
                cost = 1;
            end
            
            d(i+1, j+1) = min([d(i, j+1) + 1, ... % Deletion
                                d(i+1, j) + 1, ... % Insertion
                                d(i, j) + cost]);  % Substitution
        end
    end
    
    % Return distance
    distance = d(m+1, n+1);
end