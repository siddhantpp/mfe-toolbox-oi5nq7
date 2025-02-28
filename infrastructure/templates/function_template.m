function [results] = functionName(data, options)
%% FUNCTIONNAME Brief description of the function's purpose
%
% Detailed description of the function, including its methodology, algorithm,
% and any important concepts. This section should provide sufficient information
% for users to understand what the function does without needing to examine the code.
%
% USAGE:
%   results = functionName(data)
%   results = functionName(data, options)
%
% INPUTS:
%   data        - T by K matrix of input data
%                 Description of what each column represents
%                 First column: [Description of first column]
%                 Second column: [Description of second column]
%
%   options     - [OPTIONAL] Structure containing configuration parameters
%                 Default: [] (Uses default values for all options)
%                 Fields:
%                   options.field1 - Description [default = value1]
%                   options.field2 - Description [default = value2]
%                   options.field3 - Description [default = value3]
%
% OUTPUTS:
%   results     - Structure containing function outputs with fields:
%                   results.field1 - Description of field1
%                   results.field2 - Description of field2
%                   results.field3 - Description of field3
%
% COMMENTS:
%   Implementation notes, assumptions, numerical considerations, and important
%   details about computational methods, edge cases, or algorithmic choices.
%   - Note 1: Important implementation detail
%   - Note 2: Numerical consideration or limitation
%   - Note 3: Edge case handling
%
% EXAMPLES:
%   % Basic usage example
%   data = [1 2; 3 4; 5 6];
%   results = functionName(data);
%
%   % Example with custom options
%   options.field1 = 0.95;
%   options.field2 = 5;
%   results = functionName(data, options);
%
%   % Example showing how to interpret results
%   disp(['Result metric: ' num2str(results.field1)]);
%
% REFERENCES:
%   [1] Author, A. (Year). "Title of Paper." Journal, Volume(Issue), Pages.
%   [2] Author, B. (Year). "Title of Book." Publisher, City.
%
% SEE ALSO:
%   relatedFunction1, relatedFunction2, relatedFunction3
%
% MFE Toolbox v4.0
% Copyright (c) 2009

%% Input validation

% Validate data matrix - checks if not empty, numeric, no NaNs, all finite
data = datacheck(data, 'data');

% Check number of inputs and set default options if not provided
if nargin < 2
    options = [];
end

% Define default options structure
defaults = struct();
defaults.field1 = value1;  % Default value for field1
defaults.field2 = value2;  % Default value for field2
defaults.field3 = value3;  % Default value for field3

% Process options with defaults
options = processOptions(options, defaults);

% Validate specific parameters with different requirements
% Example 1: Scalar, non-negative parameter validation
paramOptions = struct();
paramOptions.isscalar = true;
paramOptions.isNonNegative = true;
options.field1 = parametercheck(options.field1, 'options.field1', paramOptions);

% Example 2: Parameters with bounds
paramOptions = struct();
paramOptions.lowerBound = 0;
paramOptions.upperBound = 1;
options.field2 = parametercheck(options.field2, 'options.field2', paramOptions);

% Example 3: Integer parameter validation
paramOptions = struct();
paramOptions.isInteger = true;
paramOptions.isPositive = true;
options.field3 = parametercheck(options.field3, 'options.field3', paramOptions);

% Ensure vector inputs are column vectors
if isfield(options, 'vectorField')
    options.vectorField = columncheck(options.vectorField, 'options.vectorField');
end

%% Main computation
try
    % Get dimensions of input data
    [T, K] = size(data);
    
    % Pre-allocate arrays for efficiency where appropriate
    intermediateResults = zeros(T, 1);
    
    % --------------------------------------------------------
    % Main algorithm implementation starts here
    % --------------------------------------------------------
    
    % Step 1: First computational step
    % ...code for step 1...
    
    % Step 2: Second computational step
    % ...code for step 2...
    
    % Step 3: Final computational step
    % ...code for step 3...
    
    % --------------------------------------------------------
    % Main algorithm implementation ends here
    % --------------------------------------------------------
    
    % Format and package results into output structure
    results = struct();
    results.field1 = output1;  % Description of output1
    results.field2 = output2;  % Description of output2
    results.field3 = output3;  % Description of output3
    
catch ME
    % Enhanced error handling with context
    errorID = 'MFEToolbox:functionName';
    
    % Add additional context to the error message
    if strcmp(ME.identifier, 'MATLAB:nomem')
        errorMessage = sprintf(['Error in functionName: Out of memory. ' ...
            'Try using a smaller dataset or increasing available memory.']);
    else
        errorMessage = sprintf(['Error in functionName: %s\nFunction failed ' ...
            'at line %d of file %s.'], ME.message, ME.line, ME.stack(1).file);
    end
    
    % Throw error with context
    error(errorID, errorMessage);
end

end

%% Helper function for processing options
function options = processOptions(options, defaults)
% PROCESSOPTIONS Helper function to handle default options
%
% Processes the options structure:
%   1. If options is empty, use all defaults
%   2. For each field in defaults, check if it exists in options
%   3. If field is missing in options, use value from defaults

% If options is empty, use all defaults
if isempty(options)
    options = defaults;
    return;
end

% For each field in defaults, check if it exists in options
defaultFields = fieldnames(defaults);
for i = 1:length(defaultFields)
    field = defaultFields{i};
    if ~isfield(options, field)
        % If field doesn't exist in options, use default value
        options.(field) = defaults.(field);
    end
end

% Additional validation could be added here if needed
% For example, checking for valid combinations of options
% or enforcing constraints between different options

end