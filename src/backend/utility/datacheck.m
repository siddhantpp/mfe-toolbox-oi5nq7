function data = datacheck(data, dataname)
% DATACHECK Validates numerical data for econometric and statistical functions.
%
% USAGE:
%   VALIDDATA = datacheck(DATA, DATANAME)
%
% INPUTS:
%   DATA     - Data to be validated
%   DATANAME - String containing the name of DATA, used in error messages
%
% OUTPUTS:
%   VALIDDATA - Validated data (unchanged if validation passes)
%
% COMMENTS:
%   Performs comprehensive validation ensuring that:
%     1. Data is not empty
%     2. Data is numeric
%     3. Data does not contain any NaN values
%     4. Data contains only finite values
%
%   This function is a core utility for the MFE Toolbox and helps
%   ensure computational stability and proper error handling across
%   the toolbox's econometric and statistical functions.
%
% EXAMPLES:
%   returns = datacheck(returns, 'returns');
%   parameters = datacheck(parameters, 'model parameters');
%
% See also isempty, isnumeric, isnan, isfinite, all, any

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Check if data is empty
if isempty(data)
    error('%s cannot be empty', dataname);
end

% Step 2: Verify data is numeric type
if ~isnumeric(data)
    error('%s must be numeric', dataname);
end

% Step 3: Check for presence of NaN values
if any(isnan(data(:)))
    error('%s cannot contain NaN values', dataname);
end

% Step 4: Check for presence of infinite values
if ~all(isfinite(data(:)))
    error('%s cannot contain Inf or -Inf values', dataname);
end

% Return the validated data unchanged as all validation checks have passed
end