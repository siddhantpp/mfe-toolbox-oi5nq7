function [x] = columncheck(x, name)
% COLUMNCHECK Validates if input is a column vector or converts row vector to column
%
% USAGE:
%   [X] = columncheck(X, NAME)
%
% INPUTS:
%   X     - Input data to validate/convert
%   NAME  - String name of input for error messages
%
% OUTPUTS:
%   X     - Column vector (either original or converted from row vector)
%
% COMMENTS:
%   Function checks if input is empty, non-numeric, or has invalid dimensions.
%   If input is a row vector (1×n), it will be transposed to a column vector (n×1).
%   If input has multiple columns and is not a row vector, function throws an error.
%   This function ensures consistent vector formatting for numerical stability
%   in econometric calculations.
%
% See also isempty, isnumeric, size, transpose

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Check if input is empty
if isempty(x)
    error([name ' cannot be empty']);
end

% Check if input is numeric
if ~isnumeric(x)
    error([name ' must be numeric']);
end

% Get dimensions of input
[r, c] = size(x);

% Convert row vector to column vector
if r == 1 && c > 1
    x = transpose(x);
% Throw error if input is not a vector (has multiple columns)
elseif c > 1
    error([name ' must be a column vector or a row vector']);
end