function r = stdtrnd(size, nu)
% STDTRND Generates random numbers from the standardized Student's t-distribution.
%
% USAGE:
%   R = stdtrnd(SIZE, NU)
%
% INPUTS:
%   SIZE   - Scalar, vector or 2-element vector of the form [rows cols] that 
%            specifies the dimension of random numbers to generate.
%            If SIZE is a scalar, the output will be a SIZE by 1 vector.
%            If SIZE is empty or not provided, a single random number is returned.
%   NU     - Degrees of freedom parameter, must be > 2 to ensure the existence
%            of variance (for standardization to be valid)
%
% OUTPUTS:
%   R      - Matrix of random numbers from the standardized Student's t-distribution
%            with specified degrees of freedom NU. The random numbers have mean 0
%            and variance 1.
%
% COMMENTS:
%   The standardized Student's t-distribution is a t-distribution that has been
%   scaled to have mean 0 and variance 1. This requires NU > 2, since variance
%   is only defined for degrees of freedom greater than 2.
%
%   This function supports two implementation methods:
%   1. Direct generation with MATLAB's trnd function followed by standardization
%   2. Inverse CDF method using uniform random numbers and stdtinv
%
%   Method 1 is generally faster and is used as the default implementation.
%
% EXAMPLES:
%   % Generate a single random number with 5 degrees of freedom
%   r = stdtrnd([], 5);
%
%   % Generate a 1000 by 1 vector of random numbers
%   r = stdtrnd(1000, 10);
%
%   % Generate a 100 by 5 matrix of random numbers
%   r = stdtrnd([100 5], 6);
%
% See also: stdtpdf, stdtcdf, stdtinv, trnd, rand

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate the degrees of freedom parameter (nu)
options.isscalar = true;
options.lowerBound = 2;  % Must be > 2 for standardized t-distribution
nu = parametercheck(nu, 'nu', options);

% Step 2: Process size parameter and determine dimensions of output
if nargin < 1 || isempty(size)
    % Default to 1x1 (single random number)
    rows = 1;
    cols = 1;
else
    % Check if size is valid
    size = datacheck(size, 'size');
    
    % Handle different formats of size parameter
    if isscalar(size)
        % If size is a scalar, interpret as number of rows
        rows = size;
        cols = 1;
    else
        % If size is a vector, ensure it's a column vector
        size = columncheck(size, 'size');
        
        % Extract dimensions from size parameter
        if length(size) == 1
            rows = size;
            cols = 1;
        elseif length(size) == 2
            rows = size(1);
            cols = size(2);
        else
            error('SIZE must be a scalar or a 2-element vector');
        end
    end
end

% Step 3: Generate random numbers from standardized Student's t-distribution
% Method 1: Use MATLAB's trnd and standardize
% Generate raw t random numbers with nu degrees of freedom
raw_t = trnd(nu, rows, cols);

% Standardize to ensure variance is 1 (mean is already 0)
% For a t distribution with nu df, variance = nu/(nu-2) for nu > 2
% To standardize, we scale by sqrt(nu/(nu-2))
r = raw_t * sqrt((nu-2)/nu);

% Alternative Method 2 (using inverse CDF method)
% This can be enabled if needed but is generally slower
% u = rand(rows, cols);  % Generate uniform random numbers
% r = stdtinv(u, nu);    % Transform using the inverse CDF

end