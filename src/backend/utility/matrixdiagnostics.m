function diagnostics = matrixdiagnostics(matrix, options)
%MATRIXDIAGNOSTICS Comprehensive matrix property diagnostics
%
%   DIAGNOSTICS = MATRIXDIAGNOSTICS(MATRIX) performs a comprehensive
%   diagnostic analysis of MATRIX, evaluating properties such as rank,
%   condition number, symmetry, positive definiteness, and detecting
%   numerical issues such as NaN or Inf values.
%
%   DIAGNOSTICS = MATRIXDIAGNOSTICS(MATRIX, OPTIONS) performs diagnostics
%   with the specified options:
%       OPTIONS.Tolerance - Numerical tolerance for symmetry testing
%                          (default: eps^(3/4))
%       OPTIONS.SingularityThreshold - Threshold for condition number to
%                                     determine near-singularity
%                                     (default: 1e12)
%       OPTIONS.ComputeEigenvalues - Whether to compute eigenvalues
%                                   (default: true for matrices smaller
%                                   than 500x500, false otherwise)
%       OPTIONS.MaxSizeForDet - Maximum matrix size for computing determinant
%                              (default: 100)
%
%   The returned DIAGNOSTICS structure includes:
%       .Dimensions      - Matrix dimensions [rows, cols]
%       .IsEmpty         - Whether matrix is empty
%       .IsNumeric       - Whether matrix contains numeric data
%       .IsVector        - Whether matrix is a vector
%       .IsSquare        - Whether matrix is square
%       .ContainsNaN     - Whether matrix contains NaN values
%       .ContainsInf     - Whether matrix contains Inf values
%       .NumNaN          - Number of NaN values
%       .NumInf          - Number of Inf values
%       .NumZeros        - Number of zero values
%       .Range           - [min, max] of finite values
%       .Rank            - Numerical rank (if no NaN/Inf values)
%       .ConditionNumber - Condition number (if square matrix)
%       .IsSymmetric     - Whether matrix is symmetric
%       .IsPositiveDefinite - Whether matrix is positive definite
%       .Determinant     - Matrix determinant (if small square matrix)
%       .Eigenvalues     - Eigenvalue analysis (if requested)
%
%   Example:
%       A = [1 2; 2 4];
%       diagnostics = matrixdiagnostics(A)
%
%   See also isSymmetric, isPositiveDefinite, isNearSingular, RANK, COND, DET, EIG

% Version 4.0 (28-Oct-2009)
% MFE Toolbox

% Input validation
if nargin < 1
    error('MATRIXDIAGNOSTICS:InvalidInput', 'A matrix input is required.');
end

% Default options
if nargin < 2
    options = struct();
end

if ~isstruct(options)
    error('MATRIXDIAGNOSTICS:InvalidOptions', 'OPTIONS must be a structure.');
end

% Set default options if not provided
if ~isfield(options, 'Tolerance')
    options.Tolerance = eps^(3/4); % Default tolerance for symmetry
end

if ~isfield(options, 'SingularityThreshold')
    options.SingularityThreshold = 1e12; % Default threshold for near-singularity
end

if ~isfield(options, 'MaxSizeForDet')
    options.MaxSizeForDet = 100; % Maximum size for determinant calculation
end

% Initialize diagnostics structure
diagnostics = struct();

% Basic properties
diagnostics.Dimensions = size(matrix);
diagnostics.IsEmpty = isempty(matrix);
diagnostics.IsNumeric = isnumeric(matrix);
diagnostics.IsVector = isempty(matrix) || (min(size(matrix)) == 1);
diagnostics.IsSquare = ~isempty(matrix) && (size(matrix, 1) == size(matrix, 2));
diagnostics.IsSparse = issparse(matrix);

% If not numeric, return early with limited diagnostics
if ~diagnostics.IsNumeric && ~islogical(matrix)
    warning('MATRIXDIAGNOSTICS:NonNumericInput', ...
        'Input is not numeric or logical. Limited diagnostics available.');
    return;
end

% Check for NaN and Inf values
diagnostics.ContainsNaN = any(isnan(matrix(:)));
diagnostics.ContainsInf = any(isinf(matrix(:)));
diagnostics.NumNaN = sum(isnan(matrix(:)));
diagnostics.NumInf = sum(isinf(matrix(:)));
diagnostics.NumZeros = sum(matrix(:) == 0);

% Calculate range of finite values
finiteValues = matrix(isfinite(matrix));
if ~isempty(finiteValues)
    diagnostics.Range = [min(finiteValues), max(finiteValues)];
else
    diagnostics.Range = [NaN, NaN];
end

% Numerical properties (only if no NaN/Inf)
if ~diagnostics.ContainsNaN && ~diagnostics.ContainsInf
    % Rank calculation (for all matrices)
    diagnostics.Rank = rank(matrix);
    
    % Square matrix properties
    if diagnostics.IsSquare
        % Check symmetry
        diagnostics.IsSymmetric = isSymmetric(matrix, options.Tolerance);
        
        % Condition number (for square matrices)
        try
            diagnostics.ConditionNumber = cond(matrix);
            diagnostics.IsNearSingular = isNearSingular(matrix, options.SingularityThreshold);
        catch ME
            warning('MATRIXDIAGNOSTICS:ConditionNumberFailed', ...
                'Condition number calculation failed: %s', ME.message);
            diagnostics.ConditionNumber = Inf;
            diagnostics.IsNearSingular = true;
        end
        
        % Positive definiteness (for symmetric matrices)
        if diagnostics.IsSymmetric
            diagnostics.IsPositiveDefinite = isPositiveDefinite(matrix);
        else
            diagnostics.IsPositiveDefinite = false;
        end
        
        % Determinant (for small matrices)
        if all(diagnostics.Dimensions <= options.MaxSizeForDet)
            try
                diagnostics.Determinant = det(matrix);
            catch ME
                warning('MATRIXDIAGNOSTICS:DeterminantFailed', ...
                    'Determinant calculation failed: %s', ME.message);
                diagnostics.Determinant = NaN;
            end
        end
        
        % Eigenvalue analysis (if requested)
        if ~isfield(options, 'ComputeEigenvalues')
            options.ComputeEigenvalues = all(diagnostics.Dimensions < 500);
        end
        
        if options.ComputeEigenvalues
            try
                [~, d] = eig(matrix);
                eigenvalues = diag(d);
                diagnostics.Eigenvalues = struct(...
                    'Values', eigenvalues, ...
                    'Min', min(abs(eigenvalues)), ...
                    'Max', max(abs(eigenvalues)), ...
                    'NumPositive', sum(real(eigenvalues) > 0), ...
                    'NumNegative', sum(real(eigenvalues) < 0), ...
                    'NumZero', sum(abs(eigenvalues) < eps(max(abs(eigenvalues))) * 10), ...
                    'NumComplex', sum(abs(imag(eigenvalues)) > eps(max(abs(eigenvalues))) * 10) ...
                );
            catch ME
                warning('MATRIXDIAGNOSTICS:EigenvaluesFailed', ...
                    'Eigenvalue analysis failed: %s', ME.message);
                diagnostics.Eigenvalues = struct(...
                    'Values', NaN, ...
                    'Min', NaN, ...
                    'Max', NaN, ...
                    'NumPositive', NaN, ...
                    'NumNegative', NaN, ...
                    'NumZero', NaN, ...
                    'NumComplex', NaN ...
                );
            end
        end
    end
else
    % For matrices with NaN/Inf, set numerical properties to NaN
    diagnostics.Rank = NaN;
    if diagnostics.IsSquare
        diagnostics.IsSymmetric = NaN;
        diagnostics.ConditionNumber = NaN;
        diagnostics.IsNearSingular = NaN;
        diagnostics.IsPositiveDefinite = NaN;
        diagnostics.Determinant = NaN;
    end
end

end

function result = isSymmetric(matrix, tolerance)
%ISSYMMETRIC Determines if a matrix is symmetric within a tolerance
%
%   RESULT = ISSYMMETRIC(MATRIX) checks if MATRIX is symmetric with a
%   default tolerance based on machine precision.
%
%   RESULT = ISSYMMETRIC(MATRIX, TOLERANCE) uses the specified tolerance.
%
%   A matrix is considered symmetric if the maximum absolute difference
%   between the matrix and its transpose is less than the tolerance.
%
%   Example:
%       A = [1 2; 2 1];
%       isSymmetric(A) % returns true
%
%   See also MATRIXDIAGNOSTICS

% Input validation
if nargin < 1
    error('ISSYMMETRIC:InvalidInput', 'Matrix input is required.');
end

% Check if matrix is square
if ~ismatrix(matrix) || size(matrix, 1) ~= size(matrix, 2)
    result = false;
    return;
end

% Default tolerance if not specified
if nargin < 2 || isempty(tolerance)
    tolerance = eps^(3/4);
end

% Check symmetry within tolerance
difference = abs(matrix - matrix');
maxDifference = max(difference(:));
result = maxDifference <= tolerance;

end

function result = isPositiveDefinite(matrix)
%ISPOSITIVEDEFINITE Determines if a matrix is positive definite
%
%   RESULT = ISPOSITIVEDEFINITE(MATRIX) checks if MATRIX is positive
%   definite by attempting Cholesky factorization.
%
%   A matrix is positive definite if all its eigenvalues are positive.
%   This implementation uses the Cholesky factorization as an efficient
%   test for positive definiteness.
%
%   Example:
%       A = [2 1; 1 2];
%       isPositiveDefinite(A) % returns true
%
%   See also MATRIXDIAGNOSTICS, CHOL

% Input validation
if nargin < 1
    error('ISPOSITIVEDEFINITE:InvalidInput', 'Matrix input is required.');
end

% Check if matrix is square
if ~ismatrix(matrix) || size(matrix, 1) ~= size(matrix, 2)
    result = false;
    return;
end

% Check if matrix is symmetric (required for positive definiteness)
if ~isSymmetric(matrix)
    result = false;
    return;
end

% Check if matrix is positive definite using Cholesky decomposition
try
    chol(matrix);
    result = true;
catch
    result = false;
end

end

function result = isNearSingular(matrix, threshold)
%ISNEARSINGULAR Detects if a matrix is singular or near-singular
%
%   RESULT = ISNEARSINGULAR(MATRIX) checks if MATRIX is singular or
%   near-singular based on its condition number with a default threshold.
%
%   RESULT = ISNEARSINGULAR(MATRIX, THRESHOLD) uses the specified threshold
%   for determining near-singularity.
%
%   A matrix is considered near-singular if its condition number exceeds
%   the threshold.
%
%   Example:
%       A = [1 2; 2 4]; % Rank-deficient matrix
%       isNearSingular(A) % returns true
%
%   See also MATRIXDIAGNOSTICS, COND

% Input validation
if nargin < 1
    error('ISNEARSINGULAR:InvalidInput', 'Matrix input is required.');
end

% Check if matrix is square
if ~ismatrix(matrix) || size(matrix, 1) ~= size(matrix, 2)
    error('ISNEARSINGULAR:NonSquareMatrix', 'Matrix must be square.');
end

% Default threshold if not specified
if nargin < 2 || isempty(threshold)
    threshold = 1e12; % Default threshold for near-singularity
end

% Check if matrix is near-singular based on condition number
condNumber = cond(matrix);
result = (condNumber > threshold) || isinf(condNumber);

end