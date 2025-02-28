classdef MatrixdiagnosticsTest < BaseTest
    % MatrixdiagnosticsTest Unit test class for validating the matrixdiagnostics utility function
    %
    % This test suite ensures that matrixdiagnostics correctly identifies
    % numerical properties of matrices including rank, condition number,
    % symmetry, positive definiteness, and the presence of NaN or Inf values.
    
    properties
        % Test matrices
        testMatrix
        symmetricMatrix
        positiveDefiniteMatrix
        singularMatrix
        
        % Numerical comparator
        comparator
        
        % Tolerance for numerical comparisons
        tolerance
    end
    
    methods
        function obj = MatrixdiagnosticsTest()
            % Initialize the MatrixdiagnosticsTest class instance
            obj@BaseTest();
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method is run
            setUp@BaseTest(obj);
            
            % Initialize test matrices
            obj.testMatrix = rand(4, 5);
            
            % Create a symmetric matrix (A + A')/2
            A = rand(4);
            obj.symmetricMatrix = (A + A')/2;
            
            % Create a positive definite matrix
            % A diagonal matrix with positive values is positive definite
            obj.positiveDefiniteMatrix = diag([1, 2, 3, 4]);
            
            % Create a singular matrix (not full rank)
            obj.singularMatrix = [1 2 3; 2 4 6; 3 6 9]; % Rank 1
            
            % Initialize numerical comparator
            obj.comparator = NumericalComparator();
            
            % Set default tolerance
            obj.tolerance = 1e-10;
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method is run
            tearDown@BaseTest(obj);
            
            % Clear test matrices
            obj.testMatrix = [];
            obj.symmetricMatrix = [];
            obj.positiveDefiniteMatrix = [];
            obj.singularMatrix = [];
            obj.comparator = [];
        end
        
        function testBasicDiagnostics(obj)
            % Tests that matrixdiagnostics correctly returns basic matrix properties
            
            % Get diagnostics for testMatrix
            diagnostics = matrixdiagnostics(obj.testMatrix);
            
            % Verify dimensions
            obj.assertEqual(diagnostics.Dimensions, size(obj.testMatrix), 'Dimensions should match');
            
            % Verify basic properties
            obj.assertFalse(diagnostics.IsEmpty, 'Matrix should not be empty');
            obj.assertTrue(diagnostics.IsNumeric, 'Matrix should be numeric');
            obj.assertFalse(diagnostics.IsVector, 'Matrix should not be a vector');
            obj.assertFalse(diagnostics.IsSquare, 'Non-square matrix should be detected');
            
            % Verify rank calculation
            obj.assertEqual(diagnostics.Rank, rank(obj.testMatrix), 'Rank should match MATLAB rank');
            
            % Test with a square matrix
            squareMatrix = rand(4);
            diagnostics = matrixdiagnostics(squareMatrix);
            
            obj.assertTrue(diagnostics.IsSquare, 'Square matrix should be detected');
            obj.assertFalse(diagnostics.IsEmpty, 'Matrix should not be empty');
            
            % The condition number should be calculated
            obj.assertTrue(isfield(diagnostics, 'ConditionNumber'), 'Condition number should be calculated for square matrices');
            
            % Verify condition number matches MATLAB's
            expectedCond = cond(squareMatrix);
            obj.comparator.compareMatrices(diagnostics.ConditionNumber, expectedCond, obj.tolerance);
            
            % Verify range information
            obj.assertEqual(length(diagnostics.Range), 2, 'Range should have min and max values');
            obj.assertTrue(diagnostics.Range(1) <= diagnostics.Range(2), 'Min should be <= max in range');
        end
        
        function testSymmetryDetection(obj)
            % Tests that matrixdiagnostics correctly identifies symmetric matrices
            
            % Test isSymmetric function directly
            obj.assertTrue(isSymmetric(obj.symmetricMatrix), 'Symmetric matrix should be detected');
            obj.assertFalse(isSymmetric(obj.testMatrix), 'Non-symmetric matrix should not be detected as symmetric');
            
            % Test through matrixdiagnostics
            diagnostics = matrixdiagnostics(obj.symmetricMatrix);
            obj.assertTrue(diagnostics.IsSymmetric, 'Symmetric matrix should be detected through matrixdiagnostics');
            
            % Test with a nearly symmetric matrix (with numerical error)
            nearlySymmetric = obj.symmetricMatrix;
            nearlySymmetric(1,2) = nearlySymmetric(1,2) + 1e-12; % Small perturbation
            
            % Should still be detected as symmetric with default tolerance
            obj.assertTrue(isSymmetric(nearlySymmetric), 'Nearly symmetric matrix should be detected as symmetric');
            
            % But with a very strict tolerance, it should not be symmetric
            obj.assertFalse(isSymmetric(nearlySymmetric, 1e-15), 'With strict tolerance, should not be symmetric');
            
            % Test with a non-square matrix
            options = struct('Tolerance', 1e-10);
            diagnostics = matrixdiagnostics(obj.testMatrix, options);
            
            % A non-square matrix cannot be symmetric
            if isfield(diagnostics, 'IsSymmetric')
                obj.assertFalse(diagnostics.IsSymmetric, 'Non-square matrix cannot be symmetric');
            end
        end
        
        function testPositiveDefiniteDetection(obj)
            % Tests that matrixdiagnostics correctly identifies positive definite matrices
            
            % Test isPositiveDefinite function directly
            obj.assertTrue(isPositiveDefinite(obj.positiveDefiniteMatrix), 'Positive definite matrix should be detected');
            
            % Create a non-positive definite matrix
            nonPD = eye(4);
            nonPD(1,1) = -1; % Negative eigenvalue
            
            obj.assertFalse(isPositiveDefinite(nonPD), 'Non-positive definite matrix should be detected');
            
            % Test through matrixdiagnostics
            diagnostics = matrixdiagnostics(obj.positiveDefiniteMatrix);
            obj.assertTrue(diagnostics.IsPositiveDefinite, 'Positive definite matrix should be detected through matrixdiagnostics');
            
            diagnostics = matrixdiagnostics(nonPD);
            obj.assertFalse(diagnostics.IsPositiveDefinite, 'Non-positive definite matrix should be detected through matrixdiagnostics');
            
            % Test with a symmetric but non-positive definite matrix
            symmetricNonPD = (nonPD + nonPD')/2;
            diagnostics = matrixdiagnostics(symmetricNonPD);
            obj.assertFalse(diagnostics.IsPositiveDefinite, 'Symmetric non-positive definite matrix should be detected');
            
            % Test with a non-square matrix
            % A non-square matrix cannot be positive definite
            diagnostics = matrixdiagnostics(obj.testMatrix);
            if isfield(diagnostics, 'IsPositiveDefinite')
                obj.assertFalse(diagnostics.IsPositiveDefinite, 'Non-square matrix cannot be positive definite');
            end
            
            % Test with a matrix at the boundary of positive definiteness
            boundaryPD = eye(4);
            boundaryPD(1,1) = eps; % Very small positive value
            diagnostics = matrixdiagnostics(boundaryPD);
            obj.assertTrue(diagnostics.IsPositiveDefinite, 'Boundary case should still be positive definite');
        end
        
        function testSingularityDetection(obj)
            % Tests that matrixdiagnostics correctly identifies singular or near-singular matrices
            
            % Test isNearSingular function directly
            obj.assertTrue(isNearSingular(obj.singularMatrix), 'Singular matrix should be detected');
            
            % Test with well-conditioned matrix
            wellConditioned = eye(4); % Identity matrix has condition number 1
            obj.assertFalse(isNearSingular(wellConditioned), 'Well-conditioned matrix should not be detected as singular');
            
            % Test with almost singular matrix
            almostSingular = eye(4);
            almostSingular(1,1) = 1e-14; % Very small pivot
            
            % By default, this should be detected as near-singular
            obj.assertTrue(isNearSingular(almostSingular), 'Almost singular matrix should be detected');
            
            % With a higher threshold, it should not be detected
            obj.assertFalse(isNearSingular(almostSingular, 1e16), 'With high threshold, should not be detected as singular');
            
            % Test through matrixdiagnostics
            diagnostics = matrixdiagnostics(obj.singularMatrix);
            obj.assertTrue(diagnostics.IsNearSingular, 'Singular matrix should be detected through matrixdiagnostics');
            
            diagnostics = matrixdiagnostics(wellConditioned);
            obj.assertFalse(diagnostics.IsNearSingular, 'Well-conditioned matrix should not be detected as singular through matrixdiagnostics');
            
            % Test with explicitly specified singularity threshold in options
            options = struct('SingularityThreshold', 1e10);
            diagnostics = matrixdiagnostics(almostSingular, options);
            obj.assertTrue(diagnostics.IsNearSingular, 'Should be near-singular with default threshold');
            
            options = struct('SingularityThreshold', 1e16);
            diagnostics = matrixdiagnostics(almostSingular, options);
            obj.assertFalse(diagnostics.IsNearSingular, 'Should not be near-singular with higher threshold');
        end
        
        function testNaNHandling(obj)
            % Tests that matrixdiagnostics correctly identifies and handles matrices with NaN values
            
            % Create a matrix with NaN values
            nanMatrix = obj.testMatrix;
            nanMatrix(1,1) = NaN;
            nanMatrix(2,3) = NaN;
            
            % Test with matrixdiagnostics
            diagnostics = matrixdiagnostics(nanMatrix);
            
            % Verify NaN detection
            obj.assertTrue(diagnostics.ContainsNaN, 'NaN values should be detected');
            obj.assertEqual(diagnostics.NumNaN, 2, 'Number of NaN values should be counted correctly');
            
            % Numerical properties should be set to NaN
            obj.assertTrue(isnan(diagnostics.Rank), 'Rank should be NaN for matrix with NaN values');
            
            % For square matrices with NaN, additional properties should be NaN
            nanSquare = eye(4);
            nanSquare(1,1) = NaN;
            
            diagnostics = matrixdiagnostics(nanSquare);
            obj.assertTrue(isnan(diagnostics.ConditionNumber), 'Condition number should be NaN for matrix with NaN values');
            obj.assertTrue(isnan(diagnostics.IsSymmetric), 'Symmetry property should be NaN for matrix with NaN values');
            obj.assertTrue(isnan(diagnostics.IsPositiveDefinite), 'Positive definiteness should be NaN for matrix with NaN values');
            
            % Check that range calculation excludes NaN values
            nanMatrix = ones(3);
            nanMatrix(1,1) = NaN;
            nanMatrix(2,2) = 5; % Max value
            nanMatrix(3,3) = -2; % Min value
            
            diagnostics = matrixdiagnostics(nanMatrix);
            obj.assertEqual(diagnostics.Range, [-2, 5], 'Range should exclude NaN values');
        end
        
        function testInfHandling(obj)
            % Tests that matrixdiagnostics correctly identifies and handles matrices with Inf values
            
            % Create a matrix with Inf values
            infMatrix = obj.testMatrix;
            infMatrix(1,1) = Inf;
            infMatrix(2,3) = -Inf;
            
            % Test with matrixdiagnostics
            diagnostics = matrixdiagnostics(infMatrix);
            
            % Verify Inf detection
            obj.assertTrue(diagnostics.ContainsInf, 'Inf values should be detected');
            obj.assertEqual(diagnostics.NumInf, 2, 'Number of Inf values should be counted correctly');
            
            % Numerical properties should be set to NaN
            obj.assertTrue(isnan(diagnostics.Rank), 'Rank should be NaN for matrix with Inf values');
            
            % For square matrices with Inf, additional properties should be NaN
            infSquare = eye(4);
            infSquare(1,1) = Inf;
            
            diagnostics = matrixdiagnostics(infSquare);
            obj.assertTrue(isnan(diagnostics.ConditionNumber), 'Condition number should be NaN for matrix with Inf values');
            obj.assertTrue(isnan(diagnostics.IsSymmetric), 'Symmetry property should be NaN for matrix with Inf values');
            obj.assertTrue(isnan(diagnostics.IsPositiveDefinite), 'Positive definiteness should be NaN for matrix with Inf values');
        end
        
        function testEdgeCases(obj)
            % Tests matrixdiagnostics behavior with edge case inputs
            
            % Empty matrix
            emptyMatrix = [];
            diagnostics = matrixdiagnostics(emptyMatrix);
            
            obj.assertTrue(diagnostics.IsEmpty, 'Empty matrix should be detected');
            obj.assertEqual(diagnostics.Dimensions, [0, 0], 'Empty matrix dimensions should be [0, 0]');
            
            % Scalar input
            scalarValue = 42;
            diagnostics = matrixdiagnostics(scalarValue);
            
            obj.assertTrue(diagnostics.IsSquare, 'Scalar should be detected as square');
            obj.assertTrue(diagnostics.IsVector, 'Scalar should be detected as vector');
            obj.assertEqual(diagnostics.Dimensions, [1, 1], 'Scalar dimensions should be [1, 1]');
            obj.assertEqual(diagnostics.Rank, 1, 'Scalar rank should be 1 if non-zero');
            
            % Zero scalar input
            zeroScalar = 0;
            diagnostics = matrixdiagnostics(zeroScalar);
            obj.assertEqual(diagnostics.Rank, 0, 'Zero scalar rank should be 0');
            
            % Vector inputs
            rowVector = [1, 2, 3, 4];
            colVector = [1; 2; 3; 4];
            
            rowDiag = matrixdiagnostics(rowVector);
            colDiag = matrixdiagnostics(colVector);
            
            obj.assertTrue(rowDiag.IsVector, 'Row vector should be detected as vector');
            obj.assertTrue(colDiag.IsVector, 'Column vector should be detected as vector');
            
            obj.assertEqual(rowDiag.Dimensions, [1, 4], 'Row vector dimensions should be correct');
            obj.assertEqual(colDiag.Dimensions, [4, 1], 'Column vector dimensions should be correct');
            
            % Large matrix (test performance and memory handling)
            % Use a moderately sized matrix to avoid slowing down tests
            largeMatrix = rand(50, 50);
            
            % Use a try-catch to ensure this doesn't fail due to memory issues
            try
                % Set options to avoid eigenvalue computation for large matrix
                options = struct('ComputeEigenvalues', false);
                diagnostics = matrixdiagnostics(largeMatrix, options);
                
                % Verify basic properties
                obj.assertEqual(diagnostics.Dimensions, [50, 50], 'Large matrix dimensions should be correct');
                obj.assertTrue(diagnostics.IsSquare, 'Large matrix should be detected as square');
            catch ME
                obj.assertTrue(false, ['Large matrix test failed: ', ME.message]);
            end
        end
        
        function testInvalidInputs(obj)
            % Tests that matrixdiagnostics correctly handles invalid inputs
            
            % Test with non-numeric input (cell array)
            cellInput = {1, 2; 3, 4};
            
            % Should throw an appropriate error
            obj.assertThrows(@() matrixdiagnostics(cellInput), 'MATRIXDIAGNOSTICS:NonNumericInput', ...
                'Non-numeric input should throw an error');
            
            % Test with struct input
            structInput = struct('a', 1, 'b', 2);
            
            % Should throw an appropriate error
            obj.assertThrows(@() matrixdiagnostics(structInput), 'MATRIXDIAGNOSTICS:NonNumericInput', ...
                'Struct input should throw an error');
            
            % Test with no input (should throw an error)
            obj.assertThrows(@() matrixdiagnostics(), 'MATRIXDIAGNOSTICS:InvalidInput', ...
                'No input should throw an error');
            
            % Test with invalid options
            invalidOptions = 'not a struct';
            
            % Should throw an appropriate error
            obj.assertThrows(@() matrixdiagnostics(obj.testMatrix, invalidOptions), 'MATRIXDIAGNOSTICS:InvalidOptions', ...
                'Invalid options should throw an error');
        end
    end
end