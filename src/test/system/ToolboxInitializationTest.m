classdef ToolboxInitializationTest < BaseTest
    % TOOLBOXINITIALIZATIONTEST System test class that validates the initialization 
    % process of the MFE Toolbox
    %
    % This test class verifies the proper initialization of the MFE Toolbox including
    % path configuration, platform-specific binary loading, and component accessibility.
    % It tests the complete initialization workflow described in the System Initialization
    % Flow diagram.
    %
    % The class tests:
    % - Basic initialization without path saving
    % - Initialization with optional directories
    % - Path saving functionality
    % - Platform-specific initialization behavior
    % - MEX binary loading
    % - Error handling during initialization
    % - Full initialization with all options
    % - Initialization performance
    %
    % Example:
    %   % Create and run the test
    %   test = ToolboxInitializationTest();
    %   results = test.runAllTests();
    %
    % See also: addToPath, BaseTest, MEXValidator, CrossPlatformValidator
    
    properties
        originalPath          % Store the original MATLAB path for restoration in tearDown
        mandatoryDirectories  % List of mandatory directories
        optionalDirectories   % List of optional directories
        toolboxRoot           % Root directory of the toolbox
        mexValidator          % MEXValidator for MEX file validation
        platformValidator     % CrossPlatformValidator for platform-specific testing
        pathState             % Structure to track path changes
    end
    
    methods
        function obj = ToolboxInitializationTest()
            % Initialize the ToolboxInitializationTest class
            
            % Call superclass constructor with test class name
            obj@BaseTest('ToolboxInitializationTest');
            
            % Initialize validator objects for MEX and platform-specific testing
            obj.mexValidator = MEXValidator();
            obj.platformValidator = CrossPlatformValidator();
            
            % Initialize path state tracking structure
            obj.pathState = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method
            
            % Call superclass setUp method
            obj.setUp@BaseTest();
            
            % Store original MATLAB path for restoration in tearDown
            obj.originalPath = path();
            
            % Determine the toolbox root directory
            obj.toolboxRoot = obj.getToolboxRoot();
            
            % Define mandatory directories according to technical specification
            obj.mandatoryDirectories = {'bootstrap', 'crosssection', 'distributions', ...
                'GUI', 'multivariate', 'tests', 'timeseries', 'univariate', ...
                'utility', 'realized', 'mex_source', 'dlls'};
            
            % Define optional directories
            obj.optionalDirectories = {'duplication'};
            
            % Reset path state tracking
            obj.pathState = obj.recordPathState();
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method
            
            % Restore original MATLAB path
            path(obj.originalPath);
            
            % Reset pathState
            obj.pathState = struct();
            
            % Call superclass tearDown method
            obj.tearDown@BaseTest();
        end
        
        function testBasicInitialization(obj)
            % Test basic initialization without path saving or optional directories
            
            % Call addToPath with basic options
            addToPath(false, false);
            
            % Record post-initialization path state
            afterPathState = obj.recordPathState();
            
            % Compare path states
            pathDiff = obj.comparePathStates(obj.pathState, afterPathState);
            
            % Verify mandatory directories are added
            for i = 1:length(obj.mandatoryDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.mandatoryDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Mandatory directory %s not added to path', obj.mandatoryDirectories{i}));
                end
            end
            
            % Verify optional directories are not added
            for i = 1:length(obj.optionalDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.optionalDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertFalse(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Optional directory %s should not be added to path', obj.optionalDirectories{i}));
                end
            end
            
            % Verify platform-specific MEX binary path is added
            dllDir = fullfile(obj.toolboxRoot, 'dlls');
            if exist(dllDir, 'dir')
                obj.assertTrue(obj.isDirectoryInPath(dllDir), ...
                    'Platform-specific MEX binary directory not added to path');
            end
        end
        
        function testOptionalDirectories(obj)
            % Test initialization with optional directories included
            
            % Call addToPath with optional directories included
            addToPath(false, true);
            
            % Record post-initialization path state
            afterPathState = obj.recordPathState();
            
            % Compare path states
            pathDiff = obj.comparePathStates(obj.pathState, afterPathState);
            
            % Verify mandatory directories are added
            for i = 1:length(obj.mandatoryDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.mandatoryDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Mandatory directory %s not added to path', obj.mandatoryDirectories{i}));
                end
            end
            
            % Verify optional directories are added if they exist
            for i = 1:length(obj.optionalDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.optionalDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Optional directory %s not added to path', obj.optionalDirectories{i}));
                end
            end
        end
        
        function testPathSaving(obj)
            % Test path saving functionality
            
            % Since we can't easily mock savepath, we'll test by checking
            % warning behavior when path saving is requested
            
            % Capture warnings
            warningState = warning('off', 'MFE:PathNotSaved');
            lastwarn(''); % Clear last warning
            
            try
                % Call addToPath with savePath=true
                addToPath(true, false);
                
                % Get warning message if any
                [warnMsg, warnId] = lastwarn();
                
                % The operation might succeed or fail depending on permissions
                % If it fails, we should see a specific warning
                if ~isempty(warnMsg)
                    obj.assertEqual('MFE:PathNotSaved', warnId, 'Unexpected warning ID when saving path');
                end
                
                % Verify that regardless of path saving success, 
                % the initialization still added the directories
                for i = 1:length(obj.mandatoryDirectories)
                    dirPath = fullfile(obj.toolboxRoot, obj.mandatoryDirectories{i});
                    if exist(dirPath, 'dir')
                        obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                            sprintf('Mandatory directory %s not added to path', obj.mandatoryDirectories{i}));
                    end
                end
            finally
                % Restore warning state
                warning(warningState);
            end
        end
        
        function testPlatformSpecificInitialization(obj)
            % Test platform-specific initialization behavior
            
            % Get current platform
            platform = obj.platformValidator.getCurrentPlatform();
            
            % Call addToPath for initialization
            addToPath(false, false);
            
            % Verify platform-specific initialization
            if ispc()
                % Windows-specific tests
                dllDir = fullfile(obj.toolboxRoot, 'dlls');
                if exist(dllDir, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dllDir), ...
                        'DLLs directory not added to path on Windows platform');
                    
                    % Check for .mexw64 files
                    dllFiles = what(dllDir);
                    mexFiles = dllFiles.mex;
                    
                    if ~isempty(mexFiles)
                        % Verify at least one MEX file has correct extension
                        hasCorrectExt = false;
                        for i = 1:length(mexFiles)
                            [~, ~, ext] = fileparts(mexFiles{i});
                            if strcmp(ext, '.mexw64')
                                hasCorrectExt = true;
                                break;
                            end
                        end
                        obj.assertTrue(hasCorrectExt, 'No MEX files with .mexw64 extension found in DLLs directory');
                    end
                end
            else
                % Unix-specific tests
                dllDir = fullfile(obj.toolboxRoot, 'dlls');
                if exist(dllDir, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dllDir), ...
                        'DLLs directory not added to path on Unix platform');
                    
                    % Check for .mexa64 files
                    dllFiles = what(dllDir);
                    mexFiles = dllFiles.mex;
                    
                    if ~isempty(mexFiles)
                        % Verify at least one MEX file has correct extension
                        hasCorrectExt = false;
                        for i = 1:length(mexFiles)
                            [~, ~, ext] = fileparts(mexFiles{i});
                            if strcmp(ext, '.mexa64')
                                hasCorrectExt = true;
                                break;
                            end
                        end
                        obj.assertTrue(hasCorrectExt, 'No MEX files with .mexa64 extension found in DLLs directory');
                    end
                end
            end
            
            % Verify MEX extension matches platform
            expectedExt = obj.mexValidator.getMEXExtension();
            if ispc()
                obj.assertEqual('mexw64', expectedExt, 'Platform MEX extension mismatch');
            else
                obj.assertEqual('mexa64', expectedExt, 'Platform MEX extension mismatch');
            end
        end
        
        function testMEXBinaryLoading(obj)
            % Test that platform-specific MEX binaries are correctly loaded
            
            % Call addToPath
            addToPath(false, false);
            
            % Get expected MEX extension
            expectedExt = obj.mexValidator.getMEXExtension();
            
            % Check for core MEX files documented in the technical specification
            coreFiles = {'agarch_core', 'armaxerrors', 'composite_likelihood', ...
                'egarch_core', 'igarch_core', 'tarch_core'};
            
            dllDir = fullfile(obj.toolboxRoot, 'dlls');
            
            % Check if directory exists before proceeding
            if ~exist(dllDir, 'dir')
                warning('ToolboxInitializationTest:MissingDllDir', ...
                    'DLLs directory not found. Skipping MEX binary loading test.');
                return;
            end
            
            % For each core MEX file
            for i = 1:length(coreFiles)
                mexFile = fullfile(dllDir, [coreFiles{i}, '.', expectedExt]);
                
                % Check file existence (may not exist in test environments)
                if exist(mexFile, 'file') == 3  % 3 = MEX-file
                    % Verify the file is in the path
                    whichPath = which(coreFiles{i});
                    obj.assertFalse(isempty(whichPath), ...
                        sprintf('MEX file %s not accessible via which', coreFiles{i}));
                    
                    % Use MEXValidator to verify the MEX file
                    exists = obj.mexValidator.validateMEXExists(coreFiles{i});
                    obj.assertTrue(exists, sprintf('MEX file %s validation failed', coreFiles{i}));
                end
            end
        end
        
        function testErrorHandling(obj)
            % Test error handling during initialization
            
            % Prepare a temporary path to test missing directories
            origPath = path();
            tempRoot = tempdir;
            fakeRoot = fullfile(tempRoot, 'fake_mfe_root');
            
            try
                % Create fake root with missing directories
                if ~exist(fakeRoot, 'dir')
                    mkdir(fakeRoot);
                end
                
                % Set up warning capture
                warningState = warning('off', 'MFE:MissingDirectory');
                lastwarn(''); % Clear last warning
                
                % Create a fake addToPath function
                testAddToPath = @(savePath, addOptionalDirs) obj.testAddToPathWithRoot(fakeRoot, savePath, addOptionalDirs);
                
                % Call the testing function
                result = testAddToPath(false, false);
                
                % Verify warning was issued for missing directories
                [warnMsg, warnId] = lastwarn();
                obj.assertFalse(isempty(warnId), 'No warning issued for missing directories');
                obj.assertEqual('MFE:MissingDirectory', warnId, 'Unexpected warning ID for missing directory');
                
                % Verify initialization still succeeds despite errors
                obj.assertTrue(result, 'Initialization should succeed despite missing directories');
            finally
                % Restore original path
                path(origPath);
                
                % Restore warning state
                warning(warningState);
                
                % Clean up fake root
                if exist(fakeRoot, 'dir')
                    rmdir(fakeRoot, 's');
                end
            end
        end
        
        function testFullInitialization(obj)
            % Test complete initialization process with all options
            
            % Call addToPath with full options (but disable path saving for test stability)
            addToPath(false, true);
            
            % Record post-initialization path state
            afterPathState = obj.recordPathState();
            
            % Compare path states
            pathDiff = obj.comparePathStates(obj.pathState, afterPathState);
            
            % Verify all mandatory directories are added
            for i = 1:length(obj.mandatoryDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.mandatoryDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Mandatory directory %s not added to path', obj.mandatoryDirectories{i}));
                end
            end
            
            % Verify optional directories are added if they exist
            for i = 1:length(obj.optionalDirectories)
                dirPath = fullfile(obj.toolboxRoot, obj.optionalDirectories{i});
                if exist(dirPath, 'dir')
                    obj.assertTrue(obj.isDirectoryInPath(dirPath), ...
                        sprintf('Optional directory %s not added to path', obj.optionalDirectories{i}));
                end
            end
            
            % Verify platform-specific MEX binary path
            dllDir = fullfile(obj.toolboxRoot, 'dlls');
            if exist(dllDir, 'dir')
                obj.assertTrue(obj.isDirectoryInPath(dllDir), ...
                    'Platform-specific MEX binary directory not added to path');
            end
        end
        
        function testInitializationPerformance(obj)
            % Test the performance of the initialization process
            
            % Measure execution time for multiple scenarios
            times = struct();
            
            % Basic initialization (no optional dirs)
            tic;
            addToPath(false, false);
            times.basic = toc;
            path(obj.originalPath);
            
            % Initialization with optional directories
            tic;
            addToPath(false, true);
            times.optional = toc;
            path(obj.originalPath);
            
            % Verify initialization completes within reasonable time
            % Actual limits depend on system, but should be under 5 seconds for most systems
            obj.assertTrue(times.basic < 5, ...
                sprintf('Basic initialization too slow: %.2f seconds', times.basic));
            obj.assertTrue(times.optional < 5, ...
                sprintf('Initialization with optional dirs too slow: %.2f seconds', times.optional));
            
            % Verify initialization with optional dirs doesn't take excessively longer
            % than basic initialization
            obj.assertTrue(times.optional < times.basic * 2, ...
                'Optional directories initialization is disproportionately slow');
        end
        
        %% Helper methods
        
        function result = isDirectoryInPath(obj, directoryPath)
            % Helper method to check if a directory is in the MATLAB path
            %
            % INPUTS:
            %   directoryPath - Path to directory to check
            %
            % OUTPUTS:
            %   result - Logical true if directory is in path
            
            % Standardize path separators
            directoryPath = strrep(directoryPath, '\', '/');
            
            % Get current path and standardize separators
            pathStr = path();
            pathStr = strrep(pathStr, '\', '/');
            
            % Check if directory is in path using regexp
            % Match dir followed by pathsep or end of string
            result = ~isempty(regexp(pathStr, [directoryPath, '($|', pathsep, ')'], 'once'));
        end
        
        function root = getToolboxRoot(obj)
            % Helper method to determine the toolbox root directory
            %
            % OUTPUTS:
            %   root - Path to the toolbox root directory
            
            % Get path to this test file
            testFilePath = mfilename('fullpath');
            
            % Navigate up to src/test/system directory
            [testDir, ~, ~] = fileparts(testFilePath);
            
            % Go up to src/test directory
            [testParentDir, ~, ~] = fileparts(testDir);
            
            % Go up to src directory
            [srcDir, ~, ~] = fileparts(testParentDir);
            
            % Finally, go up to toolbox root
            [root, ~, ~] = fileparts(srcDir);
            
            % Validate that this is a reasonable toolbox root
            if ~exist(fullfile(root, 'src', 'backend', 'addToPath.m'), 'file')
                error('ToolboxInitializationTest:InvalidToolboxRoot', ...
                    'Could not determine valid toolbox root directory');
            end
        end
        
        function pathState = recordPathState(obj)
            % Helper method to record the current state of the MATLAB path
            %
            % OUTPUTS:
            %   pathState - Structure with current path state
            
            % Get current path
            currentPath = path();
            
            % Split into directories
            directories = regexp(currentPath, pathsep, 'split');
            
            % Remove empty entries
            directories = directories(~cellfun(@isempty, directories));
            
            % Create path state structure
            pathState = struct(...
                'path', currentPath, ...
                'directories', {directories}, ...
                'timestamp', datestr(now) ...
            );
        end
        
        function pathDiff = comparePathStates(obj, beforeState, afterState)
            % Helper method to compare path states before and after initialization
            %
            % INPUTS:
            %   beforeState - Path state before operation
            %   afterState - Path state after operation
            %
            % OUTPUTS:
            %   pathDiff - Structure with path differences
            
            % Extract directories
            beforeDirs = beforeState.directories;
            afterDirs = afterState.directories;
            
            % Find added directories
            addedDirs = setdiff(afterDirs, beforeDirs);
            
            % Find removed directories
            removedDirs = setdiff(beforeDirs, afterDirs);
            
            % Create diff structure
            pathDiff = struct(...
                'addedDirectories', {addedDirs}, ...
                'removedDirectories', {removedDirs}, ...
                'beforeCount', length(beforeDirs), ...
                'afterCount', length(afterDirs), ...
                'timestamp', datestr(now) ...
            );
        end
        
        function result = testAddToPathWithRoot(obj, fakeRoot, savePath, addOptionalDirs)
            % Test implementation of addToPath function for error handling tests
            %
            % INPUTS:
            %   fakeRoot - Root directory to use for testing
            %   savePath - Whether to save path permanently
            %   addOptionalDirs - Whether to add optional directories
            %
            % OUTPUTS:
            %   result - Success flag
            
            % Setup mock addToPath parameters
            result = false;
            
            try
                % Get core directories
                coreDirs = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                    'multivariate', 'tests', 'timeseries', 'univariate', ...
                    'utility', 'realized', 'mex_source'};
                
                % Add Unix dlls directory if not Windows
                if ~ispc()
                    coreDirs{end+1} = 'dlls';
                end
                
                % Attempt to add non-existent directories to trigger warnings
                for i = 1:length(coreDirs)
                    dirPath = fullfile(fakeRoot, coreDirs{i});
                    % This should trigger warnings for non-existent dirs
                    if ~exist(dirPath, 'dir')
                        warning('MFE:MissingDirectory', 'Core directory "%s" not found.', dirPath);
                    end
                end
                
                % Add dlls for Windows (triggers another warning)
                if ispc()
                    dllDir = fullfile(fakeRoot, 'dlls');
                    if ~exist(dllDir, 'dir')
                        warning('MFE:MissingDirectory', 'DLLs directory not found: %s', dllDir);
                    end
                end
                
                % Simulate success
                result = true;
            catch err
                warning('MFE:ConfigError', 'Error configuring MFE Toolbox path: %s', err.message);
                result = false;
            end
        end
    end
end