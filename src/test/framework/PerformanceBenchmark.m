classdef PerformanceBenchmark
    % PERFORMANCEBENCHMARK Advanced performance benchmarking utility for the MFE Toolbox
    %
    % The PerformanceBenchmark class provides comprehensive benchmarking capabilities
    % for measuring and analyzing performance across the MFE Toolbox, with particular
    % focus on validating MEX optimization efficiency against MATLAB implementations.
    %
    % Key features:
    %   - Function execution time measurement over multiple iterations
    %   - Memory usage analysis
    %   - Implementation comparison (MATLAB vs. MEX)
    %   - Scalability testing with increasing dataset sizes
    %   - Performance profiling and bottleneck identification
    %   - Statistical analysis of performance results
    %   - Visualization of benchmark results
    %
    % This class is central to validating the performance requirements specified
    % in the technical specification, particularly for verifying MEX optimization
    % efficiency and support for large-scale dataset processing.
    %
    % Example:
    %   % Create benchmark instance
    %   benchmark = PerformanceBenchmark();
    %
    %   % Compare MATLAB and MEX implementations
    %   results = benchmark.compareImplementations(@matlab_func, @mex_func, 100, data);
    %   
    %   % Test scalability with increasing data sizes
    %   sizes = {1000, 5000, 10000, 50000};
    %   scaleResults = benchmark.compareScalability(@matlab_func, @mex_func, sizes, @generateData);
    %
    % See also: BaseTest, NumericalComparator, matrixdiagnostics
    
    properties
        NumericalComparator comparator  % For validating numerical results
        defaultIterations double        % Default number of benchmark iterations
        warmupIterations double         % Number of warmup iterations before timing
        speedupThreshold double         % Threshold for significant speedup (e.g., 1.5 for 50% improvement)
        lastBenchmarkResult struct      % Results from the most recent benchmark
        saveVisualizations logical      % Whether to save visualizations to disk
        visualizationPath char          % Path for saving visualizations
        verbose logical                 % Whether to print detailed output
    end
    
    methods
        % Constructor
        function obj = PerformanceBenchmark(options)
            % Constructor for PerformanceBenchmark class
            %
            % INPUTS:
            %   options - Structure with configuration options:
            %       .iterations - Default benchmark iterations [default: 100]
            %       .warmupIterations - Warmup iterations before timing [default: 5]
            %       .speedupThreshold - Speedup threshold for significance [default: 1.5]
            %       .saveVisualizations - Whether to save visualizations [default: false]
            %       .visualizationPath - Path for saving visualizations [default: pwd]
            %       .verbose - Detailed console output [default: false]
            %
            % OUTPUTS:
            %   obj - Initialized PerformanceBenchmark object
            
            % Create NumericalComparator instance for result validation
            obj.NumericalComparator = NumericalComparator();
            
            % Set defaultIterations to 100 or value from options
            obj.defaultIterations = 100;
            
            % Set warmupIterations to 5 or value from options
            obj.warmupIterations = 5;
            
            % Set speedupThreshold to 1.5 (50% improvement) or value from options
            obj.speedupThreshold = 1.5; % 50% improvement
            
            % Initialize empty lastBenchmarkResult structure
            obj.lastBenchmarkResult = struct();
            
            % Set saveVisualizations to false or value from options
            obj.saveVisualizations = false;
            
            % Set visualizationPath to current directory or value from options
            obj.visualizationPath = pwd;
            
            % Set verbose to false or value from options
            obj.verbose = false;
            
            % Apply custom options if provided
            if nargin > 0 && isstruct(options)
                if isfield(options, 'iterations')
                    obj.defaultIterations = options.iterations;
                end
                
                if isfield(options, 'warmupIterations')
                    obj.warmupIterations = options.warmupIterations;
                end
                
                if isfield(options, 'speedupThreshold')
                    obj.speedupThreshold = options.speedupThreshold;
                end
                
                if isfield(options, 'saveVisualizations')
                    obj.saveVisualizations = options.saveVisualizations;
                end
                
                if isfield(options, 'visualizationPath')
                    obj.visualizationPath = options.visualizationPath;
                end
                
                if isfield(options, 'verbose')
                    obj.verbose = options.verbose;
                end
            end
        end
        
        % Main benchmarking functions
        function results = benchmarkFunction(obj, func, iterations, varargin)
            % Measures the execution time of a function over multiple iterations
            %
            % INPUTS:
            %   func - Function handle to benchmark
            %   iterations - Number of iterations (optional, defaults to obj.defaultIterations)
            %   varargin - Arguments to pass to the function
            %
            % OUTPUTS:
            %   results - Struct with benchmark results including execution times and statistics
            
            % Validate input parameters (function handle, iterations)
            if ~isa(func, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First argument must be a function handle');
            end
            
            % Set iterations to defaultIterations if not specified
            if nargin < 3 || isempty(iterations)
                iterations = obj.defaultIterations;
            end
            
            % Initialize results structure
            results = struct(...
                'function', func2str(func), ...
                'iterations', iterations, ...
                'executionTimes', zeros(iterations, 1), ...
                'mean', 0, ...
                'median', 0, ...
                'min', 0, ...
                'max', 0, ...
                'std', 0, ...
                'totalTime', 0, ...
                'timestamp', now, ...
                'arguments', {varargin} ...
            );
            
            % Perform warmup runs to eliminate JIT compilation effects
            if obj.verbose
                disp(['Performing ', num2str(obj.warmupIterations), ' warmup iterations...']);
            end
            
            for i = 1:obj.warmupIterations
                func(varargin{:});
            end
            
            % Execute function for specified number of iterations with timing
            if obj.verbose
                disp(['Running ', num2str(iterations), ' benchmark iterations...']);
            end
            
            % Time each iteration
            for i = 1:iterations
                tic;
                func(varargin{:});
                results.executionTimes(i) = toc;
            end
            
            % Calculate statistics: mean, median, min, max, std of execution times
            results.mean = mean(results.executionTimes);
            results.median = median(results.executionTimes);
            results.min = min(results.executionTimes);
            results.max = max(results.executionTimes);
            results.std = std(results.executionTimes);
            results.totalTime = sum(results.executionTimes);
            
            % Store results in lastBenchmarkResult structure
            obj.lastBenchmarkResult = results;
            
            % Display results summary if verbose is true
            if obj.verbose
                disp('Benchmark results summary:');
                disp(['  Function: ', results.function]);
                disp(['  Total iterations: ', num2str(results.iterations)]);
                disp(['  Mean execution time: ', num2str(results.mean * 1000), ' ms']);
                disp(['  Median execution time: ', num2str(results.median * 1000), ' ms']);
                disp(['  Min execution time: ', num2str(results.min * 1000), ' ms']);
                disp(['  Max execution time: ', num2str(results.max * 1000), ' ms']);
                disp(['  Std dev: ', num2str(results.std * 1000), ' ms']);
                disp(['  Total time: ', num2str(results.totalTime), ' s']);
            end
            
            % Return comprehensive benchmark results structure
        end
        
        function results = compareImplementations(obj, func1, func2, iterations, varargin)
            % Compares the performance of two different implementations (e.g., MATLAB vs. MEX)
            %
            % INPUTS:
            %   func1 - Function handle for first implementation (e.g., MATLAB)
            %   func2 - Function handle for second implementation (e.g., MEX)
            %   iterations - Number of iterations (optional, defaults to obj.defaultIterations)
            %   varargin - Arguments to pass to both functions
            %
            % OUTPUTS:
            %   results - Struct with comparison results including speedup ratio and execution statistics
            
            % Validate input parameters (function handles, iterations)
            if ~isa(func1, 'function_handle') || ~isa(func2, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First two arguments must be function handles');
            end
            
            % Set iterations to defaultIterations if not specified
            if nargin < 4 || isempty(iterations)
                iterations = obj.defaultIterations;
            end
            
            % Initialize results structure
            results = struct(...
                'function1', func2str(func1), ...
                'function2', func2str(func2), ...
                'iterations', iterations, ...
                'results1', [], ...
                'results2', [], ...
                'speedupRatio', 0, ...
                'improvementPercent', 0, ...
                'meetsThreshold', false, ...
                'outputsMatch', false, ...
                'timestamp', now, ...
                'arguments', {varargin} ...
            );
            
            % Benchmark first implementation using benchmarkFunction
            if obj.verbose
                disp(['Benchmarking ', func2str(func1), '...']);
            end
            results.results1 = obj.benchmarkFunction(func1, iterations, varargin{:});
            
            % Benchmark second implementation using benchmarkFunction
            if obj.verbose
                disp(['Benchmarking ', func2str(func2), '...']);
            end
            results.results2 = obj.benchmarkFunction(func2, iterations, varargin{:});
            
            % Validate outputs match between implementations using NumericalComparator
            try
                out1 = func1(varargin{:});
                out2 = func2(varargin{:});
                
                % Compare outputs if both returned values
                if ~isempty(out1) && ~isempty(out2)
                    % Use NumericalComparator to check if outputs match
                    if isnumeric(out1) && isnumeric(out2)
                        if isscalar(out1) && isscalar(out2)
                            compResult = obj.NumericalComparator.compareScalars(out1, out2);
                            results.outputsMatch = compResult.isEqual;
                        else
                            compResult = obj.NumericalComparator.compareMatrices(out1, out2);
                            results.outputsMatch = compResult.isEqual;
                        end
                    else
                        % For non-numeric outputs, use isequal
                        results.outputsMatch = isequal(out1, out2);
                    end
                    
                    if ~results.outputsMatch && obj.verbose
                        warning('PerformanceBenchmark:OutputMismatch', 'Outputs from the two implementations do not match');
                    end
                end
            catch ME
                warning('PerformanceBenchmark:ValidationError', ...
                    'Error while validating implementation outputs: %s', ME.message);
            end
            
            % Calculate speedup ratio (time1/time2) and performance improvement percentage
            results.speedupRatio = results.results1.mean / results.results2.mean;
            results.improvementPercent = (results.speedupRatio - 1) * 100;
            
            % Determine if speedup meets or exceeds speedupThreshold
            results.meetsThreshold = results.speedupRatio >= obj.speedupThreshold;
            
            % Store comparison results in lastBenchmarkResult structure
            obj.lastBenchmarkResult = results;
            
            % Display comparison summary if verbose is true
            if obj.verbose
                disp('Implementation comparison results:');
                disp(['  Function 1: ', results.function1]);
                disp(['  Function 2: ', results.function2]);
                disp(['  Function 1 mean time: ', num2str(results.results1.mean * 1000), ' ms']);
                disp(['  Function 2 mean time: ', num2str(results.results2.mean * 1000), ' ms']);
                disp(['  Speedup ratio: ', num2str(results.speedupRatio), 'x']);
                disp(['  Improvement: ', num2str(results.improvementPercent), '%']);
                disp(['  Meets threshold (', num2str((obj.speedupThreshold-1)*100), '%): ', ...
                    num2str(results.meetsThreshold)]);
                disp(['  Outputs match: ', num2str(results.outputsMatch)]);
            end
            
            % Return comprehensive comparison results structure
        end
        
        function results = measureMemoryUsage(obj, func, varargin)
            % Measures memory usage during function execution
            %
            % INPUTS:
            %   func - Function handle to measure
            %   varargin - Arguments to pass to the function
            %
            % OUTPUTS:
            %   results - Struct with memory usage statistics including before, after, and peak usage
            
            % Validate input parameter (function handle)
            if ~isa(func, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First argument must be a function handle');
            end
            
            % Initialize results structure
            results = struct(...
                'function', func2str(func), ...
                'beforeBytes', 0, ...
                'afterBytes', 0, ...
                'peakBytes', 0, ...
                'netChange', 0, ...
                'netChangeMB', 0, ...
                'timestamp', now, ...
                'arguments', {varargin} ...
            );
            
            % Record baseline memory usage using whos
            baselineVars = whos();
            results.beforeBytes = sum([baselineVars.bytes]);
            
            % Execute function with provided arguments
            try
                if nargout(func) > 0
                    returnVal = func(varargin{:});
                    
                    % Add size of return value to the measurement
                    returnVars = whos('returnVal');
                    returnBytes = sum([returnVars.bytes]);
                else
                    func(varargin{:});
                    returnBytes = 0;
                end
            catch ME
                warning('PerformanceBenchmark:ExecutionError', ...
                    'Error during function execution: %s', ME.message);
                rethrow(ME);
            end
            
            % Record post-execution memory usage
            afterVars = whos();
            results.afterBytes = sum([afterVars.bytes]);
            
            % Calculate memory difference (allocated, freed, net change)
            results.netChange = results.afterBytes - results.beforeBytes;
            results.netChangeMB = results.netChange / (1024^2);
            
            % Analyze memory usage pattern and efficiency
            
            % Store memory usage statistics in result structure
            obj.lastBenchmarkResult = results;
            
            % Display memory usage summary if verbose is true
            if obj.verbose
                disp('Memory usage results:');
                disp(['  Function: ', results.function]);
                disp(['  Before: ', num2str(results.beforeBytes / (1024^2)), ' MB']);
                disp(['  After: ', num2str(results.afterBytes / (1024^2)), ' MB']);
                disp(['  Net change: ', num2str(results.netChangeMB), ' MB']);
            end
            
            % Return comprehensive memory usage results
        end
        
        function results = compareMemoryUsage(obj, func1, func2, varargin)
            % Compares memory usage between two different implementations
            %
            % INPUTS:
            %   func1 - Function handle for first implementation
            %   func2 - Function handle for second implementation
            %   varargin - Arguments to pass to both functions
            %
            % OUTPUTS:
            %   results - Struct with comparison of memory usage between implementations
            
            % Validate input parameters (function handles)
            if ~isa(func1, 'function_handle') || ~isa(func2, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First two arguments must be function handles');
            end
            
            % Initialize results structure
            results = struct(...
                'function1', func2str(func1), ...
                'function2', func2str(func2), ...
                'results1', [], ...
                'results2', [], ...
                'memoryRatio', 0, ...
                'efficiencyPercent', 0, ...
                'outputsMatch', false, ...
                'timestamp', now, ...
                'arguments', {varargin} ...
            );
            
            % Measure memory usage of first implementation using measureMemoryUsage
            if obj.verbose
                disp(['Measuring memory usage for ', func2str(func1), '...']);
            end
            results.results1 = obj.measureMemoryUsage(func1, varargin{:});
            
            % Measure memory usage of second implementation using measureMemoryUsage
            if obj.verbose
                disp(['Measuring memory usage for ', func2str(func2), '...']);
            end
            results.results2 = obj.measureMemoryUsage(func2, varargin{:});
            
            % Validate outputs match between implementations using NumericalComparator
            try
                out1 = func1(varargin{:});
                out2 = func2(varargin{:});
                
                % Compare outputs if both returned values
                if ~isempty(out1) && ~isempty(out2)
                    % Use NumericalComparator to check if outputs match
                    if isnumeric(out1) && isnumeric(out2)
                        if isscalar(out1) && isscalar(out2)
                            compResult = obj.NumericalComparator.compareScalars(out1, out2);
                            results.outputsMatch = compResult.isEqual;
                        else
                            compResult = obj.NumericalComparator.compareMatrices(out1, out2);
                            results.outputsMatch = compResult.isEqual;
                        end
                    else
                        % For non-numeric outputs, use isequal
                        results.outputsMatch = isequal(out1, out2);
                    end
                    
                    if ~results.outputsMatch && obj.verbose
                        warning('PerformanceBenchmark:OutputMismatch', 'Outputs from the two implementations do not match');
                    end
                end
            catch ME
                warning('PerformanceBenchmark:ValidationError', ...
                    'Error while validating implementation outputs: %s', ME.message);
            end
            
            % Calculate memory efficiency ratio and percentage difference
            if results.results2.netChange ~= 0
                results.memoryRatio = results.results1.netChange / results.results2.netChange;
                results.efficiencyPercent = (1 - (results.results2.netChange / results.results1.netChange)) * 100;
            else
                if results.results1.netChange == 0
                    results.memoryRatio = 1;
                    results.efficiencyPercent = 0;
                else
                    results.memoryRatio = Inf;
                    results.efficiencyPercent = 100;
                end
            end
            
            % Analyze relative memory efficiency between implementations
            
            % Store comparison results in result structure
            obj.lastBenchmarkResult = results;
            
            % Display memory comparison summary if verbose is true
            if obj.verbose
                disp('Memory usage comparison results:');
                disp(['  Function 1: ', results.function1]);
                disp(['  Function 2: ', results.function2]);
                disp(['  Function 1 memory change: ', num2str(results.results1.netChangeMB), ' MB']);
                disp(['  Function 2 memory change: ', num2str(results.results2.netChangeMB), ' MB']);
                disp(['  Memory efficiency ratio: ', num2str(results.memoryRatio), 'x']);
                disp(['  Efficiency improvement: ', num2str(results.efficiencyPercent), '%']);
                disp(['  Outputs match: ', num2str(results.outputsMatch)]);
            end
            
            % Return comprehensive memory comparison results
        end
        
        function results = scalabilityTest(obj, func, dataSizes, dataGenerator)
            % Tests how performance scales with increasing data sizes
            %
            % INPUTS:
            %   func - Function handle to test
            %   dataSizes - Cell array of data sizes to test
            %   dataGenerator - Function handle that generates test data given a size
            %
            % OUTPUTS:
            %   results - Struct with scalability test results with execution times per data size
            
            % Validate input parameters (function handle, data sizes, generator)
            if ~isa(func, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First argument must be a function handle');
            end
            
            if ~iscell(dataSizes)
                error('PerformanceBenchmark:InvalidInput', 'Data sizes must be provided as a cell array');
            end
            
            if ~isa(dataGenerator, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'Data generator must be a function handle');
            end
            
            % Initialize results structure to store performance by data size
            results = struct(...
                'function', func2str(func), ...
                'dataSizes', {dataSizes}, ...
                'timings', zeros(length(dataSizes), 1), ...
                'dataInfo', cell(length(dataSizes), 1), ...
                'scalingBehavior', '', ...
                'timestamp', now ...
            );
            
            % For each data size, generate test data using dataGenerator
            for i = 1:length(dataSizes)
                size = dataSizes{i};
                
                if obj.verbose
                    disp(['Testing size: ', num2str(size)]);
                end
                
                % Generate data for this size
                data = dataGenerator(size);
                
                % Record data information
                dataInfo = whos('data');
                results.dataInfo{i} = dataInfo;
                
                % Benchmark function with generated data using benchmarkFunction
                benchResult = obj.benchmarkFunction(func, obj.defaultIterations, data);
                
                % Store execution time for current data size
                results.timings(i) = benchResult.mean;
            end
            
            % Analyze scaling behavior (linear, quadratic, logarithmic)
            logSizes = log(cell2mat(dataSizes(:)));
            logTimings = log(results.timings);
            
            % Fit linear model to log-log data to determine scaling behavior
            p = polyfit(logSizes, logTimings, 1);
            scalingExponent = p(1);
            
            % Determine scaling behavior based on exponent
            if scalingExponent < 0.2
                results.scalingBehavior = 'Constant';
            elseif scalingExponent < 0.8
                results.scalingBehavior = 'Sublinear';
            elseif scalingExponent < 1.2
                results.scalingBehavior = 'Linear';
            elseif scalingExponent < 1.8
                results.scalingBehavior = 'Superlinear';
            elseif scalingExponent < 2.2
                results.scalingBehavior = 'Quadratic';
            else
                results.scalingBehavior = 'Polynomial';
            end
            
            results.scalingExponent = scalingExponent;
            
            % Generate scalability visualization if saveVisualizations is true
            if obj.saveVisualizations
                figH = obj.visualizeResults(results, 'scalability', ['Scalability Test: ', func2str(func)]);
                
                % Save figure if needed
                if obj.saveVisualizations
                    figFileName = fullfile(obj.visualizationPath, ...
                        ['scalability_', func2str(func), '_', datestr(now, 'yyyymmdd_HHMMSS'), '.fig']);
                    savefig(figH, figFileName);
                    
                    if obj.verbose
                        disp(['Saved visualization to: ', figFileName]);
                    end
                end
            end
            
            % Store as last result
            obj.lastBenchmarkResult = results;
            
            % Display summary if verbose
            if obj.verbose
                disp('Scalability test results:');
                disp(['  Function: ', results.function]);
                disp(['  Scaling behavior: ', results.scalingBehavior, ' (exponent = ', num2str(scalingExponent), ')']);
                disp('  Execution times:');
                for i = 1:length(dataSizes)
                    disp(['    Size ', num2str(dataSizes{i}), ': ', ...
                        num2str(results.timings(i) * 1000), ' ms']);
                end
            end
            
            % Return comprehensive scalability results with analysis
        end
        
        function results = compareScalability(obj, func1, func2, dataSizes, dataGenerator)
            % Compares scalability of two implementations with increasing data sizes
            %
            % INPUTS:
            %   func1 - Function handle for first implementation
            %   func2 - Function handle for second implementation
            %   dataSizes - Cell array of data sizes to test
            %   dataGenerator - Function handle that generates test data given a size
            %
            % OUTPUTS:
            %   results - Struct with comparative scalability results between implementations
            
            % Validate input parameters (function handles, data sizes, generator)
            if ~isa(func1, 'function_handle') || ~isa(func2, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First two arguments must be function handles');
            end
            
            if ~iscell(dataSizes)
                error('PerformanceBenchmark:InvalidInput', 'Data sizes must be provided as a cell array');
            end
            
            if ~isa(dataGenerator, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'Data generator must be a function handle');
            end
            
            % Initialize results structure for comparative scalability analysis
            results = struct(...
                'function1', func2str(func1), ...
                'function2', func2str(func2), ...
                'dataSizes', {dataSizes}, ...
                'timings1', zeros(length(dataSizes), 1), ...
                'timings2', zeros(length(dataSizes), 1), ...
                'speedupRatios', zeros(length(dataSizes), 1), ...
                'dataInfo', cell(length(dataSizes), 1), ...
                'scalingBehavior1', '', ...
                'scalingBehavior2', '', ...
                'speedupBehavior', '', ...
                'timestamp', now ...
            );
            
            % For each data size, generate test data using dataGenerator
            for i = 1:length(dataSizes)
                size = dataSizes{i};
                
                if obj.verbose
                    disp(['Testing size: ', num2str(size)]);
                end
                
                % Generate test data for this size
                data = dataGenerator(size);
                
                % Record data information
                dataInfo = whos('data');
                results.dataInfo{i} = dataInfo;
                
                % Benchmark both functions with the same generated data
                if obj.verbose
                    disp(['  Benchmarking ', func2str(func1), '...']);
                end
                benchResult1 = obj.benchmarkFunction(func1, obj.defaultIterations, data);
                
                if obj.verbose
                    disp(['  Benchmarking ', func2str(func2), '...']);
                end
                benchResult2 = obj.benchmarkFunction(func2, obj.defaultIterations, data);
                
                % Record mean execution times
                results.timings1(i) = benchResult1.mean;
                results.timings2(i) = benchResult2.mean;
                
                % Calculate speedup ratio for each data size
                results.speedupRatios(i) = results.timings1(i) / results.timings2(i);
            end
            
            % Analyze how speedup changes with increasing data size
            logSizes = log(cell2mat(dataSizes(:)));
            logTimings1 = log(results.timings1);
            logTimings2 = log(results.timings2);
            
            % Fit linear models to log-log data
            p1 = polyfit(logSizes, logTimings1, 1);
            p2 = polyfit(logSizes, logTimings2, 1);
            scalingExponent1 = p1(1);
            scalingExponent2 = p2(1);
            
            % Determine scaling behavior for each function
            results.scalingExponent1 = scalingExponent1;
            results.scalingExponent2 = scalingExponent2;
            
            % Map scaling exponents to behavior descriptions
            behaviorsMap = containers.Map();
            behaviorsMap([-Inf, 0.2]) = 'Constant';
            behaviorsMap([0.2, 0.8]) = 'Sublinear';
            behaviorsMap([0.8, 1.2]) = 'Linear';
            behaviorsMap([1.2, 1.8]) = 'Superlinear';
            behaviorsMap([1.8, 2.2]) = 'Quadratic';
            behaviorsMap([2.2, Inf]) = 'Polynomial';
            
            % Get keys as cell array and convert to array of start values
            keyArray = cell2mat(keys(behaviorsMap));
            keyStarts = keyArray(:,1);
            
            % Find appropriate behavior for each exponent
            [~, idx1] = max(keyStarts(keyStarts <= scalingExponent1));
            [~, idx2] = max(keyStarts(keyStarts <= scalingExponent2));
            
            % Get range for each index
            range1 = [keyStarts(idx1), keyArray(idx1, 2)];
            range2 = [keyStarts(idx2), keyArray(idx2, 2)];
            
            results.scalingBehavior1 = behaviorsMap(range1);
            results.scalingBehavior2 = behaviorsMap(range2);
            
            % Analyze speedup trend
            logSpeedups = log(results.speedupRatios);
            pSpeedup = polyfit(logSizes, logSpeedups, 1);
            speedupExponent = pSpeedup(1);
            results.speedupExponent = speedupExponent;
            
            % Determine speedup behavior
            if abs(speedupExponent) < 0.1
                results.speedupBehavior = 'Constant speedup';
            elseif speedupExponent > 0
                results.speedupBehavior = 'Increasing speedup with size';
            else
                results.speedupBehavior = 'Decreasing speedup with size';
            end
            
            % Generate comparative scalability visualization if saveVisualizations is true
            if obj.saveVisualizations
                figH = obj.visualizeResults(results, 'comparison', ...
                    ['Scalability Comparison: ', func2str(func1), ' vs. ', func2str(func2)]);
                
                % Save figure if needed
                if obj.saveVisualizations
                    figFileName = fullfile(obj.visualizationPath, ...
                        ['scalability_comparison_', datestr(now, 'yyyymmdd_HHMMSS'), '.fig']);
                    savefig(figH, figFileName);
                    
                    if obj.verbose
                        disp(['Saved visualization to: ', figFileName]);
                    end
                end
            end
            
            % Store as last result
            obj.lastBenchmarkResult = results;
            
            % Display summary if verbose
            if obj.verbose
                disp('Scalability comparison results:');
                disp(['  Function 1: ', results.function1, ' (', results.scalingBehavior1, ...
                    ', exponent = ', num2str(scalingExponent1), ')']);
                disp(['  Function 2: ', results.function2, ' (', results.scalingBehavior2, ...
                    ', exponent = ', num2str(scalingExponent2), ')']);
                disp(['  Speedup behavior: ', results.speedupBehavior, ...
                    ' (exponent = ', num2str(speedupExponent), ')']);
                disp('  Results by size:');
                for i = 1:length(dataSizes)
                    disp(['    Size ', num2str(dataSizes{i}), ':']);
                    disp(['      Function 1: ', num2str(results.timings1(i) * 1000), ' ms']);
                    disp(['      Function 2: ', num2str(results.timings2(i) * 1000), ' ms']);
                    disp(['      Speedup: ', num2str(results.speedupRatios(i)), 'x']);
                end
            end
            
            % Return comprehensive comparative scalability results with analysis
        end
        
        function results = profileOperation(obj, func, checkpoints, varargin)
            % Creates a detailed performance profile of a complex operation
            %
            % INPUTS:
            %   func - Function handle to profile
            %   checkpoints - Cell array of checkpoint names
            %   varargin - Arguments to pass to the function
            %
            % OUTPUTS:
            %   results - Struct with detailed performance profile with timing at each checkpoint
            
            % Validate input parameters (function handle, checkpoints)
            if ~isa(func, 'function_handle')
                error('PerformanceBenchmark:InvalidInput', 'First argument must be a function handle');
            end
            
            if ~iscell(checkpoints)
                error('PerformanceBenchmark:InvalidInput', 'Checkpoints must be provided as a cell array');
            end
            
            % Initialize results structure
            results = struct(...
                'function', func2str(func), ...
                'checkpoints', {checkpoints}, ...
                'timing', zeros(length(checkpoints), 1), ...
                'percentage', zeros(length(checkpoints), 1), ...
                'cumulativeTime', zeros(length(checkpoints), 1), ...
                'totalTime', 0, ...
                'bottlenecks', [], ...
                'timestamp', now, ...
                'arguments', {varargin} ...
            );
            
            % Instrument function to record timing at specified checkpoints
            try
                % Start timing
                startTime = tic;
                lastCheckpointTime = startTime;
                
                % Execute instrumented function with provided arguments
                if nargout(func) > 0
                    [~, checkpointTimes] = func(varargin{:});
                else
                    checkpointTimes = func(varargin{:});
                end
                
                % Record execution time between each checkpoint
                results.totalTime = toc(startTime);
                
                % Process checkpoint times
                if ~isempty(checkpointTimes) && length(checkpointTimes) == length(checkpoints)
                    % Copy checkpoint times
                    for i = 1:length(checkpoints)
                        results.timing(i) = checkpointTimes(i);
                    end
                else
                    warning('PerformanceBenchmark:InvalidCheckpoints', ...
                        'Invalid checkpoint timing data returned from function');
                    return;
                end
            catch ME
                warning('PerformanceBenchmark:ProfileError', ...
                    'Error during profile execution: %s', ME.message);
                rethrow(ME);
            end
            
            % Calculate percentage of total time spent in each segment
            results.percentage = (results.timing / results.totalTime) * 100;
            results.cumulativeTime = cumsum(results.timing);
            
            % Identify performance bottlenecks based on timing distribution
            bottleneckIndices = find(results.percentage > 20);
            results.bottlenecks = checkpoints(bottleneckIndices);
            
            % Generate performance profile visualization if saveVisualizations is true
            if obj.saveVisualizations
                figH = obj.visualizeResults(results, 'profile', ['Performance Profile: ', func2str(func)]);
                
                % Save figure if needed
                if obj.saveVisualizations
                    figFileName = fullfile(obj.visualizationPath, ...
                        ['profile_', func2str(func), '_', datestr(now, 'yyyymmdd_HHMMSS'), '.fig']);
                    savefig(figH, figFileName);
                    
                    if obj.verbose
                        disp(['Saved visualization to: ', figFileName]);
                    end
                end
            end
            
            % Store as last result
            obj.lastBenchmarkResult = results;
            
            % Display summary if verbose
            if obj.verbose
                disp('Performance profile results:');
                disp(['  Function: ', results.function]);
                disp(['  Total execution time: ', num2str(results.totalTime * 1000), ' ms']);
                disp('  Checkpoint timings:');
                for i = 1:length(checkpoints)
                    disp(['    ', checkpoints{i}, ': ', ...
                        num2str(results.timing(i) * 1000), ' ms (', ...
                        num2str(results.percentage(i)), '%)']);
                end
                
                if ~isempty(results.bottlenecks)
                    disp('  Bottlenecks identified:');
                    for i = 1:length(results.bottlenecks)
                        disp(['    ', results.bottlenecks{i}]);
                    end
                else
                    disp('  No significant bottlenecks identified.');
                end
            end
            
            % Return detailed performance profile with analysis
        end
        
        function figHandle = visualizeResults(obj, results, visualizationType, title)
            % Creates visualizations of benchmark results
            %
            % INPUTS:
            %   results - Results structure from a benchmark operation
            %   visualizationType - Type of visualization to create
            %   title - Title for the visualization
            %
            % OUTPUTS:
            %   figHandle - Figure handle for the created visualization
            
            % Validate input parameters (results structure, visualization type)
            if ~isstruct(results)
                error('PerformanceBenchmark:InvalidInput', 'Results must be a structure');
            end
            
            % Create appropriate figure based on visualization type
            figHandle = figure;
            
            % Process based on visualization type
            switch lower(visualizationType)
                case 'comparison'
                    % For 'comparison' type, create bar chart comparing implementations
                    if isfield(results, 'function1') && isfield(results, 'function2')
                        if isfield(results, 'dataSizes')
                            % Scalability comparison
                            subplot(2, 1, 1);
                            semilogx(cell2mat(results.dataSizes), results.timings1 * 1000, 'b-o', ...
                                'LineWidth', 2, 'MarkerSize', 8);
                            hold on;
                            semilogx(cell2mat(results.dataSizes), results.timings2 * 1000, 'r-s', ...
                                'LineWidth', 2, 'MarkerSize', 8);
                            hold off;
                            grid on;
                            xlabel('Data Size');
                            ylabel('Execution Time (ms)');
                            legend(results.function1, results.function2, 'Location', 'NorthWest');
                            title('Execution Time vs. Data Size');
                            
                            subplot(2, 1, 2);
                            semilogx(cell2mat(results.dataSizes), results.speedupRatios, 'g-d', ...
                                'LineWidth', 2, 'MarkerSize', 8);
                            grid on;
                            xlabel('Data Size');
                            ylabel('Speedup Ratio');
                            yline(obj.speedupThreshold, 'r--', ['Threshold (' num2str(obj.speedupThreshold) 'x)'], ...
                                'LineWidth', 1.5);
                            title('Speedup Ratio vs. Data Size');
                        else
                            % Simple implementation comparison
                            data = [results.results1.mean, results.results2.mean] * 1000; % Convert to ms
                            bar(data);
                            grid on;
                            xlabel('Implementation');
                            ylabel('Execution Time (ms)');
                            set(gca, 'XTickLabel', {results.function1, results.function2});
                            title('Implementation Comparison');
                            
                            % Add text for speedup
                            text(1.5, mean(data), [num2str(results.speedupRatio, '%.2f'), 'x speedup'], ...
                                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                        end
                    else
                        warning('PerformanceBenchmark:InvalidVisualization', ...
                            'Results structure not suitable for comparison visualization.');
                    end
                    
                case 'timeseries'
                    % For 'timeseries' type, create line plot of execution times
                    if isfield(results, 'executionTimes')
                        plot(results.executionTimes * 1000, 'b-', 'LineWidth', 1.5);
                        grid on;
                        xlabel('Iteration');
                        ylabel('Execution Time (ms)');
                        yline(results.mean * 1000, 'r--', 'Mean', 'LineWidth', 1.5);
                        title('Execution Time per Iteration');
                    else
                        warning('PerformanceBenchmark:InvalidVisualization', ...
                            'Results structure not suitable for timeseries visualization.');
                    end
                    
                case 'distribution'
                    % For 'distribution' type, create box plot of timing distribution
                    if isfield(results, 'executionTimes')
                        boxplot(results.executionTimes * 1000);
                        grid on;
                        ylabel('Execution Time (ms)');
                        title('Execution Time Distribution');
                    else
                        warning('PerformanceBenchmark:InvalidVisualization', ...
                            'Results structure not suitable for distribution visualization.');
                    end
                    
                case 'scalability'
                    % For 'scalability' type, create line plot of execution time vs data size
                    if isfield(results, 'dataSizes') && isfield(results, 'timings')
                        semilogx(cell2mat(results.dataSizes), results.timings * 1000, 'b-o', ...
                            'LineWidth', 2, 'MarkerSize', 8);
                        grid on;
                        xlabel('Data Size');
                        ylabel('Execution Time (ms)');
                        title(['Scalability: ', results.scalingBehavior, ...
                            ' (exponent = ', num2str(results.scalingExponent, '%.2f'), ')']);
                    else
                        warning('PerformanceBenchmark:InvalidVisualization', ...
                            'Results structure not suitable for scalability visualization.');
                    end
                    
                case 'profile'
                    % For 'profile' type, create bar chart of performance profile
                    if isfield(results, 'checkpoints') && isfield(results, 'timing')
                        bar(results.percentage);
                        grid on;
                        xlabel('Checkpoint');
                        ylabel('Percentage of Total Time (%)');
                        set(gca, 'XTickLabel', results.checkpoints);
                        xtickangle(45);
                        title('Performance Profile: Time Spent per Checkpoint');
                    else
                        warning('PerformanceBenchmark:InvalidVisualization', ...
                            'Results structure not suitable for profile visualization.');
                    end
                    
                otherwise
                    warning('PerformanceBenchmark:InvalidVisualization', ...
                        'Unknown visualization type: %s', visualizationType);
            end
            
            % Add appropriate labels, legend, and title to visualization
            if nargin >= 4 && ~isempty(title)
                sgtitle(title);
            end
            
            % Save figure to visualizationPath if saveVisualizations is true
            if obj.saveVisualizations
                figFileName = fullfile(obj.visualizationPath, ...
                    ['benchmark_', visualizationType, '_', datestr(now, 'yyyymmdd_HHMMSS'), '.fig']);
                savefig(figHandle, figFileName);
                
                if obj.verbose
                    disp(['Saved visualization to: ', figFileName]);
                end
            end
            
            % Return handle to created figure
        end
        
        function result = getLastResult(obj)
            % Returns the results from the most recent benchmark operation
            %
            % OUTPUTS:
            %   result - Last benchmark result structure
            
            % Return the lastBenchmarkResult structure
            result = obj.lastBenchmarkResult;
            
            % Return empty structure if no benchmark has been run yet
        end
        
        % Configuration methods
        function setIterations(obj, iterations)
            % Sets the default number of iterations for benchmark operations
            %
            % INPUTS:
            %   iterations - Number of iterations
            
            % Validate iterations is a positive integer
            if ~isscalar(iterations) || iterations <= 0 || iterations ~= fix(iterations)
                error('PerformanceBenchmark:InvalidInput', 'Iterations must be a positive integer');
            end
            
            % Set defaultIterations property to specified value
            obj.defaultIterations = iterations;
        end
        
        function setWarmupIterations(obj, iterations)
            % Sets the number of warmup iterations to perform before timing
            %
            % INPUTS:
            %   iterations - Number of warmup iterations
            
            % Validate iterations is a non-negative integer
            if ~isscalar(iterations) || iterations < 0 || iterations ~= fix(iterations)
                error('PerformanceBenchmark:InvalidInput', 'Warmup iterations must be a non-negative integer');
            end
            
            % Set warmupIterations property to specified value
            obj.warmupIterations = iterations;
        end
        
        function setSpeedupThreshold(obj, threshold)
            % Sets the threshold for speedup ratio to be considered significant
            %
            % INPUTS:
            %   threshold - Speedup ratio threshold (e.g., 1.5 for 50% improvement)
            
            % Validate threshold is greater than 1.0
            if ~isscalar(threshold) || threshold <= 1
                error('PerformanceBenchmark:InvalidInput', 'Speedup threshold must be a scalar greater than 1.0');
            end
            
            % Set speedupThreshold property to specified value
            obj.speedupThreshold = threshold;
        end
        
        function enableVisualizationSaving(obj, enable, path)
            % Enables saving visualizations to disk and sets output path
            %
            % INPUTS:
            %   enable - Whether to save visualizations (logical)
            %   path - Optional path for saving visualizations
            
            % Validate enable is a logical value
            if ~islogical(enable) && ~(isnumeric(enable) && (enable == 0 || enable == 1))
                error('PerformanceBenchmark:InvalidInput', 'Enable flag must be a logical value');
            end
            
            % Set saveVisualizations property to enable value
            obj.saveVisualizations = logical(enable);
            
            % If path is provided, validate it is a valid directory
            if nargin > 2 && ~isempty(path)
                if ~ischar(path) && ~isstring(path)
                    error('PerformanceBenchmark:InvalidInput', 'Path must be a string or character array');
                end
                
                % Validate path exists
                if ~exist(path, 'dir')
                    error('PerformanceBenchmark:InvalidPath', 'Specified path does not exist: %s', path);
                end
                
                % Set visualizationPath property to specified path if provided
                obj.visualizationPath = char(path);
            end
        end
        
        function setVerbose(obj, verboseFlag)
            % Sets verbose mode for detailed console output during benchmarking
            %
            % INPUTS:
            %   verboseFlag - Whether to enable verbose output (logical)
            
            % Validate verboseFlag is a logical value
            if ~islogical(verboseFlag) && ~(isnumeric(verboseFlag) && (verboseFlag == 0 || verboseFlag == 1))
                error('PerformanceBenchmark:InvalidInput', 'Verbose flag must be a logical value');
            end
            
            % Set verbose property to specified value
            obj.verbose = logical(verboseFlag);
        end
    end
end