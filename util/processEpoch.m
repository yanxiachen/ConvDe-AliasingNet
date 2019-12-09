function [net, state] = processEpoch(net, state, params, mode)
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.
%% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
    for i = 1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            state.solverState{i} = cell(1, numel(net.layers{i}.weights)) ;
            state.solverState{i}(:) = {0} ;
        end
    end
end
%% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus > 1
    parserv = ParameterServer(params.parameterServer) ;
    vl_simplenn_start_parserv(net, parserv) ;
else
    parserv = [] ;
end
%% profile
if params.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end
subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;
start = tic ;
for t=1:params.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
        fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    for s=1:params.numSubBatches
        %% get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        im = params.imdb.images.data(:,:,:,t);
        labels = params.imdb.images.label(:,:,:,t);
        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize ;
                batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(params.imdb, nextBatch) ;
        end
        if numGpus >= 1
            im = gpuArray(im) ;
        end
        if strcmp(mode, 'train')
            dzdy = 1 ;
            evalMode = 'normal' ;
        else
            dzdy = [] ;
            evalMode = 'test' ;
        end
        net.layers{end}.class = labels ;
        res = vl_simplenn_LD_train(net, im, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', params.conserveMemory, ...
            'backPropDepth', params.backPropDepth, ...
            'sync', params.sync, ...
            'cudnn', params.cudnn, ...
            'parameterServer', parserv, ...
            'holdOn', s < params.numSubBatches) ;
        %% accumulate errors
        error = sum([error, [...
            sum(double(gather(res(end).x))) ;
            reshape(params.errorFunction(params, labels, res),[],1) ; ]],2) ;
    end
    %% accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(parserv), parserv.sync() ; end
        [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
    end
    %% get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = extractStats(net, params, error / num) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s: %.3f', f, stats.(f)) ;
    end
    fprintf('\n') ;
    %% collect diagnostic statistics
    if strcmp(mode, 'train') && params.plotDiagnostics
        switchFigure(2) ; clf ;
        diagn = [res.stats] ;
        diagnvar = horzcat(diagn.variation) ;
        diagnpow = horzcat(diagn.power) ;
        subplot(2,2,1) ; barh(diagnvar) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnvar), ...
            'YTickLabel',horzcat(diagn.label), ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1], ...
            'XTick', 10.^(-5:1)) ;
        grid on ; title('Variation');
        subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnpow), ...
            'YTickLabel',{diagn.powerLabel}, ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1e5], ...
            'XTick', 10.^(-5:5)) ;
        grid on ; title('Power');
        subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
        drawnow ;
    end
end
%% Save back to state.
state.stats.(mode) = stats ;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info') ;
        profile off ;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off ;
    end
end
if ~params.saveSolverState
    state.solverState = [] ;
else
    for i = 1:numel(state.solverState)
        for j = 1:numel(state.solverState{i})
            s = state.solverState{i}{j} ;
            if isnumeric(s)
                state.solverState{i}{j} = gather(s) ;
            elseif isstruct(s)
                state.solverState{i}{j} = structfun(@gather, s, 'UniformOutput', false) ;
            end
        end
    end
end
net = vl_simplenn_move(net, 'cpu') ;