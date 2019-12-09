function res = vl_simplenn_LD_train(net, x, dzdy, res, varargin)
%VL_SIMPLENN  Evaluate a SimpleNN network.
% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
config;
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;
opts = vl_argparse(opts, varargin);
N = numel(net.layers);
LL = trainOpts.LinearLabel;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(N - opts.backPropDepth + 1, 1);
if (nargin <= 2) || isempty(dzdy)
    doder = false ;
    if opts.skipForward
        error('simplenn:skipForwardNoBackwPass', ...
            '`skipForward` valid only when backward pass is computed.');
    end
else
    doder = true ;
end
if opts.cudnn
    cudnn = {'CuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
    cudnn = {'NoCuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ;
end
switch lower(opts.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end
gpuMode = isa(x, 'gpuArray') ;
if nargin <= 3 || isempty(res)
    if opts.skipForward
        error('simplenn:skipForwardEmptyRes', ...
            'RES structure must be provided for `skipForward`.');
    end
    res = struct(...
        'x', cell(1,N+1), ...
        'dzdx', cell(1,N+1), ...
        'dzdw', cell(1,N+1), ...
        'aux', cell(1,N+1), ...
        'stats', cell(1,N+1), ...
        'time', num2cell(zeros(1,N+1)), ...
        'backwardTime', num2cell(zeros(1,N+1))) ;
end
if ~opts.skipForward
    res(1).x = x ;
    label = net.layers{end}.class;
end
%% Forward pass
for n = 1:N
    if opts.skipForward, break; end
    l = net.layers{n} ;
    res(n).time = tic ;
    switch l.type
        case 'X_org' 
            res(n+1).x = xorg (res(n).x , l.weights{1});
        case 'A_org'
            res(n+1).x = A_org(res(n).x, l.weights{1}); 
        case 'C1'
            res(n+1).x =  define_conv_1(res(n).x, l.weights{1}, l.weights{2});
        case 'H'
            res(n+1).x = zmid(LL, res(n).x, l.weights{1} );
        case 'C2'
            res(n+1).x = define_conv_2(res(n).x, l.weights{1}, l.weights{2});
        case 'A_mid'
            res(n+1).x = A_mid(res(n-3).x,res(n-4).x, res(n).x,l.weights{1},l.weights{2});
        case 'M_org'
            res(n+1).x = betaorg(res(n-5).x , res(n).x ,l.weights{1} ); 
        case 'X_mid' 
            res(n+1).x = xmid( res(n-1).x , res(n).x , res(1).x , l.weights{1});
        case 'A1_org'
            res(n+1).x = A1_org(res(n).x, res(n-1).x, l.weights{1}); 
        case 'A1_mid'
            res(n+1).x = A1_mid(res(n-3).x,res(n-4).x, res(n-5).x, res(n).x,l.weights{1},l.weights{2});
        case 'M_mid'
            res(n+1).x = betamid(res(n-6).x, res(n-5).x , res(n).x ,l.weights{1} );
        case 'M_final'
            res(n+1).x = betafinal(res(n-6).x, res(n-5).x , res(n).x ,l.weights{1} );
        case 'X_final' 
            res(n+1).x = xfinal( res(n-1).x , res(n).x , res(1).x , l.weights{1});
        case 'loss'
            res(n+1).x = rnnloss(res(n).x, label);
         otherwise
            error('Unknown layer type ''%s''.', l.type) ;
    end
           
    %% optionally forget intermediate results
    needsBProp = doder && n >= backPropLim;
    forget = opts.conserveMemory && ~needsBProp ;
    if n > 1
        lp = net.layers{N-1} ;
        % forget RELU input, even for BPROP
        forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
        forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
        forget = forget && ~lp.precious ;
    end
    if forget
        res(n).x = [] ;
    end
    if gpuMode && opts.sync
        wait(gpuDevice) ;
    end
    res(n).time = toc(res(n).time) ;
end
%% Backward pass
if doder
    res(N+1).dzdx = dzdy ;
    for n = N:-1:backPropLim
        l = net.layers{n} ;
        res(n).backwardTime = tic ;
        switch l.type
            case 'X_org'
                [res(n).dzdx{1}, res(n).dzdw{1}]  = xorg (res(1).x , l.weights{1} ,res(n+1).dzdx{1}, res(n+5).dzdx{2},res(n+6).dzdx{2});
            case 'A_org'
                [ res(n).dzdx{1}, res(n).dzdw{1} ]  = A_org(res(n).x, l.weights{1},res(n+1).dzdx{1},res(n+4).dzdx{1});
            case 'A_mid'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{4}, res(n).dzdw{1},res(n).dzdw{2} ] = A_mid(res(n-3).x,res(n-4).x, res(n).x,l.weights{1},l.weights{2},res(n+1).dzdx{3},res(n+2).dzdx{1});
            case 'M_org'
                [ res(n).dzdx{2}, res(n).dzdx{3}, res(n).dzdw{1} ] = betaorg(res(n-5).x, res(n).x, l.weights{1}, res(n+1).dzdx{2}, res(n+2).dzdx{2}, res(n+6).dzdx{3},res(n+7).dzdx{1});
            case 'M_mid'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3}, res(n).dzdw{1} ] = betamid(res(n-6).x , res(n-5).x, res(n).x, l.weights{1}, res(n+1).dzdx{2}, res(n+2).dzdx{2}, res(n+6).dzdx{3},res(n+7).dzdx{1});
            case 'X_mid' 
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1}] = xmid(res(n-1).x , res(n).x , res(1).x , l.weights{1},  res(n+1).dzdx{1},res(n+5).dzdx{2},res(n+6).dzdx{2}); 
            case 'A1_org'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1} ]  = A1_org(res(n).x, res(n-1).x, l.weights{1},res(n+1).dzdx{1},res(n+4).dzdx{1}); 
            case 'C1'
                [res(n).dzdx{1}, res(n).dzdw{1}, res(n).dzdw{2}] =  define_conv_1(res(n).x, l.weights{1}, l.weights{2},res(n+1).dzdx{1});
            case 'H'
                [res(n).dzdx{1}, res(n).dzdw{1}] = zmid(LL, res(n).x, l.weights{1}, res(n+1).dzdx{1});           
            case 'C2'
                [res(n).dzdx{1}, res(n).dzdw{1}, res(n).dzdw{2}] = define_conv_2(res(n).x, l.weights{1}, l.weights{2},res(n+1).dzdx{4});
            case 'A1_mid'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3},res(n).dzdx{4},  res(n).dzdw{1},res(n).dzdw{2} ] = A1_mid(res(n-3).x,res(n-4).x, res(n-5).x, res(n).x,l.weights{1},l.weights{2},res(n+1).dzdx{3},res(n+2).dzdx{1});
            case 'M_final'
                [ res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdx{3}, res(n).dzdw{1} ] = betafinal(res(n-6).x, res(n-5).x , res(n).x ,l.weights{1}, res(n+1).dzdx{2});
            case 'X_final'
                [res(n).dzdx{1}, res(n).dzdx{2}, res(n).dzdw{1}] = xfinal( res(n-1).x , res(n).x , res(1).x , l.weights{1}, res(n+1).dzdx{1});
            case 'loss' 
                res(n).dzdx{1} = rnnloss(res(n).x, label, 1);
        end % layers
        if opts.conserveMemory && ~net.layers{i}.precious && n ~= N
            res(n+1).dzdx = [] ;
            res(n+1).x = [] ;
        end
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(n).backwardTime = toc(res(n).backwardTime) ;
    end
    if n > 1 && n == backPropLim && opts.conserveMemory && ~net.layers{n}.precious
        res(n).dzdx = [] ;
        res(n).x = [] ;
    end
end
