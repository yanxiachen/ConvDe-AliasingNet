%% network setting
global trainOpts;
trainOpts.FilterNumber = 9;
trainOpts.FilterSize = 3;
trainOpts.Stage = 13;
trainOpts.Padding = 1;
trainOpts.LinearLabel = double(-1:0.02:1);
%% training and testing setting
trainOpts.sigma = 50;
trainOpts.Fold = 1;
trainOpts.batchSize = 1 ;
trainOpts.learningRate = 0.08 ;
trainOpts.numSubBatches = 1 ;
trainOpts.epochSize = inf;
trainOpts.prefetch = false;
trainOpts.numEpochs = 400;
trainOpts.weightDecay = 0.0005 ;
trainOpts.momentum = 0.95;
trainOpts.randomSeed = 0 ;
trainOpts.sync = false ;
trainOpts.cudnn = true ;
trainOpts.backPropDepth = +inf ;
trainOpts.continue = false ;
trainOpts.gpus = [] ;
% trainOpts.expDir = 'data/chars-experiment' ;
trainOpts.expDir = fullfile('Train_output','1DRandom4x') ;
trainOpts.train = [] ;
trainOpts.val = [] ;
trainOpts.solver = [] ;  % Empty array means use the default SGD solver
trainOpts.saveSolverState = true ;
trainOpts.nesterovUpdate = false ;
trainOpts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
trainOpts.profile = false ;
trainOpts.parameterServer.method = 'mmap' ;
trainOpts.parameterServer.prefix = 'mcn' ;
trainOpts.conserveMemory = false ;
trainOpts.backPropDepth = +inf ;
trainOpts.errorFunction = 'none';%multiclass' ;
trainOpts.errorLabels = {} ;
trainOpts.plotDiagnostics = false ;
trainOpts.plotStatistics = true;
trainOpts.postEpochFn = [] ;


