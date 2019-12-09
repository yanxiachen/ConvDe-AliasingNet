clear all;close all;clc;
config;
%% Load dataset
load('./data/Train/1DRandom-ac24-NET-4x.mat') ;
net = initialize_NET(); 
%%  Network training and validation 
tic;
[net,info] = DCTV_train(net, imdb, @getBatch, trainOpts) ;
time = toc;
time = time/3600;

function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.label(1,batch) ;
end