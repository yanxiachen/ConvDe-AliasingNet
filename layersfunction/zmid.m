function [ O,DzDw ] = zmid(p, c1, q, DzDy)
%% Nonlinear transform layer
%% z_{l} = S_PLF(c_{l}+\beta_{l})
%% The parameters are q related to the the predefined positions p;
%% Reference Copyright (c) 2017 Yan Yang
%% network setting
config;
[m,n,t,d]=size(c1);
if ~isempty(trainOpts.gpus)
    gp = 1;
else
    gp = 0;
end
I = c1;
%% The forward propagation
if nargin == 3
    I=reshape(I,m*n,t,d);
    temp = double(I);
    q = double(q);
    if gp
        temp = gpuArray(temp);
        q1 = gpuArray(q);
        p1 = gpuArray(p);
        temp2 = nnlinecu_double(p1, q1, temp);
        O = gather(temp2);
    else
        O = nnlinemex( p, q , temp); 
        O=reshape((O),m,n,t,d);
    end
end
%% The backward propagation
if nargin == 4
    I=reshape(I,m*n,t,d);
    DzDy=reshape(DzDy,m*n,t,d);
    xvar = double(I);
    yvar = double(DzDy);
    q = double(q);
    if gp
        xvar = gpuArray(xvar);
        yvar = gpuArray(yvar);
        q1 = gpuArray(q);
        p1 = gpuArray(p);
        [xgra, ygra] = nnlinecu_double(p1, q1, xvar, yvar);
        O = gather(xgra);
        O2 = O;
        DzDw = gather(ygra);
    else
        [O, DzDw] = nnlinemex(p, q, xvar, yvar); 
        O=reshape((O),m,n,t,d);
    end
end
end


