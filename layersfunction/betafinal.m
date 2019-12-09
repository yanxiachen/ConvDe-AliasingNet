function [ O, O2 ,O3, DzDw ] = betafinal( beta, x, a, eta, DzDy )

%% network setting
config;
%% The forward propagation
if nargin == 4
    O = beta + eta*(x - a);
end
%% The backward propagation
if nargin == 5
    O =  DzDy; % dbeta/dbeta(n-1)
    O2 = DzDy.*eta; % dbeta/dx
    O3 = - O2; % dbeta/da
    temp = DzDy.*(x - a); % dbeta/deta
    DzDw = sum(temp(:));
end
end
