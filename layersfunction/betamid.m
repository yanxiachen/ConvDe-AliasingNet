function [ O, O2 ,O3, DzDw ] = betamid( beta, x, a, eta, DzDy1, DzDy2, DzDy3, DzDy4 )

%% network setting
config;
%% The forward propagation
if nargin == 4
    O = beta + eta*(x - a);
end
%% The backward propagation
if nargin == 8
    DzDy =  DzDy1 +  DzDy2 + DzDy3 + DzDy4;
    O =  DzDy; % dbeta/dbeta(n-1)
    O2 = DzDy.*eta; % dbeta/dx
    O3 = -O2; % dbeta/da
    temp = DzDy.*(x - a); % dbeta/deta
    DzDw = sum(temp(:));
end
end
