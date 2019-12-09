function [ O ,O2, DzDw ] = betaorg( x, a, eta, DzDy1, DzDy2, DzDy3, DzDy4 )

%% network setting
config;
%% The forward propagation
if nargin == 3
    O = eta*(x - a);
end
%% The backward propagation
if nargin == 7
    DzDy =  DzDy1 +  DzDy2 + DzDy3 + DzDy4;
    O = DzDy.*eta; % dbeta/dx
    O2 = -O; % dbeta/da
    temp = DzDy.*(x - a); % dbeta(n)/deta(n-1)
    DzDw = sum(temp(:));
end
end
