function [O,O2, DzDw] = A1_org ( x , beta, mu2, DzDy1, DzDy2)

%% network setting
config;
% mu2 = 1;
%% The forward propagation
if nargin == 3  
    O = mu2*(x + beta);
end
%% The backward propagation
if nargin == 5
    DzDy = DzDy1 + DzDy2 ;
    %  O
    O = DzDy*mu2; % dE/dx
    O2 = O; %dE/dbeta
    % DzDw
    temp = DzDy.*(x + beta); % dE/dmu2
    DzDw = sum(temp(:));
end
end
