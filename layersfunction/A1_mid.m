function [O, O2, O3, O4, DzDw1, DzDw2] = A1_mid (a, x, beta, c2, mu1, mu2, DzDy1, DzDy2)

%% network setting
config;
%% The forward propagation
if nargin == 6
    O = mu1*a + mu2*(x + beta) - c2;
end
%% The backward propagation
if nargin == 8
    DzDy = DzDy1 + DzDy2;
    O = DzDy*mu1;   % dE/da
    O2 = DzDy*mu2;   % dE/dx
    O3 = O2;      % dE/dbeta
    O4 = -DzDy;   % dE/dc2
    % DzDw1
    temp1 = DzDy.*a; % dE/dmu1
    DzDw1 = sum(temp1(:));
    % DzDw2
    temp2 = DzDy.*(x + beta); % dE/dmu2
    DzDw2 = sum(temp2(:));
end
end