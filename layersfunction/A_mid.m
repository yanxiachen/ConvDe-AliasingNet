function [O, O2, O4, DzDw1, DzDw2] = A_mid ( a, x, c2,mu1, mu2, DzDy1, DzDy2 )

%% network setting
config;
%% The forward propagation
if nargin == 5
    O = mu1*a + mu2*x - c2;
end
%% The backward propagation
if nargin == 7
    DzDy = DzDy1 + DzDy2; 
    O = DzDy*mu1;   % dE/da
    O2 = DzDy*mu2;  % dE/dx
    O4 = -DzDy;     % dE/dc2
    % DzDw1  dE/dmu2
    temp1 = DzDy.*a; 
    DzDw1 = sum(temp1(:));
    % DzDw2  dE/dmu1
    temp2 = DzDy.*x; 
    DzDw2 = sum(temp2(:));
end
end