function [ O , DzDw ] = xorg( Y, Rho, DzDy1, DzDy2, DzDy3 )
%% The first Reconstruction layer
%% This layer has two parameters: H_{l} = sum_{m=1}^{s} gamma_{l,m}B_{m};  \rho_{l}
%% Reference Copyright (c) 2017 Yan Yang
%% network setting
config;
[m ,n,t] = size(Y);
%% prepare for the equation x(1)
load('./mask/Random/1DRandom-ac24-4x.mat')
mask = logical(ifftshift(mask));
Denom1 = zeros(m , n,t) ; Denom1(mask) = 1 ;
Denom2 = Rho;
Denom = Denom1 + Denom2;  % diagonal matrix
Denom(find( Denom  == 0)) = 1e-6;
Q1 = 1./ Denom;
%% The forward propagation
if nargin == 2
    O = real(ifft2(Y .* Q1)) ; 
end

%% The backward propagation
if nargin == 5
    DzDy = DzDy1 + DzDy2 + DzDy3;
    %O
    O = 1;
    % DzDw1
    A = (-1) * Q1 .*Q1 ;
    temp = real(DzDy.*ifft2(A.*Y));
    DzDw = sum(temp(:)); % dE/dRho
end
end
