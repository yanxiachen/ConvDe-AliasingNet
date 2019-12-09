function [O, O2, DzDw ] = xmid(a, beta, Y, Rho, DzDy1, DzDy2, DzDy3)
%% Reconstruction layer in the middle of the network.
%% This layer has two parameters: H_{l} = sum_{m=1}^{s} gamma_{l,m}B_{m};  \rho_{l}
%% This layer has two inputs: I1 = z ; I2 = \beta
%% Reference Copyright (c) 2017 Yan Yang
%% network setting
config;
[m ,n,t] = size(Y);
load('./mask/Random/1DRandom-ac24-4x.mat')
mask = logical( ifftshift(mask) );
Denom1 = zeros(m,n,t) ; Denom1(mask) = 1 ;
Denom2 = Rho;
Denom = Denom1+Denom2;
Denom(find(Denom == 0)) = 1e-6;
Q2=1./Denom;
%% The forward propagation
if nargin == 4
    Pr = Rho*(a - beta);
    O = real( ifft2(( fft2 ( Pr ) + Y ) .* Q2)); 
end
%% The backward propagation
if nargin == 7
    DzDy = DzDy1 + DzDy2 + DzDy3;
    %  O dx/da
    temp1 = Q2.*fft2(DzDy);
    O = real(Rho*ifft2(temp1));
    %  O2 dx/dbeta(n-1)
    O2 = - O;
    % DzDw1  dE/dRho
    A = (-1) * Q2 .*Q2 ;
    tp1 = Q2.*fft2(a - beta);
    tp2 = A.*(Y + fft2(Rho*(a - beta)));
    temp2 = real(DzDy.*ifft2(tp1 + tp2));
    DzDw = sum(temp2(:)); 
end
end

