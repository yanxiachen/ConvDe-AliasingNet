function Y = rnnloss(X, I, DzDy )
%% rnnloss: calculate the NMSE of restored image and original image
%% X: reconstructed image of size m*n;
%% I: ground-truth image of size m*n;

X = double(X);
I = double(I);
[m,n,t]=size(X);
S = X - I ;
I=reshape(I,m*n,t);
S1=reshape(S,m*n,t);
B=norm(I,'fro');

if nargin == 2
 
 Y = norm(S1,'fro') / B ;
elseif nargin ==3
 Y1 = norm(S1,'fro') ;   
 Y = S /(B*Y1);
else
    error('Input arguments number not proper.');
end;
end