function Y = NMSE(X, I)
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
 Y = norm(S1,'fro') / B ;
end