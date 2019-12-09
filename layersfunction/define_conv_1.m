function [ O , DzDw1,DzDw2 ] = define_conv_1( I, w,b, DzDy)
config;
fN = trainOpts.FilterNumber;
[m,n,t] = size(I);
 for i= 1:fN-1
    w1(:,:,:,i) = w(:,:,:,i);
    b1(:,i) = b(:,i);
 end
w2(:,:,1,1) = w(:,:,9);
b2 = b(:,9);
%% The forward propagation
if nargin == 3
    for  j= 1:t
        O1 =  vl_nnconv(I(:,:,j), w1,b1, ...
                                    'pad', 1, ...
                                    'stride', 1,...
                                    'dilate', 1);
         O(:,:,j,:) = O1;  
    end         
O2 = vl_nnconv(reshape(I,m*n,t), w2,b2, ...
                   'pad', 1, ...
                   'stride', 1,...
                   'dilate', 1);
O2 = reshape(O2,m,n,t);

O(:,:,:,9) = O2;
end
%% The backward propagation
if nargin == 4
    dw = 0;
    db = 0;
      for  j= 1:t
          DzDy1 = squeeze(DzDy(:,:,j,1:8));
          [O1, dw1, db1] =  vl_nnconv(I(:,:,j), w1,b1,DzDy1, ...
                                    'pad', 1, ...
                                    'stride', 1,...
                                    'dilate', 1);
           dw = dw + dw1;
           db = db + db1;
           O(:,:,j) = O1;  
      end         
    
 [O2, dw2, db2] = vl_nnconv(reshape((I),m*n,t), w2,b2,reshape((DzDy(:,:,:,9)),m*n,t), ...
                   'pad', 1, ...
                   'stride', 1,...
                   'dilate', 1);
O2 = reshape(O2,m,n,t);
O = O + O2;
% DzDw1  dc1/dw
DzDw1(:,:,:,1:8) = dw/16;
DzDw1(:,:,:,9) = dw2;
% DzDw2  dc1/db
DzDw2(:,1:8) = db/16;
DzDw2(:,9) = db2;
end
    
