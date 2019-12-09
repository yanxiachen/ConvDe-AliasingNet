function [ O , DzDw1,DzDw2 ] = define_conv_2( I, w,b, DzDy)
config;
fN = trainOpts.FilterNumber;
[m,n,t,d] = size(I);
for i= 1:fN-1
    w1(:,:,i,1) = w(:,:,i);
end
b1 = b(1);
w2(:,:,1,1) = w(:,:,9);
b2 = b(2);
%% The forward propagation
if nargin == 3
    for i= 1:t
        I1 = squeeze(I(:,:,i,1:8));
        O1 =  vl_nnconv(I1, w1,b1, ...
                                    'pad', 1, ...
                                    'stride', 1,...
                                    'dilate', 1);
        
        O(:,:,i) = O1;  
    end
    O2 = vl_nnconv(reshape(I(:,:,:,9),m*n,t), w2,b2, ...
                   'pad', 1, ...
                   'stride', 1,...
                   'dilate', 1);
    O2 = reshape(O2,m,n,t);
    O = O + O2;
end
%% The backward propagation

if nargin == 4
    dw = 0;
    db = 0;
     for i= 1:t
        I1 = squeeze(I(:,:,i,1:8));
        [O1, dw1, db1] =  vl_nnconv(I1, w1,b1,DzDy(:,:,i), ...
                                    'pad', 1, ...
                                    'stride', 1,...
                                    'dilate', 1);
         dw = dw + dw1;
         db = db + db1;  
         O(:,:,i,:) = O1;  
     end
    [O2, dw2, db2]= vl_nnconv(reshape((I(:,:,:,9)),m*n,t), w2,b2,reshape((DzDy),m*n,t), ...
                                     'pad', 1, ...
                                     'stride', 1,...
                                     'dilate', 1);
    O2 = reshape((O2),m,n,t);
    O(:,:,:,9) = O2;
    % DzDw1  dc2/dw
    DzDw1(:,:,1:8) = dw/16;
    DzDw1(:,:,9) = dw2;
    % DzDw2  dc2/db
    DzDw2(1) = db/16;
    DzDw2(2) = db2;
end
                                 
