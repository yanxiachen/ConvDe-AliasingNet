function B = filter_base( )
%%%%%%%%%%%%%%%%%%%%DCT base
 config;
fN = trainOpts.FilterNumber; % ��ʽ��6���е�L��8
fS = trainOpts.FilterSize; % 3
 fS_sqrt = fS^2;

 DCT = dctmtx(fS);
 DCT = kron(DCT, DCT);
 B = zeros(fS_sqrt, fN);
 for i = 2 : fS_sqrt
 B(:, i-1) = DCT(i, :);
 end
B(:,9)=[-0.1,-0.1,-0.1,0,0,0,0.1,0.1,0.1]';

end

