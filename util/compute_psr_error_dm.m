function [p, s, r, error] = compute_psr_error_dm(label, output)

% label = single(divide_mean(label));
% output = single(divide_mean(output));
% 
% p = compute_psnr(label,output);
% s = ssim(label,output);
% r = NMSE(label,output);
% error = abs(label - output);
 %% metrics
        error=label-output;
        mse = mean((abs(error(:))).^2);
        p = 10*log10(1/mse);
        s = ssim(abs(label),abs(output));
        r = NMSE(output,label);



