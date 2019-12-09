%% This is a cpu test code demo
%% Output: the average NMSE and PSNR and ssim over the test images.
 clc; clear;close all;
%% Load trained network
load('./Train_output/1DRandom4x/DCTV-50-lr0.01-0.1-epoch-400.mat');
vl_simplenn_display(net) ;
%% Load Test Data
folder_test='./data/Test' ;
folder_results = './results_1DRandom4x/';
DataFile = dir(fullfile(folder_test,'data*'));
DataNums = length(DataFile);
% DataNums = 50;
%% Load mask
load('./mask/Random/1DRandom-ac24-4x.mat') ;
%% matrics
recon_loss = 0;
recon_psnr = 0;
recon_ssim = 0;
recon_time = 0;
zero_loss = 0;
zero_psnr = 0;
zero_ssim = 0;
for i=1:DataNums
    data_name = DataFile(i).name;
    load(fullfile(folder_test,data_name));
    disp(data_name);
    [Nx,Ny,Nt] = size(label);
    im = abs(label);
    %% Undersampling in the k-space
    kspace_full = fft2(im);
    y = (double(kspace_full)) .* (ifftshift(mask));
    data.train = y;
    data.label = im;
    Label = label;
    im_sos = abs(sos(label));
    %% Recon image
    tic;
    res = vl_simplenn_LD_test(net, data);
    Time_Net_rec = toc;
    rec_image = res(end-1).x; % recon image
    Rec_Image = rec_image;
    rec_image_sos = abs(sos(rec_image));
    a = floor(i/10);
    b = i-a*10;
    m = num2str(a);
    n = num2str(b);
    file_name = ['result',strcat(m,n),'.mat'];
    save (strcat(folder_results,file_name),'rec_image');   
    %%  Recon matrics
    [re_PSnr,re_ssim,res_loss, error] = compute_psr_error_dm(im_sos, rec_image_sos);
    METRIC_PSNR (i) = re_PSnr;
    METRIC_ssim (i) = re_ssim;
    METRIC_loss (i) = res_loss;
    
    recon_loss = recon_loss+res_loss;
    recon_psnr = recon_psnr+re_PSnr;
    recon_ssim = recon_ssim+re_ssim;
    recon_time = recon_time+Time_Net_rec;
    %% Zero_filling matrics
    Zero_filling_rec = ifft2(y);
    rec_Zero_filling = Zero_filling_rec;
    Zero_filling_rec_sos = abs(sos(Zero_filling_rec));   
    [Zero_PSnr1,Zero_filling_rec_ssim,Zero_filling_rec_loss,error] = compute_psr_error_dm(im_sos, Zero_filling_rec_sos);
    Zero_METRIC_PSNR (i) = Zero_PSnr1;
    Zero_METRIC_ssim (i) = Zero_filling_rec_ssim;
    Zero_METRIC_loss (i) = Zero_filling_rec_loss;
    zero_loss = zero_loss+Zero_filling_rec_loss;
    zero_psnr = zero_psnr + Zero_PSnr1;
    zero_ssim = zero_ssim + Zero_filling_rec_ssim;
end
metrics.recon_loss1 = recon_loss/DataNums
metrics.recon_psnr1 = recon_psnr/DataNums
metrics.recon_ssim1 = recon_ssim/DataNums
metrics.recon_time1 = recon_time/DataNums

metrics.zero_loss1 = zero_loss/DataNums
metrics.zero_psnr1 = zero_psnr/DataNums
metrics.zero_ssim1 = zero_ssim/DataNums
file_name1 = ['result_metrics','.mat'];
save (strcat(folder_results,file_name1),'metrics');

save(strcat(folder_results,'METRIC_PSNR.mat'),'METRIC_PSNR');
save(strcat(folder_results,'METRIC_ssim.mat'),'METRIC_ssim');
save(strcat(folder_results,'METRIC_loss.mat'),'METRIC_loss');

figure;
subplot(1,3,1); imshow(abs(im_sos),[]); xlabel('label');
subplot(1,3,2); imshow(abs(Zero_filling_rec_sos),[]); xlabel('Zero-filling reconstructon result');
subplot(1,3,3); imshow(abs(rec_image_sos(:,:,1)),[]); xlabel('Generic-DCTV-Net reconstruction result');



