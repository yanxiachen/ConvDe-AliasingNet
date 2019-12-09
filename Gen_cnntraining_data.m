clear all;close all;clc;
load('./mask/Random/1DRandom-ac24-4x.mat');
DataNums = 10;
disp(numel(mask)/numel(find(mask==1)))
for k = 1:1:DataNums 
     folder = './data/Train';
     DataFile = dir(fullfile(folder,'data*'));
     data_name = DataFile(k).name;
     load(fullfile(folder,data_name)); % label
     disp(data_name)    
     im=double(abs(label));
     kspace_full = fft2(im); 
     y = (double(kspace_full)) .* (ifftshift(mask));
     images.data(:,:,:,k) = y; 
     images.label(:,:,:,k) = im; 
     images.id(1,k) = k;
end
    images.set(1,1:8) = 1; images.set(1,9:10) = 2;
    meta.sets = {'train', 'val'};
    imdb.images = images;
    imdb.meta = meta;
    save(strcat('./data/Train/', '1DRandom-ac24-NET-4x', '.mat'), 'imdb');
