clc;
clear;

%% parameters, load an image %%%%%%%%%%%%
% number of looks
L=4;
% img name
img_name='airplane45.tif'
% img resolution
img = imread(img_name);
res = size(img);
res = res(1:2);
% max of simulated noise
max_n=5;

%% speckle probability density function %%%%%%%%%%
% p=@(F) (1/gamma(L)).*(L.^L).*(F.^(L-1)).*exp(-L.*F);
p=@(F) (1/gamma(L))*(L^L)*(F^(L-1))*exp(-L*F);
r=[];
f1=linspace(0,max_n,1024);
for i=1:length(f1)
    r1=p(f1(i));
    r=[r,r1];
end

tic
%% Generate the noise sequence of the probability density function %%%%%%%%%%%%%%%%%%%
n=0;
N=res(1)*res(2);
noises=[];
while n<N
    t=rand(1)*max_n;%生成[0,8]均匀分布随机数
    pt=p(t);   %计算对应密度函数值f(t)
    rt=rand(1)*max(r);  %生成[0,m]均匀分布随机数，m需要大于概率密度函数的上界。为保证计算速度最好取上确界
    if rt<=pt     %如果随机数r小于f(t)，接纳该t并加入序列a中
        n=n+1;
        noises(n)=t;
    end
end
toc
%% compare the sequence distribution with the probability density function %%

% s=trapz(f1,r);
% plot(f1,r,'r');
% hold on;
% h=histogram(img,'Normalization','pdf');

%% Image pre-process %%%%%%
img = rgb2gray(img);
imwrite(img,'gray.jpg');
img = im2double(img);
%% add noise %%%%%%%%%%%%%
noises=noises/max(noises);
noises=reshape(noises,res);
img_with_speck = img.*noises;
% imshow(img_with_speck*2);
imwrite(img_with_speck*2,'im_with_speckle.jpg');
