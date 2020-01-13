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
    t=rand(1)*max_n;%����[0,8]���ȷֲ������
    pt=p(t);   %�����Ӧ�ܶȺ���ֵf(t)
    rt=rand(1)*max(r);  %����[0,m]���ȷֲ��������m��Ҫ���ڸ����ܶȺ������Ͻ硣Ϊ��֤�����ٶ����ȡ��ȷ��
    if rt<=pt     %��������rС��f(t)�����ɸ�t����������a��
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
