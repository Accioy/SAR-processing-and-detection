clc;
clear;

L=4;

% p=@(F) (1/gamma(L)).*(L.^L).*(F.^(L-1)).*exp(-L.*F);
p=@(F) (1/gamma(L))*(L^L)*(F^(L-1))*exp(-L*F);

r=[];
f1=linspace(0,8,1024);
for i=1:length(f1)
    r1=p(f1(i));
    r=[r,r1];
end
s=trapz(f1,r);
plot(f1,r,'r');
hold on;

n=0;
N=65536;
img=[];
while n<N
    t=rand(1)*8;%生成[0,8]均匀分布随机数
    pt=p(t);   %计算对应密度函数值f(t)
%     pt=integral(p,0,t);
    rt=rand(1)*max(r);  %生成[0,m]均匀分布随机数，m需要大于概率密度函数的上界。为保证计算速度最好取上确界
    if rt<=pt     %如果随机数r小于f(t)，接纳该t并加入序列a中
        n=n+1;
        img(n)=t;
    end
end

%以上为生成随机数列a的过程，以下为统计检验随机数列是否符合分布
h=histogram(img,'Normalization','pdf');

