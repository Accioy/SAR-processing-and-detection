
p=@(t) 1/8

n=0;
N=256*256;
img=[];
while n<N
    t=rand(1)*8;%生成[0,8]均匀分布随机数
    pt=p(t);   %计算对应密度函数值f(t)
    r=rand(1);  %生成[0,1]均匀分布随机数
    if r<=pt     %如果随机数r小于f(t)，接纳该t并加入序列a中
        n=n+1;
        img(n)=t;
    end
    n
end




r=[];
f1=linspace(0,8,1024);
for i=1:length(f1)
    r1=p(f1(i));
    r=[r,r1];
end
% plot(F,r); hold on;

s=trapz(f1,r);
%以上为生成随机数列a的过程，以下为统计检验随机数列是否符合分布
num=50;         %分100个区间统计
[x,c]=hist(img,num);
dc=8/num;
x=x/N/dc;
bar(c,x,1); hold on;
plot(f1,r,'r');