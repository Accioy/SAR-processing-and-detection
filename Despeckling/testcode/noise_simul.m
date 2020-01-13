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
    t=rand(1)*8;%����[0,8]���ȷֲ������
    pt=p(t);   %�����Ӧ�ܶȺ���ֵf(t)
%     pt=integral(p,0,t);
    rt=rand(1)*max(r);  %����[0,m]���ȷֲ��������m��Ҫ���ڸ����ܶȺ������Ͻ硣Ϊ��֤�����ٶ����ȡ��ȷ��
    if rt<=pt     %��������rС��f(t)�����ɸ�t����������a��
        n=n+1;
        img(n)=t;
    end
end

%����Ϊ�����������a�Ĺ��̣�����Ϊͳ�Ƽ�����������Ƿ���Ϸֲ�
h=histogram(img,'Normalization','pdf');

