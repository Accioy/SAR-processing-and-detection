
p=@(t) 1/8

n=0;
N=256*256;
img=[];
while n<N
    t=rand(1)*8;%����[0,8]���ȷֲ������
    pt=p(t);   %�����Ӧ�ܶȺ���ֵf(t)
    r=rand(1);  %����[0,1]���ȷֲ������
    if r<=pt     %��������rС��f(t)�����ɸ�t����������a��
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
%����Ϊ�����������a�Ĺ��̣�����Ϊͳ�Ƽ�����������Ƿ���Ϸֲ�
num=50;         %��100������ͳ��
[x,c]=hist(img,num);
dc=8/num;
x=x/N/dc;
bar(c,x,1); hold on;
plot(f1,r,'r');