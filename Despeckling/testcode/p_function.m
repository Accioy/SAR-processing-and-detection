L=1;
y=gamma(L);
F=linspace(0,8,1024);
p=[];
for i=1:length(F)
    v=1/y*L^L*F(i)^(L-1)*exp(-L*F(i));
    p=[p,v];
end
p1=[];
L=2;
y=gamma(L);
for i=1:length(F)
    v1=1/y*L^L*F(i)^(L-1)*exp(-L*F(i));
    p1=[p1,v1];
end
p2=[];
L=3;
y=gamma(L);
for i=1:length(F)
    v2=1/y*L^L*F(i)^(L-1)*exp(-L*F(i));
    p2=[p2,v2];
end
p3=[];
L=4;
y=gamma(L);
for i=1:length(F)
    v3=1/y*L^L*F(i)^(L-1)*exp(-L*F(i));
    p3=[p3,v3];
end

s=trapz(F,p); 
plot(F,p);
hold on
plot(F,p1);
hold on
plot(F,p2);
hold on
plot(F,p3);
legend('L=1','L=2','L=3','L=4');