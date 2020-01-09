L=1;
y=gamma(L);
F=linspace(0,8,1024);
p=[]
for i=1:length(F)
    p1=1/y*L^L*F(i)^(L-1)*exp(-L*F(i));
    p=[p,p1];
end
s=trapz(F,p); 
plot(F,p)