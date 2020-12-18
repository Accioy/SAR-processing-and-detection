x=linspace(-2,2,1001);
y1=[];
y2=[];
y3=[];
for i=1:1001
    y1(i)=abs(x(i));
    y2(i)=x(i)^2;
    if x(i)<-1 || x(i)>1
        y3(i)=abs(x(i))-0.5;
    else
        y3(i)=0.5*x(i)^2;
    end
end

        
plot(x,y1,'r',x,y2,'g',x,y3)
legend('FontName','Times New Roman')
legend('L1','L2','Smooth-L1')

    
    