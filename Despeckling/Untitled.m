y=[];
for x=-100:0.01:100
    y=[y,sin(3*x)];
end
z=fft(y);
f = (0:length(z)-1)*50/length(z);
plot(f,abs(z))