clear all
x = [5 10 20 30 40 60 90 130 180 250];
y1 = [31.6 13.2 7.80 7.00 7.80 7.60 9.8 11 13 16.2];
y2 = [158 66 39 35 39 38 49 55 65 81];

figure(1)
subplot(2,1,1);
plot(x,y1)
title('Error Rates, Total Number')

subplot(2,1,2);
plot(x,y2)
title('Error Rates, Percentages')