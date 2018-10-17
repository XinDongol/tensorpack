%% parameters
delta = 0.1;
a2 = 1;
a1 = 1000;
number_of_pieces = 10;  % number of pieces we want to divide
%%
b2 = delta/tanh(a2*delta);
b1 = delta/tanh(a1*delta);
thres_min = zeros(1,number_of_pieces)+1000;
amin = zeros(1,number_of_pieces);
bmin = zeros(1,number_of_pieces);
y = zeros(10,2001);
x = -1*delta:0.0001:1*delta;
y1 = b1 * tanh (a1*x);
y2 = b2 * tanh (a2*x);
% calculate the l2 distance
ddd = (sum((y1 - y2) .^ 2));
% l2 distance for each piece
per_dist = ddd/number_of_pieces;  
% iteratively search the optimal a with the least l2 dist
for i = 1:1:number_of_pieces
    for a = 1:1e-4:a1
        b = delta/tanh(a*delta);
        y(i,:) = b * tanh (a*x);
        ddd = (sum((y2 - y(i,:)) .^ 2));
        thres = abs(ddd-per_dist*i);
        if (thres<thres_min(i))
            thres_min(i) = thres;
            amin(i) = a;
            bmin(i) = b;
        end
    end
end
% plot the curves
for i = 1:1:number_of_pieces
    y(i,:) = bmin(i) * tanh (amin(i)*x);
    plot(x,y(i,:));
    hold on;
end
%x2 = 1:0.01:10;
%y2 = 2*b1*delta + 2 *(delta./tanh(x2*delta))*delta-2*delta./a1-2*delta./x2-(4/3)*a1*b1*(delta./tanh(x2*delta)).*x2*delta^3+(4/15)*(a1*b1*(delta./tanh(x2*delta)).*x2.^3+(delta./tanh(x2*delta))*b1.*x2*a1^3)*delta^5-4/63*b1*(delta./tanh(x2*delta))*a1^3.*x2.^3*delta^7-per_dist;
%plot(x2,y2)