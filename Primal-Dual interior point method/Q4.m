% Code written by Neha Birla for IMSE 505 (Fall 2019) Final
% This script is to solve problem 4
% Inter Point Method

clear all 
clc

% Defining matrices from given problem
A=[1 -1 -5 2 -1 0; 1 4 1 1 0 -1];
b=[2 4]';
c=[3 8 1 5 0 0]';

p0=[1;1];
s=c-A'*p0;
x0=[2 1 1 4 2 7]';

xsize=size(A,2);
psize=size(A,1);

a=0.995;
tol=1e-6;
x=x0;
p=p0;

while max(x.*s) > tol
    X=diag(x);
    S=diag(s);

    B=[-X\S, A';
        A, zeros(psize)];
    Q=[s; zeros(psize,1)];

    temp=B\Q;

    dx=temp(1:xsize);
    dp=temp(xsize+1:end);
    ds= -s -X\S*dx;
    rx=-x./dx;
    rs=-s./ds;
    thx = min(rx(dx<0));
    ths = min(rs(ds<0));
    th= min([1, a*thx, a*ths]);
    
    %Update
    x = x+th*dx;
    p = [p, p(:,end)+th*dp];
    s = s+th*ds; 
end

fval=b'*p;
%%
% Print the table as per sepcifications given in the question
fprintf('iter# \t(x1, x2) \t\t\tf(x)\n');
for i=1:size(p,2)
    formatSpec = '%d \t\t(%1.4f, %1.4f) \t%1.4f \n';
    fprintf(formatSpec,i-1,p(1,i),p(2,i),fval(i));  
end
