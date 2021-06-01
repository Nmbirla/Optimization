% Code written by Neha Birla for IMSE 505 (Fall 2019) mid-term
% This script is to solve problem 2 part (a)
% Minimizing the given function by steepest descent and using cubic interpolation 
% for linesearch.

clc;
clear all;
x0=[0 0]'; % initial guess of minimizer
tol2 = 1e-5; % tolerance for termination criteria

x=x0;
fval=f(x);
alpha = linesearch_cubic(x);
d=-1*grad_f(x);
i=1;
x=[x, x(:,end) + alpha(end)*d(:,end)];
fval = [fval, f(x(:,end))];

while (norm(x(:,end) - x(:,end-1)) >tol2)
    alpha =[alpha, linesearch_cubic(x(:,end))];
    d=[d, -1*grad_f(x(:,end))];
    i=i+1;
    x=[x, x(:,end) + alpha(end)*d(:,end)];
    fval = [fval, f(x(:,end))];
end
%%

figure(1);
ax=gca;
[X,Y] = meshgrid(0:0.01:1,-0.4:0.01:0.6);
Z = exp(-X)+ exp(-Y)+ X.^2 + 5*Y.^2 - 2*X.*Y;
contour(ax,X,Y,Z,exp(0.52:0.05:1.6))
hold on;
plot(x(1,1),x(2,1),'ko',x(1,end),x(2,end),'rx','markersize',10)
plot(x(1,:),x(2,:),'k')
hold off;  axis equal
xlabel('x1'); ylabel('x2');
title('steepest descent method')

%%
% verify answer using MATLAB's inbuild function fminunc (unconstraint minimization)
% x0=[0;0];
% fun=@(x)(exp(-x(1))+ exp(-x(2))+ x(1)^2 + 5*x(2)^2 - 2*x(1)*x(2));
% x2 = fminunc(fun,x0);
% fprintf('\nx* = (%1.4f, %1.4f) solution using fminunc\n',x2)

%%
% Print the table as per sepcifications given in the question
fprintf('iter# \t(x1, x2) \t\t\tf(x) \tak \t\tdk \n');
for i=1:size(x,2)-1
    formatSpec = '%d \t\t(%1.4f, %1.4f) \t%1.4f \t%1.3f \t(%1.2d, %1.2d) \n';
    fprintf(formatSpec,i-1,x(1,i),x(2,i),fval(i),alpha(i),d(1,i),d(2,i));  
end
i=i+1;
formatSpec = '%d \t\t(%1.4f, %1.4f) \t%1.4f \n';
fprintf(formatSpec,i-1,x(1,i),x(2,i),fval(i));

%%
function fx = f(x)
    fx=exp(-x(1))+ exp(-x(2))+ x(1)^2 + 5*x(2)^2 - 2*x(1)*x(2);
end

function g_alpha=g(aph,x)
    g_alpha=f(x+aph*(-1*grad_f(x)));
end

function dg = diff_g(aph,x)
    d=-1*grad_f(x);
    dg=grad_f(x+aph*d)'*d; %dg scalar output
end

function df = grad_f(x)
    % this function evaluates gradient of f=exp(-x1)+ exp(-x2)+ x1^2 + 5*(x2^2)
    % - 2*x1*x2; at a given point x, where x belongs to R^2
    df = [2*x(1) - 2*x(2) - exp(-x(1));
         10*x(2) - 2*x(1) - exp(-x(2))];
end

function alpha = linesearch_cubic(x)
    % This function performs linesearch using minimization rule. 
    % A cubic interpolation method is used. Ref pg# 725-726
    % Definition of functions used in this code
    % --> g(a) = f(x+a*d)
    % --> diff_g(a) = grad_f(x+a*d)'*d
    % Constants used in this code are s=1; beta=0.5; tolerance =1e-3;
    
    s=1;
    beta=0.5;
    
    % this if statement checks if g(s) is much greater than g(0). In that
    % case a new value of s=beta*s is assigned. Ref step1, pg#726 
    if g(s,x)/g(0,x) > 10
       s=beta*s; 
    end
    % initialize values of a and b
    b=s;
    a=0;
    
    % this while loops determines initial interval [a,b] (Step#1)
    while ~(diff_g(b,x)>=0 || g(b,x) >= g(a,x)) 
        a=b;
        b=2*b;
    end

    % this while loop calculates alpha in an iterative way (step#2)
    % 1. A cubic polynomial is fit to four values g(a), g'(a), g(b), g'(b) 
    % 2. aplha that minimizes this fitted cubic polynomial is caluclated 
    % 3. interval (a, b] is updated
    % 4. repeat until tolerance is met
    tol = 1e-6;
    while abs(b-a)>tol
        
        % fit cubic polynomial
        z= 3*(g(a,x)-g(b,x))/(b-a) + diff_g(a,x) + diff_g(b,x) ;
        w = sqrt(z^2 - diff_g(a,x)*diff_g(b,x));
        alpha = b - (diff_g(b,x) + w - z)/(diff_g(b,x)-diff_g(a,x) + 2*w)*(b-a);
        
        % check if alpha == b
        if abs(alpha-b) < tol/2
            break
        end
        
        % update the interval (a, b]
        if (diff_g(alpha,x)>=0 || g(alpha,x) >= g(a,x))
            b=alpha;
        else 
            a=alpha;
        end
    end
end
