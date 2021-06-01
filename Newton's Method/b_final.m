% Code written by Neha Birla for IMSE 505 (Fall 2019) mid-term
% This script is to solve problem 2 part (b)
% Minimizing the given function by Newton method and using Armijo Rule for 
% linesearch.

clc;
clear all;
tol2 = 5e-6; % tolerance for termination criteria
x0=[0 0]'; % initial guess of minimizer
x=x0;
fval=f(x); % function value at initial guess
alpha = armijo_rule(x);

d=-hes_mtx(x)\grad_f(x);
i=1;
x=[x, x(:,end) + alpha(end)*d(:,end)];
fval = [fval, f(x(:,end))];

while (norm(x(:,end) - x(:,end-1)) >tol2)
    alpha = [alpha, armijo_rule(x(:,end))]; % stores step size co-efficient at each iteration 
    dk = -hes_mtx(x(:,end))\grad_f(x(:,end));
    d = [d, dk]; % stores search direction at each iteration
    i = i+1;
    x = [x, x(:,end) + alpha(end)*d(:,end)]; % stores x value at each iteration 
    fval = [fval, f(x(:,end))]; % stores function value at each iteration
end
%%
% figure(1);
% ax=gca;
% [X,Y] = meshgrid(0:0.01:1,-0.4:0.01:0.6);
% Z = exp(-X)+ exp(-Y)+ X.^2 + 5*Y.^2 - 2*X.*Y;
% contour(ax,X,Y,Z,exp(0.52:0.05:1.6))
% hold on;
% plot(x(1,1),x(2,1),'ko',x(1,end),x(2,end),'rx','markersize',10)
% plot(x(1,:),x(2,:),'k')
% hold off;  axis equal
% xlabel('x1'); ylabel('x2');
% title('Newton''s method');
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

function hessian=hes_mtx(x)
% this function evaluates hessian matrix of f=exp(-x1)+ exp(-x2)+ x1^2 
% + 5*(x2^2) - 2*x1*x2; at a given point x, where x belongs to R^2
    hessian=[exp(-x(1))+2,  -2;
             -2,            exp(-x(2))+10];
end

function df = grad_f(x)
    % this function evaluates gradient of f=exp(-x1)+ exp(-x2)+ x1^2 + 5*(x2^2)
    % - 2*x1*x2; at a given point x, where x belongs to R^2
    df = [2*x(1) - 2*x(2) - exp(-x(1));
         10*x(2) - 2*x(1) - exp(-x(2))];
end

function alpha = armijo_rule(x)
    % This function minimizes the step size alpha using armijo rule and newton direction
    % Constants used s=1, sigma=0.1, beta=0.5 are given in the problem
    % Definition of functions used in this code
    % --> dk=-inv(hes_mtx(x))*grad_f(x) 
    
    s = 1;
    sigma = 0.1;
    beta= 0.5;
    
    % Newton Direction 
    dk = -hes_mtx(x)\grad_f(x);
    
    alpha = s; % Alpha Initialization
    while ~((f(x)-f(x+alpha*dk) >= -sigma*alpha*grad_f(x)'*dk))
        alpha= alpha*beta;
    end
end
