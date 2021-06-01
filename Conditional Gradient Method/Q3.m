% Code written by Neha Birla for IMSE 505 (Fall 2019) Final exam
% This script is to solve problem 3
% Conditional gradient method using Armijo rule

clc;
clear all;
tol2 = 5e-4; % tolerance for termination criteria
x0=[0 0 1]'; % initial guess of minimizer

% Aeq=[1 1 1];
% beq= 1;
% lb = [0 0 0]; 
% xs = fmincon(@f,x0,[],[],Aeq,beq,lb,[])
% fun = @(x)(0.5*(x(1)^2+x(2)^2+0.1*x(3)^2)+0.55*x(3)^2);

% % create feasible set 
% X=[];
% for i=0:0.01:1
%     for j=0:0.01:1-i
%         ele=[i, j, 1-i-j]';
%         X= [X, ele];
%     end
% end

X=[0 0 1; 0 1 0; 1 0 0]';

% First iteration
x=x0;
fval=f(x);
xb=xbar(x,X);
dk= xb-x; 
d=dk;
alpha = armijo_rule(x,dk);
xkplus1 = x(:,end) + alpha(end)*d(:,end);
x=[x, xkplus1];
fval = [fval, f(x(:,end))];

% Second iteration onwards
i=1;
while (norm(x(:,end) - x(:,end-1)) >tol2)
    
    xb=xbar(x(:,end),X);
    
    dk= xb-x(:,end); 
    d = [d, dk]; % stores search direction at each iteration
    
    alpha = [alpha, armijo_rule(x(:,end),dk)]; % stores step size co-efficient at each iteration 
   
    i = i+1;
    x = [x, x(:,end) + alpha(end)*d(:,end)]; % stores x value at each iteration 
    fval = [fval, f(x(:,end))]; % stores function value at each iteration
end

%%
% Print the table as per sepcifications given in the question
fprintf('iter# \t(x1, x2, x3) \t\t\t\tf(x) \tak \t\tdk \n');
for i=1:size(x,2)-1
    formatSpec = '%d \t\t(%1.4f, %1.4f, %1.4f) \t%1.4f \t%1.3f \t(%1.2d, %1.2d, %1.2d)\n';
    fprintf(formatSpec,i-1,x(1,i),x(2,i),x(3,i),fval(i),alpha(i),d(1,i),d(2,i),d(3,i));  
end
i=i+1;
formatSpec = '%d \t\t(%1.4f, %1.4f, %1.4f) \t%1.4f \n';
fprintf(formatSpec,i-1,x(1,i),x(2,i),x(3,i),fval(i));

%%
function xb=xbar(xk,X)
    gr=grad_f(xk)'*X;
    [~, I]=min(gr);
    xb=X(:,I);
end

function df=grad_f(x)
    df=[x(1); x(2); x(3)+0.55];
end

function fx = f(x)
    fx = 0.5*(x(1)^2+x(2)^2+0.1*x(3)^2)+0.55*x(3);
end

function alpha = armijo_rule(x,dk)
    % This function minimizes the step size alpha using armijo rule and newton direction
    % Constants used s=1, sigma=0.1, beta=0.5 are given in the problem
    % Definition of functions used in this code
    % dk --> feasible descent direction  
    
    s = 1;
    sigma = 0.1;
    beta= 0.9;
         
    alpha = s; % Alpha Initialization
    while ~((f(x)-f(x+alpha*dk) >= -sigma*alpha*grad_f(x)'*dk))
        alpha= alpha*beta;
    end
end

