% =========================================================================
% This is Newton Raphson solution of two nonlinear algebraic equations code
% count : the number of the iterations.
% cc   : if cc == 1, case01;
%        if cc == 2, case02;
% x    : the point we which to perform the evaluation.
%
% Output: the value of approximation of displacement and number of iterations.
% -------------------------------------------------------------------------
% By Jia Luo , 2021 Dec. 15th.
% =========================================================================
close all;clc ,clear;
Nd1 = @(d) 0.2.*d.^3 - 1.8.*d.^2+6.*d;
Nd2 = @(d) 0.2.*d.^3 - 2.1.*d.^2+6.*d;

%d = [0.1;0.1 ] ; %initial approximation
stop_tol = 10^-4;
stol = 0.5; % tolerence for line search, based on experienced test
maxit = 15;

f_ext = [10;0]; % external load
constant = [1.8,2.1];
t = 0:1/40:1; % load  step
n_step = length(t);
His1_d = zeros(n_step,5);
His2_d = zeros(n_step,5);

for cc = 1:2
    d = [0.1;0.1 ] ; %initial approximation
    const = constant(cc);
    for n = 1: n_step
        count = 0;
        er =1;
        s=1;

        f_t = [t(n)*f_ext(1);t(n)*f_ext(2)];
        R_0 = r(d,f_t,const);         % evaluate {f}
        % Main Modified Newton-Raphson equilibrium iteration
        Ji = JacobM(d,const);    % evaluate the Jacobian [J]
        delta_d = Ji\R_0;          % calculate {delta x} = -[J]^(-1)*{r}
      

        G0 = dot(delta_d,R_0);
        d_temp = d - delta_d;
        R_new = r(d_temp,f_t,const);        %calculate the residual
        G  = dot(delta_d,R_new);
        s= 1;

        if(abs(G) > stol*abs(G0))
            s = linesearch(G0,G,d,delta_d,stol,s,f_t,const);
        end

        d = d - s*delta_d;% calculate the new estimate based on s     
        R_new = r(d,f_t,const);        %calculate the residual
        G  = dot(delta_d,R_new);

        er0 = norm(R_new);
        s=1; 
        %--------------------------------------------
        % iteration loop
        while (er>stop_tol && count<maxit)
            count = count+1;    % increment the counter

            % ri = r(d,f_t,const);     % evaluate {r}
            % i = JacobM(d,const);     % evaluate the Jacobian [J]          
            % Do Line search if it is necessary ,solve line search scale 
            % -------------------------------------------------------------------------
            % 1.compute the search direction by multiple BFGS update algorithm.
            % -------------------------------------------------------------------------
            % delta_d = findnewd(delta_d,Ji,r0,rs,G0,G,s,count);
             [delta_d] = findnewd(delta_d,Ji,R_0,R_new,s,count);
             R_0 = R_new;
            % -------------------------------------------------------------------------
            % 2. Main line search loop. iterate to find search parameter s.
            % -------------------------------------------------------------------------       
            % Find s s.t. delta_d is orthogonal to residual vector R,
            % or the potential energy functional is minimized. 
            d_temp = d - delta_d; 
            R_new = r(d_temp,f_t,const);        %calculate the residual
            G0 = dot(delta_d,R_0);
            G  = dot(delta_d,R_new);  

            if(abs(G) > stol*abs(G0))
                s = linesearch(G0,G,d,delta_d,stol,s,f_t,const);
            end  
            
            % -------------------------------------------------------------------------
            % 3. update.
            % -------------------------------------------------------------------------
            d = d - s*delta_d;      % calculate the new estimate based on s
            %           d = d + delta_d;        % calculate the new estimate
            R_new = r(d,f_t,const);

            eri = norm(R_new);
            er = eri/er0;
            %--------------------------------------------
            %Limits the Maximum of updates 15, reinitialized the K0, pool of BFGS vectors



            %er = norm(dx)/norm(di); % approximate relative error
            %fprintf('%3g %3g %10.6g %10.6g %10.4g\n',n, count, d(1), d(2), er);
        end
        % -------------------------------------------------------------------------
        % 4. save History data.
        % -------------------------------------------------------------------------
        if cc == 1
            His1_d(n,:) = [n,count,d(1),d(2),er];
        elseif cc == 2
            His2_d(n,:) = [n,count,d(1),d(2),er];
        end
        % load step accepted, continue the Newton iteration.
    end
end

format long
%     His1_d
%     His2_d
%-------------------------------------------

figure
d1 = 0:0.1:8;
N1_exact = Nd1(d1);
N1_exp = plot(d1,N1_exact,'k','LineWidth',2);
hold on
d_h1 = His1_d(2:end,3);
N_1h = Nd1(d_h1);
N1_h = plot(d_h1,N_1h,'--bo','linewidth',2);
hold off
title("N1-d1 case01 x = 1.8 Modified N-R method");
xlabel('d1 displacement');
ylabel('N1(d)-case1 (internal force )');
legend([N1_exp ,N1_h],'N1_{exact}','N1_h',"Location","best");
exportgraphics(gca,['N-d-LS_BFGS_Hughes' '.jpg']);

M_NR_LS_BFGS_Strange_T1 = table(His1_d(:,1),His1_d(:,2),His1_d(:,3),His1_d(:,4),His1_d(:,5),'variableNames',{'Load step','Iterations ','d1','d2','Residual error'});
writetable(M_NR_LS_BFGS_Strange_T1);
M_NR_LS_BFGS_Strange_T1

figure
d2 = 0:0.1:8;
N2_exact = Nd2(d2);
N2_exp = plot(d2,N2_exact,'k','LineWidth',2);
hold on
d_h2 = His2_d(2:end,3);
N_2h = Nd2(d_h2);
N2_hp = plot(d_h2,N_2h,'--ro','linewidth',2);
hold off
title("N1-d1 case02 x = 2.1 Modified N-R method ");
xlabel('d1 displacement');
ylabel('N1(d)-case2 (internal force )');
legend([N2_exp ,N2_hp],'N2_{exact}','N2_h',"Location","best")
exportgraphics(gca,['N-d-LS_BFGS_Hughes' '.jpg']);

M_NR_LS_BFGS_Strange_T2 = table(His2_d(:,1),His2_d(:,2),His2_d(:,3),His2_d(:,4),His2_d(:,5),'variableNames',{'Load step','Iterations ','d1','d2','Residual error'});
writetable(M_NR_LS_BFGS_Strange_T2);
M_NR_LS_BFGS_Strange_T2

% =========================================================================

%compute BFGS search direction
% function [d] = findnewd(delta_d,Ji,r0,rs,G0,G,s,step_i)
function [d] = findnewd(delta_d,Ji,r0,rs,s,step_i)
% -------------------------------------------------------------------------
% 1. Form BFGS vector pairs.
% -------------------------------------------------------------------------
% deltagamma = s*(G-G0); deltaKdelta = s^2*G0;
% fact1 = 1+s*sqrt(deltagamma*deltaKdelta);
% fact2 = s/deltagamma;
delta = -s*delta_d;
gamma = rs - r0; % residual vector
alpha = s * sqrt(dot(delta_d,gamma) / dot(delta,r0));
if dot(delta_d,gamma) / dot(delta,r0)<0
   alpha = s * sqrt(-dot(delta_d,gamma) / dot(delta,r0));
end

V(:,step_i) = r0*(1+alpha) - rs;       % BFGS updates pairs
W(:,step_i) = delta / dot(delta,gamma);       % BFGS updates pairs

delta_d = rs;

% V(:,step_i) = fact1*r0 - rs;       % BFGS updates pairs
% W(:,step_i) = fact2*delta_d;       % BFGS updates pairs
% -------------------------------------------------------------------------
% 2. Inverse Update.
% -------------------------------------------------------------------------

%right side update
for i = 1:step_i
    delta_d = delta_d + dot(W(:,step_i-i+1),delta_d)*V(:,step_i-i+1);
end

%solve the intermediate system
delta_d0 = Ji \ delta_d;

%left side updates
for j = 1:step_i
    delta_d0 = delta_d0 + dot(V(:,j),delta_d0)*W(:,j);
end
d = delta_d0;

end

% =========================================================================

% The max iterations for search parameter : 5; stol = 0.5;
% w.t get si(scale parameter) s.t orthogonal to residual vector r_i+1
function [s] = linesearch(G0,G,d,delta_d,stol,s0,f_t,const)
% Step 1 find interval containing zero
d_temp = d;
sb = 0; sa = s0;
Gb = G0; Ga = G;

%sign_G = sign(Ga)*sign(Gb);
while sign(Ga)*sign(Gb) >0
    sb = sa; sa = 2*sa;
    Gb = Ga;

    d_temp = d_temp - sa*delta_d;
    r_temp = r(d_temp,f_t,const);
    Ga = dot(delta_d,r_temp);
end
step = sa; G = Ga;

%Step 2 use Illinois algorithm to find the zero
ls_count = 0 ; maxitls = 5;
while(sign(Ga)*sign(Gb) < 0 && (abs(G) > stol*abs(G0) || abs(sb -sa)) > stol*0.5*(sb+sa))
    ls_count = ls_count +1;

    step = sa - Ga*(sa - sb)/(Ga - Gb);
    d_temp = d_temp - step*delta_d;
    r_temp = r(d_temp,f_t,const); G = dot(delta_d,r_temp);

    if sign(Ga)*sign(G) > 0
        Gb = 0.5*Gb;
    else
        sb = sa; Gb = Ga;
    end
    sa = step; Ga = G;

    if(ls_count > maxitls)
        disp(' line search reach the max ls-iteration number 5');
        break
    end
end

s = step;
if s >1
    s=1;
end

end
% =========================================================================

% function r = r(d,f_t,constant)
% 
% r = [-0.2*d(1)^3+constant*d(2)^2-6*d(1)+f_t(1);d(1)-d(2)+f_t(2)];
% 
% end
% 
% function JacobM = JacobM(d,constant)
% 
% JacobM =[-0.2*3*d(1)^2-6,constant*2*d(2);1,-1];
% 
% end
% =========================================================================
function [r] = r(d,f_t,constant)
r = [0.2*d(1)^3-constant*d(2)^2+6*d(1)-f_t(1);-d(1)+d(2)-f_t(2)];
end

function [JacobM] = JacobM(d,constant)
JacobM =[0.2*3*d(1)^2+6,-constant*2*d(2);-1,1];
end
% =========================================================================

% END