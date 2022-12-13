% =========================================================================
% This is Newton Raphson solution of two nonlinear algebraic equations code
% count     : the number of the iterations. 
% cc   : if cc == 1, case01;
%        if cc == 2, case02;
% x    : the point we which to perform the evaluation.
%
% Output: the value of approximation of displacement and number of iterations.
% -------------------------------------------------------------------------
% By Jia Luo , 2021 Dec. 6th.
% =========================================================================
close all;clc ,clear;
Nd1 = @(d) 0.2.*d.^3 - 1.8.*d.^2+6.*d;
Nd2 = @(d) 0.2.*d.^3 - 2.1.*d.^2+6.*d;

%d = [0.1;0.1 ] ; %initial approximation 
stop_tol = 10^-4;

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

        f_t = [t(n)*f_ext(1);t(n)*f_ext(2)];
        r0 = r(d,f_t,const);         % evaluate {f}
        er0 = norm(r0);

        %--------------------------------------------
        % iteration loop
        while (er>stop_tol && count<maxit)
            count = count+1;    % increment the counter

            ri = r(d,f_t,const);     % evaluate {r}
            Ji = JacobM(d,const);    % evaluate the Jacobian [J]

            delta_d = -Ji\ri;        % calculate {delta x} = -[J]^(-1)*{r}
            d = d + delta_d;      % calculate the new estimate

            rii = r(d,f_t,const);

            eri = norm(rii);
            er = eri/er0;
            %er = norm(dx)/norm(di); % approximate relative error
            %fprintf('%3g %3g %10.6g %10.6g %10.4g\n',n, count, d(1), d(2), er);   
        end

        %-------------------------------------------- save History data
        if cc == 1
            His1_d(n,:) = [n,count,d(1),d(2),er];
        elseif cc == 2
            His2_d(n,:) = [n,count,d(1),d(2),er];
        end

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
title("N1-d1 case01 x = 1.8");
xlabel('d1 displacement');
ylabel('N1(d)-case1 (internal force )');
legend([N1_exp ,N1_h],'N1_{exact}','N1_h',"Location","best");
exportgraphics(gca,['N-d-case1' '.jpg']);

NR_T1 = table(His1_d(:,1),His1_d(:,2),His1_d(:,3),His1_d(:,4),His1_d(:,5),'variableNames',{'Load step','Iterations ','d1','d2','Residual error'});
writetable(NR_T1);
NR_T1

figure
d2 = 0:0.1:8;
N2_exact = Nd2(d2);
N2_exp = plot(d2,N2_exact,'k','LineWidth',2);
hold on
d_h2 = His2_d(2:end,3);
N_2h = Nd2(d_h2);
N2_hp = plot(d_h2,N_2h,'--ro','linewidth',2);
hold off
title("N1-d1 case02 x = 2.1");
xlabel('d1 displacement');
ylabel('N1(d)-case2 (internal force )');
legend([N2_exp ,N2_hp],'N2_{exact}','N2_h',"Location","best")
exportgraphics(gca,['N-d-case2' '.jpg']);

NR_T2 = table(His2_d(:,1),His2_d(:,2),His2_d(:,3),His2_d(:,4),His2_d(:,5),'variableNames',{'Load step','Iterations ','d1','d2','Residual error'});
writetable(NR_T2);
NR_T2
%--------------------------------------------
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


function r = r(d,f_t,constant)

r = [0.2*d(1)^3-constant*d(2)^2+6*d(1)-f_t(1);-d(1)+d(2)-f_t(2)];

end

function JacobM = JacobM(d,constant)

JacobM =[0.2*3*d(1)^2+6,-constant*2*d(2);-1,1];

end





























