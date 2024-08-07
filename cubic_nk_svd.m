function [f_out,X_out,residueList] = cubic_nk_svd(Y,R,iternum,N,T,gamma,Y_full,S,proj_style,thre)
%%
% Inputs:
%	    Y           : Measurements
%       R           : Number of initial frequencies
%       iternum     : Maximum number of iterations
%       N           : Number of signal dimension
%       T           : Number of snapshots
%       gamma       : Oversampling factor
%       S           : Measurement matrix
%       proj_style  : Type of measurement matrix
%       thre        : Threshold for OMP stopping criterion
% Outputs:
%       f_out       : Eitimated frequencies
%       X_out       : Eitimated gains
%       residueList : Trajectory of the energy in the residual
% Author:
%       Name        : Xiaozhi Liu (xzliu@buaa.edu.cn)

%% Initialization
f = 0:1/R:1-1/R;
f = f * 2 * pi;
D = exp(1j * (0:(N-1)).' * f) /sqrt(N);
X = zeros(R,T);
residueList = [];
epsilon = 1e-3;
R_s = 10; 

for iter = 1:iternum 
    % Sparse Coding Stage
    X = func_sp(Y,D,S,thre);
    
    % Atom Update Stage
    [D,f,X] = func_up(Y,D,f,X,gamma,R_s,S,proj_style);

    res = compute_res(D,X,Y_full,f);
    residueList = [residueList; res];
    if iter >= 10 && abs( residueList(end) - residueList(end-1) ) < epsilon
        break;
    end
end

[~,supp,~] = find(D); 
supp = unique(supp);
f_out = f(supp);
X_out = X(supp,:);
end

function res = compute_res(D,X,Y_full,f)
[~,supp,~] = find(D);
N = size(D,1);
supp = unique(supp);
f_out = f(supp);
X_out = X(supp,:);
Y_est_dl = exp(1j* (0:(N-1)).' * f_out)/sqrt(N) * X_out;
res = mean(vecnorm(Y_full - Y_est_dl) ./ vecnorm(Y_full));
end

function X = func_sp(Y,D,S,thre)
[~,T] = size(Y);
[~,R] = size(D);
X = zeros(R,T);

for j = 1:T
    X(:,j) = omp(Y(:,j),D,S,thre);
end
end

function x = omp(y,D,S,thre)
[~,R] = size(D);
res = y;
x = zeros(R,1);
D_supp = [];
supp = [];
D = S*D;

for iter = 1:R
    product = abs(D'*res);
    [~,pos]=max(product);
    D_supp = [D_supp, D(:,pos)];
    D(:,pos)=0;
    supp = [supp,pos];
    x_supp = pinv(D_supp)*y;
    res=y-D_supp*x_supp;
    if norm(res) < thre
        break;
    end
end

x(supp) = x_supp;
end

function [D_out,f_out,X_out] = func_up(Y,D,f,X,gamma,R_s,S,proj_style)
for j = 1:size(D,2)
    func_value_old = norm(Y-S*D*X,'fro'); 
    [~, data_indices, ~] = find(X(j,:));
    d = D(:,j);
    x_up = X(j,:);
    D(:,j) = 0;
    % Elimimation
    if (isempty(data_indices))
        D(:,j) = 0;
        continue;
    end
    smallX = X(:,data_indices);
    smallY = Y(:,data_indices);
    [d_new_cp,~] = ksvd_up(smallY,S*D,smallX);
    E = smallY-S*D*smallX;
    [d_refine,f_refine,x_refine] = newton_refine(E,d_new_cp,gamma,R_s,S,proj_style);
    x_up(data_indices) = x_refine;
    func_value_new = norm(Y-S*D*X-S*d_refine*x_up,'fro');
    if func_value_new < func_value_old
        f(j) = mod(f_refine,2*pi);
        D(:,j) = d_refine;
        X(j,data_indices) = x_refine;
    else
        D(:,j) = d;
    end
end

f_out = f;
D_out = D;
X_out = X;
end

function [d,x] = ksvd_up(Y,D,X)
[d,s,x] = svds(Y - D*X, 1);
x = s*x;
end

function [d,f,x] = newton_refine(E,d_svd,gamma,R_s,S,proj_style)
% Coarse Estimate
N = size(S,2);
grid_num = floor(gamma*N);
f_candidate = 0:1/grid_num:(1-1/grid_num);
f_candidate = f_candidate * 2 * pi;
D_candidate = exp(1j * (0:(N-1)).' * f_candidate) /sqrt(N);
[~,pos]=max(abs((S*D_candidate)'*d_svd));
f_coarse = f_candidate(pos);

% Cubic Newton Refinement
[f,d,x] = cubic_newton_refine(f_coarse,E,R_s,S,proj_style);
end

function [f_refine,d_refine,x_refine] = cubic_newton_refine(f_coarse,E,R_s,S,proj_style)
[M,N] = size(S);
amp = (0:N-1)';
f_up = f_coarse;
f_refine = f_coarse; 
d = exp(1j * (0:(N-1)).' * f_coarse) /sqrt(N);
d_cp = S*d;
x_coarse = 1/norm(d_cp,2)^2 * d_cp'*E;
obj_old = norm(E - d_cp*x_coarse,"fro");

switch proj_style
    case 'subsample'
        for i = 1:R_s
            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            d_cp = S*d;
            x = N/M * d_cp'*E; 

            dd_f = 1j * amp .* d;
            d2d_f = - (amp.^2) .* d;

            rmp = x*E';
            der1 = -2*real(rmp * S*dd_f);
            der2 = -2*real(rmp * S*d2d_f);

            L = 2*norm((amp.^2) .* (S'*rmp'))*sqrt((N-1)*(2*N-3)/6);
            f_up = cubic_newton_aulixiary(der1,der2,L,f_up);

            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            d_cp = S*d;
            x = N/M * d_cp'*E;

            obj_new = norm(E - d_cp*x,"fro");
            if obj_new < obj_old
                f_refine = f_up;
                obj_old = obj_new;
            end
        end
    case 'full'
        for i = 1:R_s
            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            x = d'*E; 

            dd_f = 1j * amp .* d;
            d2d_f = - (amp.^2) .* d;

            rmp = x*E';
            der1 = -2*real(rmp * dd_f); 
            der2 = -2*real(rmp * d2d_f); 

            L = 2*norm((amp.^2) .* rmp')*sqrt((N-1)*(2*N-3)/6);
            f_up = cubic_newton_aulixiary(der1,der2,L,f_up);

            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            x = d'*E;

            obj_new = norm(E - d*x,"fro");
            if obj_new < obj_old
                f_refine = f_up;
                obj_old = obj_new;
            end
        end
    otherwise
        for i = 1:R_s
            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            d_cp = S*d;
            x =  1/norm(d_cp)^2 * d_cp'*E;

            dd_f = 1j * amp .* d;
            d2d_f = - (amp.^2) .* d;

            rmp = x*E';
            der1 = norm(x)^2*2*real(d_cp'*S*dd_f) - 2*real(rmp * S*dd_f);
            der2 = norm(x)^2*2*real(d_cp'*S*d2d_f + norm(S*dd_f)^2) - 2*real(rmp * S * d2d_f);
            
            if der2 > 0
                f_up = f_up - der1/der2;
            else
                break;
            end
            d = exp(1j * (0:(N-1)).' * f_up) /sqrt(N);
            d_cp = S*d;
            x = 1/norm(d_cp)^2 * d_cp'*E;

            obj_new = norm(E - d_cp*x,"fro");
            if obj_new < obj_old
                f_refine = f_up;
                obj_old = obj_new;
            end
        end
end

d_refine = exp(1j * (0:(N-1)).' * f_refine) /sqrt(N);
d_refine_cp = S*d_refine;
x_refine = 1/norm(d_refine_cp,2)^2 *(d_refine_cp)'*E;
end

function f_up_candidate = cubic_newton_aulixiary(der1,der2,L,f_up)
Delta_f_candidate_set = [];

if (der2^2 - 4*der1*L/2) == 0
    Delta_f_1 = 1/L * (-der2);
    if Delta_f_1 >= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_1];
    end
elseif (der2^2 - 4*der1*L/2) > 0
    Delta_f_1 = 1/L * (-der2 + sqrt(der2^2 - 4*der1*L/2));
    Delta_f_2 = 1/L * (-der2 - sqrt(der2^2 - 4*der1*L/2));
    if Delta_f_1 >= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_1];
    end
    if Delta_f_2 >= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_2];
    end
end

if (der2^2 + 4*der1*L/2) == 0
    Delta_f_1 = -1/L * (-der2);
    if Delta_f_1 <= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_1];
    end
elseif (der2^2 + 4*der1*L/2) > 0
    Delta_f_1 = -1/L * (-der2 + sqrt(der2^2 + 4*der1*L/2));
    Delta_f_2 = -1/L * (-der2 - sqrt(der2^2 + 4*der1*L/2));
    if Delta_f_1 <= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_1];
    end
    if Delta_f_2 <= 0
        Delta_f_candidate_set = [Delta_f_candidate_set, Delta_f_2];
    end
end

f_up_candidate = Delta_f_candidate_set(1) + f_up;
obj_check = der1*f_up_candidate + 0.5*der2*f_up_candidate + L/6*abs(f_up_candidate)^3;

if length(Delta_f_candidate_set)>1
    for j = 2:length(Delta_f_candidate_set)
        f_up_up_candidate = Delta_f_candidate_set(j) + f_up;
        if der1*f_up_up_candidate + 0.5*der2*f_up_up_candidate + L/6*abs(f_up_up_candidate)^3 < obj_check
            f_up_candidate = f_up_up_candidate;
            obj_check = der1*f_up_up_candidate + 0.5*der2*f_up_up_candidate + L/6*abs(f_up_up_candidate)^3;
        end
    end
end
end









