%%
% This is the simulation code for 
%       An Algorithm for Designing Parametric Dictionary for Frequency Estimation     
% Author: Xiaozhi Liu (xzliu@buaa.edu.cn), Yong Xia (yxia@buaa.edu.cn)
% Affiliation: BUAA
% Date: 2024/8/1

clear; close all; clc;
rng(2024)
addpath(genpath('functions'))

% Setup
N = 64;
M = 32;
T = 1;
K = 5;
PSNR = 10;
mu = 2;
gain_true = (3*rand(K,T)+10) .* exp(1j*2*pi*rand(K,T));
omega_true = gen_separated_frequency(K,N,mu);
Y_full = exp(1j* (0:(N-1)).' * omega_true.')/sqrt(N) * gain_true;
proj_style = 'subsample';
[S,smp] = generateMeasMat(N,M,proj_style);
sigma = 10^(-PSNR/20);
Noise = sigma * (randn(N,T) + 1j*randn(N,T))/sqrt(2); 
Y_noisy = Y_full + Noise;
Y_mea = S * Y_noisy;

disp('**********Cubic NK-SVD**********')
R = N; 
gamma = 10;
thre = sqrt(M)*sigma*3;
iternum = 30;
tic
[omega_est, gain_est,residueList] = cubic_nk_svd(Y_mea, R, iternum, N, T, gamma, Y_full, S, proj_style, thre);
toc
omega_est = omega_est';
Y_est = exp(1j* (0:(N-1)).' * omega_est.')/sqrt(N) * gain_est;
nmse = mean(vecnorm(Y_full - Y_est) ./ vecnorm(Y_full));
[theta_error,beta_theta] = theta_error(omega_true, omega_est);

% Plot figures
figure;
subplot(1,2,1); 
polar(omega_true, abs(gain_true(:,1)), 'bo');
hold on; 
polar(omega_est, abs(gain_est(:,1)),'rx');
title(sprintf('Magnitude and frequency\nof each sinusoid'));
legend({'Ground truth', 'Cubic NK-SVD'}, 'Location', 'SouthOutside');
subplot(1,2,2); 
polar(angle(gain_true(:,1)), abs(gain_true(:,1)), 'bo');
hold on; 
polar(angle(gain_est(:,1)), abs(gain_est(:,1)), 'rx');
hold off
title(sprintf('Magnitude and phase\nof each sinusoid'));
legend({'Ground truth', 'Cubic NK-SVD'}, 'Location', 'SouthOutside');

figure()
xx = real(exp(2*pi*1i*(0:0.001:1)));
yy = imag(exp(2*pi*1i*(0:0.001:1)));
zz = zeros(1,length(0:0.001:1));
plot3(xx,yy,zz,'-k','LineWidth',0.002);
grid on;
set(gca, 'GridLineStyle', ':');  
set(gca, 'GridAlpha', 1);  
hold on;
xx_o = real(exp(1j*omega_true));
yy_o = imag(exp(1j*omega_true));
zz_o = (abs(gain_true).^2)/N;
stem3(xx_o,yy_o,zz_o,'xr');
xlim([-1,1]);
ylim([-1,1]);
title('Ground truth');
   
figure()
plot3(xx,yy,zz,'-k','LineWidth',0.002);
grid on;
set(gca, 'GridLineStyle', ':');  
set(gca, 'GridAlpha', 1);  
hold on;
xx_est = real(exp(1j*omega_est));
yy_est = imag(exp(1j*omega_est));
zz_est = (abs(gain_est).^2)/N;
stem3(xx_est,yy_est,zz_est,'ob');
xlim([-1,1]);
ylim([-1,1]);
title('Cubic NK-SVD');

