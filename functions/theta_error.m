function [theta_error,beta_theta] = theta_error(theta_true, theta_hat)
K = length(theta_true);
if isempty(theta_hat)
    theta_error = ones(K,1);
else
    theta_error = zeros(K, 1);
    for k = 1:K
        theta_error_k = abs(theta_hat - theta_true(k));
        theta_error(k) = min(theta_error_k);
    end
end
beta_theta = 1/K * sum(theta_error);
end