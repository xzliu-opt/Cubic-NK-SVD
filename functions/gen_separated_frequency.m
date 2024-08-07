function omega = gen_separated_frequency(K, N, mu)
omega = rand(K,1)*2*pi;
for i = 2:K
    dist_f = min(abs(omega(i) - omega(1:i-1)));
    while dist_f <= mu*2*pi/N
        omega(i) = rand(1)*2*pi;
        dist_f = min(abs(omega(i) - omega(1:i-1)));
    end
end
end