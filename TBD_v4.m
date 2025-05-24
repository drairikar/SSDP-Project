clc
clear all;
clf;
close all;

folder = 'Results';
if ~exist(folder, 'dir')
    mkdir(folder);
end

NScans = 50;
x = (1:200)';

% x_vals = linspace(1,200, 200);
dimz = length(x);

sigman = 1;
sigmaw = 2;

T = 0.5; 
t = (0:(NScans-1))*T; 

F = [1 T; 0 1];

W = [T^3/3 T^2/2; T^2/2 T]*sigmaw^2; 

Nparticles = 5000;

Nclasses = 2;
x3dB_vals = [2, 3];
alpha = -log(0.5)./(x3dB_vals/2).^2;
prior_class = [0.4, 0.6];
SNR1_dB = 20;
SNR1 = 10^(SNR1_dB/20);

x0 = 10.4;
v0 = 4.6; 

xt_true = x0 + t*v0;

sigmaq = SNR1*sigman;

s_init = [x0+20*randn(1,Nparticles); v0+30*randn(1,Nparticles)]; 
particles_class = cell(Nclasses, 1);

for c_1 = 1:Nclasses
    particles_class{c_1} = s_init;
    % particles_class{c_1} = current_particles_specific;
end

figure(103);
hold on;
for i = 1:length(alpha)
    psf = exp(-alpha(i)*x.^2);
    plot(x, psf, 'LineWidth',2, 'DisplayName',sprintf('Class %d: x_{3dB}=%.1f',i,x3dB_vals(i)))
end
xlabel('Position offset'); ylabel('Amplitude')
legend show; grid on
title('Point-Spread Functions')
filename = fullfile(folder, 'point_spread_functions_v2.png');
exportgraphics(gcf,filename, "Resolution",300);

figure(2)
plot(s_init(1,:),s_init(2,:), 'x')
title ('Particle Cloud') 
xlabel ('Position [m]')
ylabel ('Velocity [m/s]')
grid on

filename = fullfile(folder, 'particle_cloud_v2.png');
exportgraphics(gcf,filename, "Resolution",300);

x_est = zeros(1, NScans);
v_est = zeros(1,NScans);
P_class = zeros(Nclasses, NScans);

true_class = 2;      %ground truth
alpha_true = alpha(true_class);
weights_current_particles = cell(Nclasses,1);

for k = 1:NScans
    at = exp(-alpha_true*(x-xt_true(k)).^2);
    q = sigmaq/sqrt(2)*(randn+1i*randn); 
    n = sigman/sqrt(2)*(randn(dimz,1)+1i*randn(dimz,1));

    z = at*q + n; 
    
    loglik_class = zeros(Nclasses, 1);
    % particles_class = cell(Nclasses, 1);
    weights_class = cell(Nclasses, 1);

    for c = 1:Nclasses
        current_particles_specific = particles_class{c};
        % s_c = particles_class{c};
        alpha_val = alpha(c);
        A = exp(-alpha_val*(x-current_particles_specific(1,:)).^2);
        gamma = (sum(abs(A).^2,1)*sigmaq^2+sigman^2)';
    
        y = A'*z; 
        loglik = real(-log(gamma) + sigmaq^2/(sigman^2)*(abs(y).^2)./gamma);
        maxloglik = max(loglik);
         w = exp(loglik-maxloglik);
         w = w/sum(w);
         weights_class{c} = w;
         % particles_class{c} = s_init;

         loglik_class(c) = maxloglik + log(sum(w));
    
    end
    % logP_class = log(prior_class') + loglik_class;
    if k == 1
        logP_class = log(prior_class') + loglik_class;
    else
        logP_class = log(P_class(:,k-1)) + loglik_class;
    end

    maxlogP = max(logP_class);
    post_class = exp(logP_class - maxlogP);
    post_class = post_class / sum(post_class);
    P_class(:,k) = post_class;

    % --- State estimate: weighted across classes
    x_est(k) = 0; v_est(k) = 0;
    for c2 = 1:Nclasses
        particles_before_resample = particles_class{c2};
        w_c = weights_class{c2};
        x_est(k) = x_est(k) + post_class(c2) * sum(w_c .* particles_before_resample(1,:)');
        v_est(k) = v_est(k) + post_class(c2) * sum(w_c .* particles_before_resample(2,:)');
    end

    % --- Resample and propagate for next step
    for c3 = 1:Nclasses
        % c3
        % pause;
        % particles_before_resample_c = particles_class{c3};
        s_resampled = resample(particles_class{c3}, weights_class{c3}, Nparticles);
        noise = chol(W)' * randn(2, Nparticles);
        s_pred = F * s_resampled + sigmaw * noise;
        particles_class{c3} = s_pred;
    end

    % figure;
    % plot(NScans, P_class(1,:)); hold on
    % plot(NScans, P_class(2,:))

    figure(100);
    plot(particles_class{1}(1,:),particles_class{1}(2,:)); 
    xlabel('Position'); ylabel('Velocity')
    title('Particle Cloud of PSF-1')
    grid on
    filename = fullfile(folder, 'particle_cloud_psf1_v2.png');
    exportgraphics(gcf,filename, "Resolution",300);

    figure(101);
    plot(particles_class{2}(1,:),particles_class{2}(2,:)); 
    xlabel('Position'); ylabel('Velocity')
    title('Particle Cloud of PSF-2')
    grid on
    filename = fullfile(folder, 'particle_cloud_psf_2_v2.png');
    exportgraphics(gcf,filename, "Resolution",300);

    % figure(102);
    % plot(particles_class{3}(1,:),particles_class{3}(2,:)); 
    % xlabel('Position'); ylabel('Velocity')
    % grid on

    % pause;

    % Share same prediction for next iteration
    % s = particles_class{2};  % can use any class's prediction here

end

% pause;
figure(1)
plot(t, xt_true, 'b-', 'LineWidth', 2); hold on
plot(t, x_est, 'r--', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('Position [m]')
title('True vs Estimated Position')
legend('True', 'Estimated')
grid on
filename = fullfile(folder, 'True_vs_Estimated.png');
exportgraphics(gcf,filename, "Resolution",300);

% tight_layout()

figure(3)
plot(t, P_class(1,:), 'LineWidth', 2, 'DisplayName',sprintf('Class 1 (True=%d)', true_class==1)); hold on
plot(t, P_class(2,:), 'LineWidth', 2,'DisplayName',sprintf('Class 1 (True=%d)', true_class==2)); 
% plot(t, P_class(3,:), 'LineWidth',2, 'DisplayName',sprintf('Class 1 (True=%d)', true_class==3));
xlabel('Time [s]'); ylabel('P(class|Z_k)')
title('Posterior Class Probabilities')
legend('PSF Class 1', 'PSF Class 2')
grid on
filename = fullfile(folder, 'Posterior_probabilities_v2.png');
exportgraphics(gcf,filename, "Resolution",300);

