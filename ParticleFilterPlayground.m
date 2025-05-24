% This file simulates a Kalman Filter and a Particle Filter on a trivial
% linear Gaussian estimation problem. 

% Initial position and speed of an object

x = 0;
v = 50;

% Measurement Interval

T = 2;

% Observation and process noise variance

varn = 10^2;
varw = 5^2;

t = 0:T:40*T;

% Generate a set of position measurements

x_true = x+t*v;
v_true = v+0*t;
xm = x_true+sqrt(varn)*randn(size(t));



% # particles in the filter

Np = 5000;

% Initialisation of the particle cloud

x00 = 1E3*randn(1,Np);
v00 = 1E2*randn(1,Np);

si = [x00; v00];


% Initial state of the Kalman filter 

sp = [0; 0];
Qp = diag([1E6 1E4]);

% Definition of system matrices: 
% H = Observation matrix, 
% F = System Dynamics Matrix
% W = part of Process Noise Covariance matrix

H = [1 0];
F = [1 T; 0 1];
%C = [T^2/2; T];
W = [T^3/3 T^2/2; T^2/2 T];

wr = ones(1,Np)/Np;

for k=1:41
    
    k
    % KF update
    
    G = Qp*H'*inv(H*Qp*H' + varn);
    
    sf = sp + G*(xm(k)-H*sp);
    Qf = (eye(2)-G*H)*Qp;
    

   
    
    %PF update
    
    % Calculate (for all particles at the same time) the log likelihood
    % Convert (in a robust way!!) to normalized likelihoods, 
    % ie the particle weights...  
    % Then resample (wheel of fortune)
    %
    
    loglik =  (-(xm(k)-H*si).^2./2./varn);
    
    maxloglik = max(loglik);
    
    w = exp(loglik-maxloglik);
   
    w = w/sum(w);
    
    [sr, wr, indx] = resample (si,w, Np);

   
figure(1)    

   

    plot (sr(1,:),sr(2,:), '.')
    title ('Particle cloud after resampling') 

    pause
    
   mean_KF(:,k) = sf;
   mean_PF(:,k) = (mean(sr'))';
    
   figure(2)
   

   plot (1:k, mean_KF(1,1:k)-x_true(1:k), 'r+')
   hold
   plot (1:k, mean_PF(1,1:k)-x_true(1:k), 'b+')
   hold
   title ('Position estimation error of KF and PF')

   legend ('KF', 'PF')

   
figure(3)
   

   plot (1:k, mean_KF(2,1:k)-v_true(1:k), 'r+')
   hold
   plot (1:k, mean_PF(2,1:k)-v_true(1:k), 'b+')
   hold
   title ('Velocity estimation error of KF and PF')

   legend ('KF', 'PF')

    sigmas_KF = sqrt(diag(Qf));
    sigmas_PF = sqrt(diag(cov(sr')));
    
    % KF prediction
    
    sp = F*sf;
    Qp = F*Qf*F' + W*varw;
    
    %PF prediction
    
    ww = randn(2,Np);
    
    A = chol(W);
    
    si = F*sr + A'*sqrt(varw)*ww;

        
end

figure(10)

plot (xm, 'k-x')
hold
plot (mean_KF(1,:), 'b-o')
plot (mean_PF(1,:), 'r-+')
hold


