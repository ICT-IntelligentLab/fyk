% MATLAB Code: Alice-Bob Channel Probing with Pilots
% Includes pilot transmission, reception, and simple LS channel estimation


clear; clc;

%% Parameters
N = 128;                    % Number of subcarriers
Delta_f = 1e3;              % Subcarrier spacing (Hz)
fs = N * Delta_f;           % Sampling frequency
T = 1 / fs;                 % Sample period
fc = 4e9;                   % Carrier frequency (Hz)
v = 810 / 3.6;              % Speed (m/s, 810 km/h)
c = 3e8;                    % Speed of light
nu_max = v * fc / c;        % Max Doppler shift (~3000 Hz)
f_max = nu_max / Delta_f;   % Normalized max Doppler (~3)
c1 = (2 * f_max + 1) / (2 * N);  % Post-chirp parameter 

% EVA Channel Model
tau = [0, 30, 150, 310, 370, 710, 1090, 1730, 2510] * 1e-9;  % Delays (s)
powers_dB = [0, -1.5, -1.4, -3.6, -0.6, -9.1, -7, -12, -16.9];  % Powers (dB)
sigma_p2 = 10.^(powers_dB / 10);
sigma_p2 = sigma_p2 / sum(sigma_p2);  % Normalize
P = length(tau);
lp = round(tau / T);        % Normalized integer delays

% Quantization parameters 
eta = 7;                    % Levels
rho_min = -3;
rho_max = 3; 

% Pilot parameters
pilot_power = 1;            % Normalized pilot energy
SNR_pilot_dB = 20;          % Pilot SNR (dB), controls estimation quality
N0_pilot = 10^(-SNR_pilot_dB / 10) / pilot_power;  % Noise variance

num_monte = 100;            % Monte Carlo trials

%% build LTV Channel matrix H
function H = gen_H(N, lp, fp, hp)
    H = zeros(N, N);
    for p = 1:length(lp)
        % Doppler shift matrix
        Delta = diag(exp(-1j * 2 * pi * fp(p) * (0:N-1)' / N));
        % Delay circulant matrix
        shift = mod(lp(p), N);
        Pi = circshift(eye(N), shift, 2);  % Column shift for circulant 2 stands for 列
        H = H + hp(p) * Delta * Pi;
    end
end

%% Function: Extract feature and quantize to c2
function c2 = get_c2(H_diag, eta, rho_min, rho_max)
    feature = real(H_diag);  % Real part of diagonal
    alpha = (rho_max - rho_min) / (eta - 1);
    idx = floor((feature - rho_min) / alpha + 0.5);
    idx = max(0, min(eta - 1, idx));
    c2 = rho_min + idx * alpha;  % N x 1 vector

    %将feature 四舍五入到最近的量化网格点（0-7）
end

%% Main Simulation: Channel Probing with Pilots
match_rates = zeros(num_monte, 1);

for trial = 1:num_monte
    % Generate true shared channel
    theta = 2 * pi * rand(P, 1) - pi;  % Random angles
    nu_p = nu_max * cos(theta);        % Dopplers
    fp = nu_p / Delta_f;               % Normalized
    hp_true = sqrt(sigma_p2 / 2).' .* (randn(P, 1) + 1j * randn(P, 1));  % Gains
    H_true = gen_H(N, lp, fp, hp_true);  % True channel matrix 生成一百个信道 定义成AFDM的信道
    
    % Pilot signal (all-ones, unit energy)
    pilot = ones(N, 1) / sqrt(N);
    
    % Step 1: Alice sends pilot to Bob (forward channel)
    noise_B = sqrt(N0_pilot / 2) * (randn(N, 1) + 1j * randn(N, 1));
    r_B = H_true * pilot + noise_B;    % Bob receives
    % LS estimation at Bob (simple: since pilot constant, estimate effective diagonal)
    H_est_B = r_B ./ pilot;            % N x 1 (effective diagonal estimate)
    
    % Step 2: Bob sends pilot to Alice (reverse channel, reciprocity: H' )
    noise_A = sqrt(N0_pilot / 2) * (randn(N, 1) + 1j * randn(N, 1));
    r_A = H_true' * pilot + noise_A;   % Alice receives
    % LS estimation at Alice
    H_est_A = r_A ./ pilot;            % N x 1
    
    % Generate c2 from estimated diagonals
    c2_A = get_c2(H_est_A, eta, rho_min, rho_max);
    c2_B = get_c2(H_est_B, eta, rho_min, rho_max);
    
    % Compute match rate
    match_rates(trial) = mean(c2_A == c2_B);
end

%% Display Results
avg_match = mean(match_rates);
fprintf('Average Match Rate (Pilot SNR = %d dB): %.4f\n', SNR_pilot_dB, avg_match);

% Example output for last trial
disp('Example c2_A (first 10):');
disp(c2_A(1:10)');
disp('Example c2_B (first 10):');
disp(c2_B(1:10)');