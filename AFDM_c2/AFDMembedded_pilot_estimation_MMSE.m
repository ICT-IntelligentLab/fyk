% AFDM 嵌入导频信道估计 - 正确对角 LMMSE 版
clear; clc; close all;

%% 参数设置（同之前）
N = 128;
c1 = 1/(2*N);
c2 = 1/N;
modOrder = 4;
snr_dB = 20;               % 20dB 下对比明显

P = 4;
l_max = 15;
k_max = 8;

pilot_pos = floor(N/2) + 1;
pilot_val = sqrt(N)*8;     % 强导频
guard_len = 2*(l_max + 2*abs(k_max)) + 20;

%% DAFT 矩阵（同之前）
n = (0:N-1).';
Lambda_c1 = diag(exp(1j * pi * c1 * n.^2));
Lambda_c2 = diag(exp(1j * pi * c2 * n.^2));
F_dft = dftmtx(N) / sqrt(N);
F_inv = Lambda_c2 * F_dft * Lambda_c1;
F = F_inv';

%% 数据位置和发送符号（同之前）
left_end = pilot_pos - guard_len - 1;
right_start = pilot_pos + guard_len + 1;
data_idx = [1:max(0,left_end), min(N,right_start):N];
num_available = length(data_idx);
disp(['可用数据位置: ' num2str(num_available)]);

data = qammod(randi([0 modOrder-1], num_available, 1), modOrder, 'UnitAveragePower',true);

x_daft = zeros(N,1);
x_daft(data_idx) = data;
x_daft(pilot_pos) = pilot_val;

%% 生成信道 H_eff
H_eff = zeros(N,N);
for p = 1:P
    l_p = randi([0 l_max]);
    k_p = randi([-k_max k_max]);
    h_p = (randn + 1i*randn)/sqrt(2);
    doppler_phase = exp(1j * 2*pi * k_p * (0:N-1)' / N);
    delay_shift = circshift(eye(N), l_p, 1);
    H_eff = H_eff + h_p * diag(doppler_phase) * delay_shift;
end

%% 通过信道 + 噪声
y_daft_noiseless = H_eff * x_daft;
noise_power = 10^(-snr_dB/10);   % 信号功率归一化1，噪声功率
noise = sqrt(noise_power/2) * (randn(N,1) + 1i*randn(N,1));
y_daft = y_daft_noiseless + noise;

h_true = H_eff(:, pilot_pos);    % 真实响应（单位导频）

%% 方法1：阈值法
h_coarse = y_daft / pilot_val;
threshold = 0.35 * max(abs(h_coarse));
h_est_threshold = zeros(N,1);
h_est_threshold(abs(h_coarse) > threshold) = h_coarse(abs(h_coarse) > threshold);

%% 方法2：LMMSE（Wiener 滤波）
% 假设信道功率在所有索引均匀分布（保守估计，总功率1）
sigma_h2 = 1 / N;                 % 每个 DAFT 索引平均信道功率（实际稀疏，但保守）
sigma_n2 = noise_power / (pilot_val^2);  % 粗估后的等效噪声方差

% Wiener 滤波系数（对角）
wiener_gain = sigma_h2 ./ (sigma_h2 + sigma_n2);

% LMMSE 估计（直接乘系数）
h_est_lmmse = wiener_gain .* h_coarse;

%% 绘图
figure;
stem(0:N-1, abs(h_true), 'bo', 'LineWidth', 1.5, 'MarkerFaceColor','b'); hold on;
stem(0:N-1, abs(h_est_threshold), 'r--o', 'LineWidth', 1.5, 'MarkerSize',8);
stem(0:N-1, abs(h_est_lmmse), 'g-s', 'LineWidth', 1.5, 'MarkerSize',7);

legend('真实信道响应', '阈值法估计', 'LMMSE 估计');
title(['AFDM 嵌入导频信道估计 (SNR = ', num2str(snr_dB), ' dB)']);
xlabel('DAFT域索引'); ylabel('幅度');
grid on;

%% MSE
mse_threshold = mean(abs(h_true - h_est_threshold).^2);
mse_lmmse = mean(abs(h_true - h_est_lmmse).^2);
disp(['阈值法 MSE: ', num2str(mse_threshold)]);
disp(['LMMSE MSE:   ', num2str(mse_lmmse)]);
disp(['LMMSE 提升: ', num2str(10*log10(mse_threshold/mse_lmmse)), ' dB']);