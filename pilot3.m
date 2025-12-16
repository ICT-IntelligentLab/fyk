
clear; clc; close all;

N = 128;
c1 = 1/(2*N);           % 整数Doppler
c2 = 1/N;               % 可调
modOrder = 4;
snr_dB = 25;            % SNR测试鲁棒性

P = 4;
l_max = 15;
k_max = 8;              % 整数Doppler

pilot_pos = floor(N/2) + 1;
pilot_val = sqrt(N)*3;  % 将根号N的五倍值作为导频幅度（倍数可调 一般为1
guard_len = 2*(l_max + 2*abs(k_max)) + 3;  % +10也可调 一般为0

%% DAFT矩阵
n = (0:N-1).';
Lambda_c1 = diag(exp(1j * pi * c1 * n.^2));
Lambda_c2 = diag(exp(1j * pi * c2 * n.^2));
F_dft = dftmtx(N) / sqrt(N);
F_inv = Lambda_c2 * F_dft * Lambda_c1;  % IDAFT
F = F_inv';                             % DAFT

%% 可用数据位置
left_end = pilot_pos - guard_len - 1;
right_start = pilot_pos + guard_len + 1;
data_idx = [1:max(0,left_end), min(N,right_start):N];%整合区间 获取可用于存放data的长度
num_available = length(data_idx);
if num_available <= 0, error('guard太大'); end
disp(['可用数据位置: ' num2str(num_available)]);

data = qammod(randi([0 modOrder-1], num_available, 1), modOrder, 'UnitAveragePower',true);%随机生成单位QAM

%% 发送
x_daft = zeros(N,1);
x_daft(data_idx) = data;
x_daft(pilot_pos) = pilot_val;

%% delay-Doppler信道
H_eff = zeros(N,N);
for p = 1:P
    l_p = randi([0 l_max]);                  % 延迟
    k_p = randi([-k_max k_max]);             % 整数Doppler
    h_p = (randn + 1i*randn)/sqrt(2);         % 增益
    % 每路径贡献：Doppler相移 + 延迟循环移位
    doppler_phase = exp(1j * 2*pi * k_p * (0:N-1)' / N);
    delay_shift = circshift(eye(N), l_p, 1);  % 列循环移位（延迟）
    H_path = h_p * diag(doppler_phase) * delay_shift;
    H_eff = H_eff + H_path;
end

%% 通过信道
y_daft_noiseless = H_eff * x_daft;

%% 加噪声
noise_power = 10^(-snr_dB/10);
noise = sqrt(noise_power/2) * (randn(N,1) + 1i*randn(N,1));
y_daft = y_daft_noiseless + noise;

%% 嵌入单导频估计

y_pilot_response = y_daft - H_eff(:, data_idx) * data;  % 可选：减数据干扰
h_est_dd = y_daft / pilot_val;  % 直接除（guard区干净）

% 阈值提取
threshold = 0.25 * max(abs(h_est_dd));  % 根据SNR和pilot功率调
h_est_sparse = zeros(N,1);
h_est_sparse(abs(h_est_dd) > threshold) = h_est_dd(abs(h_est_dd) > threshold);

%% 真实信道稀疏表示
h_dd_effective = H_eff(:, pilot_pos) ;  % 单导频下有效稀疏信道（归一化前）

%% 图
figure;
stem(0:N-1, abs(h_dd_effective), 'bo', 'LineWidth', 1.5, 'MarkerFaceColor','b'); hold on;
stem(0:N-1, abs(h_est_sparse), 'r--o', 'LineWidth', 1.5, 'MarkerSize',8);
legend('真实有效信道 (对导频响应)', '嵌入导频估计');
title('AFDM 嵌入式单导频信道估计');
xlabel('DAFT域索引'); ylabel('幅度');
grid on;

%% MSE
mse = mean(abs(h_dd_effective - h_est_sparse).^2);
disp(['信道估计 MSE: ' num2str(mse)]);