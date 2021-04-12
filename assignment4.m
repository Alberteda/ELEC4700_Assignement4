% Name: Oritseserundede Eda 
% Student Number: 100993421
% Assignment 4

clc
clear
clearvars
set(0,'DefaultFigureWindowStyle','docked')
% First we determine the time steps for each current case within the range
% of the applied voltage from 0.1 to 10V

voltage = linspace(0.1, 10);
current = linspace(0.1, 100);

% We take a polyp_fit of 1st order for the linear p_fit 
p = polyfit(voltage, current, 1);
p_fit = p(1)*voltage+p(2);

% The R3 resistor taking the resistance value from the slope of the linear
% fit 
R3 = p(1);
figure(22)
plot(voltage, current, 'm.')
hold on
plot(voltage, p_fit)
title('The Linear Fit')
grid on

% Part 3: Programmed circuit formulation 
clc
clear
clearvars
set(0,'DefaultFigureWindowStyle','docked')

G = zeros(6, 6); 
% Resistances values:
R1 = 1;
R2 = 2;
R3 = 10;
R4 = 0.1; 
R0 = 1000; 

% Conductances values:
G1 = 1/R1;
G2 = 1/R2;
G3 = 1/R3;
G4 = 1/R4;
G0 = 1/R0;

% Additional Parameters:
vi = zeros(100, 1);
vo = zeros(100, 1);
v3 = zeros(100, 1);
a = 100;
C_val = 0.25;
L = 0.2;

G(1, 1) = 1;                                        
G(2, 1) = G1; G(2, 2) = -(G1 + G2); G(2, 6) = -1;   
G(3 ,3) = -G3; G(3, 6) = 1;                       
G(4, 3) = -a*G3; G(4, 4) = 1;                        
G(5, 5) = -(G4+G0); G(5, 4) = G4;
G(6, 2) = -1; G(6, 3) = 1;                


C = zeros(6, 6);
C(2, 1) = C_val; C(2, 2) = -C_val;
C(6, 6) = L;

F = zeros(6, 1);
v = 0;

for vin = -10:0.1:10
    v = v + 1;
    F(1) = vin;
    Vm = G\F;
    vi(v) = vin;
    vo(v) = Vm(5);
    v3(v) = Vm(3);
end

figure(1)
plot(vi, vo);
hold on;
plot(vi, v3);
title('VO and V3 for DC Sweep (Vin): -10 V to 10 V');
xlabel('Vin (V)')
ylabel('Vo (V)')
grid on
vo2 = zeros(1000, 1); 
W = zeros(1000, 1);
Avlog = zeros(1000, 1);

for freq = linspace(0, 100, 1000)
    v = v+1;
    Vm2 = (G+1j*freq*C)\F;
    W(v) = freq;
    vo2(v) = norm(Vm2(5));
    
    Avlog(v) = 20*log10(norm(Vm2(5))/10);
end 
    
figure(3)
plot(W, vo2)
hold on;
plot(W, Avlog)
grid on
title('Vo(w) dB (part C)')
xlabel('w (rad)')
ylabel('Av (dB)')

figure(4)
semilogx(W,vo2)
title('Vo(w) dB (part C)')
xlabel('w (rad)')
ylabel('Av (dB)')
grid on

w = pi;
CC = zeros(1000,1);
GG = zeros(1000,1);

for i = 1:1000
    crand = C_val + 0.05*randn();
    C(2, 1) = crand; 
    C(2, 2) = -crand;
    C(3, 3) = L;
    Vm3 = (G+(1i*w*C))\F;
    CC(i) = crand;
    GG(i) = 20*log10(abs(Vm3(5))/10);
end

figure(5)
histogram(CC)
grid on
figure(6)
grid on
hist(GG)

clc
clear
clearvars
set(0,'DefaultFigureWindowStyle','docked')

% Transient simulation parameters 
R1 = 1;
R2 = 2;
C = 0.25;
L = 0.2;
R3 = 10;
a = 100;
R4 = 0.1;
R0 = 1000;
G = zeros(7,7);
Cm = zeros(7,7);

% Conductance values
G(1,1) = 1; 
G(2,1) = -1/R1;
G(2,2) = 1/R1 + 1/R2;
G(2,6) = 1;
G(3,3) = 1/R3;
G(3,6) = -1;
G(4,3) = 1/R3;
G(4,7) = -1;
G(5,4) = -1/R4;
G(5,5) = 1/R4 + 1/R0;
G(6,2) = 1;
G(6,3) = -1;
G(7,4) = 1;
G(7,7) = -a;

% Capacitance value
Cm(2,1) = -C;
Cm(2,2) = C;
Cm(6,6) = -L;
V1 = zeros(7,1);
V2 = zeros(7,1);
V3 = zeros(7,1);
vn_1(1) = 0;
vn_2(1) = 0;
vn_3(1) = 0;
del = 0.001;
A = (Cm/del) + G;
F_index1 = zeros(7,1);
F_index2 = zeros(7,1);
F_index33 = zeros(7,1);
vo_3(1) = 0;
vo_2(1) = 0;
vo_3(1) = 0;

i = 1;
for j = del:del:1
    if j >= 0.03
        F_index1(1) = 3;
    end
    F_index2(1) = sin(2*pi*j/0.03);
    F_index33(1) = exp(-0.5*((j - 0.06)/0.03)^2);
    V1 = A\(Cm*V1/del + F_index1);
    V2 = A\(Cm*V2/del + F_index2);
    V3 = A\(Cm*V3/del + F_index33);
    vn_1(i+1) = V1(1);
    vn_2(i+1) = V2(1);
    vn_3(i+1) = V3(1);
    vo_3(i+1) = V1(5);
    vo_2(i+1) = V2(5);
    vo_3(i+1) = V3(5);
    i = i+1;
end

figure(7)
plot(0:del:1,vn_1,'b')
hold on
plot(0:del:1,vo_3,'m')
title('Voltage vs time')
xlabel('time (s)')
ylabel('Voltage')
legend('input','output')
grid on
figure(8)
plot(0:del:1,vn_2,'b')
hold on
plot(0:del:1,vo_2,'m')
title('Voltage vs time')
xlabel('time (s)')
ylabel('Voltage')
legend('input','output')
grid on
figure(9)
plot(0:del:1,vn_3,'b')
hold on
plot(0:del:1,vo_3,'m')
title('Voltage vs time')
xlabel('time (s)')
ylabel('Voltage')
legend('input','output')
grid on

% Convert to frequency domain by taking the fourier transform
step_in = fft(vn_1); %fft -> Fast fourier transform
P_mag2_in = abs(step_in/1000);
P_1_in = P_mag2_in(1:1000/2+1);
P_1_in(2:end-1) = 2*P_1_in(2:end-1);
sample_f = (1/del)*(0:(1000/2))/1000;
% Plot figure 
figure(10)
plot(sample_f,P_1_in,'b')
step_out = fft(vo_3);
P2_out = abs(step_out/1000);
P1_out = P2_out(1:1000/2+1);
P1_out(2:end-1) = 2*P1_out(2:end-1);
sample_f = (1/del)*(0:(1000/2))/1000;
hold on
plot(sample_f,P1_out,'m')
title('Frequency Plot for Step-Input')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
ylim([0 3])
legend('input','output')
grid on
% The Fourier transform gives a sine function, which makes sense because we
% are dealing with a step-signal. The filter acts like a low-pass filter with the high frequencies attenuated. 

% Taking the fourier tranform of the input voltage signal 
step_in = fft(vn_2);
P_mag2_in = abs(step_in/1000);
P_1_in = P_mag2_in(1:1000/2+1);

%The end spectrum is found and multiplied by 2
P_1_in(2:end-1) = 2*P_1_in(2:end-1);
figure(11)
plot(sample_f,P_1_in,'b')
grid on
step_out = fft(vo_2);
P2_out = abs(step_out/1000);
P1_out = P2_out(1:1000/2+1);
P1_out(2:end-1) = 2*P1_out(2:end-1);
hold on
plot(sample_f,P1_out,'m')
title('Frequency Plot for the sinusoidal input')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
ylim([0 3])
legend('input','output')
step_in = fft(vn_3); 
P_mag2_in = abs(step_in/1000);
P_1_in = P_mag2_in(1:1000/2+1);
P_1_in(2:end-1) = 2*P_1_in(2:end-1);
figure(12)
plot(sample_f,P_1_in,'b')
grid on
step_out = fft(vo_3);
P2_out = abs(step_out/1000);
P1_out = P2_out(1:1000/2+1);
P1_out(2:end-1) = 2*P1_out(2:end-1);
hold on
plot(sample_f,P1_out,'m')
title('Frequency Plot for the Gaussian input')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
ylim([0 3])
legend('input','output')

clc
clearvars
set(0,'DefaultFigureWindowStyle','docked')

R1 = 1;
R2 = 2;
R3 = 10;
R4 = 0.1;
Ro = 1000;
C = 0.25;
L = 0.2;
a = 100;
cn1 = 0.00001;
cn2 = 0.0001;
cn3 = 0.01;

% First Capacitor 
C1(1,:)=[C -C 0 0 0 0 0 0]; 
C1(2,:)=[-C C 0 0 0 0 0 0];
C1(3,:)=[0 0 cn1 0 0 0 0 0]; 
C1(4,:)=[0 0 0 0 0 0 0 0]; 
C1(5,:)=[0 0 0 0 0 0 0 0];
C1(6,:)=[0 0 0 0 0 -L 0 0]; 
C1(7,:)=[0 0 0 0 0 0 0 0]; 
C1(8,:)=[0 0 0 0 0 0 0 0]; 
% Second Capacitor
C2(1,:)=[C -C 0 0 0 0 0 0]; 
C2(2,:)=[-C C 0 0 0 0 0 0];
C2(3,:)=[0 0 cn2 0 0 0 0 0]; 
C2(4,:)=[0 0 0 0 0 0 0 0]; 
C2(5,:)=[0 0 0 0 0 0 0 0];
C2(6,:)=[0 0 0 0 0 -L 0 0]; 
C2(7,:)=[0 0 0 0 0 0 0 0]; 
C2(8,:)=[0 0 0 0 0 0 0 0]; 
% Third Capacitor
C3(1,:)=[C -C 0 0 0 0 0 0]; 
C3(2,:)=[-C C 0 0 0 0 0 0];
C3(3,:)=[0 0 cn3 0 0 0 0 0]; 
C3(4,:)=[0 0 0 0 0 0 0 0]; 
C3(5,:)=[0 0 0 0 0 0 0 0];
C3(6,:)=[0 0 0 0 0 -L 0 0]; 
C3(7,:)=[0 0 0 0 0 0 0 0]; 
C3(8,:)=[0 0 0 0 0 0 0 0]; 
% G matrices 
G(1,:)=[1 -1 0 0 0 0 0 1]; 
G(2,:)=[-1 1.5 0 0 0 1 0 0];
G(3,:)=[0 0 0.1 0 0 -1 0 0]; 
G(4,:)=[0 0 0 10 -10 0 1 0]; 
G(5,:)=[0 0 0 -10 10.0010 0 0 0];
G(6,:)=[0 1 -1 0 0 -L 0 0]; 
G(7,:)=[0 0 -10 1 0 0 0 0]; 
G(8,:)=[1 0 0 0 0 0 0 0];

F = [0;0;0;0;0;0;0;0;];

% From the lab manual: 
del = 0.001;
std = 0.03;
d = 0.06;
m = 1;
c_s = zeros(1,1000);
f_l = zeros(8,1,1000);

for i=1:1:1000
    f_l(8,1,i) = exp(-((i*del - d)/std)^2);
    f_l(3,1,i) = -0.001*randn;
    c_s(i) = f_l(3,1,i);
end

VL_1 = zeros(8,1,1000);
VL_2 = zeros(8,1,1000);
VL_3 = zeros(8,1,1000);

for i = 2:1:1000
    index1 = C1/del + G;
    index2 = C2/del + G;
    index3 = C3/del + G;
    
    VL_1(:,:,i) = index1\(C1*VL_1(:,:,i-1)/del +f_l(:,:,i));
    VL_2(:,:,i) = index2\(C1*VL_2(:,:,i-1)/del +f_l(:,:,i));
    VL_3(:,:,i) = index3\(C1*VL_3(:,:,i-1)/del +f_l(:,:,i));
end


VoL_1(1,:) = VL_1(5,1,:);
ViL_1(1,:) = VL_1(1,1,:);

VoL_2(1,:) = VL_2(5,1,:);
ViL_2(1,:) = VL_2(1,1,:);

VoL_3(1,:) = VL_3(5,1,:);
ViL_3(1,:) = VL_3(1,1,:);


figure(13)
plot((1:1000).*del, VoL_1(1,:),'b')
hold on
plot((1:1000).*del, ViL_1(1,:),'m')
title('Plot of Gaussian Pulse with added noise source for cn1 = 0.00001')
xlabel('Time (s)')
ylabel('Voltage (v)')
grid on
legend('Output Voltage','Input Voltage')
hold off

figure(14)
histogram(c_s)
title('Noise Source Histogram')
grid on
xlabel('Current in amperes')

figure(15)
FF = abs(fftshift(fft(VoL_1(1,:))));
plot(((1:length(FF))/1000)-0.5,FF,'b')
xlabel('Frequency in hertz')
ylabel('Magnitude in decibels')
grid on
xlim([-0.04 0.04])
title('Plot of Gaussian Pulse with added noise source for cn1 = 0.00001')

figure(16)
plot((1:1000).*del, VoL_2(1,:),'b')
hold on
plot((1:1000).*del, ViL_2(1,:),'m')
title('Plot of Gaussian Pulse with added noise source for cn2 = 0.0001')
xlabel('Time in seconds')
ylabel('Voltage in volts')
grid on
legend('Output Voltage','Input Voltage')
hold off

figure(17)
FF = abs(fftshift(fft(VoL_2(1,:))));
plot(((1:length(FF))/1000)-0.5,FF,'m')
xlabel('Frequency in hertz')
ylabel('Magnitude in decibels')
grid on
xlim([-0.04 0.04])
title('Plot of Gaussian Pulse with added noise source for cn2 = 0.0001')

figure(18)
plot((1:1000).*del, VoL_3(1,:),'b')
hold on
plot((1:1000).*del, ViL_3(1,:),'m')
title('Plot of Gaussian Pulse Function with added noise source and cn3 = 0.01 ')
xlabel('Time in seconds')
ylabel('Voltage in volts')
grid on
legend('Output Voltage','Input Voltage')
hold off

figure(19)
FF = abs(fftshift(fft(VoL_3(1,:))));
plot(((1:length(FF))/1000)-0.5,FF,'m')
xlabel('Frequency in hertz')
ylabel('Magnitude in decibels')
grid on
xlim([-0.04 0.04])
title('Plot of Gaussian Pulse Function with added noise source and cn3 = 0.01')

clc
clear
clearvars
set(0,'DefaultFigureWindowStyle','docked')

R1 = 1;
R2 = 2;
C = 0.25;
L = 0.2;
R3 = 10;
a = 100;
R4 = 0.1;
R0 = 1000;

G = zeros(7,7);
C_mat = zeros(7,7);
G(1,1) = 1;
G(2,1) = -1/R1;
G(2,2) = 1/R1 + 1/R2;
G(2,6) = 1;
G(3,3) = 1/R3;
G(3,6) = -1;
G(4,3) = 1/R3;
G(4,7) = -1;
G(5,4) = -1/R4;
G(5,5) = 1/R4 + 1/R0;
G(6,2) = 1;
G(6,3) = -1;
G(7,4) = 1;
G(7,7) = -a;
C_mat(2,1) = -C;
C_mat(2,2) = C;
C_mat(6,6) = -L;

F = zeros(7,1);
V = zeros(7,1);

% The Circuit with noise provided from the manual
In = 0.001; 
Cn = 0.00001; 
C_mat(3,3) = Cn;
del = 0.001;
trans = C_mat/del + G;

F = zeros(7,1);
V = zeros(7,1);
Vo_index(1) = 0;
index(1) = 0;

ii = 1;
for t = del:del:1
    F(1) = exp(-0.5*((t - 0.06)/0.03)^2);
    F(3) = In*normrnd(0,1);
     V = trans\(C_mat*V/del + F);
     index(ii + 1) = F(1);
     Vo_index(ii + 1) = V(5);
     ii = ii + 1;
end

X_input = fft(index);
P2_input = abs(X_input/1000);
P1_input = P2_input(1:1000/2+1);
P1_input(2:end-1) = 2*P1_input(2:end-1);
f = (1/del)*(0:(1000/2))/1000;

X_output = fft(Vo_index);
P2_output = abs(X_output/1000);
P1_output = P2_output(1:1000/2+1);
P1_output(2:end-1) = 2*P1_output(2:end-1);
f = (1/del)*(0:(1000/2))/1000;

C_sml = C_mat;
C_med = C_mat;
C_big = C_mat;
C_sml(3,3) = 0;
C_med(3,3) = 0.001;
C_big(3,3) = 0.1;
V_sml = zeros(7,1);
V_med = zeros(7,1);
V_big = zeros(7,1);
Vo_sml(1) = 0;
med(1) = 0;
Vout_big(1) = 0;
index(1) = 0;
ii = 1;
for t = del:del:1
     F(1) = exp(-0.5*((t - 0.06)/0.03)^2);
     F(3) = In*normrnd(0,1);
     V_sml = (C_sml/del + G)\(C_sml*V_sml/del + F);
     V_med = (C_med/del + G)\(C_med*V_med/del + F);
     V_big = (C_big/del + G)\(C_big*V_big/del + F);
     Vo_sml(ii + 1) = V_sml(5);
     med(ii + 1) = V_med(5);
     Vout_big(ii + 1) = V_big(5);
     index(ii + 1) = F(1);
     ii = ii + 1;
end

del1 = 0.01;
vin_ss(1) = 0;
vout_ss(1) = 0;
V = zeros(7,1);
ii = 1;
for t = del1:del1:1
     F(1) = exp(-0.5*((t - 0.06)/0.03)^2);
     F(3) = In*normrnd(0,1);
     V = (C_mat/del1 + G)\(C_mat*V/del1 + F);
     vout_ss(ii + 1) = V(5);
     vin_ss(ii + 1) = F(1);
     ii = ii + 1;
end

del2 = 0.1;
vin_bs(1) = 0;
vout_bs(1) = 0;
V = zeros(7,1);
ii = 1;
for t = del2:del2:1
     F(1) = exp(-0.5*((t - 0.06)/0.03)^2);
     F(3) = In*normrnd(0,1);
     V = (C_mat/del2 + G)\(C_mat*V/del2 + F);
     vout_bs(ii + 1) = V(5);
     vin_bs(ii + 1) = F(1);
     ii = ii + 1;
end

figure(20)
plot(0:del1:1,vin_ss,'b')
hold on
plot(0:del1:1,vout_ss,'m')
title('Voltage vs time with timestep = 0.003')
xlabel('time (s)')
ylabel('Voltage')
legend('input','output')
grid on

figure(21)
plot(0:del2:1,vin_bs,'m')
hold on
plot(0:del2:1,vout_bs,'b')
title('Voltage vs time with time step = 0.01')
xlabel('time (s)')
ylabel('Voltage')
legend('input','output')
grid on