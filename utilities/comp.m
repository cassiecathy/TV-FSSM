function pdiff = comp(model,model_fssm)
B = model_fssm.B;
A = model_fssm.A;
C = model_fssm.C;
Ts = model_fssm.Ts;
sigma1 =  model_fssm.sigma1;
sigma2 =  model_fssm.sigma2;
M1 = model_fssm.M1;
V1 = model_fssm.V1;

B_t = model.B;
A_t = model.A;
C_t = model.C;
Ts_t = model.Ts;
sigma1_t = model.sigma1;
sigma2_t = model.sigma2;
M1_t = model.M1;
V1_t = model.V1;

nstage = length(B);
pdiff = zeros(1,8);

temp = zeros(1,nstage);
for s = 1:nstage
    temp(s) =  norm(B{s} - B_t{s},'fro')^2/norm(B_t{s},'fro')^2;
end
pdiff(1) = mean(temp);

temp = zeros(1,nstage-1);
for s = 1:nstage-1
    temp(s) = norm(A{s} - A_t{s},'fro')^2/ norm(A_t{s},'fro')^2;
end
pdiff(2) = mean(temp(2:end));

temp = zeros(1,nstage-1);
for s = 1:nstage-1
    temp(s) = norm(C{s} - C_t{s},'fro')^2/norm(C_t{s},'fro')^2;
end
pdiff(3) = mean(temp(2:end));

% 
% temp = zeros(1,nstage-1);
% for s = 1:nstage-1
%     temp(s) = norm(F{s} - kron(C_t{s}',A_t{s}),'fro')^2/ norm(kron(C_t{s}',A_t{s}),'fro')^2;
% end
% pdiff(9) = mean(temp);

temp = zeros(1,nstage);
for s = 1:nstage
    temp(s) = norm(Ts{s} - Ts_t{s},'fro')^2/norm(Ts_t{s},'fro')^2;
%     temp(s) = max(abs(eig(Ts{s}-Ts_t{s})))/max(abs(eig(Ts_t{s})));
end
pdiff(4) = mean(temp);

temp = zeros(1,nstage);
for s = 1:nstage
%     temp(s) = (sigma1(s)-sigma1_t(s))^2/sigma1_t(s)^2;
    temp(s) = max(abs(eig(sigma1(s)-sigma1_t(s))));
end
pdiff(5) = mean(temp);

temp = zeros(1,nstage-1);
for s = 1:nstage-1
%     temp(s) = (sigma2(s)-sigma2_t(s))^2/sigma2_t(s)^2;
    temp(s) = max(abs(eig(sigma2(s)-sigma2_t(s))));
end
pdiff(6) = mean(temp);

% pdiff(5) = (trace((sigma1-sigma1_t)'*(sigma1-sigma1_t)))/(trace(sigma1_t'*sigma1_t));
% pdiff(6) = (trace((sigma2-sigma2_t)'*(sigma2-sigma2_t)))/(trace(sigma2_t'*sigma2_t));
pdiff(7) = (trace((M1-M1_t)'*(M1-M1_t)))/(trace(M1_t'*M1_t));
% pdiff(8) = (trace((V1-V1_t)'*(V1-V1_t)))/(trace(V1_t'*V1_t));
pdiff(8) =  max(abs(eig(V1-V1_t(s))));



end