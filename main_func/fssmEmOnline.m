function [model,llh] = fssmEmOnline(model,Y_new,User_options)
% Input:
% Y_new: cell(1,nstage):ntime*nsensor*1

% Output:
%   model: trained model structure

%% Input parameters checking
if nargin<2
    error('Error: Input data is missing.');
end

% The default value 
options = struct('lambda',0,'gamma',0.95);
% Input the user setting
if nargin == 3
    if isfield(User_options, 'lambda')
        options.lambda = User_options.lambda;
    end
    if isfield(User_options, 'gamma')
        options.gamma = User_options.gamma;
    end
    if isfield(User_options, 'nsam_online')
        options.nsam_online = User_options.nsam_online;
    end
end


%% EM
[Ex, Exx, Exy,llh] = Estep(model,Y_new); % E step
model = MstepOnline(Y_new, Ex, Exx, Exy,model,options); % M step
% model = Mstep(Y_new, Ex, Exx, Exy,model,options); % M step


end

