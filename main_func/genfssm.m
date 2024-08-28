function [Y,X] = genfssm(model0,nsam)
if isfield(model0, 'A')
    B = model0.B;
    A = model0.A;
    C = model0.C;
    Ts = model0.Ts;
    sigma1 = model0.sigma1;
    sigma2 = model0.sigma2;
    M1 = model0.M1;
    V1 = model0.V1;

    % B: cell(1,nstage):ntime(s)*npc(s+1)
    % A: cell(1,nstage-1):npc(s+1)*npc(s)
    % C: cell(1,nstage-1):nsensor(s)*nsensor(s+1)
    % sigma1: 
    % sigma2: 
    % M1: npc(1)*nsensor(1)
    % V1: (npc(1)*nsensor(1))*(npc(1)*nsensor(1))
    nstage = size(B,2);
    ntime = zeros(1,nstage);
    nsensor = zeros(1,nstage);
    npc = zeros(1,nstage);
    for s = 1:nstage
        [ntime(s),npc(s)] = size(B{s});
    end
    [nsensor(1),nsensor(2)] = size(C{1});
    for s = 2:nstage-1
        [~,nsensor(s+1)] = size(C{s});
    end

    B0 = cell(1,nstage);
    for s = 1:nstage
        B0{s} = kron(eye(nsensor(s))',B{s});
    end
    F = cell(1,nstage-1);
    for s = 1:nstage-1
        F{s} = kron(C{s}',A{s});
    end
    M1 = reshape(M1,[],1);

    Y = cell(1,nstage);
    X = cell(1,nstage);
    for i = 1:nsam
        X{1}(:,i) = mvnrnd(M1,V1,1);
        Y{1}(:,i) = mvnrnd(B0{1}*X{1}(:,i),kron(sigma1(1)*eye(nsensor(1)),Ts{1}),1)';
        for s = 1:nstage-1
            X{s+1}(:,i) = mvnrnd(F{s}*X{s}(:,i),sigma2(s)*eye(npc(s+1)*nsensor(s+1)),1)';
            Y{s+1}(:,i) =  mvnrnd(B0{s+1}*X{s+1}(:,i),kron(sigma1(s+1)*eye(nsensor(s+1)),Ts{s+1}),1)';
        end
    end

    for s = 1:nstage
        Y{s} = reshape(Y{s},[ntime(s),nsensor(s),nsam]);
        X{s} = reshape(X{s},[npc(s),nsensor(s),nsam]);
    end

else
    B = model0.B;
    M1 = model0.M1;
    sigma1 = model0.sigma1;
    V1 = model0.V1;
    Ts = model0.Ts;

    nstage = size(B,2);
    ntime = zeros(1,nstage);
    nsensor = zeros(1,nstage);
    npc = zeros(1,nstage);
    for s = 1:nstage
        [ntime(s),npc(s)] = size(B{s});
        [~,nsensor(s)] = size(M1{s});
    end

    B0 = cell(1,nstage);
    M0 = cell(1,nstage);
    for s = 1:nstage
        B0{s} = kron(eye(nsensor(s))',B{s});
        M0{s} = reshape(M1{s},[],1);
    end
    

    Y = cell(1,nstage);
    X = cell(1,nstage);
    for i = 1:nsam
        for s = 1:nstage
            X{s}(:,i) = mvnrnd(M0{s},V1{s},1)';
            Y{s}(:,i) =  mvnrnd(B0{s}*X{s}(:,i),kron(sigma1(s)*eye(nsensor(s)),Ts{s}),1)';
        end
    end

    for s = 1:nstage
        Y{s} = reshape(Y{s},[ntime(s),nsensor(s),nsam]);
        X{s} = reshape(X{s},[npc(s),nsensor(s),nsam]);
    end
end
    
    
end