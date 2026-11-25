
%% Shift robustness: BSM versus Block Power.
% n=8192, p=5, gap=1; A_eta=A0+eta*I, eta=0:10:100.
% Both solvers use the same normalized residual tolerance.
% Residual and angle verification are performed outside solver timing.


clear; clc; close all;
scriptDir=fileparts(mfilename('fullpath'));
if isempty(scriptDir),scriptDir=pwd;end
addpath(scriptDir,'-begin');

%% Experiment parameters
n=8192;
p=5;
gap=1;
seed=2026;
etaList=0:10:100;
tol=1e-5;
maxIter=1800;
requireAngleForPlot=false;

optsBSM=struct('epsSigma',1e-8);
optsPower=struct();

need={'generate_matrix','method_bsm','method_block_power'};
for j=1:numel(need)
    assert(exist(need{j},'file')==2, ...
        'Missing %s.m in %s',need{j},scriptDir);
end

%% Generate the matrix and initial subspace
rng(seed,'twister');
[A0,Ustar,lambda]=generate_matrix(n,p,gap,'operator');
[Q0,~]=qr(randn(n,p),0);

assert(abs(lambda(p)-lambda(p+1)-gap)<1e-9, ...
    'Unexpected leading eigengap.');

fprintf('Shift robustness: n=%d, p=%d, gap=%g, seed=%d\n', ...
    n,p,gap,seed);

names={'Block Power','BSM'};
solvers={@method_block_power,@method_bsm};
options={optsPower,optsBSM};

numEta=numel(etaList);
numMethods=numel(names);
counts=nan(numEta,numMethods);

raw=struct('Eta',{},'Algorithm',{},'Iterations',{}, ...
    'TimeSeconds',{},'MatvecColumns',{},'NormA',{}, ...
    'SolverTolAbs',{},'MaxAbsRitzResidual',{}, ...
    'MaxNormRitzResidual',{},'TanTheta',{}, ...
    'AnglePass',{},'Status',{},'Success',{},'Detail',{});

%% Shift experiments
for ie=1:numEta
    eta=etaList(ie);
    Afun=@(Z) A0(Z)+eta*Z;

    % Spectral norm of the shifted SPD matrix.
    normA=lambda(1)+eta;
    assert(eta>=0 && isfinite(normA) && normA>0, ...
        'The norm formula requires an SPD matrix and eta>=0.');

    tolAbs=tol*normA;

    fprintf('\neta=%g, ||A_eta||_2=%.12g, solver tol=%.3e\n', ...
        eta,normA,tolAbs);

    for ia=1:numMethods
        rng(seed,'twister');

        this=struct('Eta',eta,'Algorithm',names{ia}, ...
            'Iterations',NaN,'TimeSeconds',NaN, ...
            'MatvecColumns',NaN,'NormA',normA, ...
            'SolverTolAbs',tolAbs, ...
            'MaxAbsRitzResidual',NaN, ...
            'MaxNormRitzResidual',NaN, ...
            'TanTheta',NaN,'AnglePass',false, ...
            'Status','not_started','Success',false,'Detail','');

        tRun=tic;
        try
            [X,it,solverStatus,info]=solvers{ia}( ...
                Afun,Q0,tolAbs,maxIter,options{ia});

            this.TimeSeconds=toc(tRun);
            this.Iterations=it;
            this.Status=char(solverStatus);

            if isstruct(info)&&isfield(info,'MVs')
                this.MatvecColumns=info.MVs;
            end

            % Independent verification outside solver timing.
            if isequal(size(X),[n,p]) && all(isfinite(X(:)))
                [Q,~]=qr(X,0);
                AQ=Afun(Q);

                r=local_ritz_residuals(Q,AQ,p);
                this.MaxAbsRitzResidual=max(r);
                this.MaxNormRitzResidual=this.MaxAbsRitzResidual/normA;

                this.TanTheta=local_tan_max(Q,Ustar);
                this.AnglePass=isfinite(this.TanTheta) && ...
                    this.TanTheta<=tol;
            end

            if strcmp(this.Status,'converged')
                if ~isfinite(this.MaxNormRitzResidual) || ...
                        this.MaxNormRitzResidual>tol*(1+1e-5)
                    this.Status='normalized_ritz_verification_failed';
                elseif requireAngleForPlot && ~this.AnglePass
                    this.Status='angle_verification_failed';
                end
            end

            this.Success=strcmp(this.Status,'converged');

            if this.Success
                counts(ie,ia)=it;
            end

        catch ME
            if isnan(this.TimeSeconds)
                this.TimeSeconds=toc(tRun);
            end
            this.Status=['ERROR:' ME.identifier];
            this.Detail=ME.message;
        end

        raw(end+1)=this; %#ok<SAGROW>

        fprintf(['  %-14s status=%-32s it=%5g  mv=%6g  ' ...
            'time=%8.4fs  relres=%8.2e  tan=%8.2e\n'], ...
            this.Algorithm,this.Status,this.Iterations, ...
            this.MatvecColumns,this.TimeSeconds, ...
            this.MaxNormRitzResidual,this.TanTheta);
    end
end

%% Plot
figure('Color','w','Position',[120,120,800,500]);

plot(etaList,counts(:,1),'k--+', ...
    'LineWidth',1.8,'MarkerSize',8, ...
    'DisplayName','Block Power');
hold on;

plot(etaList,counts(:,2),'r--o', ...
    'LineWidth',1.8,'MarkerSize',7, ...
    'MarkerFaceColor','w','DisplayName','BSM');

grid on;
box on;
set(gca,'FontSize',12,'LineWidth',1,'XTick',etaList);

xlabel('Shift parameter \eta','FontSize',13);
ylabel('Iterations','FontSize',13);
title(sprintf('n = %d, p = %d, gap = %g',n,p,gap), ...
    'FontWeight','normal');

legend('Location','northwest');
xlim([etaList(1),etaList(end)]);

%% Results
results=table(etaList(:),counts(:,1),counts(:,2), ...
    'VariableNames',{'Eta','BlockPowerIterations','BSMIterations'});

rawTable=struct2table(raw);

fprintf('\n========== Iteration-count summary ==========\n');
disp(results);

fprintf('\n========== Detailed solver diagnostics ==========\n');
disp(rawTable);

fprintf('\nExperiment finished. No files saved.\n');

%% Local functions
function residuals=local_ritz_residuals(Q,AQ,p)
H=Q'*AQ;
H=(H+H')/2;

[Y,D]=eig(H);
d=real(diag(D));
[d,idx]=sort(d,'descend');

Y=Y(:,idx(1:p));
d=d(1:p);

X=Q*Y;
AX=AQ*Y;
R=AX-X.*d';

residuals=sqrt(sum(abs(R).^2,1)).';
end

function angle=local_tan_max(Q,Ustar)
if isempty(Q) || size(Q,1)~=size(Ustar,1) || ...
        size(Q,2)~=size(Ustar,2) || any(~isfinite(Q(:)))
    angle=Inf;
    return;
end

[Q,~]=qr(Q,0);
C=Ustar'*Q;

if min(svd(C))<1e-13
    angle=Inf;
    return;
end

Z=(Q-Ustar*C)/C;
G=Z'*Z;
G=(G+G')/2;

angle=sqrt(max(0,max(real(eig(G)))));
end
