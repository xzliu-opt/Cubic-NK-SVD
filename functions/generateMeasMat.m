function [S,samp] = generateMeasMat(N, M, proj_style)
% proj_style (optional) type of measurement matrix
%       SUPPORTED TYPES: subsample randomly : subsample (M < N)
%                        complex gaussian   : cmplx_gauss 
%                        partial Fourier    : partial_fourier
%                        regular            : full (default)

if ~exist('M','var'), M = N;
elseif isempty(M), M = N; end

if ~exist('proj_style','var'), proj_style = 'full';
elseif isempty(proj_style), proj_style = 'full'; end

switch proj_style
    case 'subsample'
        samp = randperm(N);
        samp = samp(1:M);
        samp=sort(samp,'ascend'); 
        S = eye(N);
        S = S(samp,:);
    case 'cmplx_gauss'
        S = (randn(M,N) + 1j*randn(M,N))/sqrt(2*N);
    case 'partial_fourier'
        S = 1/sqrt(N)*exp(-2j*pi/N *(0:N-1)' * (0:N-1) );
        samp = randperm(N);
        samp = samp(1:M);
        samp=sort(samp,'ascend'); 
        S = S(samp,:);
    case 'full'
        if M ~= N
            error('Number of measurements M must be equal to the length of the sinusoid N');
        end
        S = eye(N);
        samp = (1:M);
    otherwise
        error('Measurement matrix not supported');
end



