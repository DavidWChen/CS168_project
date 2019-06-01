# function [K, D,options] = constructKernel(fea_a,fea_b,options)
# % function K = constructKernel(fea_a,fea_b,options)
# %	Usage:
# %	K = constructKernel(fea_a,[],options)
# %
# %   K = constructKernel(fea_a,fea_b,options)
# %
# %	fea_a, fea_b  : Rows of vectors of data points. 
# %
# %   options       : Struct value in Matlab. The fields in options that can
# %                   be set: 
# %           KernelType  -  Choices are:
# %               'Gaussian'      - e^{-(|x-y|^2)/2t^2}
# %               'Polynomial'    - (x'*y)^d
# %               'PolyPlus'      - (x'*y+1)^d
# %               'Linear'        -  x'*y
# %
# %               t       -  parameter for Gaussian
# %               d       -  parameter for Poly
# %
# %   version 1.0 --Sep/2006 
# %
# %   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
# %

# if (~exist('options','var'))
#    options = [];
# else
#    if ~isstruct(options) 
#        error('parameter error!');
#    end
# end



# %=================================================
# if ~isfield(options,'KernelType')
#     options.KernelType = 'Gaussian';
# end

# switch lower(options.KernelType)
#     case {lower('Gaussian')}        %  e^{-(|x-y|^2)/2t^2}
#          if ~isfield(options,'t')
#              options.t = 1;
#          end
#     case {lower('Polynomial')}      % (x'*y)^d
#         if ~isfield(options,'d')
#             options.d = 2;
#         end
#     case {lower('PolyPlus')}      % (x'*y+1)^d
#         if ~isfield(options,'d')
#             options.d = 2;
#         end
#     case {lower('Linear')}      % x'*y
#     otherwise
#         error('KernelType does not exist!');
# end


# %=================================================

# switch lower(options.KernelType)
    
#     case {lower('Gaussian')}       
#         if isempty(fea_b)
#             %if we exceed limit of the data, construct using only a subset
#             %of points
#             %if size(fea_a,1) > data_limit
#             %  D = EuDist2(fea_a(randsample(nSmp,data_limit),:));
#             %else
#               D = EuDist2(fea_a,[],0);
#             %end
#         else
#             D = EuDist2(fea_a,fea_b,0);
#         end
#         if ~isfield(options,'t')
#             options.t = sqrt(mean(min(D, [], 2)));
#         end
#         K = exp(-D/(2*options.t^2));
#     case {lower('Polynomial')}     
#         if isempty(fea_b)
#             D = full(fea_a * fea_a');
#         else
#             D = full(fea_a * fea_b');
#         end
#         K = D.^options.d;
#     case {lower('PolyPlus')}     
#         if isempty(fea_b)
#             D = full(fea_a * fea_a');
#         else
#             D = full(fea_a * fea_b');
#         end
#         K = (D+1).^options.d;
#     case {lower('Linear')}     
#         if isempty(fea_b)
#             K = full(fea_a * fea_a');
#         else
#             K = full(fea_a * fea_b');
#         end
#     otherwise
#         error('KernelType does not exist!');
# end

# if isempty(fea_b)
#     K = max(K,K');
# end

import numpy as np

def constructKernel(fea_a,fea_b,options):
    options = [];
    t = 1  
    num = fea_a.shape[0]  
    D = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            D[i,j] = np.linalg.norm(np.subtract(fea_a[i],fea_a[j]))
    K = np.exp(-D/(2*t**2))
    return K

    

# if isempty(fea_b)
#     K = max(K,K');
# end
# import numpy as np
# fea = np.array([[x, x+1] for x in range(25)])
# constructKernel(fea, [], {})
