function [ux, uy, uz] = HS3D_seq(image_seq1, image_seq2, alpha, maxIters)
    % Horn-Schunck method
    % Parameter:
    %   alpha:      regularization parameter
    %   maxIters:   maxmium iterations

    if nargin == 2
        alpha = 0.01;
        maxIters = 300;
    end

    [length, height, width, depth] = size(image_seq1);
    image_seq1 = im2double(image_seq1);
    image_seq2 = im2double(image_seq2);

    % Initializing flow vectors
    ux = zeros(height,width,depth); uy = ux; uz = ux;

    % Computing image derivatives
    IxIx = ux;
    IyIx = ux;
    IzIx = ux;
    ItIx = ux;
    IxIy = ux;
    IyIy = ux;
    IzIy = ux;
    ItIy = ux;
    IxIz = ux;
    IyIz = ux;
    IzIz = ux;
    ItIz = ux;
    for l = 1:length
        [Ix, Iy, Iz, It] = imageDerivatives3D(squeeze(image_seq1(l, :, :, :)), squeeze(image_seq2(l, :, :, :)));
        IxIx = IxIx + Ix .* Ix;
        IyIx = IyIx + Iy .* Ix;
        IzIx = IzIx + Iz .* Ix;
        ItIx = ItIx + It .* Ix;
        IxIy = IxIy + Ix .* Iy;
        IyIy = IyIy + Iy .* Iy;
        IzIy = IzIy + Iz .* Iy;
        ItIy = ItIy + It .* Iy;
        IxIz = IxIz + Ix .* Iz;
        IyIz = IyIz + Iy .* Iz;
        IzIz = IzIz + Iz .* Iz;
        ItIz = ItIz + It .* Iz;
    end

    for iter = 1:maxIters
        % Gauss-Seidel equation
        ux_tmp = ux; uy_tmp = uy; uz_tmp = uz;
        ux = ux_tmp - (IxIx .* ux_tmp + IxIy .* uy_tmp + IxIz .* uz_tmp + ItIx) ./ (alpha^2 + IxIx + IyIy + IzIz);
        uy = uy_tmp - (IyIx .* ux_tmp + IyIy .* uy_tmp + IyIz .* uz_tmp + ItIy) ./ (alpha^2 + IxIx + IyIy + IzIz);
        uz = uz_tmp - (IzIx .* ux_tmp + IzIy .* uy_tmp + IzIz .* uz_tmp + ItIz) ./ (alpha^2 + IxIx + IyIy + IzIz);

        block = zeros(3, 3, 3);
        block(1, 2, 2) = 1;
        block(2, :, :) = [0, 1, 0; 1, 0, 1; 0, 1, 0];
        block(3, :, :) = block(1, :, :);
        block = block / sum(block, 'all');
        ux = convn(ux, block, 'same');
        uy = convn(uy, block, 'same');
        uz = convn(uz, block, 'same');

    end

end

function [Ix, Iy, Iz, It] = imageDerivatives3D(image1, image2)
    %This fucntion computes 3D derivatives between two 3D images.
    %
    %   Description :
    %
    %   There are four derivatives here; three along X, Y, Z axes and one along
    %   timeline axis.
    %
    %   -image1, image2 :   two subsequent images or frames
    %   -dx, dy, dz : vectors along X, Y and Z axes respectively
    %   -dt : vectors along timeline axis
    %   -Ix, Iy, Iz : derivatives along X, Y and Z axes respectively
    %   -It : derivatives along timeline axis
    %
    %   Author : Mohammad Mustafa
    %   By courtesy of The University of Nottingham and
    %   Mirada Medical Limited, Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    dx = zeros(2, 2, 2);
    dx(:, :, 1) = [-1 -1; 1 1]; dx(:, :, 2) = [-1 -1; 1 1];
    dx = 0.25 * dx;

    dy = zeros(2, 2, 2);
    dy(:, :, 1) = [-1 1; -1 1]; dy(:, :, 2) = [-1 1; -1 1];
    dy = 0.25 * dy;

    dz = zeros(2, 2, 2);
    dz(:, :, 1) = [-1 -1; -1 -1]; dz(:, :, 2) = [1 1; 1 1];
    dz = 0.25 * dz;

    dt = ones(2, 2, 2);
    dt = 0.25 * dt;

    % Computing derivatives
    Ix = 0.5 * (convn(image1, dx) + convn(image2, dx));
    Iy = 0.5 * (convn(image1, dy) + convn(image2, dy));
    Iz = 0.5 * (convn(image1, dz) + convn(image2, dz));
    It = 0.5 * (convn(image1, dt) - convn(image2, dt));

    % Adjusting sizes
    Ix = Ix(1:size(Ix, 1) - 1, 1:size(Ix, 2) - 1, 1:size(Ix, 3) - 1);
    Iy = Iy(1:size(Iy, 1) - 1, 1:size(Iy, 2) - 1, 1:size(Iy, 3) - 1);
    Iz = Iz(1:size(Iz, 1) - 1, 1:size(Iz, 2) - 1, 1:size(Iz, 3) - 1);
    It = It(1:size(It, 1) - 1, 1:size(It, 2) - 1, 1:size(It, 3) - 1);

end
