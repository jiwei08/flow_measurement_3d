function [ux, uy, uz] = LK3D(image1, image2, r)
    %This function estimates deformations between two subsequent 3-D images
    %using Lucas-Kanade optical flow equation.
    %
    %   Description :
    %
    %   -image1, image2 :   two subsequent images or frames
    %   -r : radius of the neighbourhood, default value is 2.
    %
    %   Reference :
    %   Lucas, B. D., Kanade, T., 1981. An iterative image registration
    %   technique with an application to stereo vision. In: Proceedings of the
    %   7th international joint conference on Artificial intelligence - Volume 2.
    %   Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, pp. 674-679.
    %
    %   Author : Mohammad Mustafa
    %   By courtesy of The University of Nottingham and Mirada Medical Limited,
    %   Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    %  Default parameter
    if nargin == 2
        r = 5;
    end

    [height, width, depth] = size(image1);
    image1 = im2double(image1);
    image2 = im2double(image2);

    % Initializing flow vectors
    ux = zeros(size(image1)); uy = ux; uz = ux;

    % Computing image derivatives
    [Ix, Iy, Iz, It] = imageDerivatives3D(image1, image2);

    for i = (r + 1):(height - r)

        for j = (r + 1):(width - r)

            for k = (r + 1):(depth - r)

                blockofIx = Ix(i - r:i + r, j - r:j + r, k - r:k + r);
                blockofIy = Iy(i - r:i + r, j - r:j + r, k - r:k + r);
                blockofIz = Iz(i - r:i + r, j - r:j + r, k - r:k + r);
                blockofIt = It(i - r:i + r, j - r:j + r, k - r:k + r);

                A = zeros(3, 3);
                B = zeros(3, 1);

                A(1, 1) = sum(sum(sum(blockofIx.^2)));
                A(1, 2) = sum(sum(sum(blockofIx .* blockofIy)));
                A(1, 3) = sum(sum(sum(blockofIx .* blockofIz)));

                A(2, 1) = sum(sum(sum(blockofIy .* blockofIx)));
                A(2, 2) = sum(sum(sum(blockofIy.^2)));
                A(2, 3) = sum(sum(sum(blockofIy .* blockofIz)));

                A(3, 1) = sum(sum(sum(blockofIz .* blockofIx)));
                A(3, 2) = sum(sum(sum(blockofIz .* blockofIy)));
                A(3, 3) = sum(sum(sum(blockofIz.^2)));

                B(1, 1) = sum(sum(sum(blockofIx .* blockofIt)));
                B(2, 1) = sum(sum(sum(blockofIy .* blockofIt)));
                B(3, 1) = sum(sum(sum(blockofIz .* blockofIt)));

                invofA = pinv(A);

                V = invofA * (-B);
                ux(i, j, k) = V(1, 1);
                uy(i, j, k) = V(2, 1);
                uz(i, j, k) = V(3, 1);
            end

        end

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
