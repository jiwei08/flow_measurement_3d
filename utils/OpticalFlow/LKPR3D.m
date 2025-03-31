function [ux, uy, uz] = LKPR3D(image1, image2, r, numlevels, iterations, sigma)
    %This function estimates deformation between two subsequent 3-D images using
    %Lucas-Kanade optical flow equation with pyramidal approach.
    %
    %   Description :
    %
    %   -image1, image2 : two subsequent images or frames.
    %   -r : redius of neighbourhood, default value is 2.
    %   -numlevels : the number of levels in pyramid, default value is 2.
    %   -iterations : number of iterations in refinement, default value is 1.
    %   -sigma : standard deviation of Gaussian function, default value is 0.5.
    %
    %   Reference:
    %   Lucas, B. D., Kanade, T., 1981. An iterative image registration
    %   technique with an application to stereo vision. In: Proceedings of the
    %   7th international joint conference on Artificial intelligence - Volume 2.
    %   Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, pp. 674-679.
    %
    %   Author: Mohammad Mustafa
    %   By courtesy of The University of Nottingham and Mirada Medical Limited,
    %   Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    % Default parameters
    if nargin == 5
        sigma = 0.5;
    elseif nargin == 4
        iterations = 1; sigma = 0.5;
    elseif nargin == 3
        numlevels = 2; iterations = 1; sigma = 0.5;
    elseif nargin == 2
        r = 2; numlevels = 2; iterations = 1; sigma = 0.5;
    end

    currentImage1 = image1; currentImage2 = image2;

    % s contains sizes of each pyramid levels
    s = zeros(1, 3, numlevels);
    s(:, :, 1) = (size(image1));

    % Each of pyramid levels will have resized image
    pyrIm1 = zeros(size(image1, 1), size(image1, 2), size(image1, 3), numlevels);
    pyrIm2 = pyrIm1;
    pyrIm1(:, :, :, 1) = currentImage1;
    pyrIm2(:, :, :, 1) = currentImage2;

    % Building pyramid by downsampling
    for i = 2:numlevels
        currentImage1 = sampling3D(currentImage1, 1);
        currentImage2 = sampling3D(currentImage2, 1);
        % Adjusting the size
        pyrIm1(1:size(currentImage1, 1), 1:size(currentImage1, 2), ...
        1:size(currentImage1, 3), i) = currentImage1;
        pyrIm2(1:size(currentImage2, 1), 1:size(currentImage2, 2), ...
            1:size(currentImage2, 3), i) = currentImage2;
        s(:, :, i) = size(currentImage1);
    end

    % Base operation
    currentImage1 = pyrIm1(1:s(1, 1, numlevels), 1:s(1, 2, numlevels), ...
    1:s(1, 3, numlevels), numlevels);
    currentImage2 = pyrIm2(1:s(1, 1, numlevels), 1:s(1, 2, numlevels), ...
        1:s(1, 3, numlevels), numlevels);

    [ux, uy, uz] = LKW3D(currentImage1, currentImage2, r, sigma);

    % Refining flow vectors
    if iterations > 0

        for i = 1:iterations
            [ux, uy, uz] = refinedLK3D(ux, uy, uz, currentImage1, currentImage2, r, sigma);
        end

    end

    % Operations at higher levels of pyramids

    for i = (numlevels - 1):-1:1
        % Size and magnitudes of flow vectors are upsampled
        disp(['level ' num2str(i)])
        temp = 2 * sampling3D(ux, 2);
        uxInitial = temp(1:s(1, 1, i), 1:s(1, 2, i), 1:s(1, 3, i));
        temp = 2 * sampling3D(uy, 2);
        uyInitial = temp(1:s(1, 1, i), 1:s(1, 2, i), 1:s(1, 3, i));
        temp = 2 * sampling3D(uz, 2);
        uzInitial = temp(1:s(1, 1, i), 1:s(1, 2, i), 1:s(1, 3, i));

        currentImage1 = pyrIm1(1:s(1, 1, i), 1:s(1, 2, i), 1:s(1, 3, i), i);
        currentImage2 = pyrIm2(1:s(1, 1, i), 1:s(1, 2, i), 1:s(1, 3, i), i);

        [ux, uy, uz] = refinedLK3D(uxInitial, uyInitial, uzInitial, ...
            currentImage1, currentImage2, r, sigma);

        if iterations > 0

            for j = 1:iterations
                disp(['iteration ' num2str(j)])
                [ux, uy, uz] = refinedLK3D(ux, uy, uz, currentImage1, currentImage2, 2, sigma);
            end

        end

    end

end

function [ux, uy, uz] = refinedLK3D(uxIn, uyIn, uzIn, image1, image2, r, sigma)
    %This function refines Lucas-Kanade 3-D optical flow using previous estimates.
    %
    %   Description :
    %
    %   -uxIn,uyIn,uzIn : initial estimates of optical flow along 3 principal axes.
    %   -image1, image2 : two subsequent images or frames.
    %   -r : redius of neighbourhood, default value is 2.
    %   -sigma : standard deviation of Gaussian function, default value is 0.5.
    %   -ww : wighted window used in least square equation; it gives more
    %         weight to the central pixel.
    %
    %   Author : Mohammad Mustafa
    %   By courtesy of The University of Nottingham and Mirada Medical Limited,
    %   Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    % Default parameters
    if nargin == 5
        r = 2; sigma = 0.5;
    elseif nargin == 6
        sigma = 0.5;
    end

    % displacment vectors should have round numbers of displacement along axes.
    uxIn = round(uxIn); uyIn = round(uyIn); uzIn = round(uzIn);

    s = size(image1);

    % Initializing flow vectors
    ux = zeros(s); uy = ux; uz = ux;

    % Gaussian wighted window to be used in least square equation
    ww = gaussianKernel3D(r, sigma);

    for i = r + 1:s(1) - r

        for j = r + 1:s(2) - r

            for k = r + 1:s(3) - r
                currentImage1 = image1(i - r:i + r, j - r:j + r, k - r:k + r);

                %using initial displacement vectors for refinement
                % for X axis
                minC = uxIn(i, j, k) + j - r;
                maxC = uxIn(i, j, k) + j + r;
                % for Y axis
                minR = uyIn(i, j, k) + i - r;
                maxR = uyIn(i, j, k) + i + r;
                % for Z axis
                minD = uzIn(i, j, k) + k - r;
                maxD = uzIn(i, j, k) + k + r;

                if minR >= 1 && maxR <= s(1) && minC >= 1 && maxC <= s(2) ...
                        && minD >= 1 && maxD <= s(3)
                    currentImage2 = image2(minR:maxR, minC:maxC, minD:maxD);

                    [Ix, Iy, Iz, It] = imageDerivatives3D(currentImage1, currentImage2);

                    % least square equation with weighted window
                    A = zeros(3, 3);
                    B = zeros(3, 1);
                    A(1, 1) = sum(sum(sum((Ix.^2) .* ww)));
                    A(1, 2) = sum(sum(sum((Ix .* Iy) .* ww)));
                    A(1, 3) = sum(sum(sum((Ix .* Iz) .* ww)));

                    A(2, 1) = sum(sum(sum((Iy .* Ix) .* ww)));
                    A(2, 2) = sum(sum(sum((Iy.^2) .* ww)));
                    A(2, 3) = sum(sum(sum((Iy .* Iz) .* ww)));

                    A(3, 1) = sum(sum(sum((Iz .* Ix) .* ww)));
                    A(3, 2) = sum(sum(sum((Iz .* Iy) .* ww)));
                    A(3, 3) = sum(sum(sum((Iz.^2) .* ww)));

                    B(1, 1) = sum(sum(sum((Ix .* It) .* ww)));
                    B(2, 1) = sum(sum(sum((Iy .* It) .* ww)));
                    B(3, 1) = sum(sum(sum((Iz .* It) .* ww)));

                    invofA = pinv(A);

                    V = invofA * (-B);
                    ux(i, j, k) = V(1, 1);
                    uy(i, j, k) = V(2, 1);
                    uz(i, j, k) = V(3, 1);
                end

            end

        end

    end

    ux = ux + uxIn;
    uy = uy + uyIn;
    uz = uz + uzIn;

end

function output = sampling3D(input, option)
    %This function will downsample or upsample a 3-D image by a factor of 2.
    %   Description :
    %
    %   -input : input 3-D matrix
    %   =option : options; 1 for downsample and 2 for upsample
    %   -output : downsampled 3-D image
    %
    %   Author: Mohammad Mustafa
    %   By courtesy of The University of Nottingham and Mirada Medical Limited,
    %   Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    if option == 1
        % Downsampling by a factor of 2
        input = imfilter(input, gaussianKernel3D(2, 1.5));
        output = input(1:2:size(input, 1), 1:2:size(input, 2), 1:2:size(input, 3));
    elseif option == 2
        % Upsampling by a factor of 2
        output = zeros(2 * size(input));

        for i = 1:size(input, 1)

            for j = 1:size(input, 2)

                for k = 1:size(input, 3)
                    output(2 * i - 1:2 * i, 2 * j - 1:2 * j, 2 * k - 1:2 * k) = input(i, j, k);
                end

            end

        end

        output = imfilter(output, gaussianKernel3D(2, 1.5));

    else
        error('Option needs to be either 1 or 2.');
    end

end

function h = gaussianKernel3D(r, sigma)
    %This function creates a pre-defined 3-D Gaussian kernel.
    %
    %   Description :
    %
    %   h = gaussianKernel3D(r,sigma) returns a rotationally symmetric Gaussian
    %   kernel h of size 2*r+1 with standard deviation sigma (positive). The
    %   default size of r is 1 and the default sigma is 0.5.
    %
    %   Author : Mohammad Mustafa
    %   By courtesy of The University of Nottingham and Mirada Medical Limited,
    %   Oxford, UK
    %
    %   Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    %   3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    %
    %   June 2012

    if nargin < 1
        r = 1; sigma = 0.5;
    elseif nargin == 1
        sigma = 0.5;
    end

    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);
    arg =- (x .* x + y .* y + z .* z) / (2 * sigma * sigma);

    h = exp(arg);

    if sum(h(:)) ~= 0,
        h = h / sum(h(:));
    end;

end

function [Ix, Iy, Iz, It] = imageDerivatives3D(image1, image2)
    %This fuction computes 3D derivatives between two 3D images.
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

    % Aadjusting sizes
    Ix = Ix(1:size(Ix, 1) - 1, 1:size(Ix, 2) - 1, 1:size(Ix, 3) - 1);
    Iy = Iy(1:size(Iy, 1) - 1, 1:size(Iy, 2) - 1, 1:size(Iy, 3) - 1);
    Iz = Iz(1:size(Iz, 1) - 1, 1:size(Iz, 2) - 1, 1:size(Iz, 3) - 1);
    It = It(1:size(It, 1) - 1, 1:size(It, 2) - 1, 1:size(It, 3) - 1);

end
