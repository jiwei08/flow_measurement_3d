function plotFunc = plotFuncs
    plotFunc.plot_mat3d = @plot_mat3d;
    plotFunc.plot_gif3d = @plot_gif3d;
    plotFunc.plot_gif3d_seqfile = @plot_gif3d_seqfile;
    plotFunc.plot_flow = @plot_flow;
end

function plot_mat3d(filename, outputFolder)

    if nargin == 1
        isplot = true;
    elseif isempty(outputFolder)
        isplot = false;
        outputFolder = './tmp/';
    else
        isplot = false;
    end

    rst = load(filename);
    [~,name,~]=fileparts(filename);

    if isfield(rst, 'rst')
        sourceMask = rst.rst;
        rst = rmfield(rst, 'rst');
        outname=[name, '_mask_inv.png'];
    end

    if isfield(rst, 'sourceMask')
        sourceMask = rst.sourceMask;
        rst = rmfield(rst, 'sourceMask');
        outname=[name, '.png'];
    end

    if ~isplot
        figure('visible', 'off');
    else
        figure();
    end

    [Nx, Ny, Nz] = size(sourceMask);
    isosurface(1:Nx, 1:Ny, 1:Nz, permute(sourceMask, [2, 1, 3]), 0.8, 'FaceColor', 'red');
    axis image; box on;
    % xlabel('x'); ylabel('y'); zlabel('z');

    % [x, y, z] = meshgrid(1:Nx, 1:Ny, 0);
    % surface(x, y, z, max(permute(sourceMask, [2, 1, 3]), [], 3), 'EdgeColor', 'none', 'FaceAlpha', 1);

    [x, y, z] = meshgrid(1:Nx, 1:Ny, 0);
    surface(x, y, z, 0.3 * ones(Ny, Nx), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.2);

    % [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
    % surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), ...
    %     reshape(max(permute(sourceMask, [2, 1, 3]), [], 2), [Ny, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

    % [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
    % surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), ...
    %     0.3 * ones(Ny, Nz), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

    [x, y, z] = meshgrid(1:Nx, Ny, 1:Nz);
    surface(reshape(x, [Nx, Nz]), reshape(y, [Nx, Nz]), reshape(z, [Nx, Nz]), ...
        0.3 * ones(Nx, Nz), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

    % [x, y, z] = meshgrid(1:Nx, Ny, 1:Nz);
    % surface(reshape(x, [Nx, Nz]), reshape(y, [Nx, Nz]), reshape(z, [Nx, Nz]), ...
    %     reshape(max(permute(sourceMask, [2, 1, 3]), [], 1), [Nx, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

    [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
    surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), ...
        reshape(max(permute(sourceMask, [2, 1, 3]), [], 2), [Ny, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

    caxis([0, 1]);
    colormap default;
    colorbar;

    light('position', [-1, 0, 0]);
    % camproj('perspective');
    % view(0, 90);

    set(gca, 'xticklabel', [], 'yticklabel', [], 'zticklabel', []);
    xlabel('Transverse'); ylabel('Flow direction'); zlabel('Depth');
    grid on

    if ~isplot
        exportgraphics(gca, fullfile(outputFolder, outname));
    end

    field = fieldnames(rst);

    for i = 1:length(field)

        if ~isplot
            figure('visible', 'off');
        else
            figure();
        end

        key = field{i};
        semilogy(rst.(key) / rst.(key)(1), 'LineWidth', 2);
        xlabel('iter');

        switch key
            case 'res'
                ylabel('$\|Ax_k-b\|/\|b\|$', 'Interpreter', 'latex');
            case 'res_model'
                ylabel('$\|x_k-x_*\|/\|x_*\|$', 'Interpreter', 'latex');
            otherwise
                ylabel('error');
        end

        ax = gca();
        ax.FontSize = 13;
        ax.Title = [];

        if ~isplot
            exportgraphics(gca, fullfile(outputFolder, [name,'_',key, '.png']));
        end

    end

    % matlab -nosplash -nodesktop -r "addpath(genpath('utils'));plot_mat3d('./rst.mat');exit;"
end

function plot_gif3d_seqfile(filepre,L,outputFolder)
    preind=strfind(filepre,'/');
    filenamepre=filepre(preind(end)+1:end-1);

    if nargin == 2 || isempty(outputFolder)
        outputFolder = filepre(1:preind(end));
    end
    for loop=0:L
        disp(['Ploting ',num2str(loop),' mask...'])
        h=figure('visible','off');
        filename=[filepre,num2str(loop),'.mat'];
        [~,name,~]=fileparts(filename);
        data=load(filename);

        if isfield(data, 'rst')
            sourceMask = data.rst;
            data = rmfield(data, 'rst');
            outname=[filenamepre, '_mask_inv.gif'];
        end
    
        if isfield(data, 'sourceMask')
            sourceMask = data.sourceMask;
            data = rmfield(data, 'sourceMask');
            outname=[filenamepre, '.gif'];
        end

        [Nx, Ny, Nz] = size(sourceMask);
        isosurface(1:Nx, 1:Ny, 1:Nz, permute(sourceMask, [2, 1, 3]), 0.8, 'FaceColor', 'red');
        axis image; box on;

        [x, y, z] = meshgrid(1:Nx, 1:Ny, 0);
        surface(x, y, z, 0.3 * ones(Ny, Nx), ...
            'EdgeColor', 'none', 'FaceAlpha', 0.2);

        [x, y, z] = meshgrid(1:Nx, Ny, 1:Nz);
        surface(reshape(x, [Nx, Nz]), reshape(y, [Nx, Nz]), reshape(z, [Nx, Nz]), ...
            0.3 * ones(Nx, Nz), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
        surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), ...
            reshape(max(permute(sourceMask, [2, 1, 3]), [], 2), [Ny, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

        caxis([0, 1]);
        colormap default;
        colorbar;

        light('position', [-1, 0, 0]);

        set(gca, 'xticklabel', [], 'yticklabel', [], 'zticklabel', []);
        xlabel('Transverse'); ylabel('Flow direction'); zlabel('Depth');
        grid on

        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if loop == 0
            imwrite(imind, cm, fullfile(outputFolder,outname), 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
        else
            imwrite(imind, cm, fullfile(outputFolder,outname), 'gif', 'WriteMode', 'append', 'DelayTime', 0.3);
        end
    end
end
function plot_gif3d(filename, outputFolder)

    if nargin == 1
        outputFolder = './';
    elseif isempty(outputFolder)
        outputFolder = './';
    end

    seq_mask = load(filename);
    seq_mask = seq_mask.seq_inv;

    % seq_mask = seq_mask .* (seq_mask > 0.8);

    [L, Nx, Ny, Nz] = size(seq_mask);

    for loop = 1:L
        h = figure('visible', 'off');
        cur_mask = squeeze(seq_mask(loop, :, :, :));
        isosurface(1:Nx, 1:Ny, 1:Nz, permute(cur_mask, [2, 1, 3]), 0.8, 'FaceColor', 'red');
        axis image; box on;
        xlabel('x'); ylabel('y'); zlabel('z');

        [x, y, z] = meshgrid(1:Nx, 1:Ny, 0);
        surface(x, y, z, 0.3 * ones(Ny, Nx), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
        % surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), 0.3 * ones(Ny, Nz), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % [x, y, z] = meshgrid(1:Nx, Ny, 1:Nz);
        % surface(reshape(x, [Nx, Nz]), reshape(y, [Nx, Nz]), reshape(z, [Nx, Nz]), reshape(max(permute(cur_mask, [2, 1, 3]), [], 1), [Nx, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

        [x, y, z] = meshgrid(1:Nx, Ny, 1:Nz);
        surface(reshape(x, [Nx, Nz]), reshape(y, [Nx, Nz]), reshape(z, [Nx, Nz]), ...
            0.3 * ones(Nx, Nz), 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        [x, y, z] = meshgrid(Nx, 1:Ny, 1:Nz);
        surface(reshape(x, [Ny, Nz]), reshape(y, [Ny, Nz]), reshape(z, [Ny, Nz]), ...
            reshape(max(permute(cur_mask, [2, 1, 3]), [], 2), [Ny, Nz]), 'EdgeColor', 'none', 'FaceAlpha', 1);

        caxis([0, 1]);
        colormap default;
        colorbar;

        light('position', [-1, 0, 0]);
        camproj('perspective');
        % view([0, 90])

        set(gca, 'xticklabel', [], 'yticklabel', [], 'zticklabel', []);
        grid on

        % write to the GIF file
        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if loop == 1
            imwrite(imind, cm, fullfile(outputFolder,'seq_inv.gif'), 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, fullfile(outputFolder,'seq_inv.gif'), 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end

    end

end

function plot_flow(flow, outputFolder)

    if nargin == 1
        outputFolder = './';
    elseif isempty(outputFolder)
        outputFolder = './';
    end

    ux = squeeze(flow(:, :, :, 1));
    uy = squeeze(flow(:, :, :, 2));
    uz = squeeze(flow(:, :, :, 3));

    f = 2;
    x = ux(1:f:size(ux, 1), 1:f:size(ux, 2), 1:f:size(ux, 3));
    y = uy(1:f:size(uy, 1), 1:f:size(uy, 2), 1:f:size(uy, 3));
    z = uz(1:f:size(uz, 1), 1:f:size(uz, 2), 1:f:size(uz, 3));

    % n = sqrt(x.^2 + y.^2 + z.^2); x = x .* (n > 0.7); y = y .* (n > 0.7); z = z .* (n > 0.7);

    [X, Y, Z] = meshgrid(1:size(x, 2), 1:size(x, 1), 1:size(x, 3));
    quiver3(X, Y, Z, permute(x, [2, 1, 3]), permute(y, [2, 1, 3]), permute(z, [2, 1, 3])); axis([1 size(x, 2) 1 size(x, 1) 1 size(x, 3)]);
    xlabel('x'); ylabel('y'); zlabel('z');
    camproj('perspective');
    exportgraphics(gca, fullfile(outputFolder,'flow.png'));
end
