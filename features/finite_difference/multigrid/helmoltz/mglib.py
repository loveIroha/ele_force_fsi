def compute_multigrid_levels_2D(Nx, Ny, width, height):
    num_levels = 0
    level_dh = []
    level_dim = []
    dx = width/Nx
    dy = height/Ny
    while True:
        num_levels += 1
        level_dh.append((dx, dy))
        level_dim.append((Nx, Ny))
        print(f"num_levels : {num_levels}, level_dh  : {dx:.4e}, {dy:.4e}")
        print(f"num_levels : {num_levels}, level_dim : {Nx}, {Ny}")
        if Nx % 2 == 1 or Ny % 2 == 1:
            break
        if Nx * Ny < 17:
            break
        Nx, Ny = (Nx // 2, Ny // 2)
        dx, dy = (dx * 2, dy * 2)
    if num_levels <= 1:
        raise Exception("the problem is too small for the multigrid solver.")
    return num_levels, level_dh, level_dim