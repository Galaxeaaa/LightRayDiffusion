import torch

def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    # I_eps = torch.zeros_like(I_min_cov.sum(dim=-3)) + 1e-10
    # p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # I_min_cov.sum(dim=-3): torch.Size([1, 1, 3, 3])
    # sum_proj: torch.Size([1, 1, 3, 1])

    # p_intersect = np.linalg.lstsq(I_min_cov.sum(dim=-3).numpy(), sum_proj.numpy(), rcond=None)[0]

    if torch.any(torch.isnan(p_intersect)):
        print("Warning: intersect_skew_lines_high_dim: p_intersect is nan")
        return None, None
    return p_intersect, r

def pluckerRays2Point(plucker_rays):
    rays = torch.tensor(plucker_rays)
    mask = torch.linalg.norm(rays, dim=-1) > 1e-5
    d = rays[..., :3]
    m = rays[..., 3:]
    p = torch.cross(d, m, dim=-1)
    p_intersect, _ = intersect_skew_lines_high_dim(p, d, mask)

    return p_intersect

if __name__ == "__main__":
    rays = torch.tensor([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
    ], dtype=torch.float32)
    rays = rays.unsqueeze(0).repeat(3, 1, 1)
    print(pluckerRays2Point(rays))