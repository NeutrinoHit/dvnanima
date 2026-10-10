"""Physics tests of the wormhole film (run: python3 test_wormhole_camera.py)."""
import numpy as np
import wormhole_physics as wp

rng = np.random.default_rng(1)


def test_closed_form_vs_quadrature():
    worst = 0.0
    for _ in range(400):
        lc = rng.uniform(0.0, 12.0)
        psi = rng.uniform(0.01, np.pi - 0.01)
        thr, a = wp.sweep(np.array([psi]), lc, 1.0)
        thr2, a2 = wp.sweep_quadrature(psi, lc, 1.0)
        if abs(np.sin(psi) * np.hypot(lc, 1.0) - 1.0) < 2e-3 or (abs(psi - np.pi / 2) < 0.1 and lc > 1.0):
            continue          # skip the log peak at p = b and the tangent rays (the quadrature is the weak side there)
        assert bool(thr[0]) == thr2, (lc, psi)
        worst = max(worst, abs(a[0] - a2))
    assert worst < 2e-6, worst


def test_flat_limit():
    for lc in (3.0, 8.0):
        for psi in (0.1, 0.7, 1.4):
            _, a = wp.sweep(np.array([psi]), lc, 1e-5)
            assert abs(a[0] - (np.pi - psi)) < 1e-4
        _, a = wp.sweep(np.array([2.5]), lc, 1e-5)          # outward: straight line, swept angle pi - psi
        assert abs(a[0] - (np.pi - 2.5)) < 1e-4


def test_critical_cone():
    for lc in (0.0, 1.0, 3.0, 9.0):
        pc = wp.critical_angle(lc, 1.0)
        thr, _ = wp.sweep(np.array([pc - 1e-6, pc + 1e-6]), lc, 1.0)
        if lc > 0:
            assert thr[0] and not thr[1]


def test_throat_symmetry():
    psi = np.linspace(0.05, 1.5, 7)
    _, a_in = wp.sweep(psi, 0.0, 1.0)
    _, a_out = wp.sweep(np.pi - psi, 0.0, 1.0)
    assert np.allclose(a_in, a_out, atol=1e-9)


def test_energy_estimate_of_the_book():
    c, G = 299792458.0, 6.67430e-11
    e = wp.side_energy_si(1000.0, c, G)
    assert abs(e / 1e47 + 0.95) < 0.01
    assert abs(2 * e / 1.8e47) < 1.1                           # about the rest energy of the Sun


def test_surface():
    b = 1.3
    l = np.linspace(-4, 4, 9)
    z = wp.throat_z_of_l(l, b)
    assert np.allclose(wp.surface_radius(z, b) ** 2, l ** 2 + b ** 2)


def test_ray_path_matches_sweep():
    for lc, psi in ((4.0, 0.9), (4.0, 0.2), (2.0, 1.3), (1.0, 0.5), (6.0, 2.4)):
        thr, ang = wp.sweep(np.array([psi]), lc, 1.0)
        u, ph, through = wp.ray_path(psi, lc, 1.0, 14.0, 4000)
        assert through == bool(thr[0])
        assert abs(ph[-1] - ang[0]) < 2e-3, (lc, psi, ph[-1], ang[0])


def test_window_radius_in_the_picture():
    import wormhole_view as wv
    import wormhole_sky as sk
    W, H = 320, 180
    fov = 70.0
    for l in (2.0, 5.0, 9.0):
        _, far = wv.render_view(l, 0.3, 0.0, 0.0, 0.0, W, H, fov=fov)
        row = far[H // 2]
        xs = np.where(row)[0]
        f = 0.5 * W / np.tan(np.radians(fov) / 2)
        r_px = 0.5 * (xs.max() - xs.min() + 1)
        r_expected = f * np.tan(wp.critical_angle(l, 1.0))
        assert abs(r_px - r_expected) < 1.5, (l, r_px, r_expected)


def test_continuity_through_the_throat():
    import wormhole_view as wv
    W, H = 160, 90
    a, fa = wv.render_view(0.02, 1.0, 0.0, 0.0, 0.0, W, H, fov=80.0)
    b_, fb = wv.render_view(-0.02, 1.0, 0.0, 0.0, 0.0, W, H, fov=80.0)
    assert fa.all() and fb.all()                     # the same (other) sky ahead on both sides of the throat
    assert np.abs(a - b_).mean() < 0.03


def test_look_back_shows_our_sky_in_a_window():
    import wormhole_view as wv
    W, H = 160, 90
    _, far = wv.render_view(-5.0, 0.0, np.pi, 0.0, 0.0, W, H, fov=70.0)
    assert (~far).any() and far.any()                 # camera on the far side looking back: a window with the other (our) sky inside
    _, far2 = wv.render_view(-5.0, 0.0, 0.0, 0.0, 0.0, W, H, fov=70.0)
    assert far2.all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
