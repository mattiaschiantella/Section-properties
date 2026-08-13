import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def rects_overlap(r1, r2):
    """Axis-aligned bounding box overlap test between two rectangles."""
    x0_1, y0_1, b1, h1 = r1
    x0_2, y0_2, b2, h2 = r2
    x1_1, y1_1 = x0_1 + b1, y0_1 + h1
    x1_2, y1_2 = x0_2 + b2, y0_2 + h2

    if x1_1 <= x0_2 or x1_2 <= x0_1:
        return False
    if y1_1 <= y0_2 or y1_2 <= y0_1:
        return False
    return True


def compute_geometry(rectangles):
    """Area, centroid, centroidal moments of inertia and principal axes.
    Returns plain Python floats only (JSON-serializable, needed for
    dcc.Store)."""
    A_tot = 0.0
    x_num = 0.0
    y_num = 0.0

    for x0, y0, b, h in rectangles:
        A = b * h
        xc = x0 + b / 2
        yc = y0 + h / 2

        A_tot += A
        x_num += A * xc
        y_num += A * yc

    x_bar = x_num / A_tot
    y_bar = y_num / A_tot

    Ix = 0.0
    Iy = 0.0
    Ixy = 0.0

    for x0, y0, b, h in rectangles:
        A = b * h
        xc = x0 + b / 2
        yc = y0 + h / 2

        dx = xc - x_bar
        dy = yc - y_bar

        Ix += (b * h**3) / 12 + A * dy**2
        Iy += (h * b**3) / 12 + A * dx**2
        Ixy += A * dx * dy

    # Principal axes of inertia
    if abs(Ix - Iy) < 1e-9 and abs(Ixy) < 1e-9:
        theta_p = 0.0
    else:
        theta_p = 0.5 * np.arctan2(-2 * Ixy, (Ix - Iy))

        # Keep the 'x' principal axis aligned with I1 (I_max)
        if Ix < Iy:
            theta_p += np.pi / 2
        theta_p = (theta_p + np.pi / 2) % np.pi - np.pi / 2

    R = np.sqrt(((Ix - Iy) / 2) ** 2 + Ixy**2)
    I_mean = (Ix + Iy) / 2
    I1 = I_mean + R
    I2 = I_mean - R

    return {
        "A_tot": float(A_tot), "x_bar": float(x_bar), "y_bar": float(y_bar),
        "Ix": float(Ix), "Iy": float(Iy), "Ixy": float(Ixy),
        "theta_p": float(theta_p), "I1": float(I1), "I2": float(I2),
    }


def format_geometry_output(geom):
    lines = []
    lines.append(f"Total area = {geom['A_tot']:.4f} mm\u00b2")
    lines.append(f"Centroid x\u0304 = {geom['x_bar']:.4f} mm")
    lines.append(f"Centroid y\u0304 = {geom['y_bar']:.4f} mm")
    lines.append("")
    lines.append("Centroidal moments of inertia:")
    lines.append(f"Ix  = {geom['Ix']:.4f} mm\u2074")
    lines.append(f"Iy  = {geom['Iy']:.4f} mm\u2074")
    lines.append(f"Ixy = {geom['Ixy']:.4f} mm\u2074")
    lines.append("")
    lines.append("Principal axes of inertia (x, y):")
    lines.append(f"theta_p (counter-clockwise) = {np.degrees(geom['theta_p']):.4f} deg")
    lines.append(f"I1 (max) = {geom['I1']:.4f} mm\u2074")
    lines.append(f"I2 (min) = {geom['I2']:.4f} mm\u2074")
    return lines


# =====================================================================
# STRESSES
# =====================================================================

def sigma_bending_at(x, y, Mx, My, geom):
    Ix, Iy, Ixy = geom["Ix"], geom["Iy"], geom["Ixy"]
    Delta = Ix * Iy - Ixy**2
    xc = x - geom["x_bar"]
    yc = y - geom["y_bar"]
    return ((Iy * Mx + Ixy * My) * yc - (Ix * My + Ixy * Mx) * xc) / Delta


def sigma_total_at(x, y, Mx, My, N, geom):
    return N / geom["A_tot"] + sigma_bending_at(x, y, Mx, My, geom)


def compute_stress_extremes(rectangles, Mx, My, N, geom):
    sigma_max = -np.inf
    sigma_min = np.inf
    loc_max = None
    loc_min = None

    for x0, y0, b, h in rectangles:
        corners = [(x0, y0), (x0 + b, y0), (x0, y0 + h), (x0 + b, y0 + h)]
        for (x, y) in corners:
            s = sigma_total_at(x, y, Mx, My, N, geom)
            if s > sigma_max:
                sigma_max = s
                loc_max = (x, y)
            if s < sigma_min:
                sigma_min = s
                loc_min = (x, y)

    return float(sigma_max), loc_max, float(sigma_min), loc_min


# =====================================================================
# SHEAR STRESSES (approximate Jourawski / Zhuravskii method)
#
# Classical assumption: at a given cut, the shear stress is uniform
# across the width (horizontal cut) or height (vertical cut) of the
# cut, i.e. tau = V * S / (I * b). Vy and Vx are treated independently
# (horizontal cuts for Vy using Ix, vertical cuts for Vx using Iy) and
# combined as a vector magnitude when both are present. This is an
# engineering approximation, not a rigorous 2D coupled treatment.
# =====================================================================

def _width_at_y(rectangles, y):
    """Total material width at height y (sum over all rectangles whose
    y-range contains y -- handles disjoint pieces, e.g. flanges)."""
    w = 0.0
    for x0, y0, b, h in rectangles:
        if y0 <= y <= y0 + h:
            w += b
    return w


def _height_at_x(rectangles, x):
    """Total material height at position x (sum over all rectangles
    whose x-range contains x)."""
    h_tot = 0.0
    for x0, y0, b, h in rectangles:
        if x0 <= x <= x0 + b:
            h_tot += h
    return h_tot


def _Sx_below(rectangles, y, y_bar):
    """First moment about the centroidal x-axis of the part of the
    section with y' <= y (a horizontal cut at height y)."""
    S = 0.0
    for x0, y0, b, h in rectangles:
        if y <= y0:
            continue
        y_bottom = min(y, y0 + h)
        hh = y_bottom - y0
        if hh <= 0:
            continue
        area = b * hh
        yc = y0 + hh / 2
        S += area * (yc - y_bar)
    return S


def _Sy_left(rectangles, x, x_bar):
    """First moment about the centroidal y-axis of the part of the
    section with x' <= x (a vertical cut at position x)."""
    S = 0.0
    for x0, y0, b, h in rectangles:
        if x <= x0:
            continue
        x_right = min(x, x0 + b)
        bb = x_right - x0
        if bb <= 0:
            continue
        area = bb * h
        xc = x0 + bb / 2
        S += area * (xc - x_bar)
    return S


def tau_from_Vy(rectangles, y, Vy, geom):
    """Shear stress at height y produced by a vertical shear force Vy
    (horizontal cut, uniform-across-width assumption)."""
    w = _width_at_y(rectangles, y)
    if w <= 0:
        return 0.0
    S = _Sx_below(rectangles, y, geom["y_bar"])
    return Vy * S / (geom["Ix"] * w)


def tau_from_Vx(rectangles, x, Vx, geom):
    """Shear stress at position x produced by a horizontal shear force
    Vx (vertical cut, uniform-across-height assumption)."""
    h_tot = _height_at_x(rectangles, x)
    if h_tot <= 0:
        return 0.0
    S = _Sy_left(rectangles, x, geom["x_bar"])
    return Vx * S / (geom["Iy"] * h_tot)


# =====================================================================
# PLOTLY HELPERS
# =====================================================================

def _axis_ref(fig, row, col):
    """Return ('x', 'y'), ('x2', 'y2'), ... for the given subplot cell."""
    sp = fig.get_subplot(row, col)
    xref = sp.xaxis.plotly_name.replace("axis", "")
    yref = sp.yaxis.plotly_name.replace("axis", "")
    return xref, yref


def _view_range(xmin, xmax, ymin, ymax, margin_fraction=0.20):
    """Margin + Navier / Scienza delle Costruzioni convention: x axis
    increasing to the LEFT, y axis increasing DOWNWARD (reversed ranges)."""
    margin_x = margin_fraction * max(xmax - xmin, 1e-6)
    margin_y = margin_fraction * max(ymax - ymin, 1e-6)
    xlo, xhi = xmin - margin_x, xmax + margin_x
    ylo, yhi = ymin - margin_y, ymax + margin_y
    return [xhi, xlo], [yhi, ylo]  # reversed ranges


def _rect_shapes(rectangles, fillcolor="lightgray", opacity=0.6, line_color="black"):
    return [
        dict(type="rect", x0=x0, y0=y0, x1=x0 + b, y1=y0 + h,
             line=dict(color=line_color, width=1),
             fillcolor=fillcolor, opacity=opacity, layer="above")
        for x0, y0, b, h in rectangles
    ]


def _axes_annotations(origin, angle, label_x, label_y, scale, color="black", xref="x", yref="y"):
    x0, y0 = origin
    dx = scale * np.cos(angle)
    dy = scale * np.sin(angle)

    def arrow(xt, yt, text):
        return dict(
            x=xt, y=yt, ax=x0, ay=y0, axref=xref, ayref=yref,
            xref=xref, yref=yref,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=color, text=text, font=dict(size=12, color=color),
            standoff=0,
        )

    return [arrow(x0 + dx, y0 + dy, label_x), arrow(x0 - dy, y0 + dx, label_y)]


def _infinite_line_points(point, direction, xr, yr):
    x0, y0 = point
    dx, dy = direction
    norm = np.hypot(dx, dy)
    if norm < 1e-12:
        return None
    dx, dy = dx / norm, dy / norm
    span = 2 * max(abs(xr[1] - xr[0]), abs(yr[1] - yr[0]), 1.0)
    return [x0 - span * dx, x0 + span * dx], [y0 - span * dy, y0 + span * dy]


def _moment_arrow(fig, center, phi, length, color="purple", xref="x", yref="y", row=None, col=None):
    """Moment vector: a plain shaft ending in two consecutive arrowheads
    on the SAME side (chevron), pointing toward the positive direction
    of (Mx, My) as defined by the reference system."""
    dirx, diry = np.cos(phi), np.sin(phi)
    base = (center[0] - length / 2 * dirx, center[1] - length / 2 * diry)
    tip = (center[0] + length / 2 * dirx, center[1] + length / 2 * diry)

    chevron = 0.22 * length
    p1 = (tip[0] - chevron * dirx, tip[1] - chevron * diry)
    p2 = (tip[0] - 2 * chevron * dirx, tip[1] - 2 * chevron * diry)

    trace_kwargs = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(go.Scatter(
        x=[base[0], p2[0]], y=[base[1], p2[1]], mode="lines",
        line=dict(color=color, width=3), hoverinfo="skip", showlegend=False,
    ), **trace_kwargs)

    annotations = []
    for start, end in [(p2, p1), (p1, tip)]:
        annotations.append(dict(
            x=end[0], y=end[1], ax=start[0], ay=start[1], axref=xref, ayref=yref,
            xref=xref, yref=yref, showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2.5, arrowcolor=color, text="",
        ))
    annotations.append(dict(
        x=tip[0], y=tip[1], text="M", showarrow=False, xref=xref, yref=yref,
        font=dict(size=13, color=color, family="Arial Black"),
        xanchor="left", yanchor="middle",
    ))
    return annotations


def _sigma_diagram_ribbon(rectangles, geom, Mx, My, N, sigma_min, sigma_max):
    """
    Build the classic Navier linear diagram (zero at the neutral axis,
    growing away from it) as a ribbon polygon in the SAME (x, y) plane
    as the section: a straight line through the centroid, perpendicular
    to the neutral axis, offset sideways by a fixed clearance (so it
    doesn't sit on top of the material) plus an amount proportional to
    sigma at each point along that line.

    Returns a dict with the polygon coordinates, or None if there's no
    bending (uniform stress, nothing meaningful to draw).
    """
    Ix, Iy, Ixy = geom["Ix"], geom["Iy"], geom["Ixy"]
    Delta = Ix * Iy - Ixy**2
    k2 = -(Ix * My + Ixy * Mx) / Delta
    k3 = (Iy * Mx + Ixy * My) / Delta
    norm = np.hypot(k2, k3)
    if norm < 1e-9:
        return None

    n_hat = np.array([k2, k3]) / norm
    t_hat = np.array([-n_hat[1], n_hat[0]])

    corners = []
    for x0, y0, b, h in rectangles:
        corners += [(x0, y0), (x0 + b, y0), (x0, y0 + h), (x0 + b, y0 + h)]
    corners = np.array(corners)
    proj_n = corners @ n_hat
    proj_t = corners @ t_hat
    n_min_s, n_max_s = proj_n.min(), proj_n.max()
    t_max_s = proj_t.max()

    centroid = np.array([geom["x_bar"], geom["y_bar"]])
    t_bar = centroid @ t_hat

    s = np.linspace(n_min_s, n_max_s, 60)
    pts = np.outer(s, n_hat) + t_bar * t_hat
    sigma_s = sigma_total_at(pts[:, 0], pts[:, 1], Mx, My, N, geom)

    span_n = max(n_max_s - n_min_s, 1e-6)
    vmax = max(abs(sigma_min), abs(sigma_max), 1e-9)
    scale = 0.22 * span_n / vmax
    gap = 0.06 * span_n

    # baseline offset large enough to clear the section on every sample
    # (both the tension and compression bulges), so nothing overlaps
    clearance = (t_max_s - t_bar) + gap + scale * vmax
    baseline_t = t_bar + clearance

    outer_t = baseline_t + scale * sigma_s
    outer_pts = np.outer(s, n_hat) + np.outer(outer_t, t_hat)
    base_pts = np.outer(s, n_hat) + baseline_t * t_hat

    return dict(s=s, sigma=sigma_s, outer=outer_pts, base=base_pts, n_hat=n_hat, t_hat=t_hat)


def _add_sigma_ribbon_trace(fig, ribbon, row=None, col=None):
    """Split the ribbon at the zero crossing (if any) so tension and
    compression get different (semi-transparent) colors."""
    s, sigma, outer, base = ribbon["s"], ribbon["sigma"], ribbon["outer"], ribbon["base"]
    kwargs = dict(row=row, col=col) if row is not None else {}

    sign = np.sign(sigma)
    change = np.where(np.diff(sign) != 0)[0]
    splits = [0] + [i + 1 for i in change] + [len(s)]

    for i0, i1 in zip(splits[:-1], splits[1:]):
        if i1 - i0 < 2:
            continue
        seg_sigma = sigma[i0:i1]
        color = "rgba(200,30,30,0.28)" if seg_sigma.mean() >= 0 else "rgba(30,60,200,0.28)"
        line_color = "rgba(150,20,20,0.8)" if seg_sigma.mean() >= 0 else "rgba(20,40,150,0.8)"
        poly_x = list(outer[i0:i1, 0]) + list(base[i0:i1, 0][::-1])
        poly_y = list(outer[i0:i1, 1]) + list(base[i0:i1, 1][::-1])
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y, mode="lines", fill="toself",
            line=dict(color=line_color, width=1.5), fillcolor=color,
            hoverinfo="skip", showlegend=False,
        ), **kwargs)

    fig.add_trace(go.Scatter(
        x=[outer[0, 0], outer[-1, 0]], y=[outer[0, 1], outer[-1, 1]], mode="markers",
        marker=dict(color="black", size=4), showlegend=False, hoverinfo="skip",
    ), **kwargs)

    return [
        dict(x=outer[0, 0], y=outer[0, 1], text=f"{sigma[0]:.1f} MPa", showarrow=False,
             font=dict(size=10), xshift=-6, yshift=-6),
        dict(x=outer[-1, 0], y=outer[-1, 1], text=f"{sigma[-1]:.1f} MPa", showarrow=False,
             font=dict(size=10), xshift=6, yshift=6),
    ]


def _tau_diagram_ribbon(coord_samples, tau_samples, axis, xmin, xmax, ymin, ymax):
    """
    Build a tau(y) or tau(x) ribbon, attached directly to the outside
    edge of the section's bounding box (so it never overlaps the
    material) and growing further outward proportionally to tau.

    axis="y": ribbon runs along y, attached past x=xmax, bulges in +x.
    axis="x": ribbon runs along x, attached past y=ymax, bulges in +y.

    Uses the magnitude of tau (not the signed value): the sign of tau
    from the S/I/b formula is a convention artifact, not a physically
    meaningful direction here, so the diagram always bulges outward,
    zero at the free edges and maximum near the neutral axis.
    """
    tau_abs = np.abs(tau_samples)
    vmax = max(np.max(tau_abs), 1e-9)

    if axis == "y":
        span = max(ymax - ymin, 1e-6)
        gap = 0.06 * span
        scale = 0.22 * span / vmax
        base = xmax + gap
        outer = base + scale * tau_abs
        base_pts = np.column_stack([np.full_like(coord_samples, base), coord_samples])
        outer_pts = np.column_stack([outer, coord_samples])
    else:
        span = max(xmax - xmin, 1e-6)
        gap = 0.06 * span
        scale = 0.22 * span / vmax
        base = ymax + gap
        outer = base + scale * tau_abs
        base_pts = np.column_stack([coord_samples, np.full_like(coord_samples, base)])
        outer_pts = np.column_stack([coord_samples, outer])

    return dict(coord=coord_samples, tau=tau_samples, outer=outer_pts, base=base_pts)


def _add_tau_ribbon_trace(fig, ribbon, color="rgba(255,140,0,0.30)", line_color="darkorange"):
    outer, base = ribbon["outer"], ribbon["base"]
    poly_x = list(outer[:, 0]) + list(base[::-1, 0])
    poly_y = list(outer[:, 1]) + list(base[::-1, 1])
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y, mode="lines", fill="toself",
        line=dict(color=line_color, width=1.5), fillcolor=color,
        hoverinfo="skip", showlegend=False,
    ))


# =====================================================================
# FIGURE BUILDERS
# =====================================================================

def build_geometry_figure(rectangles, geom=None):
    fig = go.Figure()

    if not rectangles:
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                           height=520, margin=dict(l=20, r=20, t=20, b=20))
        return fig

    xs = [r[0] for r in rectangles]
    ys = [r[1] for r in rectangles]
    x1s = [r[0] + r[2] for r in rectangles]
    y1s = [r[1] + r[3] for r in rectangles]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)
    axis_scale = 0.12 * max(xmax - xmin, ymax - ymin, 1.0)

    shapes = _rect_shapes(rectangles)
    annotations = _axes_annotations((0, 0), 0, "x'", "y'", axis_scale)

    if geom is not None:
        fig.add_trace(go.Scatter(
            x=[geom["x_bar"]], y=[geom["y_bar"]], mode="markers",
            marker=dict(color="red", size=9), showlegend=False,
            hovertemplate="centroid<br>x=%{x:.2f} mm, y=%{y:.2f} mm<extra></extra>",
        ))
        annotations += _axes_annotations(
            (geom["x_bar"], geom["y_bar"]), geom["theta_p"], "x", "y", axis_scale
        )

    xr, yr = _view_range(xmin, xmax, ymin, ymax)

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=xr, title="x [mm]", zeroline=False),
        yaxis=dict(range=yr, title="y [mm]", zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=30, t=30, b=50), height=520,
    )
    return fig


def build_stress_figure(rectangles, geom, Mx, My, N):
    sigma_max, loc_max, sigma_min, loc_min = compute_stress_extremes(rectangles, Mx, My, N, geom)

    xs = [r[0] for r in rectangles]
    ys = [r[1] for r in rectangles]
    x1s = [r[0] + r[2] for r in rectangles]
    y1s = [r[1] + r[3] for r in rectangles]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)

    nx, ny = 220, 220
    X = np.linspace(xmin, xmax, nx)
    Y = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(X, Y)

    inside = np.zeros_like(XX, dtype=bool)
    for x0, y0, b, h in rectangles:
        inside |= (XX >= x0) & (XX <= x0 + b) & (YY >= y0) & (YY <= y0 + h)

    sigma_field = sigma_total_at(XX, YY, Mx, My, N, geom)
    sigma_masked = np.where(inside, sigma_field, np.nan)

    vmax = max(abs(sigma_max), abs(sigma_min), 1e-9)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=X, y=Y, z=sigma_masked,
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="\u03c3 [MPa]"),
        hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>\u03c3=%{z:.2f} MPa<extra></extra>",
    ))

    shapes = _rect_shapes(rectangles, fillcolor="rgba(0,0,0,0)", opacity=1.0)
    annotations = []

    Ix, Iy, Ixy = geom["Ix"], geom["Iy"], geom["Ixy"]
    Delta = Ix * Iy - Ixy**2
    k2 = -(Ix * My + Ixy * Mx) / Delta
    k3 = (Iy * Mx + Ixy * My) / Delta
    has_bending = (abs(k2) > 1e-12) or (abs(k3) > 1e-12)

    # ---------------- Navier diagram ribbon (same plot, offset outward) ----------------
    ribbon = _sigma_diagram_ribbon(rectangles, geom, Mx, My, N, sigma_min, sigma_max) if has_bending else None

    # expand the view to include the ribbon, so it doesn't get clipped
    view_xmin, view_xmax, view_ymin, view_ymax = xmin, xmax, ymin, ymax
    if ribbon is not None:
        all_x = np.concatenate([ribbon["outer"][:, 0], ribbon["base"][:, 0]])
        all_y = np.concatenate([ribbon["outer"][:, 1], ribbon["base"][:, 1]])
        view_xmin = min(view_xmin, all_x.min())
        view_xmax = max(view_xmax, all_x.max())
        view_ymin = min(view_ymin, all_y.min())
        view_ymax = max(view_ymax, all_y.max())

    xr, yr = _view_range(view_xmin, view_xmax, view_ymin, view_ymax)

    if has_bending:
        a, b_coef = k2, k3
        c = -N / geom["A_tot"] + k2 * geom["x_bar"] + k3 * geom["y_bar"]

        if abs(b_coef) > abs(a):
            x_p = geom["x_bar"]
            y_p = (c - a * x_p) / b_coef
        else:
            y_p = geom["y_bar"]
            x_p = (c - b_coef * y_p) / a

        direction_na = (-b_coef, a)
        pts = _infinite_line_points((x_p, y_p), direction_na, xr, yr)
        if pts:
            fig.add_trace(go.Scatter(
                x=pts[0], y=pts[1], mode="lines",
                line=dict(color="green", dash="dash", width=2),
                name="Neutral axis", hoverinfo="skip",
            ))

        phi = np.arctan2(My, Mx)
        pts = _infinite_line_points(
            (geom["x_bar"], geom["y_bar"]), (np.cos(phi), np.sin(phi)), xr, yr
        )
        if pts:
            fig.add_trace(go.Scatter(
                x=pts[0], y=pts[1], mode="lines",
                line=dict(color="darkorange", dash="dashdot", width=2),
                name="Bending moment direction", hoverinfo="skip",
            ))

        L_ref = 0.18 * max(xmax - xmin, ymax - ymin, 1.0)
        annotations += _moment_arrow(fig, (geom["x_bar"], geom["y_bar"]), phi, L_ref, color="purple")

        if ribbon is not None:
            annotations += _add_sigma_ribbon_trace(fig, ribbon)

    fig.add_trace(go.Scatter(
        x=[geom["x_bar"]], y=[geom["y_bar"]], mode="markers",
        marker=dict(color="black", size=7), showlegend=False,
        hovertemplate="centroid<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[loc_max[0]], y=[loc_max[1]], mode="markers",
        marker=dict(symbol="triangle-up", color="white", size=11, line=dict(color="black", width=1.5)),
        showlegend=False,
        hovertemplate=f"max = {sigma_max:.1f} MPa<extra></extra>",
    ))
    annotations.append(dict(
        x=loc_max[0], y=loc_max[1], text=f"max = {sigma_max:.1f} MPa",
        showarrow=False, font=dict(size=11, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1,
        xanchor="left", yanchor="bottom", xshift=10, yshift=10,
    ))

    fig.add_trace(go.Scatter(
        x=[loc_min[0]], y=[loc_min[1]], mode="markers",
        marker=dict(symbol="triangle-down", color="white", size=11, line=dict(color="black", width=1.5)),
        showlegend=False,
        hovertemplate=f"min = {sigma_min:.1f} MPa<extra></extra>",
    ))
    annotations.append(dict(
        x=loc_min[0], y=loc_min[1], text=f"min = {sigma_min:.1f} MPa",
        showarrow=False, font=dict(size=11, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1,
        xanchor="left", yanchor="top", xshift=10, yshift=-10,
    ))

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=xr, title="x [mm]", zeroline=False),
        yaxis=dict(range=yr, title="y [mm]", zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=30, t=30, b=50), height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig, sigma_max, loc_max, sigma_min, loc_min


# =====================================================================
# PER-RECTANGLE (thin-wall / built-up section) shear analysis
#
# Standard result (see e.g. Hibbeler, Mechanics of Materials): along a
# segment PERPENDICULAR to V, shear flow varies LINEARLY; along a
# segment PARALLEL to V, it varies PARABOLICALLY. The divisor (wall
# thickness) is always that piece's OWN short side, regardless of
# which force (Vx or Vy) is being analyzed -- what changes is which
# moment/inertia (Sx/Ix for Vy, Sy/Iy for Vx) is used, and that alone
# determines whether the local profile comes out linear or quadratic.
# =====================================================================

def _long_axis(rect):
    """'x' if the rectangle is wider than tall (flange-like), else 'y'."""
    x0, y0, b, h = rect
    return "x" if b >= h else "y"


def _find_attachments(rect, other_rects, axis):
    """
    For 'rect' whose long axis is 'axis' ('x' or 'y'), find where OTHER
    rectangles with the OPPOSITE long axis touch it (share an edge along
    its span). Returns a sorted list of attachment center positions
    along the span.
    """
    x0, y0, b, h = rect
    attachments = []
    for orect in other_rects:
        if orect == rect:
            continue
        ox0, oy0, ob, oh = orect
        if axis == "x":
            if _long_axis(orect) == "x":
                continue  # only perpendicular neighbours attach meaningfully
            touches = abs(oy0 - (y0 + h)) < 1e-6 or abs((oy0 + oh) - y0) < 1e-6
            if touches:
                xa, xb = max(x0, ox0), min(x0 + b, ox0 + ob)
                if xb > xa:
                    attachments.append((xa + xb) / 2)
        else:
            if _long_axis(orect) == "y":
                continue
            touches = abs(ox0 - (x0 + b)) < 1e-6 or abs((ox0 + ob) - x0) < 1e-6
            if touches:
                ya, yb = max(y0, oy0), min(y0 + h, oy0 + oh)
                if yb > ya:
                    attachments.append((ya + yb) / 2)
    return sorted(set(attachments))


def _tent_profile(s0, s1, thickness, arm_const, attach_positions, n=40):
    """
    Local first-moment profile Q(s) for a piece PERPENDICULAR to V,
    swept along its own span [s0, s1]: zero at the free tips (and at
    the midpoints between consecutive attachments, if more than one),
    rising linearly toward each attachment -- the classic 'tent' shape.
    Returns (s_samples, Q_samples) with Q in mm^3 (still needs *V/I).
    """
    if attach_positions:
        mids = [(a + b) / 2 for a, b in zip(attach_positions[:-1], attach_positions[1:])]
        zero_points = [s0] + mids + [s1]
    else:
        zero_points = [s0, s1]

    s_samples = np.linspace(s0, s1, n)
    zp = np.array(sorted(set(zero_points)))
    Q_samples = np.zeros_like(s_samples)
    for i, s in enumerate(s_samples):
        idx = np.clip(np.searchsorted(zp, s), 1, len(zp) - 1)
        left, right = zp[idx - 1], zp[idx]
        d = min(s - left, right - s)
        Q_samples[i] = thickness * d * arm_const
    return s_samples, Q_samples


def _perpendicular_piece_profile(rect, rectangles, geom, direction):
    """
    Local linear tau(s) profile for a piece PERPENDICULAR to the given
    shear direction ('Vy' or 'Vx') -- i.e. a 'flange' for that
    direction. Returns (coord_samples, tau_samples, thickness, is_x_sweep).
    """
    x0, y0, b, h = rect
    if direction == "Vy":
        # perpendicular to Vy means long axis = x
        s0, s1 = x0, x0 + b
        thickness = h
        arm_const = (y0 + h / 2) - geom["y_bar"]
        attachments = _find_attachments(rect, rectangles, axis="x")
        s_samples, Q = _tent_profile(s0, s1, thickness, arm_const, attachments)
        return s_samples, Q, thickness, True
    else:
        # perpendicular to Vx means long axis = y
        s0, s1 = y0, y0 + h
        thickness = b
        arm_const = (x0 + b / 2) - geom["x_bar"]
        attachments = _find_attachments(rect, rectangles, axis="y")
        s_samples, Q = _tent_profile(s0, s1, thickness, arm_const, attachments)
        return s_samples, Q, thickness, False


def _choose_offset_sign(rect, rectangles, axis):
    """Pick whichever side has more real clearance -- checked against the
    actual neighbouring rectangles, not just the bounding-box edge (being
    at the bounding-box edge means that side is wide open, not blocked)."""
    x0, y0, b, h = rect
    others = [r for r in rectangles if r != rect]

    if axis == "y":
        blockers_up = [oy0 + oh for ox0, oy0, ob, oh in others
                       if ox0 < x0 + b and ox0 + ob > x0 and oy0 + oh <= y0 + 1e-9]
        blockers_down = [oy0 for ox0, oy0, ob, oh in others
                         if ox0 < x0 + b and ox0 + ob > x0 and oy0 >= y0 + h - 1e-9]
        clearance_up = (y0 - max(blockers_up)) if blockers_up else float("inf")
        clearance_down = (min(blockers_down) - (y0 + h)) if blockers_down else float("inf")
        return -1 if clearance_up >= clearance_down else 1
    else:
        blockers_left = [ox0 + ob for ox0, oy0, ob, oh in others
                         if oy0 < y0 + h and oy0 + oh > y0 and ox0 + ob <= x0 + 1e-9]
        blockers_right = [ox0 for ox0, oy0, ob, oh in others
                          if oy0 < y0 + h and oy0 + oh > y0 and ox0 >= x0 + b - 1e-9]
        clearance_left = (x0 - max(blockers_left)) if blockers_left else float("inf")
        clearance_right = (min(blockers_right) - (x0 + b)) if blockers_right else float("inf")
        return -1 if clearance_left >= clearance_right else 1


def _shear_ribbons_for_direction(rectangles, geom, V, direction, section_bbox, color, line_color):
    """
    Build one ribbon polygon PER RECTANGLE for the given shear force
    direction ('Vy' or 'Vx'): a local linear tent for pieces
    perpendicular to V, or the (already validated) global Jourawski
    profile for pieces parallel to V. Each ribbon is attached directly
    beside its own rectangle, offset to whichever side has more room.

    Returns (traces, annotations, tau_max, loc_max, view_extra) where
    view_extra is (xmin,xmax,ymin,ymax) to fold into the view range.
    """
    sxmin, sxmax, symin, symax = section_bbox
    span_ref = max(sxmax - sxmin, symax - symin, 1.0)
    gap = 0.05 * span_ref

    traces = []
    tau_max = 0.0
    loc_max = (geom["x_bar"], geom["y_bar"])
    view_xmin, view_xmax, view_ymin, view_ymax = sxmin, sxmax, symin, symax

    # normalize scale across ALL pieces so relative magnitudes stay meaningful
    all_abs = []
    per_piece = []
    for rect in rectangles:
        x0, y0, b, h = rect
        perpendicular = (_long_axis(rect) == "x") if direction == "Vy" else (_long_axis(rect) == "y")
        if perpendicular:
            s, Q, thickness, is_x_sweep = _perpendicular_piece_profile(rect, rectangles, geom, direction)
            if direction == "Vy":
                tau = V * Q / (geom["Ix"] * thickness)
            else:
                tau = V * Q / (geom["Iy"] * thickness)
        else:
            if direction == "Vy":
                s = np.linspace(y0, y0 + h, 40)
                tau = np.array([tau_from_Vy(rectangles, yy, V, geom) for yy in s])
            else:
                s = np.linspace(x0, x0 + b, 40)
                tau = np.array([tau_from_Vx(rectangles, xx, V, geom) for xx in s])
            is_x_sweep = (direction == "Vx")
        per_piece.append((rect, s, tau, is_x_sweep))
        all_abs.append(np.max(np.abs(tau)) if len(tau) else 0.0)

    vmax = max(max(all_abs) if all_abs else 0.0, 1e-9)
    scale = 0.22 * span_ref / vmax

    per_piece_annotations = []

    for rect, s, tau, is_x_sweep in per_piece:
        x0, y0, b, h = rect
        tau_abs = np.abs(tau)
        piece_max = float(tau_abs.max()) if tau_abs.size else 0.0
        i_peak = int(np.argmax(tau_abs)) if tau_abs.size else 0

        if piece_max > tau_max:
            tau_max = piece_max
            if is_x_sweep:
                loc_max = (float(s[i_peak]), y0 + h / 2)
            else:
                loc_max = (x0 + b / 2, float(s[i_peak]))

        if is_x_sweep:
            # ribbon runs along x (s = x), bulges in y
            sign = _choose_offset_sign(rect, rectangles, axis="y")
            base_y = (y0 - gap) if sign < 0 else (y0 + h + gap)
            outer_y = base_y + sign * scale * tau_abs
            poly_x = list(s) + list(s[::-1])
            poly_y = list(outer_y) + [base_y] * len(s)
            view_ymin = min(view_ymin, outer_y.min())
            view_ymax = max(view_ymax, outer_y.max())
            peak_x, peak_y = float(s[i_peak]), float(outer_y[i_peak])
            hover_x, hover_y = s, outer_y
        else:
            # ribbon runs along y (s = y), bulges in x
            sign = _choose_offset_sign(rect, rectangles, axis="x")
            base_x = (x0 - gap) if sign < 0 else (x0 + b + gap)
            outer_x = base_x + sign * scale * tau_abs
            poly_x = list(outer_x) + [base_x] * len(s)
            poly_y = list(s) + list(s[::-1])
            view_xmin = min(view_xmin, outer_x.min())
            view_xmax = max(view_xmax, outer_x.max())
            peak_x, peak_y = float(outer_x[i_peak]), float(s[i_peak])
            hover_x, hover_y = outer_x, s

        traces.append(go.Scatter(
            x=poly_x, y=poly_y, mode="lines", fill="toself",
            line=dict(color=line_color, width=1.5), fillcolor=color,
            hoverinfo="skip", showlegend=False,
        ))

        # thin invisible-ish line along the outer edge, just to enable
        # hovering and read tau at any point along this piece
        traces.append(go.Scatter(
            x=list(hover_x), y=list(hover_y), mode="lines",
            line=dict(color=line_color, width=0.5),
            customdata=tau, hovertemplate=f"{direction}<br>\u03c4=%{{customdata:.2f}} MPa<extra></extra>",
            showlegend=False,
        ))

        if piece_max > 1e-6:
            per_piece_annotations.append(dict(
                x=peak_x, y=peak_y, text=f"{piece_max:.1f}",
                showarrow=False, font=dict(size=9, color=line_color),
                xanchor="left" if is_x_sweep is False else "center",
                yanchor="middle" if is_x_sweep is False else ("top" if sign < 0 else "bottom"),
                xshift=4 if is_x_sweep is False else 0,
            ))

    global_annotation = [dict(
        x=loc_max[0], y=loc_max[1], text=f"{direction}: max \u2248 {tau_max:.1f} MPa",
        showarrow=False, font=dict(size=10, color="black"),
        bgcolor="white", bordercolor=line_color, borderwidth=1,
        xanchor="left", yanchor="bottom", xshift=8, yshift=8,
    )] if tau_max > 1e-9 else []

    annotations = per_piece_annotations + global_annotation

    return traces, annotations, tau_max, loc_max, (view_xmin, view_xmax, view_ymin, view_ymax)


def build_shear_figure(rectangles, geom, Vx, Vy):
    """Per-rectangle Jourawski shear analysis, drawn IN THE SAME plot as
    the section: each rectangle gets its own diagram attached right
    beside it -- a linear 'tent' if that piece is perpendicular to the
    shear force being analyzed, or the standard parabolic-type profile
    if it's parallel to it. Vx, Vy already in N."""
    xs = [r[0] for r in rectangles]
    ys = [r[1] for r in rectangles]
    x1s = [r[0] + r[2] for r in rectangles]
    y1s = [r[1] + r[3] for r in rectangles]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)
    section_bbox = (xmin, xmax, ymin, ymax)

    has_vy = abs(Vy) > 1e-9
    has_vx = abs(Vx) > 1e-9

    fig = go.Figure()
    shapes = _rect_shapes(rectangles, fillcolor="lightgray", opacity=0.5)

    all_traces = []
    all_annotations = []
    tau_max_vy = tau_max_vx = 0.0
    loc_max_vy = loc_max_vx = (geom["x_bar"], geom["y_bar"])
    view_xmin, view_xmax, view_ymin, view_ymax = xmin, xmax, ymin, ymax

    if has_vy:
        traces, anns, tau_max_vy, loc_max_vy, extra = _shear_ribbons_for_direction(
            rectangles, geom, Vy, "Vy", section_bbox,
            color="rgba(0,100,0,0.30)", line_color="darkgreen",
        )
        all_traces += traces
        all_annotations += anns
        view_xmin = min(view_xmin, extra[0]); view_xmax = max(view_xmax, extra[1])
        view_ymin = min(view_ymin, extra[2]); view_ymax = max(view_ymax, extra[3])

    if has_vx:
        traces, anns, tau_max_vx, loc_max_vx, extra = _shear_ribbons_for_direction(
            rectangles, geom, Vx, "Vx", section_bbox,
            color="rgba(255,140,0,0.30)", line_color="darkorange",
        )
        all_traces += traces
        all_annotations += anns
        view_xmin = min(view_xmin, extra[0]); view_xmax = max(view_xmax, extra[1])
        view_ymin = min(view_ymin, extra[2]); view_ymax = max(view_ymax, extra[3])

    for t in all_traces:
        fig.add_trace(t)

    fig.add_trace(go.Scatter(
        x=[geom["x_bar"]], y=[geom["y_bar"]], mode="markers",
        marker=dict(color="black", size=7), showlegend=False,
        hovertemplate="centroid<extra></extra>",
    ))

    tau_max = max(tau_max_vy, tau_max_vx)
    loc_max = loc_max_vy if tau_max_vy >= tau_max_vx else loc_max_vx

    xr, yr = _view_range(view_xmin, view_xmax, view_ymin, view_ymax)

    fig.update_layout(
        shapes=shapes, annotations=all_annotations,
        xaxis=dict(range=xr, title="x [mm]", zeroline=False),
        yaxis=dict(range=yr, title="y [mm]", zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=30, t=30, b=50), height=560,
    )

    return fig, tau_max, loc_max
