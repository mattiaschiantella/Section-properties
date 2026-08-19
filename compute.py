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
    lines.append("")
    lines.append(f"Centroid coordinates")
    lines.append(f" x' = {geom['x_bar']:.4f} mm")
    lines.append(f" y' = {geom['y_bar']:.4f} mm")
    lines.append("")
    lines.append("Centroidal moments of inertia:")
    lines.append(f"Ix'x'  = {geom['Ix']:.4f} mm\u2074")
    lines.append(f"Iy'y'  = {geom['Iy']:.4f} mm\u2074")
    lines.append(f"Ix'y' = {geom['Ixy']:.4f} mm\u2074")
    lines.append("")
    lines.append("Principal axes of inertia (x, y):")
    lines.append(f"θx' (counter-clockwise if positive) = {np.degrees(geom['theta_p']):.4f} deg")
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


def _excluded_intervals(rect, rectangles):
    """
    List of (a, b) sub-intervals along this piece's OWN long-axis span
    where a PERPENDICULAR neighbour's own tip touches it -- i.e. the
    neighbour terminates INTO this piece's middle (e.g. a web ending at
    a flange). That sub-range (as wide as the neighbour's own
    thickness) is the small corner shared by both pieces: it belongs to
    the neighbour's zone, not to this piece's own independent length,
    so it's excluded here (from J, from the local Jourawski tent, and
    from the visualization) to avoid counting it twice.
    """
    x0, y0, b, h = rect
    own_axis = _long_axis(rect)
    intervals = []
    for orect in rectangles:
        if orect == rect:
            continue
        ox0, oy0, ob, oh = orect
        if _long_axis(orect) == own_axis:
            continue  # only a perpendicular neighbour forms a corner
        if own_axis == "x":
            touches = abs(oy0 + oh - y0) < 1e-6 or abs(oy0 - (y0 + h)) < 1e-6
            if touches:
                a, bnd = max(x0, ox0), min(x0 + b, ox0 + ob)
                if bnd > a:
                    intervals.append((a, bnd))
        else:
            touches = abs(ox0 + ob - x0) < 1e-6 or abs(ox0 - (x0 + b)) < 1e-6
            if touches:
                a, bnd = max(y0, oy0), min(y0 + h, oy0 + oh)
                if bnd > a:
                    intervals.append((a, bnd))

    intervals.sort()
    merged = []
    for a, bnd in intervals:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], bnd))
        else:
            merged.append((a, bnd))
    return merged


def _free_segments(s0, s1, excluded):
    """[s0, s1] minus the excluded sub-intervals, as a list of
    (seg_start, seg_end, is_true_tip_start, is_true_tip_end)."""
    segs = []
    cursor = s0
    bounds = sorted(excluded)
    for a, b in bounds:
        if a > cursor:
            segs.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < s1:
        segs.append((cursor, s1))
    return [(a, b, abs(a - s0) < 1e-6, abs(b - s1) < 1e-6) for a, b in segs]


def _tent_profile(s0, s1, thickness, arm_const, excluded, n=40):
    """
    Local first-moment profile Q(s) for a piece PERPENDICULAR to V,
    swept along its own span [s0, s1] MINUS the excluded sub-intervals
    (skipped entirely -- no samples there, since that sub-range belongs
    to the attached perpendicular piece, not to this one).

    Each free segment ramps from 0 at a TRUE free tip (s0 or s1) up to
    a maximum at the excluded interval's edge; a segment with no true
    tip on either side (sandwiched between two excluded intervals)
    ramps from 0 at its own midpoint outward to each edge.

    Returns (s_samples, Q_samples), Q in mm^3 (still needs *V/I).
    """
    segments = _free_segments(s0, s1, excluded)
    all_s, all_Q = [], []
    for seg_start, seg_end, is_tip_start, is_tip_end in segments:
        if seg_end - seg_start < 1e-9:
            continue
        n_seg = max(int(round(n * (seg_end - seg_start) / max(s1 - s0, 1e-9))), 4)
        seg_s = np.linspace(seg_start, seg_end, n_seg)
        if is_tip_start and not is_tip_end:
            d = seg_s - seg_start
        elif is_tip_end and not is_tip_start:
            d = seg_end - seg_s
        elif is_tip_start and is_tip_end:
            d = np.minimum(seg_s - seg_start, seg_end - seg_s)
        else:
            mid = (seg_start + seg_end) / 2
            d = np.abs(seg_s - mid)
        all_s.append(seg_s)
        all_Q.append(thickness * d * arm_const)

    if not all_s:
        return np.array([]), np.array([])
    return np.concatenate(all_s), np.concatenate(all_Q)


def _attachment_zero_points(s0, s1, excluded):
    """Boundaries (true tips + excluded-interval edges + sandwiched
    midpoints) used to tell, for a given sample, which free segment and
    which 'branch' (toward which edge) it belongs to -- for the
    flow-direction logic."""
    zero_points = []
    for seg_start, seg_end, is_tip_start, is_tip_end in _free_segments(s0, s1, excluded):
        if is_tip_start:
            zero_points.append(seg_start)
        if is_tip_end:
            zero_points.append(seg_end)
        if not is_tip_start and not is_tip_end:
            zero_points.append((seg_start + seg_end) / 2)
    if not zero_points:
        zero_points = [s0, s1]
    return np.array(sorted(set(zero_points)))


def _branch_sign(s, zp):
    """+1 if s is on the 'ascending' (left) side of its local zero-point
    bracket -- i.e. increasing s moves AWAY from the nearest zero point
    -- else -1 (s is moving TOWARD the next zero point as it increases)."""
    idx = np.clip(np.searchsorted(zp, s), 1, len(zp) - 1)
    left, right = zp[idx - 1], zp[idx]
    return 1.0 if (s - left) <= (right - s) else -1.0


def _perpendicular_piece_profile(rect, rectangles, geom, direction):
    """
    Local linear tau(s) profile for a piece PERPENDICULAR to the given
    shear direction ('Vy' or 'Vx') -- i.e. a 'flange' for that
    direction. Returns (coord_samples, tau_samples, thickness, is_x_sweep).
    """
    x0, y0, b, h = rect
    excluded = _excluded_intervals(rect, rectangles)
    if direction == "Vy":
        # perpendicular to Vy means long axis = x
        s0, s1 = x0, x0 + b
        thickness = h
        arm_const = (y0 + h / 2) - geom["y_bar"]
        s_samples, Q = _tent_profile(s0, s1, thickness, arm_const, excluded)
        return s_samples, Q, thickness, True
    else:
        # perpendicular to Vx means long axis = y
        s0, s1 = y0, y0 + h
        thickness = b
        arm_const = (x0 + b / 2) - geom["x_bar"]
        s_samples, Q = _tent_profile(s0, s1, thickness, arm_const, excluded)
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


def _split_at_gaps(s):
    """Indices where consecutive samples jump by much more than the
    local step -- i.e. an excluded interval was skipped. Returns a list
    of (start_idx, end_idx) index ranges, each a contiguous run."""
    if len(s) < 2:
        return [(0, len(s))]
    diffs = np.diff(s)
    step = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    breaks = np.where(diffs > 3 * step)[0]
    bounds = [0] + list(breaks + 1) + [len(s)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


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
        per_piece.append((rect, s, tau, is_x_sweep, perpendicular))
        all_abs.append(np.max(np.abs(tau)) if len(tau) else 0.0)

    vmax = max(max(all_abs) if all_abs else 0.0, 1e-9)
    scale = 0.22 * span_ref / vmax

    per_piece_annotations = []
    flow_annotations = []

    for rect, s, tau, is_x_sweep, perpendicular in per_piece:
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
            view_ymin = min(view_ymin, outer_y.min())
            view_ymax = max(view_ymax, outer_y.max())
            peak_x, peak_y = float(s[i_peak]), float(outer_y[i_peak])
            hover_x, hover_y = s, outer_y
            mid_y = y0 + h / 2
            segments = [(list(s[i0:i1]) + list(s[i0:i1][::-1]),
                         list(outer_y[i0:i1]) + [base_y] * (i1 - i0))
                        for i0, i1 in _split_at_gaps(s)]
            zp = _attachment_zero_points(s[0], s[-1], _excluded_intervals(rect, rectangles)) if perpendicular else None
            flow_points = []
            for i in np.linspace(2, len(s) - 3, 4).astype(int):
                if abs(tau[i]) < 1e-9:
                    continue
                if perpendicular:
                    dx = _branch_sign(s[i], zp) * (1.0 if tau[i] < 0 else -1.0)
                else:
                    dx = -1.0 if tau[i] >= 0 else 1.0
                flow_points.append((float(s[i]), mid_y, (dx, 0.0)))
        else:
            # ribbon runs along y (s = y), bulges in x
            sign = _choose_offset_sign(rect, rectangles, axis="x")
            base_x = (x0 - gap) if sign < 0 else (x0 + b + gap)
            outer_x = base_x + sign * scale * tau_abs
            view_xmin = min(view_xmin, outer_x.min())
            view_xmax = max(view_xmax, outer_x.max())
            peak_x, peak_y = float(outer_x[i_peak]), float(s[i_peak])
            hover_x, hover_y = outer_x, s
            mid_x = x0 + b / 2
            segments = [(list(outer_x[i0:i1]) + [base_x] * (i1 - i0),
                         list(s[i0:i1]) + list(s[i0:i1][::-1]))
                        for i0, i1 in _split_at_gaps(s)]
            zp = _attachment_zero_points(s[0], s[-1], _excluded_intervals(rect, rectangles)) if perpendicular else None
            flow_points = []
            for i in np.linspace(2, len(s) - 3, 4).astype(int):
                if abs(tau[i]) < 1e-9:
                    continue
                if perpendicular:
                    dy = _branch_sign(s[i], zp) * (1.0 if tau[i] < 0 else -1.0)
                else:
                    dy = -1.0 if tau[i] >= 0 else 1.0
                flow_points.append((mid_x, float(s[i]), (0.0, dy)))

        arrow_len = 0.10 * span_ref
        for px, py, (dx, dy) in flow_points:
            flow_annotations.append(dict(
                x=px + arrow_len / 2 * dx, y=py + arrow_len / 2 * dy,
                ax=px - arrow_len / 2 * dx, ay=py - arrow_len / 2 * dy,
                axref="x", ayref="y", xref="x", yref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.8,
                arrowcolor=line_color, text="",
            ))

        for poly_x, poly_y in segments:
            traces.append(go.Scatter(
                x=poly_x, y=poly_y, mode="lines", fill="toself",
                line=dict(color=line_color, width=1.5), fillcolor=color,
                hoverinfo="skip", showlegend=False,
            ))

        # thin invisible-ish line along the outer edge, just to enable
        # hovering and read tau at any point along this piece (split at
        # gaps too, so it doesn't bridge across an excluded interval)
        for i0, i1 in _split_at_gaps(s):
            traces.append(go.Scatter(
                x=list(hover_x[i0:i1]), y=list(hover_y[i0:i1]), mode="lines",
                line=dict(color=line_color, width=0.5),
                customdata=tau[i0:i1], hovertemplate=f"{direction}<br>\u03c4=%{{customdata:.2f}} MPa<extra></extra>",
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

    annotations = flow_annotations + per_piece_annotations + global_annotation

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


# =====================================================================
# TORSION ON OPEN THIN-WALLED SECTIONS (Saint-Venant, membrane analogy)
#
# For a section built from thin rectangular segments (length L_i,
# thickness t_i << L_i):
#   J = (1/3) * sum(L_i * t_i^3)
#   tau(n) = 2*T*n / J    for n in [-t_i/2, +t_i/2]  (n measured from the
#                          segment's own mid-thickness line)
#   tau_max,i = T*t_i / J  (at each segment's own surface)
#
# Shear stress is CONSTANT along a segment's length and varies LINEARLY
# across its own thickness, zero at the mid-line. Each segment is
# treated independently (no continuity/connectivity needed, unlike the
# flexural Jourawski case) since Saint-Venant torsion of an open
# section has no circulating shear flow between segments.
#
# Sign convention: positive T = counterclockwise, using a standard
# right-handed frame with z pointing out of the page (toward the
# viewer). For each rectangle, n is measured along z_hat x s_hat, where
# s_hat is the segment's own long-axis direction (+x for horizontal
# pieces, +y for vertical pieces).
# =====================================================================

def _torsion_effective_span(rect, rectangles):
    """Returns (s0, s1, thickness, excluded): this rectangle's own full
    span along its long axis, its thickness, and the list of
    sub-intervals to exclude (shared corners with attached perpendicular
    pieces -- see _excluded_intervals)."""
    x0, y0, b, h = rect
    own_axis = _long_axis(rect)
    s0, s1 = (x0, x0 + b) if own_axis == "x" else (y0, y0 + h)
    thickness = min(b, h)
    excluded = _excluded_intervals(rect, rectangles)
    return s0, s1, thickness, excluded


def compute_torsion_constant(rectangles):
    """J = sum(L_i * t_i^3) / 3, using each rectangle's own long side as
    length (net of any shared corner with an attached perpendicular
    piece) and short side as thickness."""
    J = 0.0
    for rect in rectangles:
        s0, s1, thickness, excluded = _torsion_effective_span(rect, rectangles)
        length = (s1 - s0) - sum(b - a for a, b in excluded)
        J += max(length, 0.0) * thickness**3 / 3.0
    return J


def _torsion_n_field(rectangles, XX, YY):
    """For every grid point, the signed distance n from its own
    rectangle's mid-thickness line (NaN outside all rectangles, and
    NaN in the small corner shared with a perpendicular attachment)."""
    n_field = np.full(XX.shape, np.nan)
    for rect in rectangles:
        x0, y0, b, h = rect
        s0, s1, _, excluded = _torsion_effective_span(rect, rectangles)
        if b >= h:
            mask = (XX >= x0) & (XX <= x0 + b) & (YY >= y0) & (YY <= y0 + h)
            for a, bnd in excluded:
                mask &= ~((XX >= a) & (XX <= bnd))
            # x-oriented: s_hat = +x, n_hat = s_hat x z = -y (positive T = counterclockwise)
            y_mid = y0 + h / 2
            n_field[mask] = -(YY[mask] - y_mid)
        else:
            mask = (XX >= x0) & (XX <= x0 + b) & (YY >= y0) & (YY <= y0 + h)
            for a, bnd in excluded:
                mask &= ~((YY >= a) & (YY <= bnd))
            # y-oriented: s_hat = +y, n_hat = s_hat x z = +x
            x_mid = x0 + b / 2
            n_field[mask] = (XX[mask] - x_mid)
    return n_field


def build_torsion_figure(rectangles, geom, T):
    """T already in N*mm. Returns fig, tau_max, loc_max, J."""
    J = compute_torsion_constant(rectangles)

    xs = [r[0] for r in rectangles]
    ys = [r[1] for r in rectangles]
    x1s = [r[0] + r[2] for r in rectangles]
    y1s = [r[1] + r[3] for r in rectangles]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)

    nx, ny = 260, 260
    X = np.linspace(xmin, xmax, nx)
    Y = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(X, Y)

    n_field = _torsion_n_field(rectangles, XX, YY)
    tau_field = 2.0 * T * n_field / J if J > 0 else np.zeros_like(n_field)

    # per-segment exact max (occurs at each segment's own surface, n=+-t/2)
    tau_max = 0.0
    loc_max = (geom["x_bar"], geom["y_bar"])
    per_seg_annotations = []
    flow_annotations = []
    for x0, y0, b, h in rectangles:
        thickness = min(b, h)
        seg_tau_max = abs(T) * thickness / J if J > 0 else 0.0
        if seg_tau_max > tau_max:
            tau_max = seg_tau_max
            loc_max = (x0 + b / 2, y0 + h / 2)
        if seg_tau_max > 1e-6:
            per_seg_annotations.append(dict(
                x=x0 + b / 2, y=y0 + h / 2, text=f"{seg_tau_max:.1f}",
                showarrow=False, font=dict(size=9, color="black"),
            ))

    vmax = max(tau_max, 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=X, y=Y, z=tau_field,
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="\u03c4_t [MPa]"),
        hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>\u03c4=%{z:.2f} MPa<extra></extra>",
    ))

    shapes = _rect_shapes(rectangles, fillcolor="rgba(0,0,0,0)", opacity=1.0)

    # ---------------- linear ribbon per rectangle, same style as the sigma diagram ----------------
    span_ref = max(xmax - xmin, ymax - ymin, 1.0)
    gap = 0.05 * span_ref
    scale = 0.20 * span_ref / vmax
    clearance = gap + scale * vmax
    view_xmin, view_xmax, view_ymin, view_ymax = xmin, xmax, ymin, ymax

    for x0, y0, b, h in rectangles:
        rect = (x0, y0, b, h)
        if b >= h:
            s = np.linspace(y0, y0 + h, 20)
            n = -(s - (y0 + h / 2))
            tau_s = 2.0 * T * n / J if J > 0 else np.zeros_like(n)
            sign = _choose_offset_sign(rect, rectangles, axis="x")
            base = (x0 - clearance) if sign < 0 else (x0 + b + clearance)
            outer = base + sign * scale * tau_s
            poly_x, poly_y = list(outer) + [base] * len(s), list(s) + list(s[::-1])
            hover_x, hover_y = outer, s
            view_xmin, view_xmax = min(view_xmin, outer.min()), max(view_xmax, outer.max())
        else:
            s = np.linspace(x0, x0 + b, 20)
            n = s - (x0 + b / 2)
            tau_s = 2.0 * T * n / J if J > 0 else np.zeros_like(n)
            sign = _choose_offset_sign(rect, rectangles, axis="y")
            base = (y0 - clearance) if sign < 0 else (y0 + h + clearance)
            outer = base + sign * scale * tau_s
            poly_x, poly_y = list(s) + list(s[::-1]), list(outer) + [base] * len(s)
            hover_x, hover_y = s, outer
            view_ymin, view_ymax = min(view_ymin, outer.min()), max(view_ymax, outer.max())

        arrow_len = 0.08 * span_ref
        Tsign = 1.0 if T >= 0 else -1.0
        s0, s1, _, excluded = _torsion_effective_span(rect, rectangles)
        if abs(T) > 1e-9 and J > 0:
            free_segs = [(a, b) for a, b, *_ in _free_segments(s0, s1, excluded) if b - a > 1e-6]
            if b >= h:
                for seg_a, seg_b in free_segs:
                    xs_edge = np.linspace(seg_a + 0.1 * (seg_b - seg_a), seg_b - 0.1 * (seg_b - seg_a), 3)
                    for xa in xs_edge:
                        flow_annotations.append(dict(  # top edge (y0): tau sign = +Tsign
                            x=xa + arrow_len / 2 * Tsign, y=y0, ax=xa - arrow_len / 2 * Tsign, ay=y0,
                            axref="x", ayref="y", xref="x", yref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor="black", text="",
                        ))
                        flow_annotations.append(dict(  # bottom edge (y0+h): tau sign = -Tsign
                            x=xa - arrow_len / 2 * Tsign, y=y0 + h, ax=xa + arrow_len / 2 * Tsign, ay=y0 + h,
                            axref="x", ayref="y", xref="x", yref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor="black", text="",
                        ))
            else:
                for seg_a, seg_b in free_segs:
                    ys_edge = np.linspace(seg_a + 0.1 * (seg_b - seg_a), seg_b - 0.1 * (seg_b - seg_a), 3)
                    for ya in ys_edge:
                        flow_annotations.append(dict(  # right edge (x0+b): tau sign = +Tsign
                            x=x0 + b, y=ya + arrow_len / 2 * Tsign, ax=x0 + b, ay=ya - arrow_len / 2 * Tsign,
                            axref="x", ayref="y", xref="x", yref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor="black", text="",
                        ))
                        flow_annotations.append(dict(  # left edge (x0): tau sign = -Tsign
                            x=x0, y=ya - arrow_len / 2 * Tsign, ax=x0, ay=ya + arrow_len / 2 * Tsign,
                            axref="x", ayref="y", xref="x", yref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor="black", text="",
                        ))

        mid = len(s) // 2
        for i0, i1, col, lcol in [(0, mid + 1, "rgba(30,60,200,0.28)", "rgba(20,40,150,0.8)"),
                                   (mid, len(s), "rgba(200,30,30,0.28)", "rgba(150,20,20,0.8)")]:
            if b >= h:
                px = list(outer[i0:i1]) + [base] * (i1 - i0)
                py = list(s[i0:i1]) + list(s[i0:i1][::-1])
            else:
                px = list(s[i0:i1]) + list(s[i0:i1][::-1])
                py = list(outer[i0:i1]) + [base] * (i1 - i0)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", fill="toself",
                line=dict(color=lcol, width=1.2), fillcolor=col,
                hoverinfo="skip", showlegend=False,
            ))

        fig.add_trace(go.Scatter(
            x=list(hover_x), y=list(hover_y), mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", width=0.5),
            customdata=tau_s, hovertemplate="\u03c4=%{customdata:.2f} MPa<extra></extra>",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[geom["x_bar"]], y=[geom["y_bar"]], mode="markers",
        marker=dict(color="black", size=7), showlegend=False,
        hovertemplate="centroid<extra></extra>",
    ))

    annotations = flow_annotations + per_seg_annotations + [dict(
        x=loc_max[0], y=loc_max[1], text=f"max \u2248 {tau_max:.2f} MPa",
        showarrow=False, font=dict(size=11, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1,
        xanchor="left", yanchor="bottom", xshift=8, yshift=8,
    )] if tau_max > 1e-9 else flow_annotations + per_seg_annotations

    xr, yr = _view_range(view_xmin, view_xmax, view_ymin, view_ymax)

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=xr, title="x [mm]", zeroline=False),
        yaxis=dict(range=yr, title="y [mm]", zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=30, t=30, b=50), height=560,
    )

    return fig, tau_max, loc_max, J
