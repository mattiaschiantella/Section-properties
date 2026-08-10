import numpy as np
import plotly.graph_objects as go


# =====================================================================
# GEOMETRY (identical math to the Flask/Tkinter versions)
# =====================================================================

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


def _axes_annotations(origin, angle, label_x, label_y, scale, color="black"):
    x0, y0 = origin
    dx = scale * np.cos(angle)
    dy = scale * np.sin(angle)

    def arrow(xt, yt, text):
        return dict(
            x=xt, y=yt, ax=x0, ay=y0, axref="x", ayref="y", xref="x", yref="y",
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


def _moment_arrow(fig, center, phi, length, color="purple"):
    """Moment vector: a plain shaft ending in two consecutive arrowheads
    on the SAME side (chevron), pointing toward the positive direction
    of (Mx, My) as defined by the reference system."""
    dirx, diry = np.cos(phi), np.sin(phi)
    base = (center[0] - length / 2 * dirx, center[1] - length / 2 * diry)
    tip = (center[0] + length / 2 * dirx, center[1] + length / 2 * diry)

    chevron = 0.22 * length
    p1 = (tip[0] - chevron * dirx, tip[1] - chevron * diry)
    p2 = (tip[0] - 2 * chevron * dirx, tip[1] - 2 * chevron * diry)

    fig.add_trace(go.Scatter(
        x=[base[0], p2[0]], y=[base[1], p2[1]], mode="lines",
        line=dict(color=color, width=3), hoverinfo="skip", showlegend=False,
    ))

    annotations = []
    for start, end in [(p2, p1), (p1, tip)]:
        annotations.append(dict(
            x=end[0], y=end[1], ax=start[0], ay=start[1], axref="x", ayref="y",
            xref="x", yref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2.5, arrowcolor=color, text="",
        ))
    annotations.append(dict(
        x=tip[0], y=tip[1], text="M", showarrow=False,
        font=dict(size=13, color=color, family="Arial Black"),
        xanchor="left", yanchor="middle",
    ))
    return annotations


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

    xr, yr = _view_range(xmin, xmax, ymin, ymax)

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
                name="Bending axis (moment direction)", hoverinfo="skip",
            ))

        L_ref = 0.18 * max(xmax - xmin, ymax - ymin, 1.0)
        annotations += _moment_arrow(fig, (geom["x_bar"], geom["y_bar"]), phi, L_ref, color="purple")

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
        margin=dict(l=50, r=30, t=30, b=50), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig, sigma_max, loc_max, sigma_min, loc_min


def build_shear_figure(rectangles, geom, Vx, Vy):
    """Approximate Jourawski shear stress map. Vx, Vy already in N."""
    xs = [r[0] for r in rectangles]
    ys = [r[1] for r in rectangles]
    x1s = [r[0] + r[2] for r in rectangles]
    y1s = [r[1] + r[3] for r in rectangles]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)

    nx, ny = 220, 220
    X = np.linspace(xmin, xmax, nx)
    Y = np.linspace(ymin, ymax, ny)

    # per-row / per-column profiles (each depends on a single coordinate)
    tau_y_row = np.array([tau_from_Vy(rectangles, y, Vy, geom) for y in Y])  # (ny,)
    tau_x_col = np.array([tau_from_Vx(rectangles, x, Vx, geom) for x in X])  # (nx,)

    XX, YY = np.meshgrid(X, Y)
    tau_y_field = np.tile(tau_y_row.reshape(-1, 1), (1, nx))
    tau_x_field = np.tile(tau_x_col.reshape(1, -1), (ny, 1))
    tau_field = np.sqrt(tau_y_field**2 + tau_x_field**2)

    inside = np.zeros_like(XX, dtype=bool)
    for x0, y0, b, h in rectangles:
        inside |= (XX >= x0) & (XX <= x0 + b) & (YY >= y0) & (YY <= y0 + h)

    tau_masked = np.where(inside, tau_field, np.nan)

    # approximate max: found numerically on this grid (tau is not linear,
    # so unlike sigma we can't guarantee the extremum sits at a corner)
    if np.all(np.isnan(tau_masked)):
        tau_max = 0.0
        loc_max = (geom["x_bar"], geom["y_bar"])
    else:
        idx = np.nanargmax(tau_masked)
        iy, ix = np.unravel_index(idx, tau_masked.shape)
        tau_max = float(tau_masked[iy, ix])
        loc_max = (float(X[ix]), float(Y[iy]))

    fig = go.Figure()

    vmax = max(tau_max, 1e-9)
    fig.add_trace(go.Heatmap(
        x=X, y=Y, z=tau_masked,
        colorscale="Viridis", zmin=0, zmax=vmax,
        colorbar=dict(title="\u03c4 [MPa]"),
        hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>\u03c4=%{z:.2f} MPa<extra></extra>",
    ))

    shapes = _rect_shapes(rectangles, fillcolor="rgba(0,0,0,0)", opacity=1.0)

    fig.add_trace(go.Scatter(
        x=[geom["x_bar"]], y=[geom["y_bar"]], mode="markers",
        marker=dict(color="black", size=7), showlegend=False,
        hovertemplate="centroid<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[loc_max[0]], y=[loc_max[1]], mode="markers",
        marker=dict(symbol="triangle-up", color="white", size=11, line=dict(color="black", width=1.5)),
        showlegend=False,
        hovertemplate=f"max \u2248 {tau_max:.1f} MPa<extra></extra>",
    ))
    annotations = [dict(
        x=loc_max[0], y=loc_max[1], text=f"max \u2248 {tau_max:.1f} MPa",
        showarrow=False, font=dict(size=11, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1,
        xanchor="left", yanchor="bottom", xshift=10, yshift=10,
    )]

    xr, yr = _view_range(xmin, xmax, ymin, ymax)

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=xr, title="x [mm]", zeroline=False),
        yaxis=dict(range=yr, title="y [mm]", zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=30, t=30, b=50), height=520,
    )

    return fig, tau_max, loc_max

