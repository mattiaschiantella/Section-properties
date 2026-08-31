import threading
import time
import os
import requests
import dash
from dash import dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from compute import (
    rects_overlap, compute_geometry, format_geometry_output,
    compute_stress_extremes, build_geometry_figure, build_stress_figure,
    build_shear_figure, build_torsion_figure, compute_torsion_constant,
)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Section Analyzer"
server = app.server  # needed by gunicorn/Render: exposes the underlying Flask app

# =====================================================================
# KEEP-ALIVE SELF-PING THREAD
# Prevents Render free instances from sleeping due to 15-minute inactivity.
# =====================================================================
def keep_alive():
    """Periodically pings the application endpoint to keep the Render service awake."""
    url = os.environ.get("RENDER_EXTERNAL_URL", "https://section-properties-d6ej.onrender.com/")
    
    print(f"[Keep-Alive] Thread started for URL: {url}")
    
    # Initial delay before starting the ping loop
    time.sleep(30)
    
    while True:
        try:
            response = requests.get(url, timeout=10)
            print(f"[Keep-Alive] Ping sent to {url} - Status Code: {response.status_code}")
        except Exception as e:
            print(f"[Keep-Alive] Ping failed: {e}")
        
        # Ping every 10 minutes (600 seconds)
        time.sleep(600)


# Start the background thread in daemon mode
ping_thread = threading.Thread(target=keep_alive, daemon=True)
ping_thread.start()


INPUT_STYLE = {"width": "90px", "marginRight": "8px"}
BTN_STYLE = {"marginRight": "8px", "marginTop": "6px"}
# Minor-gridline spacing (major tick / 5, see assets/grid_snap.js) for the
# app's default empty-canvas view (see build_geometry_figure's 300mm-span
# empty branch, major step 50 -> minor step 10); used to seed
# store-grid-step before the client has had a chance to report the real,
# live value (i.e. before any zoom/pan has happened yet).
DEFAULT_GRID_STEP = 10.0


def _make_banner(lines, width=156):
    """Build a comment-style ASCII banner with '#' perfectly right-aligned.

    Padding is computed here (not typed by hand) so the border can never
    get out of alignment, regardless of font/line length.
    """
    border = "#" * (width + 4)  # "# " + content + " #"
    out = [border]
    for line in lines:
        if line == "":
            body = ""
        else:
            body = line
        # content area is (width) chars wide, left-aligned, padded with spaces
        padded = body.ljust(width)
        out.append(f"# {padded} #")
    out.append(border)
    return "\n".join(out)


AUTHOR_BANNER = _make_banner([
    "This script was written by",
    "    Mattia Schiantella, PhD",
    "    Department of Civil and Environmental Engineering, University of Perugia, Italy",
    "    e-mail: mattia.schiantella@unipg.it",
    "",
    "Disclaimer:",
    "The author does not guarantee that the script is free from errors; the app is intended for educational purposes, please report any bug by sending an e-mail.",
])


# =====================================================================
# LAYOUT
# =====================================================================

app.layout = html.Div([

    # ---------------- persistent state ----------------
    dcc.Store(id="store-rects", data=[]),
    dcc.Store(id="store-pending", data=None),
    dcc.Store(id="store-geom", data=None),
    dcc.Store(id="store-stress", data=None),
    dcc.Store(id="store-output", data=[]),
    # [xdtick, ydtick] currently on screen for geom-graph, kept in sync by a
    # clientside callback so the server can snap a freshly-drawn rectangle
    # to the SAME grid the user was actually looking at (dtick changes with
    # zoom -- see assets/grid_snap.js for the live in-drag snapping, and
    # DEFAULT_GRID_STEP below for why this starts pre-seeded).
    dcc.Store(id="store-grid-step", data=[DEFAULT_GRID_STEP, DEFAULT_GRID_STEP]),
    # Set True by assets/middle_click_pan.js for the entire duration of a
    # middle-mouse-button drag (mousedown to mouseup), so the server-side
    # add_rectangle_from_drawing callback can unconditionally refuse to
    # add a rectangle while it's set -- a plain client-side revert of the
    # drawn shape isn't reliable on its own here: a relayoutData request
    # for the accidental shape can already be in flight to the server by
    # the time the revert fires, so the server needs its own gate rather
    # than trusting the client's visual state alone.
    dcc.Store(id="store-suppress-draw", data=False),

    html.H2("Section Properties and Stress Analysis"),

    # ---------------- author / credits box ----------------
    html.Pre(AUTHOR_BANNER, style={
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "padding": "10px 14px",
        "marginBottom": "16px",
        "backgroundColor": "#f8f9fa",
        "fontFamily": "monospace",
        "fontSize": "0.85em",
        "lineHeight": "1.3",
        "display": "inline-block",
        "whiteSpace": "pre",
    }),

    html.Div([

        # ---------------- LEFT: controls ----------------
        html.Div([

            html.Fieldset([
                html.Legend("Geometry input"),
                html.Div([
                    dcc.Input(id="in-x0", type="number", placeholder="x0 [mm]", style={**INPUT_STYLE, "width": "100px"}),
                    dcc.Input(id="in-y0", type="number", placeholder="y0 [mm]", style={**INPUT_STYLE, "width": "100px"}),
                    dcc.Input(id="in-b", type="number", placeholder="base [mm]", style={**INPUT_STYLE, "width": "100px"}),
                    dcc.Input(id="in-h", type="number", placeholder="height [mm]", style={**INPUT_STYLE, "width": "100px"}),
                ]),
                html.Div([
                    html.Button("Add", id="btn-add", n_clicks=0, style=BTN_STYLE),
                    html.Button("Compute geometry", id="btn-compute-geom", n_clicks=0, style=BTN_STYLE),
                ]),
                html.Div(id="form-error", style={"color": "red", "marginTop": "4px"}),
                html.P([
                    "Reference system: x positive to the LEFT, y positive DOWNWARD",
                    html.Br(),
                    "(Structural mechanics De Saint Venant's convention).",
                    html.Br(),
                    "x0 and y0 are the coordinates of the top-right vertex of the rectangle.",
                    html.Br(),
                    "x0, y0, base, height are in mm.",
                ], style={"color": "gray", "fontSize": "0.85em", "marginTop": "8px"}),
            ]),

            # Static, always-present component: avoids the classic Dash pitfall
            # of dynamically recreating "confirm/cancel" buttons that don't
            # exist in the initial layout. The browser's own Cancel button
            # never touches the server, so it can never add the rectangle.
            dcc.ConfirmDialog(
                id="confirm-overlap",
                message=("The current rectangle is overlapping the previous ones. "
                          "If you continue, the shared part will be counted twice "
                          "in the area, centroid and moment of inertia calculations. "
                          "Press OK to add it anyway, or Cancel to discard it."),
            ),

            html.Fieldset([
                html.Legend("Defined rectangles"),
                html.Div(id="rect-list-div"),
                html.Div([
                    dcc.Dropdown(id="del-dropdown", options=[], placeholder="Select to delete",
                                 style={"width": "260px", "display": "inline-block"}),
                    html.Button("Delete selected", id="btn-delete", n_clicks=0,
                                style={**BTN_STYLE, "verticalAlign": "top", "marginLeft": "8px"}),
                    html.Button("Delete all", id="btn-delete-all", n_clicks=0,
                                style={**BTN_STYLE, "verticalAlign": "top", "marginLeft": "8px"}),
                ]),
            ]),

            html.Fieldset([
                html.Legend("Internal actions"),
                html.Div([
                    dcc.Input(id="in-Mx", type="number", placeholder="Mx [kN\u00b7m]", style=INPUT_STYLE),
                    dcc.Input(id="in-My", type="number", placeholder="My [kN\u00b7m]", style=INPUT_STYLE),
                    dcc.Input(id="in-N", type="number", placeholder="N [kN]", style=INPUT_STYLE),
                ]),
                html.Div([
                    dcc.Input(id="in-Vx", type="number", placeholder="Vx [kN]", style=INPUT_STYLE),
                    dcc.Input(id="in-Vy", type="number", placeholder="Vy [kN]", style=INPUT_STYLE),
                    dcc.Input(id="in-T", type="number", placeholder="Mz [kN\u00b7m]", style=INPUT_STYLE),
                ], style={"marginTop": "6px"}),
                html.P([
                    "Shear stress is evaluated separately for Vx e Vy components.",
                    html.Br(),
                    "Torsional shear stress can be evaluated only for open cross-sections.",
                    html.Br(),
                    "The torsional moment Mz is considered positive if counterclockwise.",
                ], style={"color": "gray", "fontSize": "0.8em", "marginTop": "6px"}),
                html.Button("Compute stresses", id="btn-compute-stress", n_clicks=0, style=BTN_STYLE),
            ]),

            html.Fieldset([
                html.Legend("Results"),
                html.Pre(id="output-text", style={"background": "#f5f5f5", "padding": "10px",
                                                    "minHeight": "150px", "whiteSpace": "pre-wrap"}),
            ]),

        ], style={"flex": "1", "minWidth": "360px", "paddingRight": "20px"}),

        # ---------------- RIGHT: plots ----------------
        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Section geometry", children=[
                    html.P([
                        "Tip: use the rectangle-draw tool ▢ in the graph toolbar above "
                        "to sketch a rectangle directly on the plot (snaps to the grid "
                        "lines currently on screen — zoom in for a finer grid).",
                    ], style={"color": "gray", "fontSize": "0.8em", "margin": "4px 0"}),
                    dcc.Graph(
                        id="geom-graph",
                        config={
                            "displaylogo": False,
                            "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                            "scrollZoom": True,
                        },
                    ),
                ]),
                dcc.Tab(label="Normal stress map \u03c3 [MPa]", children=[
                    dcc.Graph(id="stress-graph", config={"displaylogo": False, "scrollZoom": True}),
                ]),
                dcc.Tab(label="Shear stress map \u03c4 [MPa]", children=[
                    dcc.Graph(id="shear-graph", config={"displaylogo": False, "scrollZoom": True}),
                ]),
                dcc.Tab(label="Torsional shear \u03c4_t [MPa]", children=[
                    dcc.Graph(id="torsion-graph", config={"displaylogo": False, "scrollZoom": True}),
                ]),
            ]),
        ], style={"flex": "2", "minWidth": "500px"}),

    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start"}),

], style={"margin": "20px", "fontFamily": "sans-serif"})


# =====================================================================
# CALLBACKS
# =====================================================================

# Keeps store-grid-step in sync with whatever grid spacing geom-graph is
# actually showing right now. Uses the same niceStep() helper exposed by
# assets/grid_snap.js (rather than Plotly's own xaxis.dtick/yaxis.dtick)
# because Plotly only resolves those AFTER the first zoom/pan relayout --
# they're still absent on the very first draw right after page load.
app.clientside_callback(
    """
    function(relayoutData) {
        const gd = document.getElementById("geom-graph");
        const plot = gd ? gd.querySelector(".js-plotly-plot") : null;
        const fl = plot && plot._fullLayout;
        const niceStep = window.__gridSnapNiceStep;
        if (!fl || !fl.xaxis || !fl.yaxis || !niceStep) {
            return window.dash_clientside.no_update;
        }
        const xStep = niceStep(Math.abs(fl.xaxis.range[1] - fl.xaxis.range[0]));
        const yStep = niceStep(Math.abs(fl.yaxis.range[1] - fl.yaxis.range[0]));
        return [xStep, yStep];
    }
    """,
    Output("store-grid-step", "data"),
    Input("geom-graph", "relayoutData"),
)


@app.callback(
    Output("store-rects", "data"),
    Output("store-pending", "data"),
    Output("store-geom", "data"),
    Output("store-stress", "data"),
    Output("form-error", "children"),
    Output("confirm-overlap", "displayed"),
    Output("in-x0", "value"),
    Output("in-y0", "value"),
    Output("in-b", "value"),
    Output("in-h", "value"),
    Input("btn-add", "n_clicks"),
    State("in-x0", "value"), State("in-y0", "value"),
    State("in-b", "value"), State("in-h", "value"),
    State("store-rects", "data"),
    prevent_initial_call=True,
)
def add_rectangle(n_clicks, x0, y0, b, h, rects):
    rects = rects or []

    if None in (x0, y0, b, h):
        return (no_update, no_update, no_update, no_update,
                "Invalid input values", False,
                no_update, no_update, no_update, no_update)

    new_rect = [x0, y0, b, h]
    if any(rects_overlap(tuple(new_rect), tuple(r)) for r in rects):
        # don't append yet: store it as pending and pop the confirm dialog.
        # Nothing is written to store-rects here, so a Cancel click (which
        # never even reaches the server) cannot possibly add the rectangle.
        return (no_update, new_rect, no_update, no_update, "", True,
                no_update, no_update, no_update, no_update)

    rects = rects + [new_rect]
    # rectangle accepted: keep the input fields as they are (do NOT reset
    # them to None here — see note above about dcc.Input losing a value of
    # 0 on a None -> 0 transition).
    return rects, None, None, None, "", False, no_update, no_update, no_update, no_update


@app.callback(
    Output("store-rects", "data", allow_duplicate=True),
    Output("store-pending", "data", allow_duplicate=True),
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Output("in-x0", "value", allow_duplicate=True),
    Output("in-y0", "value", allow_duplicate=True),
    Output("in-b", "value", allow_duplicate=True),
    Output("in-h", "value", allow_duplicate=True),
    Input("confirm-overlap", "submit_n_clicks"),
    State("store-pending", "data"), State("store-rects", "data"),
    prevent_initial_call=True,
)
def confirm_add(submit_n_clicks, pending, rects):
    # This callback only fires when the user presses OK in the native
    # browser dialog. Pressing Cancel (or closing it) never triggers any
    # server callback at all, so store-rects is untouched in that case.
    if not pending:
        raise PreventUpdate
    rects = (rects or []) + [pending]
    # keep the input fields as they are (do NOT reset to None, see note
    # in add_rectangle about the None -> 0 transition losing the value 0)
    return rects, None, None, None, no_update, no_update, no_update, no_update


@app.callback(
    Output("store-rects", "data", allow_duplicate=True),
    Output("store-pending", "data", allow_duplicate=True),
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Output("form-error", "children", allow_duplicate=True),
    Output("confirm-overlap", "displayed", allow_duplicate=True),
    Input("geom-graph", "relayoutData"),
    State("store-rects", "data"),
    State("store-grid-step", "data"),
    State("store-suppress-draw", "data"),
    prevent_initial_call=True,
)
def add_rectangle_from_drawing(relayout_data, rects, grid_step, suppress_draw):
    """Turn a rectangle sketched with the graph's draw tool into a new
    entry, snapped to whichever grid was actually on screen at draw time
    (grid_step, kept in sync client-side -- see the clientside callback
    above and assets/grid_snap.js, which also live-snaps the cursor
    itself while dragging). A browser MouseEvent's clientX/clientY are
    only ever whole CSS pixels, so the raw coordinates that arrive here
    carry a little sub-pixel noise even though the drag was already
    snapped -- rounding to grid_step here is what removes that, rather
    than a plain float-noise rounding. Plotly reports a full 'shapes'
    array (rather than a partial 'shapes[i].x0'-style key) only right
    after a NEW shape is added via the draw tool, which is exactly the
    event this callback wants to react to; a longer array than the
    current rect count is the signal that a shape was actually added (as
    opposed to e.g. one being erased). suppress_draw is True for the
    whole duration of a middle-mouse-button pan drag (see
    assets/middle_click_pan.js): Plotly's drawrect handling doesn't
    check which button was pressed, and a client-side-only revert of an
    accidentally-drawn shape can lose the race against this very
    callback already being in flight to the server -- checking the flag
    server-side is what actually closes that race, rather than trusting
    the client's shapes array alone."""
    rects = rects or []
    if suppress_draw:
        raise PreventUpdate
    if not relayout_data or "shapes" not in relayout_data:
        raise PreventUpdate

    shapes = relayout_data["shapes"]
    if len(shapes) <= len(rects):
        raise PreventUpdate

    new_shape = shapes[-1]
    x0s, x1s = new_shape.get("x0"), new_shape.get("x1")
    y0s, y1s = new_shape.get("y0"), new_shape.get("y1")
    if None in (x0s, x1s, y0s, y1s):
        raise PreventUpdate

    x_step, y_step = grid_step or [DEFAULT_GRID_STEP, DEFAULT_GRID_STEP]

    def snap(v, step):
        return round(v / step) * step if step else v

    x0, x1 = sorted([snap(x0s, x_step), snap(x1s, x_step)])
    y0, y1 = sorted([snap(y0s, y_step), snap(y1s, y_step)])
    b, h = x1 - x0, y1 - y0
    if b <= 0 or h <= 0:
        raise PreventUpdate

    new_rect = [x0, y0, b, h]
    if any(rects_overlap(tuple(new_rect), tuple(r)) for r in rects):
        return no_update, new_rect, no_update, no_update, "", True

    rects = rects + [new_rect]
    return rects, None, None, None, "", False


@app.callback(
    Output("rect-list-div", "children"),
    Output("del-dropdown", "options"),
    Input("store-rects", "data"),
)
def render_rect_list(rects):
    rects = rects or []
    if not rects:
        return html.P("No rectangles yet."), []

    header = html.Tr([html.Th(c) for c in ["#", "x0", "y0", "base", "height"]])
    rows = [
        html.Tr([html.Td(i)] + [html.Td(v) for v in r])
        for i, r in enumerate(rects)
    ]
    table = html.Table([header] + rows, style={"borderCollapse": "collapse", "marginBottom": "8px"})

    options = [
        {"label": f"{i}: x0={r[0]}, y0={r[1]}, b={r[2]}, h={r[3]}", "value": i}
        for i, r in enumerate(rects)
    ]
    return table, options


@app.callback(
    Output("store-rects", "data", allow_duplicate=True),
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Input("btn-delete", "n_clicks"),
    State("del-dropdown", "value"), State("store-rects", "data"),
    prevent_initial_call=True,
)
def delete_rectangle(n_clicks, idx, rects):
    rects = rects or []
    if idx is None or not (0 <= idx < len(rects)):
        raise PreventUpdate
    rects = rects[:idx] + rects[idx + 1:]
    return rects, None, None


@app.callback(
    Output("store-rects", "data", allow_duplicate=True),
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Input("btn-delete-all", "n_clicks"),
    State("store-rects", "data"),
    prevent_initial_call=True,
)
def delete_all_rectangles(n_clicks, rects):
    if not rects:
        raise PreventUpdate
    return [], None, None


@app.callback(
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-output", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Input("btn-compute-geom", "n_clicks"),
    State("store-rects", "data"),
    prevent_initial_call=True,
)
def compute_geometry_cb(n_clicks, rects):
    if not rects:
        raise PreventUpdate
    geom = compute_geometry([tuple(r) for r in rects])
    return geom, format_geometry_output(geom), None


@app.callback(
    Output("store-stress", "data", allow_duplicate=True),
    Output("store-output", "data", allow_duplicate=True),
    Input("btn-compute-stress", "n_clicks"),
    State("in-Mx", "value"), State("in-My", "value"), State("in-N", "value"),
    State("in-Vx", "value"), State("in-Vy", "value"), State("in-T", "value"),
    State("store-rects", "data"), State("store-geom", "data"),
    prevent_initial_call=True,
)
def compute_stress_cb(n_clicks, Mx_kNm, My_kNm, N_kN, Vx_kN, Vy_kN, T_kNm, rects, geom):
    if not rects or not geom:
        raise PreventUpdate

    # an empty field is treated as zero (no contribution), same idea as
    # leaving any of these out entirely
    Mx_kNm = Mx_kNm if Mx_kNm is not None else 0.0
    My_kNm = My_kNm if My_kNm is not None else 0.0
    N_kN = N_kN if N_kN is not None else 0.0
    Vx_kN = Vx_kN if Vx_kN is not None else 0.0
    Vy_kN = Vy_kN if Vy_kN is not None else 0.0
    T_kNm = T_kNm if T_kNm is not None else 0.0

    Mx = Mx_kNm * 1.0e6
    My = My_kNm * 1.0e6
    N = N_kN * 1.0e3
    Vx = Vx_kN * 1.0e3
    Vy = Vy_kN * 1.0e3
    T = T_kNm * 1.0e6

    sigma_max, loc_max, sigma_min, loc_min = compute_stress_extremes(
        [tuple(r) for r in rects], Mx, My, N, geom
    )

    # rebuilt fresh every time (geometry text + this run's stress block only),
    # so repeated clicks refresh the results instead of stacking them
    lines = format_geometry_output(geom) + [
        "",
        "---- Normal stresses (biaxial bending + axial force) ----",
        f"Mx = {Mx_kNm:.4f} kN\u00b7m, My = {My_kNm:.4f} kN\u00b7m, N = {N_kN:.4f} kN",
        f"Sigma max (tension)     = {sigma_max:.4f} MPa  at "
        f"(x={loc_max[0]:.3f}, y={loc_max[1]:.3f}) mm",
        f"Sigma min (compression) = {sigma_min:.4f} MPa  at "
        f"(x={loc_min[0]:.3f}, y={loc_min[1]:.3f}) mm",
    ]

    if Vx != 0.0 or Vy != 0.0:
        _, tau_max, loc_tau_max = build_shear_figure([tuple(r) for r in rects], geom, Vx, Vy)
        lines += [
            "",
            "---- Shear stresses (approximate Jourawski method) ----",
            f"Vx = {Vx_kN:.4f} kN, Vy = {Vy_kN:.4f} kN",
            f"Tau max (approx, numeric) = {tau_max:.4f} MPa  at "
            f"(x={loc_tau_max[0]:.3f}, y={loc_tau_max[1]:.3f}) mm",
        ]

    if T != 0.0:
        _, tau_t_max, loc_t_max, J = build_torsion_figure([tuple(r) for r in rects], geom, T)
        lines += [
            "",
            "---- Torsional shear stress (open thin-walled section) ----",
            f"Mz = {T_kNm:.4f} kN\u00b7m (positive = counterclockwise)",
            f"J = {J:.4f} mm\u2074",
            f"Tau_t max = {tau_t_max:.4f} MPa  at (x={loc_t_max[0]:.3f}, y={loc_t_max[1]:.3f}) mm",
        ]

    return [Mx_kNm, My_kNm, N_kN, Vx_kN, Vy_kN, T_kNm], lines


@app.callback(
    Output("output-text", "children"),
    Input("store-output", "data"),
)
def render_output(lines):
    return "\n".join(lines or [])


@app.callback(
    Output("geom-graph", "figure"),
    Input("store-rects", "data"),
    Input("store-geom", "data"),
)
def render_geom_graph(rects, geom):
    rects = [tuple(r) for r in (rects or [])]
    return build_geometry_figure(rects, geom)


@app.callback(
    Output("stress-graph", "figure"),
    Input("store-stress", "data"),
    Input("store-geom", "data"),
    State("store-rects", "data"),
)
def render_stress_graph(stress_inputs, geom, rects):
    if not stress_inputs or not geom or not rects:
        return {}
    rects = [tuple(r) for r in rects]
    Mx_kNm, My_kNm, N_kN, Vx_kN, Vy_kN, T_kNm = stress_inputs
    Mx, My, N = Mx_kNm * 1.0e6, My_kNm * 1.0e6, N_kN * 1.0e3
    fig, *_ = build_stress_figure(rects, geom, Mx, My, N)
    return fig


@app.callback(
    Output("shear-graph", "figure"),
    Input("store-stress", "data"),
    Input("store-geom", "data"),
    State("store-rects", "data"),
)
def render_shear_graph(stress_inputs, geom, rects):
    if not stress_inputs or not geom or not rects:
        return {}
    rects = [tuple(r) for r in rects]
    Mx_kNm, My_kNm, N_kN, Vx_kN, Vy_kN, T_kNm = stress_inputs
    Vx, Vy = Vx_kN * 1.0e3, Vy_kN * 1.0e3
    fig, *_ = build_shear_figure(rects, geom, Vx, Vy)
    return fig


@app.callback(
    Output("torsion-graph", "figure"),
    Input("store-stress", "data"),
    Input("store-geom", "data"),
    State("store-rects", "data"),
)
def render_torsion_graph(stress_inputs, geom, rects):
    if not stress_inputs or not geom or not rects:
        return {}
    rects = [tuple(r) for r in rects]
    Mx_kNm, My_kNm, N_kN, Vx_kN, Vy_kN, T_kNm = stress_inputs
    T = T_kNm * 1.0e6
    fig, *_ = build_torsion_figure(rects, geom, T)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
