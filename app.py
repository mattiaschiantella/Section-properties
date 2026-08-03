import dash
from dash import dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from compute import (
    rects_overlap, compute_geometry, format_geometry_output,
    compute_stress_extremes, build_geometry_figure, build_stress_figure,
)

app = dash.Dash(__name__)
app.title = "Section Analyzer"

INPUT_STYLE = {"width": "90px", "marginRight": "8px"}
BTN_STYLE = {"marginRight": "8px", "marginTop": "6px"}


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

    html.H2("Section Analyzer - Dash"),

    html.Div([

        # ---------------- LEFT: controls ----------------
        html.Div([

            html.Fieldset([
                html.Legend("Geometry input"),
                html.Div([
                    dcc.Input(id="in-x0", type="number", placeholder="x0 [mm]", style=INPUT_STYLE),
                    dcc.Input(id="in-y0", type="number", placeholder="y0 [mm]", style=INPUT_STYLE),
                    dcc.Input(id="in-b", type="number", placeholder="base [mm]", style=INPUT_STYLE),
                    dcc.Input(id="in-h", type="number", placeholder="height [mm]", style=INPUT_STYLE),
                ]),
                html.Div([
                    html.Button("Add", id="btn-add", n_clicks=0, style=BTN_STYLE),
                    html.Button("Compute geometry", id="btn-compute-geom", n_clicks=0, style=BTN_STYLE),
                ]),
                html.Div(id="form-error", style={"color": "red", "marginTop": "4px"}),
            ]),

            html.Div(id="warning-div"),

            html.Fieldset([
                html.Legend("Defined rectangles"),
                html.Div(id="rect-list-div"),
                html.Div([
                    dcc.Dropdown(id="del-dropdown", options=[], placeholder="Select to delete",
                                 style={"width": "260px", "display": "inline-block"}),
                    html.Button("Delete selected", id="btn-delete", n_clicks=0,
                                style={**BTN_STYLE, "verticalAlign": "top", "marginLeft": "8px"}),
                ]),
            ]),

            html.Fieldset([
                html.Legend("Internal actions"),
                html.Div([
                    dcc.Input(id="in-Mx", type="number", value=0, placeholder="Mx [kN\u00b7m]", style=INPUT_STYLE),
                    dcc.Input(id="in-My", type="number", value=0, placeholder="My [kN\u00b7m]", style=INPUT_STYLE),
                    dcc.Input(id="in-N", type="number", value=0, placeholder="N [kN]", style=INPUT_STYLE),
                ]),
                html.P("Units: x0, y0, base, height in mm. Mx, My in kN\u00b7m. N in kN. "
                       "Resulting stresses are given in MPa.",
                       style={"color": "gray", "fontSize": "0.85em"}),
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
                    dcc.Graph(id="geom-graph", config={"displaylogo": False}),
                ]),
                dcc.Tab(label="Normal stress map \u03c3 [MPa]", children=[
                    dcc.Graph(id="stress-graph", config={"displaylogo": False}),
                ]),
            ]),
        ], style={"flex": "2", "minWidth": "500px"}),

    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start"}),

], style={"margin": "20px", "fontFamily": "sans-serif"})


# =====================================================================
# CALLBACKS
# =====================================================================

@app.callback(
    Output("store-rects", "data"),
    Output("store-pending", "data"),
    Output("store-geom", "data"),
    Output("store-stress", "data"),
    Output("form-error", "children"),
    Input("btn-add", "n_clicks"),
    State("in-x0", "value"), State("in-y0", "value"),
    State("in-b", "value"), State("in-h", "value"),
    State("store-rects", "data"),
    prevent_initial_call=True,
)
def add_rectangle(n_clicks, x0, y0, b, h, rects):
    rects = rects or []

    if None in (x0, y0, b, h):
        return no_update, no_update, no_update, no_update, "Invalid input values"

    new_rect = [x0, y0, b, h]
    if any(rects_overlap(tuple(new_rect), tuple(r)) for r in rects):
        # don't append yet: wait for explicit confirmation
        return no_update, new_rect, no_update, no_update, ""

    rects = rects + [new_rect]
    return rects, None, None, None, ""


@app.callback(
    Output("store-rects", "data", allow_duplicate=True),
    Output("store-pending", "data", allow_duplicate=True),
    Output("store-geom", "data", allow_duplicate=True),
    Output("store-stress", "data", allow_duplicate=True),
    Input("btn-confirm-add", "n_clicks"),
    State("store-pending", "data"), State("store-rects", "data"),
    prevent_initial_call=True,
)
def confirm_add(n_clicks, pending, rects):
    if not pending:
        raise PreventUpdate
    rects = (rects or []) + [pending]
    return rects, None, None, None


@app.callback(
    Output("store-pending", "data", allow_duplicate=True),
    Input("btn-cancel-add", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_add(n_clicks):
    return None


@app.callback(
    Output("warning-div", "children"),
    Input("store-pending", "data"),
)
def show_warning(pending):
    if not pending:
        return None
    return html.Div([
        html.P("The current rectangle is overlapping the previous ones. If you continue, "
               "the shared part will be counted twice in the area, centroid and moment "
               "of inertia calculations."),
        html.Button("Add anyway", id="btn-confirm-add", n_clicks=0, style=BTN_STYLE),
        html.Button("Cancel", id="btn-cancel-add", n_clicks=0, style=BTN_STYLE),
    ], style={"background": "#fff3cd", "border": "1px solid #ffe69c", "padding": "10px",
              "marginBottom": "10px"})


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
    State("store-rects", "data"), State("store-geom", "data"), State("store-output", "data"),
    prevent_initial_call=True,
)
def compute_stress_cb(n_clicks, Mx_kNm, My_kNm, N_kN, rects, geom, output_lines):
    if not rects or not geom:
        raise PreventUpdate
    if None in (Mx_kNm, My_kNm, N_kN):
        raise PreventUpdate

    Mx = Mx_kNm * 1.0e6
    My = My_kNm * 1.0e6
    N = N_kN * 1.0e3

    sigma_max, loc_max, sigma_min, loc_min = compute_stress_extremes(
        [tuple(r) for r in rects], Mx, My, N, geom
    )

    lines = (output_lines or []) + [
        "",
        "---- Normal stresses (biaxial bending + axial force) ----",
        f"Mx = {Mx_kNm:.4f} kN\u00b7m, My = {My_kNm:.4f} kN\u00b7m, N = {N_kN:.4f} kN",
        f"Sigma max (tension)     = {sigma_max:.4f} MPa  at "
        f"(x={loc_max[0]:.3f}, y={loc_max[1]:.3f}) mm",
        f"Sigma min (compression) = {sigma_min:.4f} MPa  at "
        f"(x={loc_min[0]:.3f}, y={loc_min[1]:.3f}) mm",
    ]

    return [Mx_kNm, My_kNm, N_kN], lines


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
    Mx_kNm, My_kNm, N_kN = stress_inputs
    Mx, My, N = Mx_kNm * 1.0e6, My_kNm * 1.0e6, N_kN * 1.0e3
    fig, *_ = build_stress_figure(rects, geom, Mx, My, N)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
