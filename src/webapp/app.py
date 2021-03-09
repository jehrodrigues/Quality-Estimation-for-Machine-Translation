import base64
import io

import numpy as np
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table.Format as Format

import plotly.express as px

# from src.webapp.utils import discrete_background_color_bins
from src.models.crosslingual_cosine_similarity import CrosslingualCosineSimilarity

# Model is loaded in the main function
cosine_similarity_model = None

# Languages to be supported
languages = sorted(
    [
        "English",
        "Italian",
        "Russian",
        "Chinese",
        "Japanese",
        "Arabic",
        "Spanish",
        "Portuguese",
        "Korean",
        "French",
    ]
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, url_base_pathname='/TranslationQualityDemo/')

server = app.server

# App's layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("definedcrowd.png"),
                    style={"height": "10%", "width": "10%", "float": "left"},
                ),
                html.H2(
                    "Machine Translation Quality Estimation (MTQE) Demo",
                    style={"position": "relative", "top": "18px", "left": "10px"},
                ),
            ]
        ),
        html.Hr(),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div(
            id="source-choices",
            children=[
                html.Label(
                    [
                        "Source Language",
                        dcc.Dropdown(
                            id="source-language-dropdown",
                            options=[{"label": l, "value": l} for l in languages],
                            value="English",
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "position": "relative",
                        "min-width": "200px",
                    },
                ),
                html.Label(
                    [
                        "Source Column",
                        dcc.Dropdown(
                            id="source-column-dropdown",
                            options=[{"label": "Source Text", "value": "Source Text"}],
                            value="Source Text",
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "position": "relative",
                        "min-width": "200px",
                    },
                ),
            ],
        ),
        html.Div(
            id="target-choices",
            children=[
                html.Label(
                    [
                        "Target Language",
                        dcc.Dropdown(
                            id="target-language-dropdown",
                            options=[{"label": l, "value": l} for l in languages],
                            value="English",
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "position": "relative",
                        "min-width": "200px",
                    },
                ),
                html.Label(
                    [
                        "Target Column",
                        dcc.Dropdown(
                            id="target-column-dropdown",
                            options=[{"label": "Target Text", "value": "Target Text"}],
                            value="Target Text",
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "position": "relative",
                        "min-width": "200px",
                    },
                ),
            ],
        ),
        html.Div(
            [
                html.H3(
                    [
                        f"Average Quality Score: ",
                        html.B("--", id="average_score"),
                    ]
                ),
                dash_table.DataTable(
                    id="datatable-translation-quality",
                    columns=[
                        {"name": "Source Text", "id": "Source Text"},
                        {"name": "Target Text", "id": "Target Text"},
                        {"name": "Quality Score", "id": "Quality Score"},
                        {"name": "Confidence", "id": "Confidence"},
                    ],
                    data=[
                        {
                            "Source Text": "",
                            "Target Text": "",
                            "Quality Score": "",
                            "Confidence": "",
                        }
                    ],
                    editable=False,
                    sort_action="native",
                    sort_mode="multi",
                    style_cell={
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=20,
                    export_format="csv",
                    export_headers="display",
                ),
            ],
            style={
                "margin-left": "10px",
                "margin-rigth": "10px",
                "margin-top": "20px",
                "margin-bottom": "10px",
            },
        ),
        html.Br(),
        html.Div(
            dcc.Graph(id="distribution-graph"),
        ),
    ]
)

# Main callback event - push changes from file update and drodown's selection
@app.callback(
    [
        Output("datatable-translation-quality", "data"),
        Output("datatable-translation-quality", "columns"),
        Output("source-column-dropdown", "options"),
        Output("target-column-dropdown", "options"),
        Output("distribution-graph", "figure"),
        Output("average_score", "children"),
    ],
    [
        Input("upload-data", "contents"),
        Input("source-column-dropdown", "value"),
        Input("target-column-dropdown", "value"),
    ],
    [
        State("upload-data", "filename"),
        State("datatable-translation-quality", "data"),
        State("datatable-translation-quality", "columns"),
        State("distribution-graph", "figure"),
    ],
)
def update_output(
    contents, source_column, target_column, filename, rows, columns, figure
):
    if not dash.callback_context.triggered:
        raise PreventUpdate

    average_score = "--"
    column_options = [{"label": c["name"], "value": c["name"]} for c in columns]
    df = pd.DataFrame(columns=columns).from_dict(rows)

    # read file contents
    if contents:
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        try:
            if ".csv" in filename:
                # Assume that the user uploaded a CSV file
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=",")
                except Exception:
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=";")
            elif ".tsv" in filename:
                # Assume that the user uploaded a TSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t", warn_bad_lines=True, error_bad_lines=False)
            elif ".xls" in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)

        # Updated data should only contain translation pairs
        df.fillna('', inplace=True)
        df = df.astype(str)

    # Check if the model was correcly loaded
    if cosine_similarity_model is None:
        raise Exception("Quality Model not loaded...")

    df["Quality Score"] = 0
    df["Confidence"] = "-"

    # Check if the source and target columns are valid
    if source_column in df.columns and target_column in df.columns:

        df["Quality Score"] = df.apply(
            # changing the precision in this function because the formater for DataTable is really limited
            lambda row: float(
                "{:.2f}".format(
                    cosine_similarity_model.predict(
                        row[source_column], row[target_column]
                    )
                )
            )
            if (len(row[source_column]) != 0 and len(row[target_column]) != 0) else 0,
            axis=1,
        )

        # By now, confidence is measured according to the source sentence length
        mean_sentence_size = np.mean(df[source_column].apply(len))
        std_sentence_size = np.std(df[source_column].apply(len))
        df[f"Confidence"] = df.apply(
            lambda row: "2 - High"
            if len(row[source_column]) < (mean_sentence_size - std_sentence_size)
            else "1 - Medium"
            if len(row[source_column]) <= (mean_sentence_size + std_sentence_size)
            else "0 - Low",
            axis=1,
        )

    average_score = np.mean(df["Quality Score"])

    # Colors using conditional formatters
    # styles = discrete_background_color_bins(df, n_bins=8, columns=["Quality Score"])

    # Statistics Chart
    figure = px.histogram(df, x="Quality Score", marginal="rug", nbins=20)

    figure.update_layout(
        title_text="Quality Scores",  # title of plot
        xaxis_title_text="Score",  # xaxis label
        yaxis_title_text="Frequency",  # yaxis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
    )

    # update DataTable and Dropdown's data
    rows = df.to_dict("records")
    column_options = [{"label": i, "value": i} for i in df.columns]
    columns = [{"name": i, "id": i} for i in df.columns]

    return rows, columns, column_options, column_options, figure, f"{average_score:.2f}"


if __name__ == "__main__":
    # Load the crosslingual semantic similarity model
    cosine_similarity_model = CrosslingualCosineSimilarity()
    app.run_server(debug=False, host="0.0.0.0",)
