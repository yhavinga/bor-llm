import os
import pandas as pd
import re
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

# Set the folder where benchmark results are stored
RESULTS_DIR = "./benchmark_results"

# Create the Dash application
app = dash.Dash(__name__, title="Benchmark Comparison")

# Define the layout of the application
app.layout = html.Div([
    html.H1("Text Simplification Model Comparison"),
    
    # File selection section
    html.Div([
        html.Div([
            html.H3("Model A"),
            dcc.Dropdown(id='model-a-dropdown', placeholder="Select Model A"),
        ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}),
        
        html.Div([
            html.H3("Model B"),
            dcc.Dropdown(id='model-b-dropdown', placeholder="Select Model B"),
        ], style={'width': '45%', 'display': 'inline-block'}),
        
        html.Button('Compare', id='compare-button', n_clicks=0, 
                   style={'margin-top': '20px', 'font-size': '16px'}),
    ], style={'margin-bottom': '20px'}),
    
    # Overall statistics section
    html.Div([
        html.H2("Comparison Summary"),
        html.Div(id='summary-stats'),
        
        # SARI score distribution graph
        dcc.Graph(id='sari-distribution'),
        
        # Length comparison graph
        dcc.Graph(id='length-comparison')
    ]),
    
    # Side-by-side comparison controls
    html.Div([
        html.H2("Side-by-Side Comparison"),
        html.Div([
            html.Div([
                html.Label("Sort by:"),
                dcc.Dropdown(
                    id='sort-field-dropdown',
                    options=[
                        {'label': 'SARI Score', 'value': 'sari'},
                        {'label': 'Record ID', 'value': 'id'},
                        {'label': 'Original Text Length', 'value': 'orig_len'},
                        {'label': 'Model A Output Length', 'value': 'a_len'},
                        {'label': 'Model B Output Length', 'value': 'b_len'},
                    ],
                    value='sari'
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),
            
            html.Div([
                html.Label("Sort order:"),
                dcc.RadioItems(
                    id='sort-order-radio',
                    options=[
                        {'label': 'Ascending', 'value': 'asc'},
                        {'label': 'Descending', 'value': 'desc'},
                    ],
                    value='desc',
                    inline=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Label("Filter by SARI difference:"),
            dcc.RangeSlider(
                id='sari-diff-slider',
                min=0,
                max=100,
                step=1,
                marks={i: f'{i}' for i in range(0, 101, 10)},
                value=[0, 100]
            ),
        ], style={'width': '50%', 'margin': '0 auto', 'padding': '10px'}),
    ]),
    
    # Side-by-side comparison table
    html.Div(id='comparison-table-container', style={'overflow': 'auto'}),
    
    # Store component to keep the loaded data
    dcc.Store(id='model-a-data'),
    dcc.Store(id='model-b-data'),
])

# Callback to populate model dropdowns
@callback(
    [Output('model-a-dropdown', 'options'),
     Output('model-b-dropdown', 'options')],
    Input('model-a-dropdown', 'search_value')  # Just a trigger
)
def update_model_options(_):
    if not os.path.exists(RESULTS_DIR):
        return [], []
    
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_results.csv')]
    
    # Extract model names from filenames for better display
    options = []
    for f in files:
        # Better model name extraction
        # Remove the _100_results.csv part
        model_name = re.sub(r'_\d+_results\.csv$', '', f)
        
        # Format the name in a more readable way
        formatted_name = model_name.replace('-', ' ')
        
        options.append({'label': formatted_name, 'value': f})
    
    return options, options

# Helper function to format model names consistently
def format_model_name(filename):
    """Extract and format model name from results filename"""
    # Remove the _results.csv suffix
    model_name = re.sub(r'_\d+_results\.csv$', '', filename)
    return model_name.replace('-', ' ')

# Callback to load the model data when selected
@callback(
    [Output('model-a-data', 'data'),
     Output('model-b-data', 'data')],
    Input('compare-button', 'n_clicks'),
    [State('model-a-dropdown', 'value'),
     State('model-b-dropdown', 'value')]
)
def load_model_data(n_clicks, model_a, model_b):
    if n_clicks == 0 or not model_a or not model_b:
        raise PreventUpdate
    
    # Load data for model A
    a_path = os.path.join(RESULTS_DIR, model_a)
    df_a = pd.read_csv(a_path)
    
    # Load data for model B
    b_path = os.path.join(RESULTS_DIR, model_b)
    df_b = pd.read_csv(b_path)
    
    # Make sure we're comparing the same examples by matching on Original text
    merged = pd.merge(
        df_a, df_b, 
        on='Original', 
        suffixes=('_A', '_B')
    )
    
    # Add lengths for sorting
    merged['Original_Length'] = merged['Original'].str.len()
    merged['A_Length'] = merged['Model Output_A'].str.len()
    merged['B_Length'] = merged['Model Output_B'].str.len()
    merged['SARI_Diff'] = abs(merged['SARI Score_A'] - merged['SARI Score_B'])
    
    # Add an ID column
    merged['ID'] = range(1, len(merged) + 1)
    
    return merged.to_dict('records'), {"model_a": model_a, "model_b": model_b}

# Callback to update summary statistics
@callback(
    Output('summary-stats', 'children'),
    [Input('model-a-data', 'data'),
     Input('model-b-data', 'data')]
)
def update_summary(model_a_data, model_b_info):
    if not model_a_data or not model_b_info:
        raise PreventUpdate
    
    df = pd.DataFrame(model_a_data)
    model_a = model_b_info["model_a"]
    model_b = model_b_info["model_b"]
    
    # Calculate summary statistics
    avg_sari_a = df['SARI Score_A'].mean()
    avg_sari_b = df['SARI Score_B'].mean()
    
    # Count where each model performs better
    a_better = (df['SARI Score_A'] > df['SARI Score_B']).sum()
    b_better = (df['SARI Score_B'] > df['SARI Score_A']).sum()
    tie = (df['SARI Score_A'] == df['SARI Score_B']).sum()
    
    # Format the model names for display
    model_a_name = format_model_name(model_a)
    model_b_name = format_model_name(model_b)
    
    return html.Div([
        html.Div([
            html.H3(f"Average SARI Scores"),
            html.Table([
                html.Tr([
                    html.Th("Model", style={'text-align': 'left', 'padding': '8px'}),
                    html.Th("Average SARI", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td(model_a_name, style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{avg_sari_a:.2f}", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td(model_b_name, style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{avg_sari_b:.2f}", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Difference", style={'text-align': 'left', 'padding': '8px', 'font-weight': 'bold'}),
                    html.Td(f"{abs(avg_sari_a - avg_sari_b):.2f}", 
                           style={'text-align': 'right', 'padding': '8px', 'font-weight': 'bold'})
                ])
            ], style={'border-collapse': 'collapse', 'width': '100%'})
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Performance Comparison"),
            html.Table([
                html.Tr([
                    html.Th("Result", style={'text-align': 'left', 'padding': '8px'}),
                    html.Th("Count", style={'text-align': 'right', 'padding': '8px'}),
                    html.Th("Percentage", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td(f"{model_a_name} better", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{a_better}", style={'text-align': 'right', 'padding': '8px'}),
                    html.Td(f"{100 * a_better / len(df):.1f}%", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td(f"{model_b_name} better", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{b_better}", style={'text-align': 'right', 'padding': '8px'}),
                    html.Td(f"{100 * b_better / len(df):.1f}%", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Tie", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{tie}", style={'text-align': 'right', 'padding': '8px'}),
                    html.Td(f"{100 * tie / len(df):.1f}%", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Total", style={'text-align': 'left', 'padding': '8px', 'font-weight': 'bold'}),
                    html.Td(f"{len(df)}", style={'text-align': 'right', 'padding': '8px', 'font-weight': 'bold'}),
                    html.Td("100%", style={'text-align': 'right', 'padding': '8px', 'font-weight': 'bold'})
                ])
            ], style={'border-collapse': 'collapse', 'width': '100%'})
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '5%'})
    ])

# Callback to update SARI distribution graph
@callback(
    Output('sari-distribution', 'figure'),
    [Input('model-a-data', 'data'),
     Input('model-b-data', 'data')]
)
def update_sari_distribution(model_a_data, model_b_info):
    if not model_a_data or not model_b_info:
        raise PreventUpdate
    
    df = pd.DataFrame(model_a_data)
    model_a = model_b_info["model_a"]
    model_b = model_b_info["model_b"]
    
    # Format the model names for display
    model_a_name = format_model_name(model_a)
    model_b_name = format_model_name(model_b)
    
    # Create histogram for SARI score distributions
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['SARI Score_A'],
        name=model_a_name,
        opacity=0.7,
        bingroup=1
    ))
    fig.add_trace(go.Histogram(
        x=df['SARI Score_B'],
        name=model_b_name,
        opacity=0.7,
        bingroup=1
    ))
    
    fig.update_layout(
        title="SARI Score Distribution",
        xaxis_title="SARI Score",
        yaxis_title="Count",
        barmode='overlay',
        bargap=0.1
    )
    
    return fig

# Callback to update length comparison graph
@callback(
    Output('length-comparison', 'figure'),
    [Input('model-a-data', 'data'),
     Input('model-b-data', 'data')]
)
def update_length_comparison(model_a_data, model_b_info):
    if not model_a_data or not model_b_info:
        raise PreventUpdate
    
    df = pd.DataFrame(model_a_data)
    model_a = model_b_info["model_a"]
    model_b = model_b_info["model_b"]
    
    # Format the model names for display
    model_a_name = format_model_name(model_a)
    model_b_name = format_model_name(model_b)
    
    # Create scatter plot comparing original vs simplified lengths
    fig = go.Figure()
    
    # Add reference line (x=y)
    max_len = max(df['Original_Length'].max(), df['A_Length'].max(), df['B_Length'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_len],
        y=[0, max_len],
        mode='lines',
        name='No Simplification',
        line=dict(color='gray', dash='dash')
    ))
    
    # Add points for model A
    fig.add_trace(go.Scatter(
        x=df['Original_Length'],
        y=df['A_Length'],
        mode='markers',
        name=model_a_name,
        marker=dict(size=8)
    ))
    
    # Add points for model B
    fig.add_trace(go.Scatter(
        x=df['Original_Length'],
        y=df['B_Length'],
        mode='markers',
        name=model_b_name,
        marker=dict(size=8)
    ))
    
    # Calculate and display trendlines
    fig.update_layout(
        title="Text Length Comparison",
        xaxis_title="Original Text Length (characters)",
        yaxis_title="Simplified Text Length (characters)",
        legend_title="Model"
    )
    
    return fig

# Callback to update the comparison table
@callback(
    Output('comparison-table-container', 'children'),
    [Input('model-a-data', 'data'),
     Input('model-b-data', 'data'),
     Input('sort-field-dropdown', 'value'),
     Input('sort-order-radio', 'value'),
     Input('sari-diff-slider', 'value')]
)
def update_comparison_table(model_a_data, model_b_info, sort_field, sort_order, sari_diff_range):
    if not model_a_data or not model_b_info:
        raise PreventUpdate
    
    df = pd.DataFrame(model_a_data)
    model_a = model_b_info["model_a"]
    model_b = model_b_info["model_b"]
    
    # Filter by SARI difference
    df = df[(df['SARI_Diff'] >= sari_diff_range[0]) & (df['SARI_Diff'] <= sari_diff_range[1])]
    
    # Sort the data according to user selection
    ascending = (sort_order == 'asc')
    
    if sort_field == 'sari':
        df = df.sort_values(by='SARI Score_A', ascending=ascending)
    elif sort_field == 'id':
        df = df.sort_values(by='ID', ascending=ascending)
    elif sort_field == 'orig_len':
        df = df.sort_values(by='Original_Length', ascending=ascending)
    elif sort_field == 'a_len':
        df = df.sort_values(by='A_Length', ascending=ascending)
    elif sort_field == 'b_len':
        df = df.sort_values(by='B_Length', ascending=ascending)
    
    # Format the model names for display
    model_a_name = format_model_name(model_a)
    model_b_name = format_model_name(model_b)
    
    # Prepare data for the table
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'ID': row['ID'],
            'Original': row['Original'],
            f"{model_a_name} Output": row['Model Output_A'],
            f"{model_a_name} SARI": f"{row['SARI Score_A']:.2f}",
            f"{model_b_name} Output": row['Model Output_B'],
            f"{model_b_name} SARI": f"{row['SARI Score_B']:.2f}",
            'Reference': row['Reference_A'],  # Both should be the same
            'SARI Diff': f"{abs(row['SARI Score_A'] - row['SARI Score_B']):.2f}"
        })
    
    return dash_table.DataTable(
        id='comparison-table',
        columns=[
            {'name': 'ID', 'id': 'ID'},
            {'name': 'Original', 'id': 'Original'},
            {'name': f"{model_a_name} Output", 'id': f"{model_a_name} Output"},
            {'name': f"{model_a_name} SARI", 'id': f"{model_a_name} SARI"},
            {'name': f"{model_b_name} Output", 'id': f"{model_b_name} Output"},
            {'name': f"{model_b_name} SARI", 'id': f"{model_b_name} SARI"},
            {'name': 'Reference', 'id': 'Reference'},
            {'name': 'SARI Diff', 'id': 'SARI Diff'}
        ],
        data=table_data,
        style_table={
            'height': '900px',
            'overflowY': 'auto',
            'width': '100%'
        },
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '50px',  # Use minWidth instead of width
            'maxWidth': '300px',  # Add maxWidth for better column sizing
        },
        style_cell_conditional=[
            {'if': {'column_id': 'ID'}, 'width': '5%'},
            {'if': {'column_id': 'Original'}, 'width': '20%'},
            {'if': {'column_id': f"{model_a_name} Output"}, 'width': '20%'},
            {'if': {'column_id': f"{model_a_name} SARI"}, 'width': '10%'},
            {'if': {'column_id': f"{model_b_name} Output"}, 'width': '20%'},
            {'if': {'column_id': f"{model_b_name} SARI"}, 'width': '10%'},
            {'if': {'column_id': 'Reference'}, 'width': '20%'},
            {'if': {'column_id': 'SARI Diff'}, 'width': '10%'}
        ],
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in table_data
        ],
        tooltip_duration=None,
        page_size=50
    )

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    app.run_server(debug=True, port=8050)