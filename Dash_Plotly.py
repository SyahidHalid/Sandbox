#pip install dash pandas plotly

#D:\Git Repository>mkdir Sandbox && cd Sandbox
#D:\Git Repository\mkdir Sandbox>python3 -m venv venv
#D:\Git Repository\mkdir Sandbox>venv\Scripts\activate.bat
#(venv) D:\Git Repository\Sandbox>

# from dash import dash, dcc, html
# import pandas as pd
    
#===============================================================================================
# data = pd.DataFrame({
#     'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
#     'Category': ['Electronics', 'Clothing', 'Books', 'Home'] * 25,
#     'Sales': [abs(int(x)) for x in np.random.normal(loc=100, scale=40, size=100)],
#     'Revenue': [abs(int(x)) for x in np.random.normal(loc=1000, scale=200, size=100)],
# })

# app = dash.Dash(__name__)

# app.layout = html.Div(
#     children=[
#         html.H1(children="Syahid",),
#         html.P(
#             children="Syahid mana"
#             "syahid ap"
#             "situ syahid",
#         ),
#         dcc.Graph(
#             figure={
#                 "data":[
#                     {
#                         "x":data["Date"],
#                         "y":data["Sales"],
#                         "type":"lines",
#                     },
#                 ],
#                 "layout":{"title":"Avergae balh"},
#             },
#         ),
#         dcc.Graph(
#             figure={
#                 "data":[
#                     {
#                         "x":data["Date"],
#                         "y":data["Revenue"],
#                         "type":"lines",
#                     },
#                 ],
#                 "layout":{"title":"Avergae Revenue"},
#             },
#         ),
#     ]
# )

# if __name__ == "__main__":
#     app.run+_server(debug=True)
    
#===============================================================================================
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Sample dataset
df = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'Category': ['Electronics', 'Clothing', 'Books', 'Home'] * 25,
    'Sales': [abs(int(x)) for x in np.random.normal(loc=100, scale=40, size=100)],
    'Revenue': [abs(int(x)) for x in np.random.normal(loc=1000, scale=200, size=100)],
})

# Dash app initialization
app = dash.Dash(__name__)
app.title = "Sales Dashboard with Filters and Tabs"

# Layout with Tabs
app.layout = html.Div([
    html.H2("Interactive Sales Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.Label("Select Category:"),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
                    value='Electronics'
                ),
                html.Br(),
                dcc.Graph(id='sales-bar'),
            ])
        ]),

        dcc.Tab(label='Revenue Trends', children=[
            html.Div([
                html.Label("Select Category:"),
                dcc.Dropdown(
                    id='revenue-filter',
                    options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
                    value='Electronics'
                ),
                html.Br(),
                dcc.Graph(id='revenue-line'),
            ])
        ])
    ])
])

# Callbacks for tab 1 (Sales Bar)
@app.callback(
    Output('sales-bar', 'figure'),
    Input('category-filter', 'value')
)
def update_sales_chart(selected_category):
    filtered_df = df[df['Category'] == selected_category]
    fig = px.bar(
        filtered_df, x='Date', y='Sales', title=f"Sales for {selected_category}",
        labels={'Sales': 'Units Sold'}, template='plotly_dark'
    )
    return fig

# Callbacks for tab 2 (Revenue Line)
@app.callback(
    Output('revenue-line', 'figure'),
    Input('revenue-filter', 'value')
)
def update_revenue_chart(selected_category):
    filtered_df = df[df['Category'] == selected_category]
    fig = px.line(
        filtered_df, x='Date', y='Revenue', title=f"Revenue Trend for {selected_category}",
        labels={'Revenue': 'Revenue ($)'}, template='plotly_dark'
    )
    return fig

# Run app
if __name__ == '__main__':
    app.run(debug=True)

#run> python Dash_plotly.py
