
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from src.financial_market_optimizer import FinancialMarketOptimizer

app = dash.Dash(__name__)

optimizer = FinancialMarketOptimizer(initial_capital=100000, num_assets=5)

app.layout = html.Div([
    html.H1('Financial Market Optimizer'),
    dcc.Graph(id='portfolio-allocation'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('portfolio-allocation', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    market_data = np.random.rand(5)  # Simulating real-time market data
    optimal_allocation = optimizer.optimize_portfolio(market_data)
    
    labels = [f'Asset {i+1}' for i in range(len(optimal_allocation))]
    
    return {
        'data': [go.Pie(labels=labels, values=optimal_allocation)],
        'layout': go.Layout(
            title='Optimal Portfolio Allocation',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h")
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
