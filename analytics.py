import plotly.graph_objects as go
import pandas as pd


class AnalyticsEngine:
    @staticmethod
    def create_progress_chart(sessions):
        if not sessions:
            return None

        df = pd.DataFrame([{
            'Day': s.day_number,
            'Completed': 1 if s.is_completed else 0,
            'Topic': s.topic
        } for s in sessions])

        fig = go.Figure(data=[
            go.Bar(name='Progress', x=df['Day'], y=df['Completed'], marker_color='#00FFAA')
        ])

        fig.update_layout(
            title="Study Consistency Tracker",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig