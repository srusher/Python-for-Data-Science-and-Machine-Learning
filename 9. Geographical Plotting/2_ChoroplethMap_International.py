import chart_studio
chart_studio.tools.set_credentials_file(username='Sjrush27',api_key='cUhMKoXMlMiM71C4NUE3')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

df = pd.read_csv('C:/Users/sjrus/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2014_World_GDP')

df.head()

data = dict(type='choropleth',locations=df['CODE'],z=df['GDP (BILLIONS)'],text=df['COUNTRY'],colorbar={'title':'GDP in Billions USD'})

layout = dict(title='2014 Global GDP',geo=dict(showframe=False,projection={'type':'kavrayskiy7'}))
choromap3 = go.Figure(data=[data],layout=layout)
plot(choromap3)