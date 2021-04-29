# Chart-Studio is streamlined to host plots generated from plotly

# Setting up offline mode #

## Use plotly.io.write_html() to create and standalone HTML that is saved locally and opened inside your web browser.

## Use plotly.io.show() when working offline in a Jupyter Notebook to display the plot in the notebook.

## Example script:

# import plotly.graph_objects as go
# import plotly.io as pio

# fig = go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1]))
# fig.update_layout(title_text='hello world')
# pio.write_html(fig, file='hello_world.html', auto_open=True)

################

# For ONLINE use only
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

data = dict(type='choropleth', locations=['AZ','CA','NY'],locationmode='USA-states',colorscale='Portland',text=['text 1','text 2','text 3'],z=[1.0,2.0,3.0],colorbar={'title':'Colorbar Title Goes Here'})

layout = dict(geo={'scope':'usa'})
choromap = go.Figure(data=[data],layout=layout)

py.iplot(choromap,filename='basic-line',auto_open=True)
# ^^ this plotting method will generate the same map as the method below

plot(choromap)

df = pd.read_csv('C:/Users/sjrus/Desktop/Machine Learning/1. Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2011_US_AGRI_Exports')

df.head()

data = dict(type='choropleth',colorscale='plasma',locations=df['code'],locationmode='USA-states',z=df['total exports'],text=df['text'],colorbar={'title':'Millions USD'},marker=dict(line=dict(color='rgb(255,255,255)',width=2)))

layout=dict(title='2011 US Agriculture Exports by State',geo=dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'))

choromap2 = go.Figure(data=[data],layout=layout)

plot(choromap2)