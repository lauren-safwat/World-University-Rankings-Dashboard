#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/lauren-safwat/World-University-Rankings-Dashboard/blob/main/World_University_Rankings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_trich_components as dtc

import plotly.express as px
import plotly.graph_objs as go

from itertools import cycle


# In[5]:


unis = pd.read_csv('dataset/World_University_Rankings.csv')


# In[6]:


unis.head(10)


# In[7]:


unis.info()


# In[8]:

assets_path = os.getcwd() +'/assets'
app = Dash(__name__, assets_folder=assets_path, external_stylesheets=[dbc.themes.BOOTSTRAP, app.get_asset_url('css/style.css')], suppress_callback_exceptions=True)


# In[9]:


unis[unis['Type']=='Public'].count()['University']


# # **Distribution of universities across the world page**

# In[10]:


uniban = [
    dbc.CardHeader(style = {'background-color':'#045275'}),
    dbc.CardBody(
        [
            html.P("Universities"),
            html.H5(
                unis['University'].nunique(),
            ),
        ]
    )
]

countryban = [
    dbc.CardHeader(style = {'background-color':'#2d6a7e'}),
    dbc.CardBody(
        [
            html.P("Countries"),
            html.H5(
                unis['Country'].nunique(),
            ),
        ]
    )
]

privateban = [
    dbc.CardHeader(style = {'background-color':'#64bea1'}),
    dbc.CardBody(
        [
            html.P("Private Universities"),
            html.H5(
                unis[unis['Type']=='Private'].count()['University'],
            ),
        ]
    )
]

publicban = [
    dbc.CardHeader(style = {'background-color':'#b0e3a5'}),
    dbc.CardBody(
        [
            html.P("Public Universities"),
            html.H5(
                unis[unis['Type']=='Public'].count()['University'],
            ),
        ]
    )
]


# In[11]:


world = dbc.Container([
    dbc.Row(html.H1('🎓 Distribution of Universities Across the World'),id='page_title'),
     dbc.Row(
    [
        dbc.Col(dbc.Card(uniban, outline=True)),
        dbc.Col(dbc.Card(countryban,  outline=True)),
        dbc.Col(dbc.Card(privateban,  outline=True)),
       dbc.Col(dbc.Card(publicban,  outline=True)),
    ],
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='map', figure={}), width=9),
        dbc.Col(html.Div(id='table', style={'maxHeight': '410px', 'overflowY': 'scroll'}),width=3)
    ], id='map_table'),

    dbc.Row([
        dbc.Col(dcc.Slider(id='mapSlider',
                  min=2017,
                  max=2022,
                  value=2022,
                  step=None,
                  marks={i: str(i) for i in range(2017, 2023)}
                  ), width=6),
             
        dbc.Col(dcc.Dropdown([str(region) for region in unis['Region'].unique()], placeholder='Select Region', id='map_region'), width=2),
        dbc.Col(dcc.Dropdown(placeholder='Select Country', id='map_country'),width=2),
        dbc.Col(dcc.Checklist([
            {'label': ' Private Universities', 'value': 'Private'},
            {'label': ' Public Universities', 'value': 'Public'},
        ],
        id='checklist'
      ),width=2)
    ],id='controls_row'),
    

    dbc.Row([
        dbc.Col(
            [
             dbc.Row(dcc.Graph(id='bar_uniCount', figure={})),],
            width=4
        ),

        dbc.Col(
            dbc.Row(dcc.Graph(id='bar_intCount', figure={})),
            width=4
        ),
        dbc.Col(
            dbc.Row(dcc.Graph(id='pie_type', figure={})),
            width=4
        )
      ]),
    
    ])


# In[12]:


@app.callback(
    Output('map_country', 'options'),
    Output('map', 'figure'),
    Output("table", "children"),
    Input('mapSlider', 'value'),
    Input('map_region', 'value'),
    Input('map_country', 'value'),
    Input('checklist', 'value')
)


def updateMap(year, region, country,priv):
    year = 'Rank_' + str(year)
    df = unis[unis[year] > 0].sort_values(year, axis=0)
    print
    zoom = 0.1

    if region:
        zoom = 1.3
        df = df[df['Region'] == region]

    if country:
        zoom = 2.5
        df = df[df['Country'] == country]
    if(priv):
      df=df[df['Type'].isin(priv)]

    
    fig = px.scatter_mapbox(df[:100],
                            lon='Longitude',
                            lat='Latitude',
                            color=year,
                            hover_name='University',
                            hover_data={'Latitude':False, 'Longitude':False, 'Country':True,'City':True},
                            zoom=zoom,
                            mapbox_style='carto-positron',
                            color_continuous_scale=px.colors.sequential.deep_r,
                            )
    
    fig.update_traces(marker = {'size':10, 'opacity':0.5}, selector={'type': 'scattermapbox'})

    countries = [{'label': str(i), 'value':str(i)} for i in df['Country'].unique()]

    table = dbc.Table.from_dataframe(df[[year, 'University']], striped=True, bordered=True, hover=True, responsive=True)
    
    return countries, fig, table


# In[13]:


@app.callback(
    Output('bar_uniCount', 'figure'),
    Input('map_region', 'value'),
    Input('checklist', 'value')
)

def updateBar1(region, priv):
    title = 'University Count in the World'
    x=unis.groupby(['Region'])['University'].count()
    y=x.keys()
    text1 = 'Region'    
    df=unis
    if(priv):
      df=df[df['Type'].isin(priv)]
      x=df.groupby(['Region'])['University'].count()
      y=x.keys()
    if region:
        title = 'University Count in '+str(region)
        df = df[df['Region'] == region]
        x=df.groupby(['Country'])['University'].count()
        y=x.keys()
        text1 = 'Country'
    sortd = sorted(zip(x,y))
    x = [i[0] for i in sortd]
    y = [i[1] for i in sortd]
        
    fig = go.Figure(go.Bar(
                x=list(x),
                y=list(y),
                orientation='h',
                text=x,
                hovertemplate = "<br>"+text1+": %{y} </br> Count:%{text}<extra></extra>",
                marker={
                'color': x,
                'colorscale': 'bluyl'
                }))
    fig.update_layout(title=title,
                      barmode='group',
                      bargap=0.0,
                      bargroupgap=0.0
                     )
    return fig


# In[14]:


@app.callback(
    Output('bar_intCount', 'figure'),
    Input('map_region', 'value'),
    Input('checklist', 'value')
)

def updateBar2(region,priv):
    title = 'International Students in the World'
    x=unis.groupby(['Region'])['International_Students'].sum()
    y=x.keys()
    text1 = 'Region'    
    df=unis
    if(priv):
      df=df[df['Type'].isin(priv)]
      x=df.groupby(['Region'])['International_Students'].sum()
      y=x.keys()

    if region:
        title = 'International Students in '+str(region)
        text1 = 'Country'
        df = df[df['Region'] == region]
        x=df.groupby(['Country'])['International_Students'].sum()
        y=x.keys()
    sortd = sorted(zip(x,y))
    x = [i[0] for i in sortd]
    y = [i[1] for i in sortd]
    
    fig = go.Figure(go.Bar(
                x=list(x),
                y=list(y),
                orientation='h',
                text=x,
                hovertemplate = "<br>"+text1+": %{y} </br> Count:%{text}<extra></extra>",
                marker={
                'color': x,
                'colorscale': 'bluyl'
                }))
    fig.update_layout(title=title,
                      barmode='group',
                      bargap=0.0,
                      bargroupgap=0.0)
    return fig


# In[15]:


@app.callback(
    Output('pie_type', 'figure'),
    Input('map_region', 'value'),
    Input('map_country', 'value')
)

def updatePie(region,country):
    title = 'Public VS Private Universities in the World'
    df=unis
    labels=['Private','Public']  
    text = 'World'      

    if region:
        title = 'Public VS Private Universities in '+region
        df = unis[unis['Region'] == region]
        text = region  
    if country :
      title = 'Public VS Private Universities in '+country
      df = df[df['Country'] == country]
      text=country
    
    values=[df[df['Type']=='Private'].count()['University'],df[df['Type']=='Public'].count()['University']]
    countries = [{'label': str(i), 'value':str(i)} for i in df['Country'].unique()]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                             text=values,
                             hole=.3,
                             marker = dict(colors = ['rgb(148,212,180)','rgb(4,82,117,255)']),
                             textinfo='label+percent',
                             insidetextorientation='radial',
                             hovertemplate = "<br>Type: %{label} </br> Count:%{value}<extra></extra>",
                             )])
    fig.update_layout(title=title)
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                        x=0,
                                        y=-0.12,
                                        showarrow=False,
                                        text=text,
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    return fig


# # **Universities page**

# In[16]:


uni = dbc.Container([
      dbc.Row([
          dbc.Col(html.H1('👩🏻‍🎓 Ranking of Top 5 Universities Over the World', id='page2_title'))
      ]),

      dbc.Row([
          dbc.Col(dcc.Dropdown([str(region) for region in unis['Region'].unique()], placeholder='Select Region',multi=True, id='line_region_uni'),width=4),
          dbc.Col(dcc.Dropdown(placeholder='Select Country',multi=True, id='line_country_uni'),width=4),
          dbc.Col(dcc.Dropdown(placeholder='Select University',multi=True, id='line_uni_uni'),width=4)
      ], className='dropDowns'),  

      dbc.Row([
          html.H3('Ranking of Universites over the Last 6 Years'),
          dcc.Graph(id='line_uni', figure={})
      ]),

      dbc.Row([
          dbc.Col(dtc.Carousel([],
              id='carousel',
              slides_to_scroll=1,
              swipe_to_slide=True,
              autoplay=False,
              arrows=True
          ))
      ])
    
])


# In[17]:


@app.callback(
    Output('line_uni_uni', 'options'),
    Output('line_country_uni', 'options'),
    Output('line_uni', 'figure'),
    Input('line_region_uni', 'value'),
    Input('line_country_uni', 'value'),
    Input('line_uni_uni', 'value')
)

def updateLine(region, country, university):
    df = unis
    
    palette = cycle(px.colors.cyclical.Phase)
    palette2=cycle(px.colors.cyclical.Phase)

    if region :
        df = unis[unis['Region'].isin(region) ]
    if country :
        df = df[df['Country'].isin(country)]
    if not university:
        university = list(df[df['Rank_2022']>0].iloc[0:5, 0].values)
    fig = go.Figure()
    for uni in university:
      x=np.arange(2017, 2023)
      y=df[df['University']==uni].iloc[:,11:17].values.flatten().tolist()
      x = [x[i] for i in range(len(y)) if y[i]>0]
      y = [y[i] for i in range(len(y)) if y[i]>0]
      
      fig.add_trace(go.Scatter(x=x, y=y, name=uni,
                          line_shape='linear',
                          line_color=next(palette),
                          hovertemplate = "<br>Year : %{x} </br> Rank:%{y}<extra></extra>"))
      fig.add_trace(go.Scatter(x=[x[y.index(np.min(y))]], y=[np.min(y)],name=str(min(y)),
                              mode = 'markers',
                              marker_symbol = 'circle',
                              marker_color=next(palette2),
                              marker_size = 10,
                              hovertemplate = "<br>Year : %{x} </br> Rank:%{y}<extra></extra>",
                              showlegend=False))

    fig.update_layout(
        yaxis=dict(
            autorange='reversed',
        ),
        xaxis=dict(
            dtick=1,
            range=[2016.1, 2022.9]
        ),
      legend=dict(
          title_font_family="Times New Roman",
          font=dict(
              family="Courier",
              size=12,
              color="black"
          ),
        
      )
  )
    
    countries = [{'label': str(i), 'value':str(i)} for i in df['Country'].unique()]
    universities = [{'label': str(i), 'value':str(i)} for i in df['University'].unique()]
    return universities, countries, fig



# In[18]:


@app.callback(
    Output('carousel', 'children'),
    Input('line_region_uni', 'value'),
    Input('line_country_uni', 'value'),
    Input('line_uni_uni', 'value')
)

def updateCards(region, country, university):
    df = unis

    if region :
        df = df[df['Region'].isin(region)]
    if country :
        df = df[df['Country'].isin(country)]

    universities = df[df['Rank_2022']>0][0:5]

    if university:  
        universities = df[df['University'].isin(university)]

    labels=['International Students', 'National Students']

    cards = []

    for i in range(universities.shape[0]):

        fig = go.Figure()
        international_students = universities.iloc[i, 9]
        students = universities.iloc[i, 10]
        values = [international_students, students-international_students]

        fig.add_trace(go.Pie(labels=labels, values=values,
                                text=values,
                                hole=.3,
                                  marker = dict(colors = ['rgb(148,212,180)','rgb(4,82,117,255)']),
                                textinfo='percent',
                                insidetextorientation='radial',
                                hovertemplate = "<br>%{label} </br> Count:%{value}<extra></extra>",)
        )

        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))

        uni_data = []
        if universities.iloc[i, 2]:
            uni_data.append(html.Li('City: ' + universities.iloc[i, 2]))
        uni_data.append(html.Li('Country: ' + universities.iloc[i, 1]))
        uni_data.append(html.Li('Region: ' + universities.iloc[i, 3]))
        uni_data.append(html.Li('Type: ' + universities.iloc[i, 6] + ' University'))
       
        card_content = [

            html.A([

                dbc.CardHeader([

                    dbc.Row([

                        dbc.Col(html.Img(id='uni_logo', src = app.get_asset_url('images/' + universities.iloc[i, 0] + '.jpg')), width=2),

                        dbc.Col(html.H5(universities.iloc[i, 0])) 

                    ])

                ], id='card_header')

            ], href=universities.iloc[i, 5], target='_blank'),



            dbc.CardBody([

                html.Ul(uni_data, id='uni_data'),

                dcc.Graph(figure = fig, style={"height": "70%", "width": "100%"})

            ]),

        ]

        cards.append(dbc.Card(card_content, outline=True,id='uni_card'))

    
    return cards



# # **Dashboard Layout**

# In[19]:


app.layout = html.Div(children=[
     dbc.Navbar(children=[
        dbc.Col([
            html.Img(id='logo', src=app.get_asset_url('images/World2.png')),
            html.H1('World University Rankings', id='title')
        ], width=8),

        dbc.Col(dbc.Tabs(id='tabs', active_tab='tab-1', children=[
            dbc.Tab(label="World Overview", tab_id='tab-1'),
            dbc.Tab(label="Universities", tab_id='tab-2')
        ]))
     ], id='nav', sticky = 'top'),
   
     html.Div(id='content')
], className='dashboard')


# In[20]:


@app.callback(
    Output('content', 'children'),
    Input('tabs', 'active_tab'),
)

def display_content(active_tab):
    if active_tab == 'tab-1':
        return world
    return uni


# In[21]:

server = app.server

