from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import base64
def embeddable_image(data):
    img_data = (data * 255).astype(np.uint8).reshape(100,100)
    image = Image.fromarray(img_data, mode='L')
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
from bokeh.resources import CDN
from bokeh.embed import file_html
output_notebook()

def show_embedding(emb,alldat,res,savefig=False,showfig=True):
    emb.rename(columns={ emb.columns[0]: "x", emb.columns[1]: "y"}, inplace = True)
    emb['turns'] = [str(x) for x in alldat.turns]
    emb['index'] = [str(x) for x in alldat.index]
    emb['image'] = list(map(embeddable_image, res))

    datasource = ColumnDataSource(emb)
    #color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in digits.target_names],
    #palette=Spectral10)

    color_mapping = CategoricalColorMapper(factors=np.unique(emb['specie']).tolist(), 
                                          palette=['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B'])

    plot_figure = figure(title='UMAP projection on diffusion components of all tracks',
                         plot_width=900,plot_height=900,tools=('pan, wheel_zoom, reset'))

    plot_figure.title.text_font_size = '18pt'

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px'>@specie</span>
            <span style='font-size: 16px'>turns: @turns</span>
            <span style='font-size: 16px'>idx: @index</span>
        </div>
    </div>
    """))
    plot_figure.circle('x','y',source=datasource,color=dict(field='specie', transform=color_mapping),
                       line_alpha=0.6,fill_alpha=0.6,size=4)
    
    if savefig:
        output_file("UMAP_exploration.html",mode="inline")
        save(plot_figure,resources="inline")
    
    if showfig:
        show(plot_figure)
    
from eyediagram._brescount import bres_segments_count
def rasterizelines(ldat,dim=100):

    df=ldat[["Position X","Position Y"]]    
    df.loc[:,"Position X"]=df.loc[:,"Position X"].values-min(df.loc[:,"Position X"].values)
    df.loc[:,"Position Y"]=df.loc[:,"Position Y"].values-min(df.loc[:,"Position Y"].values)

    df=df/max(max(df["Position X"]),max(df["Position Y"]))
    df=df*dim
    df=df.drop_duplicates(subset=['Position X', 'Position Y'])

    starts=df.iloc[::1, :2]
    ends=df.iloc[1::1, :2]
    starts.index=range(0,len(starts))
    ends.index=range(0,len(starts)-1)
    df=pd.concat([starts,ends],1)
    img = np.zeros((dim, dim), dtype=np.int32)
    sticks=df.loc[1:,:].values.astype(np.int32)[:-1,:]


    bres_segments_count(sticks, img)
    img=img>0
    return(img.flatten().astype(int))