import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import namedtuple
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn import metrics
import lightgbm as lgb
import streamlit as st


def make_binned_averages(data, feature, target, bw, bmin=0, bmax=0):
    sf = data.copy()

    if bmax == 0:
        bmax = max(sf[feature])

    bins = np.arange(bmin, bmax + bw, step=bw)
    n, b = np.histogram(sf[feature], bins=bins)
    m = (b[1:] + b[:-1]) / 2

    sf.loc[:, 'binrange'] = pd.cut(sf[feature], bins=bins)

    g = sf.groupby('binrange')[target]

    nr = g.value_counts()[g.value_counts().index.get_level_values(1) == 0].tolist()
    nc = n - nr
    ntup = namedtuple("BinnedData", ["r", "means", "churned", "retained", "bins"])

    return ntup(m, g.mean(), nc, nr, bins)


if __name__ == '__main__':

    # st. set_page_config(layout="wide")

    df = pd.read_csv('ibm_telco_customer_churn.csv')

    check_nans = df.isna().sum()
    print(check_nans[check_nans != 0].to_string())

    churn_fraction = df['Churn Value'].mean()

    fig = px.histogram(x=df['Churn Label'], color=df['Churn Label'],
                       title=f'Churned Percent = {churn_fraction*100:.3f}%',)

    st.plotly_chart(fig, theme="streamlit")

    df = df[df['Churn Reason'] != 'Deceased']

    wcfig, wcax = plt.subplots()
    text = df['Churn Reason'].dropna().str.cat()

    wc_kws = dict(collocations=False, width=1920, height=1080, background_color=None, mode='RGBA')

    word_cloud = WordCloud(**wc_kws).generate(text)

    wcfig = px.imshow(word_cloud, aspect='auto', )
    wcfig.update_xaxes(visible=False)
    wcfig.update_yaxes(visible=False)

    sfig = make_subplots(rows=1, cols=2)

    histfig = go.Histogram(x=df['Churn Reason'])

    sfig.add_trace(histfig, row=1, col=1)
    sfig.add_trace(wcfig.data[0], row=1, col=2)

    sfig.update_xaxes(row=1, col=1, categoryorder='total descending')
    sfig.update_xaxes(row=1, col=2, visible=False)
    sfig.update_yaxes(row=1, col=2, visible=False)

    st.plotly_chart(sfig, theme="streamlit", use_container_width=True)

    temporal_features = ["Monthly Charges", "Tenure Months", "Total Charges"]
    target = 'Churn Value'

    df['Total Charges'].replace(' ', 0, inplace=True)
    df['Total Charges'] = df['Total Charges'].astype(float)

    f0d = make_binned_averages(df, temporal_features[0], target, bw=5, bmin=15, bmax=120)
    f1d = make_binned_averages(df, temporal_features[1], target, bw=2, bmin=-0.5)

    # h0dat = px.histogram(x=f0d.feature, color=f0d.target).data
    # h1dat = px.histogram(x=f1d.feature, color=f1d.target).data

    tpfig = make_subplots(rows=2, cols=2)

    tpfig.add_trace(go.Scatter(x=f0d.r, y=f0d.means, mode='lines'), row=1, col=1)
    tpfig.add_trace(go.Scatter(x=f1d.r, y=f1d.means, mode='lines'), row=2, col=1)

    tpfig.add_trace(go.Bar(x=f0d.bins, y=f0d.retained, offsetgroup=0), row=1, col=2)
    tpfig.add_trace(go.Bar(x=f0d.bins, y=f0d.churned, base=f0d.retained, offsetgroup=0),
                    row=1, col=2)

    tpfig.add_trace(go.Bar(x=f1d.bins, y=f1d.retained, offsetgroup=0), row=2, col=2)
    tpfig.add_trace(go.Bar(x=f1d.bins, y=f1d.churned, base=f1d.retained, offsetgroup=0),
                    row=2, col=2)

    st.plotly_chart(tpfig, use_container_width=True)

    binary_features = ["Senior Citizen", "Partner", "Dependents", "Paperless Billing", "Phone Service"]

    online_features = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',]

    target = 'Churn Value'

    h = df.loc[:, binary_features].apply(lambda col: col.value_counts()).T.fillna(0)
    hf = df.loc[:, binary_features].apply(lambda col: df.groupby(col)[target].mean()).fillna(0).T

    bpfig = make_subplots(rows=1, cols=2)

    bpfig.add_trace(go.Bar(y=h.loc[:, 'No'], offsetgroup=0), row=1, col=1)
    bpfig.add_trace(go.Bar(y=h.loc[:, 'Yes'], base=h['No'], offsetgroup=0), row=1, col=1)

    bpfig.add_trace(go.Bar(y=hf.loc[:, 'Yes']), row=1, col=2)
    bpfig.add_trace(go.Bar(y=hf.loc[:, 'No']), row=1, col=2)


    st.plotly_chart(bpfig, use_container_width=True)