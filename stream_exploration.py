import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import plotly.tools as ptls
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

    # g = sf.groupby('binrange')[target]

    # nr = g.value_counts()[g.value_counts().index.get_level_values(1) == 0].tolist()
    # nc = n - nr
    # ntup = namedtuple("BinnedData", ["r", "means", "churned", "retained", "bins"])

    # return ntup(m, g.mean(), nc, nr, bins)
    g = sf.groupby('binrange')[target].mean()

    ntup = namedtuple("BinnedData", ["r", "means", "feature", "target", "bins"])

    return ntup(m, g, sf[feature], sf[target], bins)


if __name__ == '__main__':

    st. set_page_config(layout="wide")
    pd.options.plotting.backend = "plotly"

    df = pd.read_csv('ibm_telco_customer_churn.csv')

    check_nans = df.isna().sum()
    print(check_nans[check_nans != 0].to_string())

    df = df[df['Churn Reason'] != 'Deceased']

    tab1, tab2, tab3 = st.tabs(["Churn Reasons", "Charts", "Model Results"])

    # with tab1:
    #     col0, col1 = st.columns(2)
    #     histfig = px.histogram(x=df['Churn Reason']).update_xaxes(categoryorder='total descending')
    #     histfig.update_layout(title='Churn Reasons')
    #     col0.plotly_chart(histfig, use_container_width=True)

    #     text = df['Churn Reason'].dropna().str.cat()

    #     wc_kws = dict(collocations=False, width=1920, height=1080,
    #                   background_color=None, mode='RGBA')

    #     word_cloud = WordCloud(**wc_kws).generate(text)

    #     wcfig = px.imshow(word_cloud, aspect='auto', )
    #     wcfig.update_xaxes(visible=False)
    #     wcfig.update_yaxes(visible=False)
    #     wcfig.update_layout(title='Churn Reason Word Cloud')

    #     col1.plotly_chart(wcfig, use_container_width=True)

    with tab2:
        col0, col1 = st.columns(2)
        churn_fraction = df['Churn Value'].mean()

        fig = px.histogram(x=df['Churn Label'], color=df['Churn Label'],
                           title=f'Churned Percent = {churn_fraction*100:.3f}%',)

        st.plotly_chart(fig, theme="streamlit")

        target = 'Churn Value'

        temporal_features = ["Monthly Charges", "Tenure Months", "Total Charges"]
        binary_features = ["Senior Citizen", "Partner", "Dependents", "Paperless Billing",
                           "Phone Service"]
        multiopt_features = ['Internet Service', 'Contract', 'Payment Method']
        online_features = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                           'Streaming TV', 'Streaming Movies']

        df['Total Charges'].replace(' ', 0, inplace=True)
        df['Total Charges'] = df['Total Charges'].astype(float)

        f0d = make_binned_averages(df, temporal_features[0], target, bw=5, bmin=15, bmax=120)
        f1d = make_binned_averages(df, temporal_features[1], target, bw=2, bmin=-0.5)

        fig0 = px.line(x=f0d.r, y=f0d.means, markers=True)
        fig1 = px.histogram(x=f0d.feature, color=f0d.target, nbins=30)

        col0.plotly_chart(fig0, use_container_width=True)
        col0.plotly_chart(fig1, use_container_width=True)

        fig0 = px.line(x=f1d.r, y=f1d.means, markers=True)
        fig1 = px.histogram(x=f1d.feature, color=f1d.target)

        col1.plotly_chart(fig0, use_container_width=True)
        col1.plotly_chart(fig1, use_container_width=True)

        h = df.loc[:, binary_features].apply(lambda col: col.value_counts()).T.fillna(0)
        k = df.loc[:, binary_features].apply(lambda col: df.groupby(col)[target].mean()).fillna(0).T

        fig0 = h[['Yes', 'No']].plot.bar()
        fig1 = k[['Yes', 'No']].plot.bar().update_layout(barmode='group')

        col0.plotly_chart(fig0, use_container_width=True)
        col0.plotly_chart(fig1, use_container_width=True)

        lst_df = []
        for feature in multiopt_features:
            g = df.loc[:, [feature, 'Churn Value']].groupby(feature).mean().reset_index()
            g.rename(columns={feature: 'feature'}, inplace=True)
            g['topic'] = [feature] * len(g)
            lst_df.append(g)

        cf = pd.concat(lst_df)

        counts = df.loc[:, multiopt_features].apply(pd.Series.value_counts)

        counts = counts.reindex(cf.feature)

        fig0 = counts[multiopt_features].plot.bar()
        fig1 = cf.plot.line(x='feature', y='Churn Value', color='topic', markers=True)

        col1.plotly_chart(fig0, use_container_width=True)
        col1.plotly_chart(fig1, use_container_width=True)

    with tab3:
        col0, col1 = st.columns(2)
        gbm = lgb.Booster(model_file='lgbm_model.pkl')

        z = [[0.41, 0.09], [0.027, 0.47]]
        z = z[::-1]
        xl = ['Not Churned', 'Churned']
        yl = xl[::-1]

        zl = [[str(y) for y in x] for x in z]

        hmapfig = ff.create_annotated_heatmap(z, x=xl, y=yl,
                                              annotation_text=zl, colorscale='Viridis')
        hmapfig.update_layout(title='Confusion Matrix', yaxis_title='True Label',
                              xaxis_title='Predicted Label')
        col0.plotly_chart(hmapfig, use_container_width=True)

        imf = pd.DataFrame(columns=['feature', 'importance'])
        imf.loc[:, 'feature'] = gbm.feature_name()
        imf.loc[:, 'importance'] = gbm.feature_importance(importance_type='gain')
        imf.sort_values(by='importance', ascending=False, inplace=True)

        impfig = px.bar(x=imf.feature.iloc[:15], y=imf.importance.iloc[:15])
        impfig.update_xaxes(categoryorder='total descending')

        impfig.update_layout(title='15 Most Important Features', xaxis_title="Feature",
                             yaxis_title="Importance")
        col1.plotly_chart(impfig, use_container_width=True)
