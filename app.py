import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from datetime import datetime as dt, timedelta
from dateutil.relativedelta import relativedelta

import requests
from io import BytesIO

# Podesi radni direktorijum i putanju za funkcije
#sys.path.append(r'\\backupsrv.mozzart.co.yu\Analitika\SAPB\Danka\1. Analiza igraca i bonusa\0. FunkcijeIgraci')
#os.chdir(r'C:\Users\aleksandar.radojevic\Desktop\CLTV_app')


def get_data(cohorte):
    query = f"""
        SELECT * FROM acar.member_cltv
        WHERE cohorte = '{cohorte}'
    """
    data = nf.sqlQuery(query)
    return data

def transform_cohorte_data(data_cohorte, igrac_id_col='mem_id'):
    """
    Transformiše DataFrame sa podacima o igračima u dugi format, gde se za svakog igrača
    generišu podaci po proizvodima, modelima i KPI-jima, uključujući stvarne vrednosti.
    """
    if igrac_id_col not in data_cohorte.columns:
        raise Exception(f"Nema kolone za identifikaciju igrača ({igrac_id_col}) u data_cohorte!")

    proizvodi = ['sk', 'sl', 'v', 'k', 'b', 'l6']
    modeli = ['v1', 'v2', 'v3']
    kpi_list = ['ggr1', 'amount_cum', 'amount_real_cum']
    base_columns = [igrac_id_col, 'dt_date_pred', 'mesec_predikcije', 'cohorte', 'RFM_cat']
    df_list = []

    # 1. Obrada po proizvodima i modelima (v1, v2, v3)
    for prod in proizvodi:
        for model in modeli:
            for kpi in kpi_list:
                df_temp = data_cohorte[base_columns].copy()
                df_temp['proizvod'] = prod
                df_temp['model'] = model
                df_temp['KPI'] = kpi

                col_name = {
                    'ggr1': f'model_{model}_{prod}',
                    'amount_cum': f'model_amount_{model}_{prod}',
                    'amount_real_cum': f'model_amount_real_{model}_{prod}'
                }.get(kpi, None)

                df_temp['vrednost'] = data_cohorte.get(col_name, np.nan)

                real_col_name = {
                    'ggr1': f'ggr1_{prod}_cum_realno',
                    'amount_cum': f'amount_{prod}_cum_realno',
                    'amount_real_cum': f'amount_real_{prod}_cum_real'
                }.get(kpi, None)

                df_temp['realno'] = data_cohorte.get(real_col_name, np.nan)

                df_list.append(df_temp)

    # 2. Dodavanje "realno" kao model po proizvodima
    for prod in proizvodi:
        for kpi in kpi_list:
            df_temp = data_cohorte[base_columns].copy()
            df_temp['proizvod'] = prod
            df_temp['model'] = 'realno'
            df_temp['KPI'] = kpi

            col_name = {
                'ggr1': f'ggr1_{prod}_cum_realno',
                'amount_cum': f'amount_{prod}_cum_realno',
                'amount_real_cum': f'amount_real_{prod}_cum_real'
            }.get(kpi, None)

            df_temp['vrednost'] = data_cohorte.get(col_name, np.nan)
            df_temp['realno'] = df_temp['vrednost']

            df_list.append(df_temp)

    # 3. Obrada za total po modelima
    for model in modeli:
        for kpi in kpi_list:
            df_temp = data_cohorte[base_columns].copy()
            df_temp['proizvod'] = 'total'
            df_temp['model'] = model
            df_temp['KPI'] = kpi

            col_name = {
                'ggr1': f'model_{model}',
                'amount_cum': f'model_amount_{model}',
                'amount_real_cum': f'model_amount_real_{model}'
            }.get(kpi, None)

            df_temp['vrednost'] = data_cohorte.get(col_name, np.nan)

            real_col_name = {
                'ggr1': 'ggr1_cum_realno',
                'amount_cum': 'amount_cum_realno',
                'amount_real_cum': 'amount_real_cum_realno'
            }.get(kpi, None)

            df_temp['realno'] = data_cohorte.get(real_col_name, np.nan)

            df_list.append(df_temp)

    # 4. Dodavanje "realno" kao model za total
    for kpi in kpi_list:
        df_temp = data_cohorte[base_columns].copy()
        df_temp['proizvod'] = 'total'
        df_temp['model'] = 'realno'
        df_temp['KPI'] = kpi

        col_name = {
            'ggr1': 'ggr1_cum_realno',
            'amount_cum': 'amount_cum_realno',
            'amount_real_cum': 'amount_real_cum_realno'
        }.get(kpi, None)

        df_temp['vrednost'] = data_cohorte.get(col_name, np.nan)
        df_temp['realno'] = df_temp['vrednost']

        df_list.append(df_temp)

    data_final_long = pd.concat(df_list, ignore_index=True)
    return data_final_long

def prikazi_filtere_i_tabelu(data_final):
    st.sidebar.header("Filteri")

    def safe_unique(df, kolona):
        if kolona in df.columns:
            return ["Sve"] + sorted(df[kolona].dropna().unique().tolist())
        else:
            return ["Sve"]

    cohorte_list = safe_unique(data_final, 'cohorte')
    cohorte = st.sidebar.selectbox("Cohorta", cohorte_list)

    # Prvo filtriraj po cohorti, da ne radimo nepotrebno na celom datasetu
    if cohorte != "Sve":
        data_final = data_final[data_final['cohorte'] == cohorte]

    data_final_long = transform_cohorte_data(data_final, igrac_id_col='mem_id')

    mesec_predikcije_list = safe_unique(data_final_long, 'mesec_predikcije')
    proizvod_list = safe_unique(data_final_long, 'proizvod')
    kpi_list = safe_unique(data_final_long, 'KPI')

    mesec_predikcije = st.sidebar.selectbox("Mesec predikcije", mesec_predikcije_list)
    proizvod = st.sidebar.selectbox("Proizvod", proizvod_list)
    kpi = st.sidebar.selectbox("KPI", kpi_list)

    # Filtriraj podatke
    filtered_data = data_final_long.copy()
    if cohorte != "Sve" and 'cohorte' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['cohorte'] == cohorte]
    if mesec_predikcije != "Sve" and 'mesec_predikcije' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['mesec_predikcije'] == mesec_predikcije]
    if proizvod != "Sve" and 'proizvod' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['proizvod'] == proizvod]
    if kpi != "Sve" and 'KPI' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['KPI'] == kpi]

    if filtered_data.empty:
        st.warning("Nema podataka za izabrane parametre!")
        return {
            'cohorte': cohorte,
            'mesec_predikcije': mesec_predikcije,
            'proizvod': proizvod,
            'kpi': kpi
        }

    # --- PRVA TABELA: agregatno po cohorti, dt_date_pred, model (v1, v2, v3, realno) + razlike ---
    st.subheader("Agregatne vrednosti po cohorti, datumu i modelu (v1, v2, v3, realno) + razlike")
    modeli_prikaz = ['v1', 'v2', 'v3', 'realno']
    agg_df = (
        filtered_data[filtered_data['model'].isin(modeli_prikaz)]
        .groupby(['cohorte', 'dt_date_pred', 'model'])[['vrednost', 'realno']]
        .sum()
        .reset_index()
    )
    # Pivotiranje da dobijemo kolone v1, v2, v3, realno
    agg_pivot = agg_df.pivot_table(
        index=['cohorte', 'dt_date_pred'],
        columns='model',
        values='vrednost',
        aggfunc='sum'
    ).reset_index()

    # Dodaj i realno kao posebnu kolonu (ako nije već u vrednostima)
    if 'realno' not in agg_pivot.columns and 'realno' in agg_df['model'].unique():
        realno_df = agg_df[agg_df['model'] == 'realno'][['cohorte', 'dt_date_pred', 'realno']]
        realno_df = realno_df.rename(columns={'realno': 'realno'})
        agg_pivot = pd.merge(agg_pivot, realno_df, on=['cohorte', 'dt_date_pred'], how='left')

    # Dodaj razlike v1, v2, v3 u odnosu na realno
    prikaz_razlike = st.sidebar.radio("Prikaži razliku kao:", ("Broj", "Procenat"), horizontal=True)
    for col in ['v1', 'v2', 'v3']:
        if col in agg_pivot.columns and 'realno' in agg_pivot.columns:
            if prikaz_razlike == "Broj":
                agg_pivot[f"{col}_diff"] = agg_pivot[col] - agg_pivot['realno']
            else:
                # Procenat: (model - realno) / realno * 100
                agg_pivot[f"{col}_diff"] = np.where(
                    agg_pivot['realno'] == 0,
                    np.nan,
                    (agg_pivot[col] - agg_pivot['realno']) / agg_pivot['realno'] * 100
                )

    # Formatiranje brojeva
    for col in modeli_prikaz:
        if col in agg_pivot.columns:
            agg_pivot[col] = agg_pivot[col].apply(lambda x: "" if pd.isna(x) else f"{int(round(x)):,}".replace(",", "."))
    for col in ['v1_diff', 'v2_diff', 'v3_diff']:
        if col in agg_pivot.columns:
            if prikaz_razlike == "Broj":
                agg_pivot[col] = agg_pivot[col].apply(lambda x: "" if pd.isna(x) else f"{int(round(x)):,}".replace(",", "."))
            else:
                agg_pivot[col] = agg_pivot[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}%")

    # Prikaz tabele sa razlikama
    st.dataframe(agg_pivot, use_container_width=True)

    # --- GRAFIK: v1, v2, v3, realno po dt_date_pred ---
    st.subheader("Grafik: v1, v2, v3 i realno po datumu predikcije")
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for col in ['v1', 'v2', 'v3', 'realno']:
            if col in agg_pivot.columns:
                # Pretvori vrednosti nazad u brojeve (ukloni formatiranje)
                vrednosti = pd.to_numeric(agg_pivot[col].replace("", np.nan).str.replace(".", "", regex=False), errors='coerce')
                fig.add_trace(go.Scatter(
                    x=agg_pivot['dt_date_pred'],
                    y=vrednosti,
                    mode='lines+markers',
                    name=col
                ))
        fig.update_layout(
            yaxis_title="Vrednost",
            xaxis_title="Datum predikcije",
            legend_title="Model",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Ne mogu da prikažem grafik: {e}")

    # --- DRUGA TABELA: po mem_id, cohorte, dt_date_pred, v1, v2, v3, realno ---
    st.subheader("Vrednosti po igraču (mem_id), cohorti i datumu za izabrane filtere")
    try:
        pivot_df = (
            filtered_data[filtered_data['model'].isin(modeli_prikaz)]
            .pivot_table(
                index=['mem_id', 'cohorte', 'dt_date_pred'],
                columns='model',
                values='vrednost',
                aggfunc='sum'
            )
        )
    except Exception as e:
        st.error(f"Greška u pivotiranju po mem_id: {str(e)}")
        return {
            'cohorte': cohorte,
            'mesec_predikcije': mesec_predikcije,
            'proizvod': proizvod,
            'kpi': kpi
        }

    prikaz_df = pivot_df.reset_index()
    prikaz_df = prikaz_df[['mem_id', 'cohorte', 'dt_date_pred'] + [col for col in modeli_prikaz if col in prikaz_df.columns]]

    # Formatiraj brojeve za prikaz
    for col in modeli_prikaz:
        if col in prikaz_df.columns:
            prikaz_df[col] = prikaz_df[col].apply(lambda x: "" if pd.isna(x) else f"{int(round(x)):,}".replace(",", "."))

    st.dataframe(prikaz_df, use_container_width=True)

    return {
        'cohorte': cohorte,
        'mesec_predikcije': mesec_predikcije,
        'proizvod': proizvod,
        'kpi': kpi
    }

# Podesi izgled aplikacije i sidebar
st.set_page_config(page_title='CLTV', page_icon=Image.open('slika1.jpg'), layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>CLTV - vrednost igrača</h1>",
    unsafe_allow_html=True
)

st.sidebar.image(
    Image.open('slika1.jpg'),
    caption='cltv - mozzart',
    use_container_width=True
)

@st.cache_data(show_spinner=False)
def kesirani_podaci():
    file_url = "https://www.googleapis.com/drive/v3/files/1tNumrMMqtqdZRaT3xv4P_v1Pi_Cn646T?alt=media&key=AIzaSyCgMSiy_V85irRI2g634p2n4cNg6UNW7YI"
    response = requests.get(file_url)
    data = pd.read_parquet(BytesIO(response.content))
    # data = pd.read_parquet(r'../CLTV_app/data/CLVT - per player - 2025-01 all.parquet')
    return data

data_all = kesirani_podaci()
data_final = data_all.copy()

# Glavna stranica
data_result = prikazi_filtere_i_tabelu(data_final)




