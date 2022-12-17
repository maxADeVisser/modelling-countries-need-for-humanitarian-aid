import streamlit as st
import pandas as pd
import os
import plotly_express as px

def style_df(df):
    """Style dataframe"""
    cell_hover = {
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
    }
    row_hover = {
        'selector': 'tr:hover',
        'props': [('background-color', '#ffffb3')]
    }
    
    numeric_columns = df.select_dtypes(include=['int8','int16','int32','int64', 'float16', 'float32', 'float64']).columns
    styler = df.style
    
    styler\
        .set_caption("INSERT CAPTION")\
        .background_gradient(axis="rows", cmap="Blues", subset=['child_mort', 'exports', 'health', 'imports', 'income', 'life_expec', 'total_fer', 'gdpp'])\
        .set_table_styles([cell_hover, row_hover])\
        .format({
            "exports": "{:}%",
            "health": "{:}%",
            "imports": "{:}%",
            })\
        .highlight_max(color="red", subset=numeric_columns)\
        .highlight_min(color="lightgreen", subset=numeric_columns)\
        .bar(subset=["inflation"], align="zero", color=["red", "lightgreen"]) # bar chart
        #.highlight_quantile(q_left=0.25, q_right=0.75, subset="imports", color="red") # highlight quantiles
        #.highlight_between(left=70, right=80, subset="life_expec", color="yellow") # highlight range of values

    return styler

def feature_mapping(features: list):
    return features.map({
        "child_mort": "Child Mortality",
        "exports": "Exports",
        "health": "Health",
        "imports": "Imports",
        "income": "Income",
        "inflation": "Inflation",
        "life_expec": "Life Expectancy",
        "total_fer": "Total Fertility",
        "gdpp": "GDP per capita"
    })

# st.markdown(MARKDOWNSTRING)
# st.header("Header")
# st.latex(LATEXSTRING)


if __name__ == "__main__":    
    DIR = os.path.dirname(__file__)

    df = pd.read_csv(f"{DIR}/../data/country-data.csv").set_index("country")
    countries = list(df.index)
    features = df.columns
    
    st.title("Country Data Dashboard")
    
    #left, right  = st.columns(2)
    
    country_selection = st.sidebar.multiselect("Select a country(s)", countries, 
                                               default=countries[:3])
    feature_selection = st.sidebar.radio("Select a feature", features)
    
    if country_selection:
        country_s = ""
        for country in country_selection:
            country_s += f"{country}, "
            
        selection_df = df.loc[country_selection]
        
        st.write(f"Selected countries: {country_s}")
        #right.dataframe(selection_df) # TODO style this dataframe with correct formating and units

        
        fig = px.bar(selection_df, x=selection_df.index, y=feature_selection)
        st.plotly_chart(figure_or_data=fig, 
                          theme="streamlit")
        
        st.write("Here is the dataframe (sorted by child_mort):")
        st.dataframe(selection_df.sort_values("child_mort"))
