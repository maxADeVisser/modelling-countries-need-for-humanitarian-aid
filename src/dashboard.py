"""
To run dashboard lcoally, install streamlit:
pip install streamlit

Then run dashboard.py from commandline like this:
streamlit run dashboard.py

"""

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
    
    #numeric_columns = df.select_dtypes(include=['int8','int16','int32','int64', 'float16', 'float32', 'float64']).columns
    styler = df.style
    
    styler\
        .set_caption("INSERT CAPTION")\
        .background_gradient(axis="rows", cmap="Blues", subset=['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'])\
        .set_table_styles([cell_hover, row_hover])\
        .format({
            "exports": "{:}%",
            "health": "{:}%",
            "imports": "{:}%",
            })\
        .bar(subset=["inflation"], align="zero", color=["red", "lightgreen"]) # bar chart
        #.highlight_max(color="red", subset=numeric_columns)\
        #.highlight_min(color="lightgreen", subset=numeric_columns)\
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
    
    # sidebar
    feature_selection = st.sidebar.radio("Select a feature", features)
    st.title(f"Inspecting {feature_selection} attribute")
    country_selection = st.sidebar.multiselect("Select a country(s)", 
                                               countries, 
                                               default=countries[:5])
    
    selection_df = df.loc[country_selection].sort_values(feature_selection,ascending=True)
    

    value_selection = st.sidebar.slider(label=f"Select a value for {feature_selection}", 
                                            min_value=int(selection_df[feature_selection].min()),
                                            max_value=int(selection_df[feature_selection].max()),
                                            value=int(selection_df[feature_selection].mean()),
                                            )
    
    # TODO make a slider for slider that changes for according to feature selected (min/max values).
    # When moving the slider, the 5 countries closets to the slider value on both sides should be selected.
    # filtered_df = df.iloc[(df[feature_selection]-value_selection).abs().argsort()[:5]]
    # st.dataframe(filtered_df)
    

    country_s = ""
    for country in country_selection:
        country_s += f"{country}, "
    
    with st.expander(f"Countries selected for exploration:"):
        st.write(country_s)

    fig = px.bar(selection_df, 
                    x=selection_df.index, 
                    y=feature_selection)
    st.plotly_chart(figure_or_data=fig, 
                        theme="streamlit", 
                        use_container_width=True)
    
    st.write(f"Dataframe sorted ascending by {feature_selection}:")
    st.dataframe(style_df(selection_df),
                 use_container_width=True) # TODO style this dataframe with correct formating and units
