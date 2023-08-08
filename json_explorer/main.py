from io import BytesIO
import streamlit as st
import json
import pandas as pd

from json_explorer.analyzer import Analyzer
from json_explorer.constants import HANDLED_TYPES

st.set_page_config(layout="wide")

f = st.file_uploader("Upload your JSON file.")

if not f:
    st.stop()

data = json.load(f)

def construct_collation(analyzer: Analyzer):
    for type, fields in analyzer._value_lookup.items():
        if type not in HANDLED_TYPES:
            continue
        for key in fields:
            left, right = st.columns(2)
            analyzed = analyzer.collated[key]
            left.markdown(f"# {key}")
            left.markdown(analyzed.stats())
            try:
                right.bokeh_chart(analyzed.chart(key=key))
            except NotImplementedError:
                right.warning(f"No Chart Implemented for {type}")
            except ValueError as e:
                right.warning(e)

analyzer = Analyzer(data=data)

st.write(f"Successfully ingested JSON file {f.name} with {len(data)} entries.")

with st.expander("Preview Raw Data..."):
    st.write("Showing preview of the first 5 rows as a table...")
    st.table(pd.read_json(BytesIO(json.dumps(data[:5]).encode())))

    st.write("Showing preview of the first 5 rows as a dict...")
    st.json(data[:5])

if st.button("Analyze JSON Data..."):
    analyzer.analyze()

    with st.expander(f"Analyzed data for top level fields..."):
        construct_collation(analyzer=analyzer)

    for name, sub in analyzer.sub_analyzers.items():
        with st.expander(f"Analyzed data for `{name}` fields..."):
            construct_collation(analyzer=sub)

