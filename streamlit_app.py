import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Configuración de la página ---
st.set_page_config(page_title="TEO App", layout="wide")

# --- Funciones Matemáticas ---
def max_min_composition(matrix1, matrix2):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    if cols1 != rows2:
        return None
    result_matrix = np.zeros((rows1, cols2))
    for i in range(rows1):
        for k in range(cols2):
            max_val = 0
            for j in range(cols1):
                min_val = min(matrix1[i, j], matrix2[j, k])
                if min_val > max_val:
                    max_val = min_val
            result_matrix[i, k] = max_val
    return result_matrix

def subtract_matrices(matrix_m2, matrix_m1):
    return np.maximum(0, matrix_m2 - matrix_m1)

# --- Funciones de Visualización ---
def draw_incidence_graph(matrix, source_labels, target_labels, title):
    G = nx.DiGraph()
    source_nodes = [f"S_{label}" for label in source_labels]
    target_nodes = [f"T_{label}" for label in target_labels]
    
    G.add_nodes_from(source_nodes, bipartite=0)
    G.add_nodes_from(target_nodes, bipartite=1)
    
    pos = {}
    pos.update((node, (1, -i)) for i, node in enumerate(source_nodes))
    pos.update((node, (2.5, -i)) for i, node in enumerate(target_nodes))
    
    edge_labels = {}
    for i, s_label in enumerate(source_labels):
        s_node = source_nodes[i]
        for j, t_label in enumerate(target_labels):
            t_node = target_nodes[j]
            weight = matrix[i, j]
            if weight > 0:
                G.add_edge(s_node, t_node, weight=weight)
                edge_labels[(s_node, t_node)] = f"{weight:.2f}"
                
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='skyblue', node_size=1000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='lightgreen', node_size=1000, ax=ax)
    
    labels = {node: node.split('_')[1] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8, ax=ax)
    
    plt.title(title)
    plt.axis('off')
    return fig

# --- Interfaz de Usuario (UI) ---
st.title("Teoría de los Efectos Olvidados (TEO)")
st.markdown("Plataforma interactiva para el análisis de incidencias indirectas.")

modo = st.radio("Seleccione el tipo de análisis:", ["1. Matriz Cuadrada (Auto-incidencia)", "2. Dos Matrices (A -> B -> C)"])

if "1. Matriz Cuadrada" in modo:
    st.header("Análisis de Matriz Única")
    archivo_m1 = st.file_uploader("Suba su matriz CSV (A -> A)", type=['csv'])
    
    if archivo_m1:
        df = pd.read_csv(archivo_m1, index_col=0)
        st.write("### Matriz Original (M1)")
        st.dataframe(df)
        
        if st.button("Calcular Efectos Olvidados"):
            matriz_np = df.values
            etiquetas = list(df.index)
            
            # Cálculos
            m2 = max_min_composition(matriz_np, matriz_np)
            efectos = subtract_matrices(m2, matriz_np)
            
            # Resultados M2
            st.write("### Matriz de Segundo Orden (M2)")
            df_m2 = pd.DataFrame(m2, index=etiquetas, columns=etiquetas)
            st.dataframe(df_m2)
            st.pyplot(draw_incidence_graph(m2, etiquetas, etiquetas, "Red de 2do Orden"))
            
            # Resultados Efectos Olvidados
            st.write("### Matriz de Efectos Olvidados (M2 - M1)")
            df_efectos = pd.DataFrame(efectos, index=etiquetas, columns=etiquetas)
            st.dataframe(df_efectos)
            
            # Modo Inspector Integrado
            st.divider()
            st.subheader("Modo Inspector")
            elemento = st.selectbox("Seleccione un elemento para inspeccionar sus efectos olvidados (Causa ->):", etiquetas)
            if elemento:
                st.write(df_efectos.loc[elemento][df_efectos.loc[elemento] > 0].sort_values(ascending=False))

elif "2. Dos Matrices" in modo:
    st.header("Análisis Encadenado")
    col1, col2 = st.columns(2)
    with col1:
        archivo_m1 = st.file_uploader("Sube Matriz 1 (A -> B)", type=['csv'])
    with col2:
        archivo_m2 = st.file_uploader("Sube Matriz 2 (B -> C)", type=['csv'])
        
    if archivo_m1 and archivo_m2:
        df1 = pd.read_csv(archivo_m1, index_col=0)
        df2 = pd.read_csv(archivo_m2, index_col=0)
        
        st.write("Matrices cargadas correctamente. Verifique que las dimensiones internas coincidan.")
        if st.button("Calcular Composición (A -> C)"):
            resultado = max_min_composition(df1.values, df2.values)
            if resultado is not None:
                df_res = pd.DataFrame(resultado, index=df1.index, columns=df2.columns)
                st.write("### Matriz Resultante")
                st.dataframe(df_res)
                st.pyplot(draw_incidence_graph(resultado, list(df1.index), list(df2.columns), "Red Resultante"))
            else:
                st.error("Las dimensiones de las matrices no son compatibles.")
