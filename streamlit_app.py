import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Teoría de los Efectos Olvidados", layout="wide", page_icon="📊")

# --- FUNCIONES MATEMÁTICAS ---
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

def load_and_clean_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.clip(0, 1)
    return df

# --- FUNCIONES DE VISUALIZACIÓN ---
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
                edge_labels[(s_node, t_node)] = f"{weight:.3f}"
                
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='#87CEEB', node_size=1200, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='#98FB98', node_size=1200, edgecolors='black', ax=ax)
    
    labels = {node: node.split('_')[1] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    return fig

def draw_focused_graph(center_node, connected_nodes, weights, title, mode='source'):
    G = nx.DiGraph()
    G.add_node(center_node, layer=0)
    
    edge_labels = {}
    for node, w in zip(connected_nodes, weights):
        G.add_node(node, layer=1)
        if mode == 'source':
            G.add_edge(center_node, node, weight=w)
            edge_labels[(center_node, node)] = f"{w:.3f}"
        else:
            G.add_edge(node, center_node, weight=w)
            edge_labels[(node, center_node)] = f"{w:.3f}"
            
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#FFFACD', edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=25, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    return fig

# --- INTERFAZ DE USUARIO (UI) ---
st.title("🧩 Teoría de los Efectos Olvidados (TEO)")
st.markdown("Automatización de incidencias de orden superior y descubrimiento de efectos ocultos en sistemas complejos.")

modo = st.radio(
    "Seleccione la topología de su análisis:", 
    ["1. Sistemas Cerrados (Matriz Cuadrada / Auto-incidencia)", "2. Sistemas Encadenados (Dos Matrices A -> B -> C)"],
    horizontal=True
)

st.divider()

if "1." in modo:
    st.header("Análisis de Sistemas Cerrados (A → A)")
    archivo_m1 = st.file_uploader("Suba su Matriz de Incidencias (.csv)", type=['csv'])
    
    if archivo_m1:
        df = load_and_clean_csv(archivo_m1)
        labels = list(df.index)
        matriz_np = df.values
        
        st.subheader("1. Matriz Original (Directa)")
        st.dataframe(df.style.format("{:.3f}"))
        
        # Guardamos en memoria si el botón fue presionado
        if st.button("🚀 Procesar Efectos Olvidados", type="primary"):
            st.session_state['procesado_m1'] = True
            
        # Si está en memoria, mostramos todo (esto evita que desaparezca)
        if st.session_state.get('procesado_m1', False):
            m2 = max_min_composition(matriz_np, matriz_np)
            efectos = subtract_matrices(m2, matriz_np)
            
            df_m2 = pd.DataFrame(m2, index=labels, columns=labels)
            df_efectos = pd.DataFrame(efectos, index=labels, columns=labels)
            
            tab1, tab2 = st.tabs(["📊 Matriz de Segundo Orden (M²)", "🔍 Matriz de Efectos Olvidados (M² - M)"])
            
            with tab1:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df_m2.style.format("{:.3f}"))
                with col2:
                    st.pyplot(draw_incidence_graph(m2, labels, labels, "Red de Incidencias (2do Orden)"))
            
            with tab2:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.dataframe(df_efectos.style.format("{:.3f}"))
                with col2:
                    st.pyplot(draw_incidence_graph(efectos, labels, labels, "Red de Efectos Olvidados"))
            
            # --- MODO INSPECTOR ---
            st.divider()
            st.header("🕵️‍♂️ Modo Inspector Focalizado")
            st.markdown("Aísle un elemento para analizar su comportamiento como Causa (emisor) o como Efecto (receptor).")
            
            elemento = st.selectbox("Seleccione el elemento a inspeccionar:", ["-- Seleccione --"] + labels)
            
            if elemento != "-- Seleccione --":
                colA, colB = st.columns(2)
                
                with colA:
                    st.markdown(f"### '{elemento}' como CAUSA")
                    valores_causa = df_efectos.loc[elemento]
                    df_causa = pd.DataFrame({'Efecto': labels, 'Incidencia': valores_causa})
                    df_causa = df_causa[df_causa['Incidencia'] > 0].sort_values(by='Incidencia', ascending=False)
                    
                    if not df_causa.empty:
                        st.dataframe(df_causa.style.format("{:.3f}"), hide_index=True)
                        st.pyplot(draw_focused_graph(elemento, df_causa['Efecto'].tolist(), df_causa['Incidencia'].tolist(), f"Impactos indirectos de '{elemento}'", mode='source'))
                    else:
                        st.info(f"'{elemento}' no genera efectos olvidados hacia otros nodos.")

                with colB:
                    st.markdown(f"### '{elemento}' como EFECTO")
                    valores_efecto = df_efectos[elemento]
                    df_efecto = pd.DataFrame({'Causa': labels, 'Incidencia': valores_efecto})
                    df_efecto = df_efecto[df_efecto['Incidencia'] > 0].sort_values(by='Incidencia', ascending=False)
                    
                    if not df_efecto.empty:
                        st.dataframe(df_efecto.style.format("{:.3f}"), hide_index=True)
                        st.pyplot(draw_focused_graph(elemento, df_efecto['Causa'].tolist(), df_efecto['Incidencia'].tolist(), f"Causas ocultas que inciden en '{elemento}'", mode='target'))
                    else:
                        st.info(f"Ningún nodo genera efectos olvidados hacia '{elemento}'.")

elif "2." in modo:
    st.header("Análisis de Sistemas Encadenados (A → B → C)")
    
    col1, col2 = st.columns(2)
    with col1:
        archivo_m1 = st.file_uploader("1️⃣ Suba Matriz 1 (A → B) [.csv]", type=['csv'])
    with col2:
        archivo_m2 = st.file_uploader("2️⃣ Suba Matriz 2 (B → C) [.csv]", type=['csv'])
        
    if archivo_m1 and archivo_m2:
        df1 = load_and_clean_csv(archivo_m1)
        df2 = load_and_clean_csv(archivo_m2)
        
        if len(df1.columns) != len(df2.index):
            st.error(f"⚠️ Error de dimensión: Las columnas de M1 ({len(df1.columns)}) no coinciden con las filas de M2 ({len(df2.index)}).")
        else:
            st.success("Matrices compatibles cargadas con éxito.")
            
            st.markdown("**(Opcional)** Para calcular los Efectos Olvidados, puede subir la matriz de incidencia directa (A → C).")
            archivo_m_directa = st.file_uploader("3️⃣ Suba Matriz Directa (A → C) [.csv] (Opcional)", type=['csv'])
            
            # Guardamos en memoria si el botón fue presionado
            if st.button("🚀 Procesar Composición", type="primary"):
                st.session_state['procesado_m2'] = True
                
            # Si está en memoria, mostramos todo
            if st.session_state.get('procesado_m2', False):
                resultado = max_min_composition(df1.values, df2.values)
                labels_A = list(df1.index)
                labels_C = list(df2.columns)
                df_res = pd.DataFrame(resultado, index=labels_A, columns=labels_C)
                
                tab1, tab2 = st.tabs(["📊 Matriz Resultante de Composición", "🔍 Efectos Olvidados"])
                
                with tab1:
                    st.write("Resultado de la composición Max-Min (A → C):")
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.dataframe(df_res.style.format("{:.3f}"))
                    with c2:
                        st.pyplot(draw_incidence_graph(resultado, labels_A, labels_C, "Red de Composición Max-Min"))
                
                with tab2:
                    if archivo_m_directa:
                        df_dir = load_and_clean_csv(archivo_m_directa)
                        if df_dir.shape == df_res.shape:
                            efectos = subtract_matrices(resultado, df_dir.values)
                            df_efectos = pd.DataFrame(efectos, index=labels_A, columns=labels_C)
                            
                            c1, c2 = st.columns([1, 1.5])
                            with c1:
                                st.dataframe(df_efectos.style.format("{:.3f}"))
                            with c2:
                                st.pyplot(draw_incidence_graph(efectos, labels_A, labels_C, "Red de Efectos Olvidados"))
                            
                            # --- MODO INSPECTOR ---
                            st.divider()
                            st.header("🕵️‍♂️ Modo Inspector")
                            tipo_inspeccion = st.radio("¿Qué desea inspeccionar?", ["Elemento Causa (Conjunto A)", "Elemento Efecto (Conjunto C)"], horizontal=True)
                            
                            if "Causa" in tipo_inspeccion:
                                elemento = st.selectbox("Seleccione un elemento Causa:", ["-- Seleccione --"] + labels_A)
                                if elemento != "-- Seleccione --":
                                    valores = df_efectos.loc[elemento]
                                    df_foc = pd.DataFrame({'Efecto Final (C)': labels_C, 'Incidencia': valores})
                                    df_foc = df_foc[df_foc['Incidencia'] > 0].sort_values(by='Incidencia', ascending=False)
                                    st.dataframe(df_foc.style.format("{:.3f}"), hide_index=True)
                                    if not df_foc.empty:
                                        st.pyplot(draw_focused_graph(elemento, df_foc['Efecto Final (C)'].tolist(), df_foc['Incidencia'].tolist(), f"Impactos de '{elemento}'", mode='source'))
                            else:
                                elemento = st.selectbox("Seleccione un elemento Efecto:", ["-- Seleccione --"] + labels_C)
                                if elemento != "-- Seleccione --":
                                    valores = df_efectos[elemento]
                                    df_foc = pd.DataFrame({'Causa Raíz (A)': labels_A, 'Incidencia': valores})
                                    df_foc = df_foc[df_foc['Incidencia'] > 0].sort_values(by='Incidencia', ascending=False)
                                    st.dataframe(df_foc.style.format("{:.3f}"), hide_index=True)
                                    if not df_foc.empty:
                                        st.pyplot(draw_focused_graph(elemento, df_foc['Causa Raíz (A)'].tolist(), df_foc['Incidencia'].tolist(), f"Causas que impactan a '{elemento}'", mode='target'))
                        else:
                            st.error("Las dimensiones de la matriz directa no coinciden con el resultado.")
                    else:
                        st.info("Suba la matriz directa (A → C) en el paso anterior para habilitar el cálculo de Efectos Olvidados y el Modo Inspector.")
