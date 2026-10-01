# ============================================================
# ANÁLISE DE INTELIGÊNCIA COMPUTACIONAL — COMPLETA
# Regressão, Classificação, Agrupamento, SÉRIE TEMPORAL
# + Pré-análise com recomendações
# + Execução de TODOS os estimadores disponíveis
# + Ajuste manual universal de hiperparâmetros
# + Exportação e comparação de funções matemáticas
# + Gráficos dinâmicos com seleção de modelos
# + Seção de explicações por algoritmo
# + Compatibilidade entre versões do Streamlit
# ============================================================

from __future__ import annotations

import inspect
import io
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "A dependência 'plotly' não está instalada. Instale com: pip install plotly"
    ) from exc

import streamlit as st

# ----- sklearn -----
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cluster import (
    AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN,
    KMeans, MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression,
    LogisticRegression, Ridge, RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error,
    normalized_mutual_info_score, precision_score, r2_score,
    recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, ParameterGrid, ParameterSampler,
    RandomizedSearchCV, StratifiedKFold, cross_validate,
    train_test_split, TimeSeriesSplit,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.utils.discovery import all_estimators

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None

try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None

# ----- Série Temporal (opcionais) -----
try:
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as SM_ETS
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


# ============================================================
# COMPATIBILIDADE ENTRE VERSÕES DO STREAMLIT
# ============================================================

def _supports_width_kwarg() -> bool:
    try:
        return "width" in inspect.signature(st.button).parameters
    except Exception:
        return False


_HAS_WIDTH = _supports_width_kwarg()


def ui_button(label: str, **kwargs):
    if _HAS_WIDTH:
        return st.button(label, width="stretch", **kwargs)
    kwargs.pop("width", None)
    return st.button(label, use_container_width=True, **kwargs)


def ui_download_button(label: str, data, file_name: str, mime: str, **kwargs):
    if _HAS_WIDTH:
        return st.download_button(
            label, data, file_name=file_name, mime=mime, width="stretch", **kwargs
        )
    return st.download_button(
        label, data, file_name=file_name, mime=mime,
        use_container_width=True, **kwargs,
    )


def ui_dataframe(df, **kwargs):
    kwargs.pop("width", None)
    if _HAS_WIDTH:
        return st.dataframe(df, width="stretch", **kwargs)
    return st.dataframe(df, use_container_width=True, **kwargs)


def ui_plotly_chart(fig, **kwargs):
    kwargs.pop("width", None)
    if _HAS_WIDTH:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    return st.plotly_chart(fig, use_container_width=True, **kwargs)


# ============================================================
# CONSTANTES
# ============================================================

DEFAULT_RANDOM_STATE = 42

CURATED_CLASSIFIERS = [
    "LogisticRegression", "LinearDiscriminantAnalysis", "GaussianNB",
    "KNeighborsClassifier", "SVC", "DecisionTreeClassifier",
    "RandomForestClassifier", "ExtraTreesClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier",
    "AdaBoostClassifier", "RidgeClassifier",
]

CURATED_REGRESSORS = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge",
    "SVR", "KNeighborsRegressor", "DecisionTreeRegressor",
    "RandomForestRegressor", "ExtraTreesRegressor",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "AdaBoostRegressor",
]

CURATED_CLUSTERERS = [
    "KMeans", "MiniBatchKMeans", "AgglomerativeClustering", "DBSCAN",
    "GaussianMixture", "Birch", "MeanShift", "SpectralClustering",
    "AffinityPropagation", "OPTICS", "KohonenSOM",
]

# Lista de modelos de série temporal nativos (via statsmodels/prophet)
CURATED_TS_NATIVE = [
    "ARIMA", "SARIMAX", "ExponentialSmoothing", "Prophet",
]

# Lista de modelos tabulares usados via lag features
CURATED_TS_LAG_MODELS = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "RandomForestRegressor", "ExtraTreesRegressor",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "KNeighborsRegressor", "DecisionTreeRegressor",
]


# ============================================================
# EXPLICAÇÕES DOS ALGORITMOS
# ============================================================

ALGORITHM_EXPLANATIONS: dict[str, dict[str, str]] = {
    # ... [mantidas as explicações existentes] ...
    "LinearRegression": {"tipo": "Regressão",
        "descricao": "Ajusta um hiperplano minimizando a soma dos quadrados dos resíduos.",
        "como_funciona": "Resolve (XᵀX)⁻¹Xᵀy via SVD/QR.",
        "quando_usar": "Baseline, relação linear, baixa dimensionalidade.",
        "limitacoes": "Sensível a outliers e multicolinearidade."},
    "Ridge": {"tipo": "Regressão",
        "descricao": "Regressão linear com penalização L2.",
        "como_funciona": "Minimiza ‖y − Xβ‖² + α‖β‖².",
        "quando_usar": "Multicolinearidade, p ≈ n ou p > n.",
        "limitacoes": "Não faz seleção de variáveis."},
    "Lasso": {"tipo": "Regressão",
        "descricao": "Regressão linear com penalização L1.",
        "como_funciona": "Coordinate descent em ‖y − Xβ‖² + α‖β‖₁.",
        "quando_usar": "Alta dimensionalidade com seleção de features.",
        "limitacoes": "Instável com features correlacionadas."},
    "ElasticNet": {"tipo": "Regressão",
        "descricao": "Combina L1 e L2.",
        "como_funciona": "Coordinate descent generalizado.",
        "quando_usar": "Features correlacionadas + seleção.",
        "limitacoes": "Dois hiperparâmetros."},
    "BayesianRidge": {"tipo": "Regressão",
        "descricao": "Ridge com estimação bayesiana de α e λ.",
        "como_funciona": "Maximiza evidência marginal iterativamente.",
        "quando_usar": "Quando não se quer escolher α manualmente.",
        "limitacoes": "Assume ruído gaussiano."},
    "SVR": {"tipo": "Regressão",
        "descricao": "Support Vector Regression com tubo ε.",
        "como_funciona": "Minimiza ½‖w‖² + C·Σ(ξ+ξ*) sujeito a |y−f(x)| ≤ ε+ξ.",
        "quando_usar": "Alta dimensionalidade, não linearidade.",
        "limitacoes": "Sensível à escala; tuning de C, ε, γ."},
    "KNeighborsRegressor": {"tipo": "Regressão",
        "descricao": "Média dos k vizinhos mais próximos.",
        "como_funciona": "Agrega y dos k vizinhos.",
        "quando_usar": "Padrões locais, baixa dimensão.",
        "limitacoes": "Maldição da dimensionalidade."},
    "DecisionTreeRegressor": {"tipo": "Regressão",
        "descricao": "Árvore que particiona em regiões e prevê média.",
        "como_funciona": "Escolhe corte que minimiza erro quadrático.",
        "quando_usar": "Interpretabilidade, não linearidade.",
        "limitacoes": "Alta variância."},
    "RandomForestRegressor": {"tipo": "Regressão",
        "descricao": "Ensemble de árvores com bootstrap.",
        "como_funciona": "Média das árvores.",
        "quando_usar": "Dados tabulares, não linearidade.",
        "limitacoes": "Menos interpretável."},
    "ExtraTreesRegressor": {"tipo": "Regressão",
        "descricao": "Random Forest com cortes aleatórios.",
        "como_funciona": "Thresholds aleatórios em cada nó.",
        "quando_usar": "Quando se quer mais aleatoriedade.",
        "limitacoes": "Pode ter viés."},
    "GradientBoostingRegressor": {"tipo": "Regressão",
        "descricao": "Boosting: árvores corrigem erros sequencialmente.",
        "como_funciona": "Fₘ = Fₘ₋₁ + η·hₘ(x) sobre resíduos.",
        "quando_usar": "Alta acurácia tabular.",
        "limitacoes": "Sensível a overfitting."},
    "HistGradientBoostingRegressor": {"tipo": "Regressão",
        "descricao": "Boosting com histogramas.",
        "como_funciona": "Discretiza features em bins.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Menos preciso em pequenos."},
    "AdaBoostRegressor": {"tipo": "Regressão",
        "descricao": "Boosting adaptativo.",
        "como_funciona": "Aumenta peso dos erros maiores.",
        "quando_usar": "Com regressor base simples.",
        "limitacoes": "Sensível a outliers."},
    "XGBRegressor": {"tipo": "Regressão",
        "descricao": "Gradient boosting otimizado.",
        "como_funciona": "Similar a GBR com regularização L1/L2.",
        "quando_usar": "Alta performance tabular.",
        "limitacoes": "Muitos hiperparâmetros."},
    "LGBMRegressor": {"tipo": "Regressão",
        "descricao": "LightGBM: boosting leaf-wise com histogramas.",
        "como_funciona": "Cresce folha com maior ganho.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Overfit em pequenos."},
    "CatBoostRegressor": {"tipo": "Regressão",
        "descricao": "CatBoost com ordered boosting.",
        "como_funciona": "Trata categóricas nativamente.",
        "quando_usar": "Muitas categóricas.",
        "limitacoes": "Lento."},
    "LogisticRegression": {"tipo": "Classificação",
        "descricao": "Modelo linear probabilístico.",
        "como_funciona": "P(y=1|x) = 1/(1+exp(−βᵀx)).",
        "quando_usar": "Baseline linear, probabilidades.",
        "limitacoes": "Fronteira linear."},
    "LinearDiscriminantAnalysis": {"tipo": "Classificação",
        "descricao": "Assume gaussianas com mesma covariância.",
        "como_funciona": "Projeção de Fisher.",
        "quando_usar": "Classes gaussianas.",
        "limitacoes": "Assume normalidade."},
    "GaussianNB": {"tipo": "Classificação",
        "descricao": "Naive Bayes gaussiano.",
        "como_funciona": "P(y|x) ∝ P(y)·∏P(xᵢ|y).",
        "quando_usar": "Baseline rápido.",
        "limitacoes": "Assume independência."},
    "KNeighborsClassifier": {"tipo": "Classificação",
        "descricao": "Votação dos k vizinhos.",
        "como_funciona": "Moda das classes dos k vizinhos.",
        "quando_usar": "Padrões locais.",
        "limitacoes": "Sensível à escala."},
    "SVC": {"tipo": "Classificação",
        "descricao": "SVM com margem máxima.",
        "como_funciona": "Maximiza 2/‖w‖ com kernels.",
        "quando_usar": "Alta dimensão.",
        "limitacoes": "Lento em datasets grandes."},
    "DecisionTreeClassifier": {"tipo": "Classificação",
        "descricao": "Árvore com critério de impureza.",
        "como_funciona": "Gini/entropia.",
        "quando_usar": "Interpretabilidade.",
        "limitacoes": "Overfit."},
    "RandomForestClassifier": {"tipo": "Classificação",
        "descricao": "Ensemble de árvores.",
        "como_funciona": "Votação majoritária.",
        "quando_usar": "Dados tabulares.",
        "limitacoes": "Menos interpretável."},
    "ExtraTreesClassifier": {"tipo": "Classificação",
        "descricao": "Random Forest com cortes aleatórios.",
        "como_funciona": "Thresholds aleatórios.",
        "quando_usar": "Mais aleatoriedade.",
        "limitacoes": "Pode ter viés."},
    "GradientBoostingClassifier": {"tipo": "Classificação",
        "descricao": "Boosting de árvores.",
        "como_funciona": "Árvores nos resíduos da log-loss.",
        "quando_usar": "Alta acurácia.",
        "limitacoes": "Sensível a overfitting."},
    "HistGradientBoostingClassifier": {"tipo": "Classificação",
        "descricao": "Boosting com histogramas.",
        "como_funciona": "Discretização em bins.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Menos preciso em pequenos."},
    "AdaBoostClassifier": {"tipo": "Classificação",
        "descricao": "Boosting adaptativo.",
        "como_funciona": "Aumenta peso dos erros.",
        "quando_usar": "Com classificadores fracos.",
        "limitacoes": "Sensível a ruído."},
    "RidgeClassifier": {"tipo": "Classificação",
        "descricao": "Classificador linear com L2.",
        "como_funciona": "Ridge por classe.",
        "quando_usar": "Alta dimensão.",
        "limitacoes": "Sem probabilidades."},
    "XGBClassifier": {"tipo": "Classificação",
        "descricao": "XGBoost para classificação.",
        "como_funciona": "Boosting regularizado.",
        "quando_usar": "Alta performance.",
        "limitacoes": "Muitos hiperparâmetros."},
    "LGBMClassifier": {"tipo": "Classificação",
        "descricao": "LightGBM para classificação.",
        "como_funciona": "Boosting leaf-wise.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Overfit em pequenos."},
    "CatBoostClassifier": {"tipo": "Classificação",
        "descricao": "CatBoost para classificação.",
        "como_funciona": "Ordered boosting.",
        "quando_usar": "Muitas categóricas.",
        "limitacoes": "Lento."},
    "KMeans": {"tipo": "Agrupamento",
        "descricao": "Minimiza inércia (distância ao centroide).",
        "como_funciona": "Alterna atribuição e recálculo de centroides.",
        "quando_usar": "Clusters esféricos, k conhecido.",
        "limitacoes": "Assume esferas; k fixo."},
    "MiniBatchKMeans": {"tipo": "Agrupamento",
        "descricao": "KMeans com mini-batches.",
        "como_funciona": "Atualiza com subamostras.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Menos preciso."},
    "AgglomerativeClustering": {"tipo": "Agrupamento",
        "descricao": "Hierárquico aglomerativo.",
        "como_funciona": "Merge bottom-up.",
        "quando_usar": "Hierarquia desejada.",
        "limitacoes": "O(n²) memória."},
    "DBSCAN": {"tipo": "Agrupamento",
        "descricao": "Density-based.",
        "como_funciona": "Regiões densas; ruído = -1.",
        "quando_usar": "Clusters arbitrários com ruído.",
        "limitacoes": "Sensível a eps."},
    "GaussianMixture": {"tipo": "Agrupamento",
        "descricao": "Mistura gaussiana via EM.",
        "como_funciona": "Estima médias, covariâncias, pesos.",
        "quando_usar": "Clusters elípticos, probabilidades.",
        "limitacoes": "Assume gaussianidade."},
    "Birch": {"tipo": "Agrupamento",
        "descricao": "Árvore CF (Clustering Feature).",
        "como_funciona": "Constrói árvore e agrupa folhas.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Menos preciso."},
    "MeanShift": {"tipo": "Agrupamento",
        "descricao": "Busca modos de densidade.",
        "como_funciona": "Move pontos para média local.",
        "quando_usar": "Sem k fixo.",
        "limitacoes": "Lento; bandwidth crítico."},
    "SpectralClustering": {"tipo": "Agrupamento",
        "descricao": "Autovetores da Laplaciana.",
        "como_funciona": "Grafo + autovetores + KMeans.",
        "quando_usar": "Clusters não convexos.",
        "limitacoes": "O(n³)."},
    "AffinityPropagation": {"tipo": "Agrupamento",
        "descricao": "Troca de mensagens entre pontos.",
        "como_funciona": "Responsabilidade e disponibilidade.",
        "quando_usar": "Sem k fixo.",
        "limitacoes": "Lento."},
    "OPTICS": {"tipo": "Agrupamento",
        "descricao": "Variação do DBSCAN com densidades variáveis.",
        "como_funciona": "Ordena pontos por densidade.",
        "quando_usar": "Densidades variadas.",
        "limitacoes": "Parâmetros xi e min_samples."},
    "HDBSCAN": {"tipo": "Agrupamento",
        "descricao": "DBSCAN hierárquico sem eps.",
        "como_funciona": "Constrói hierarquia de densidade.",
        "quando_usar": "Densidades variadas com ruído.",
        "limitacoes": "min_cluster_size crítico."},
    "KohonenSOM": {"tipo": "Agrupamento",
        "descricao": "Self-Organizing Map (Mapa de Kohonen).",
        "como_funciona": "BMU + atualização de vizinhos com kernel gaussiano.",
        "quando_usar": "Visualização de alta dimensionalidade.",
        "limitacoes": "Grade e iterações precisam ser escolhidas."},
    # ----- SÉRIE TEMPORAL -----
    "ARIMA": {"tipo": "Série Temporal",
        "descricao": "AutoRegressive Integrated Moving Average — modela séries estacionárias combinando componentes AR (defasagens), I (diferenciação) e MA (erros defasados).",
        "como_funciona": "ARIMA(p,d,q): diferencia a série d vezes para torná-la estacionária; ajusta AR(p) nos valores defasados e MA(q) nos resíduos. Estimado por máxima verossimilhança.",
        "quando_usar": "Séries univariadas estacionárias ou que se tornam estacionárias após diferenciação; sem sazonalidade forte.",
        "limitacoes": "Sensível a outliers; requer escolha de p, d, q; não lida com múltiplas séries relacionadas sem extensão."},
    "SARIMAX": {"tipo": "Série Temporal",
        "descricao": "Extensão sazonal do ARIMA com componentes exógenas (SARIMA + eXogenous).",
        "como_funciona": "ARIMA(p,d,q)×(P,D,Q,s) + regressores exógenos X. Estima sazonalidade e efeitos de variáveis externas por máxima verossimilhança.",
        "quando_usar": "Séries com sazonalidade clara (diária, semanal, anual) e/ou que dependem de variáveis externas.",
        "limitacoes": "Muitos parâmetros — requer tuning cuidadoso; lento em séries longas."},
    "ExponentialSmoothing": {"tipo": "Série Temporal",
        "descricao": "Suavização exponencial (Holt-Winters) com nível, tendência e sazonalidade.",
        "como_funciona": "Média ponderada exponencialmente dos valores passados com componentes de tendência e sazonalidade.",
        "quando_usar": "Séries com tendência e/ou sazonalidade; boa performance com dados recentes mais relevantes.",
        "limitacoes": "Assume estrutura aditiva ou multiplicativa fixa; requer definição do período sazonal."},
    "Prophet": {"tipo": "Série Temporal",
        "descricao": "Modelo aditivo da Meta que decompõe a série em tendência + sazonalidades + feriados + ruído.",
        "como_funciona": "Tendência piecewise-linear (ou logística) + Fourier para sazonalidades + efeitos de feriados. Estimado por Stan (regressão bayesiana).",
        "quando_usar": "Séries de negócio com sazonalidade múltipla, feriados, outliers, dados faltantes.",
        "limitacoes": "Não adequado para séries curtas ou alta frequência; menos preciso que ARIMA em séries bem-comportadas."},
    "NaiveForecaster": {"tipo": "Série Temporal",
        "descricao": "Forecast ingênuo: último valor, média ou valor sazonal.",
        "como_funciona": "ŷ(t+h) = y(t), média(y) ou y(t+h-s).",
        "quando_usar": "Baseline obrigatório em qualquer estudo de previsão.",
        "limitacoes": "Não captura tendências nem padrões complexos."},
    "AutoARIMA": {"tipo": "Série Temporal",
        "descricao": "ARIMA com seleção automática de (p,d,q) e (P,D,Q,s) por critério de informação.",
        "como_funciona": "Busca stepwise ou por heurística sobre o espaço de ordens, escolhendo por AIC/BIC.",
        "quando_usar": "Quando não se quer escolher manualmente p,d,q.",
        "limitacoes": "Pode sobreajustar; busca pode ser lenta."},
    "ThetaForecaster": {"tipo": "Série Temporal",
        "descricao": "Método Theta: decompõe a série em duas linhas theta e extrapola.",
        "como_funciona": "Separa tendência e componente theta, aplica suavização.",
        "quando_usar": "Séries com tendência estável.",
        "limitacoes": "Assume tendência constante."},
    "PolynomialTrendForecaster": {"tipo": "Série Temporal",
        "descricao": "Ajusta um polinômio de grau k ao tempo e extrapola.",
        "como_funciona": "Regressão polinomial sobre índice temporal.",
        "quando_usar": "Tendência não linear bem definida.",
        "limitacoes": "Extrapolações podem divergir rapidamente."},
}


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise IC — Completa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ESTADO
# ============================================================

def init_state() -> None:
    defaults = {
        "df_raw": None, "encoding_config": None, "encoding_preview": None,
        "results": None, "task_type": "regression", "encoding_ready": False,
        "prediction_result": None, "clustering_result": None,
        "pre_analysis_reg": None, "pre_analysis_clf": None, "pre_analysis_cluster": None,
        "ts_results": None, "ts_pre_analysis": None, "ts_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# TRANSFORMERS CUSTOMIZADOS
# ============================================================

class IQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str = "keep", factor: float = 1.5):
        self.mode = mode
        self.factor = factor

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        self.lower_ = np.empty(self.n_features_in_, dtype=float)
        self.upper_ = np.empty(self.n_features_in_, dtype=float)
        self.median_ = np.empty(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                self.lower_[j] = -np.inf
                self.upper_[j] = np.inf
                self.median_[j] = 0.0
                continue
            q1, q3 = np.nanpercentile(finite, [25, 75])
            iqr = q3 - q1
            self.lower_[j] = q1 - self.factor * iqr
            self.upper_[j] = q3 + self.factor * iqr
            self.median_[j] = float(np.nanmedian(finite))
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()
        if self.mode == "keep":
            return X_arr
        low, up = self.lower_, self.upper_
        mask_low = X_arr < low
        mask_up = X_arr > up
        if self.mode == "winsorize":
            X_arr = np.where(mask_low, low, X_arr)
            X_arr = np.where(mask_up, up, X_arr)
        elif self.mode == "median":
            mask = mask_low | mask_up
            X_arr[mask] = np.broadcast_to(self.median_, X_arr.shape)[mask]
        return X_arr


class ManualMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: dict[str, float] | None = None, unknown_value: float = -1.0):
        self.mapping = mapping or {}
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        out = np.empty(arr.shape, dtype=float)
        for idx, value in np.ndenumerate(arr):
            key = "missing" if pd.isna(value) else str(value)
            out[idx] = float(self.mapping.get(key, self.unknown_value))
        return out


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.10, max_features: int | None = None):
        self.threshold = threshold
        self.max_features = max_features

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_sr = pd.Series(y).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)
        y_num = pd.to_numeric(y_sr, errors="coerce")
        if y_num.isna().all():
            y_num = pd.Series(pd.factorize(y_sr.astype(str))[0], dtype=float)
        else:
            y_num = y_num.fillna(y_num.median())
        corr_values = {}
        for col in X_df.columns:
            x = pd.to_numeric(X_df[col], errors="coerce")
            corr_values[col] = float(x.corr(y_num)) if x.notna().sum() > 1 else 0.0
        corr = pd.Series(corr_values).abs().fillna(0).sort_values(ascending=False)
        if corr.empty:
            raise ValueError("Nenhuma variável disponível após o pré-processamento.")
        if self.max_features is not None and int(self.max_features) > 0:
            chosen = list(corr.head(int(self.max_features)).index)
        else:
            chosen = list(corr[corr >= float(self.threshold)].index)
        if not chosen:
            chosen = [corr.index[0]]
        self.selected_indices_ = [X_df.columns.get_loc(c) for c in chosen]
        self.selected_features_ = [str(c) for c in chosen]
        self.n_features_in_ = X_df.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X)
        return arr[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class KohonenClusterer(BaseEstimator):
    def __init__(self, som_x: int = 5, som_y: int = 5, sigma: float = 1.0,
                 learning_rate: float = 0.5, num_iteration: int = 1000,
                 random_state: int | None = None,
                 neighborhood_function: str = "gaussian"):
        self.som_x = som_x
        self.som_y = som_y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.random_state = random_state
        self.neighborhood_function = neighborhood_function
        self.som_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        if MiniSom is None:
            raise ImportError("MiniSom não está instalado. pip install minisom")
        X = np.asarray(X, dtype=float)
        self.som_ = MiniSom(
            self.som_x, self.som_y, X.shape[1],
            sigma=self.sigma, learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            random_seed=self.random_state,
        )
        self.som_.random_weights_init(X)
        self.som_.train_random(X, self.num_iteration)
        self.labels_ = self._assign_labels(X)
        return self

    def _assign_labels(self, X):
        labels = []
        for x in X:
            bmu = self.som_.winner(x)
            labels.append(bmu[0] * self.som_y + bmu[1])
        return np.asarray(labels)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def get_params(self, deep=True):
        return {
            "som_x": self.som_x, "som_y": self.som_y, "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "random_state": self.random_state,
            "neighborhood_function": self.neighborhood_function,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ============================================================
# UTILITÁRIOS DE DADOS
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore", dtype=float)


def detect_categorical_columns(df: pd.DataFrame, max_unique_int: int = 20) -> list[str]:
    cats: list[str] = []
    for col in df.columns:
        try:
            s = df[col]
            if (pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)):
                cats.append(col); continue
            if pd.api.types.is_integer_dtype(s):
                nun = s.nunique(dropna=True)
                if 1 < nun <= max_unique_int:
                    cats.append(col)
        except Exception:
            continue
    return cats


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or str(series.dtype) == "string"):
            continue
        s = series.astype(str).str.strip()
        s_num = (s.str.replace(r"R\$\s*", "", regex=True)
                 .str.replace(r"%\s*$", "", regex=True)
                 .str.replace(" ", "", regex=False))
        direct = pd.to_numeric(s_num, errors="coerce")
        brazilian = pd.to_numeric(
            s_num.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce")
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        if direct.notna().sum() / non_null >= 0.90:
            df[col] = direct
        elif brazilian.notna().sum() / non_null >= 0.90:
            df[col] = brazilian
    return df


def load_dataframe(uploaded_file, source_type: str, uci_id: int | None = None) -> pd.DataFrame:
    if source_type == "UCI":
        if fetch_ucirepo is None:
            raise ImportError("Pacote 'ucimlrepo' não está instalado.")
        ds = fetch_ucirepo(id=int(uci_id))
        if getattr(ds.data, "features", None) is not None and getattr(ds.data, "targets", None) is not None:
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        else:
            df = ds.data.original
        return coerce_numeric(df)
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo foi enviado.")
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = uploaded_file.read().decode("utf-8", errors="replace").lstrip("\ufeff")
        first = raw.splitlines()[0] if raw else ""
        counts = {";": first.count(";"), ",": first.count(","),
                  "\t": first.count("\t"), "|": first.count("|")}
        sep = max(counts, key=counts.get) if max(counts.values()) else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Use CSV, XLS/XLSX ou JSON.")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    return coerce_numeric(df)


def parse_list(text: str) -> list[Any]:
    values: list[Any] = []
    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue
        low = token.lower()
        if low in {"scale", "auto"}:
            values.append(low); continue
        try:
            value = float(token)
            if value.is_integer() and "." not in token and "e" not in low:
                value = int(value)
            values.append(value)
        except ValueError as exc:
            raise ValueError(f"Valor inválido: '{token}'.") from exc
    if not values:
        raise ValueError("A lista de hiperparâmetros não pode ficar vazia.")
    return values


def validate_feature_selection(df: pd.DataFrame, features: list[str], target: str) -> None:
    if target not in df.columns:
        raise ValueError("A variável alvo não existe no dataset.")
    if not features:
        raise ValueError("Selecione pelo menos uma variável preditora.")
    if target in features:
        raise ValueError("A variável alvo não pode estar entre as preditoras.")
    if len(df[features].columns) != len(set(features)):
        raise ValueError("Existem variáveis preditoras duplicadas.")


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > up)).sum())
        rows.append([col, q1, q3, iqr, low, up, n, 100 * n / len(s)])
    return pd.DataFrame(rows, columns=["Variável", "Q1", "Q3", "IQR",
                                       "Limite inf.", "Limite sup.", "Outliers", "%"])


# ============================================================
# SÉRIE TEMPORAL — FUNÇÕES ESPECÍFICAS
# ============================================================

def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Detecta colunas que podem ser datas (object/string parseable)."""
    candidates = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(col); continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(20)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= 0.8 * len(sample):
                    candidates.append(col)
            except Exception:
                continue
    return candidates


def make_ts_dataframe(df, time_col, target_col, exog_cols, n_lags,
                      date_format=None):
    """Constrói dataframe de série temporal com lag features."""
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce",
                                     format=date_format)
    data = data.dropna(subset=[time_col, target_col])
    data = data.sort_values(time_col).reset_index(drop=True)

    # Lag features da variável alvo
    y_series = pd.to_numeric(data[target_col], errors="coerce")
    data = data.loc[y_series.notna()].reset_index(drop=True)
    y_series = pd.to_numeric(data[target_col], errors="coerce")

    lag_df = pd.DataFrame(index=data.index)
    for lag in range(1, int(n_lags) + 1):
        lag_df[f"{target_col}_lag{lag}"] = y_series.shift(lag)

    # Features exógenas (se houver)
    exog_df = pd.DataFrame(index=data.index)
    if exog_cols:
        exog_df = data[exog_cols].copy()

    X = pd.concat([lag_df, exog_df], axis=1)
    y = y_series.copy()

    # Remove linhas com NaN nos lags
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = data.loc[valid, time_col].reset_index(drop=True)

    return X, y, dates


def temporal_split(X, y, dates, test_size=0.2):
    """Split temporal — sem embaralhar."""
    n = len(y)
    n_test = max(1, int(n * test_size))
    n_train = n - n_test
    return (X.iloc[:n_train].reset_index(drop=True),
            X.iloc[n_train:].reset_index(drop=True),
            y.iloc[:n_train].reset_index(drop=True),
            y.iloc[n_train:].reset_index(drop=True),
            dates.iloc[:n_train].reset_index(drop=True),
            dates.iloc[n_train:].reset_index(drop=True))


def evaluate_ts_tabular(X_tr, y_tr, X_te, y_te, dates_te, models,
                         cv=5, search_method="grid", n_iter=20,
                         random_state=42):
    """Avalia regressores tabulares via lag features com TimeSeriesSplit."""
    results = {}
    n_splits = max(2, int(cv))
    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    progress = st.progress(0)
    status = st.empty()
    items = list(models.items())
    for i, (name, cfg) in enumerate(items, start=1):
        status.info(f"Treinando **{name}** ({i}/{len(items)})...")
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", clone(cfg["estimator"])),
            ])
            best = None
            best_params = {}
            cv_scores = np.array([])
            if cfg["params"]:
                if search_method == "grid":
                    search = GridSearchCV(pipe, cfg["params"], scoring="r2",
                                          cv=ts_cv, n_jobs=-1, refit=True,
                                          error_score="raise")
                elif search_method == "random":
                    n_rand = randomized_iteration_limit(cfg["params"], n_iter)
                    search = RandomizedSearchCV(pipe, cfg["params"], n_iter=n_rand,
                                                scoring="r2", cv=ts_cv, n_jobs=-1,
                                                refit=True, random_state=random_state,
                                                error_score="raise")
                else:
                    manual = {k: [v[0]] for k, v in cfg["params"].items()}
                    search = GridSearchCV(pipe, manual, scoring="r2", cv=ts_cv,
                                          n_jobs=-1, refit=True, error_score="raise")
                search.fit(X_tr, y_tr)
                best = search.best_estimator_
                best_params = search.best_params_
                idx_best = search.best_index_
                cv_scores = np.asarray([
                    search.cv_results_[f"split{k}_test_score"][idx_best]
                    for k in range(n_splits) if
                    f"split{k}_test_score" in search.cv_results_
                ], dtype=float)
            else:
                best = pipe.fit(X_tr, y_tr)

            y_pred = best.predict(X_te)
            mae = float(mean_absolute_error(y_te, y_pred))
            mse = float(mean_squared_error(y_te, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_te, y_pred)) if len(y_te) > 1 else float("nan")

            # MAPE e SMAPE com proteção para zeros
            y_te_arr = np.asarray(y_te, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            mask = np.abs(y_te_arr) > 1e-9
            mape = float(np.mean(np.abs((y_te_arr[mask] - y_pred_arr[mask])
                                         / y_te_arr[mask])) * 100) if mask.any() else None
            denom = (np.abs(y_te_arr) + np.abs(y_pred_arr)) / 2
            mask2 = denom > 1e-9
            smape = float(np.mean(np.abs(y_te_arr[mask2] - y_pred_arr[mask2])
                                   / denom[mask2]) * 100) if mask2.any() else None

            results[name] = {
                "best_model": best, "best_params": best_params,
                "cv_scores": cv_scores,
                "cv_mean": float(cv_scores.mean()) if cv_scores.size else None,
                "cv_std": float(cv_scores.std()) if cv_scores.size else None,
                "test_mae": mae, "test_rmse": rmse, "test_mse": mse,
                "test_r2": r2, "test_mape": mape, "test_smape": smape,
                "y_pred": y_pred, "dates_te": dates_te,
                "family": "tabular",
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(i / len(items))
    status.success("✅ Previsão tabular concluída!")
    progress.empty()
    return results


def evaluate_ts_native(y_tr_full, y_te_full, dates_tr, dates_te, models,
                       n_splits=5):
    """Avalia modelos nativos de série temporal (ARIMA/SARIMAX/ETS/Prophet)."""
    results = {}
    progress = st.progress(0)
    status = st.empty()
    items = list(models.items())

    for i, (name, cfg) in enumerate(items, start=1):
        status.info(f"Ajustando **{name}** ({i}/{len(items)})...")
        try:
            y_tr = pd.Series(np.asarray(y_tr_full, dtype=float),
                              index=pd.DatetimeIndex(dates_tr))
            y_te = pd.Series(np.asarray(y_te_full, dtype=float),
                              index=pd.DatetimeIndex(dates_te))
            y_full = pd.concat([y_tr, y_te])
            n_tr = len(y_tr)

            # Walk-forward CV simples
            cv_scores = []
            min_train = max(5, n_tr // (n_splits + 1))
            for k in range(1, n_splits + 1):
                train_end = min_train * k
                if train_end >= n_tr:
                    break
                y_fold_tr = y_full.iloc[:train_end]
                y_fold_val = y_full.iloc[train_end:min(train_end + min_train, n_tr)]
                if len(y_fold_val) < 1:
                    continue
                try:
                    fold_pred = _fit_ts_native(name, cfg["params"], y_fold_tr,
                                                steps=len(y_fold_val),
                                                freq=y_fold_tr.index)
                    cv_scores.append(float(r2_score(y_fold_val, fold_pred))
                                     if len(y_fold_val) > 1 else np.nan)
                except Exception:
                    pass

            # Fit final em toda a série de treino
            y_pred = _fit_ts_native(name, cfg["params"], y_tr,
                                     steps=len(y_te), freq=y_te.index)

            mae = float(mean_absolute_error(y_te, y_pred))
            mse = float(mean_squared_error(y_te, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_te, y_pred)) if len(y_te) > 1 else float("nan")

            y_te_arr = np.asarray(y_te, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            mask = np.abs(y_te_arr) > 1e-9
            mape = float(np.mean(np.abs((y_te_arr[mask] - y_pred_arr[mask])
                                         / y_te_arr[mask])) * 100) if mask.any() else None
            denom = (np.abs(y_te_arr) + np.abs(y_pred_arr)) / 2
            mask2 = denom > 1e-9
            smape = float(np.mean(np.abs(y_te_arr[mask2] - y_pred_arr[mask2])
                                   / denom[mask2]) * 100) if mask2.any() else None

            cv_arr = np.asarray([s for s in cv_scores if np.isfinite(s)],
                                 dtype=float)
            results[name] = {
                "best_model": None, "best_params": cfg["params"],
                "cv_scores": cv_arr,
                "cv_mean": float(cv_arr.mean()) if cv_arr.size else None,
                "cv_std": float(cv_arr.std()) if cv_arr.size else None,
                "test_mae": mae, "test_rmse": rmse, "test_mse": mse,
                "test_r2": r2, "test_mape": mape, "test_smape": smape,
                "y_pred": np.asarray(y_pred, dtype=float),
                "dates_te": dates_te,
                "family": "native",
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(i / len(items))
    status.success("✅ Previsão nativa concluída!")
    progress.empty()
    return results


def _fit_ts_native(name, params, y_train, steps, freq):
    """Ajusta um modelo nativo e retorna previsão."""
    if name in {"ARIMA", "AutoARIMA"}:
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels não instalado.")
        order = params.get("order", (1, 1, 1))
        model = SM_ARIMA(y_train, order=order)
        fit = model.fit()
        return np.asarray(fit.forecast(steps=steps), dtype=float)

    if name == "SARIMAX":
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels não instalado.")
        order = params.get("order", (1, 1, 1))
        seasonal_order = params.get("seasonal_order", (0, 0, 0, 0))
        model = SM_SARIMAX(y_train, order=order,
                            seasonal_order=seasonal_order)
        fit = model.fit(disp=False)
        return np.asarray(fit.forecast(steps=steps), dtype=float)

    if name == "ExponentialSmoothing":
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels não instalado.")
        trend = params.get("trend", "add")
        seasonal = params.get("seasonal", None)
        seasonal_periods = params.get("seasonal_periods", None)
        model = SM_ETS(y_train, trend=trend, seasonal=seasonal,
                        seasonal_periods=seasonal_periods)
        fit = model.fit()
        return np.asarray(fit.forecast(steps=steps), dtype=float)

    if name == "Prophet":
        if not HAS_PROPHET:
            raise ImportError("prophet não instalado.")
        df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        m = Prophet(
            yearly_seasonality=params.get("yearly_seasonality", "auto"),
            weekly_seasonality=params.get("weekly_seasonality", "auto"),
            daily_seasonality=params.get("daily_seasonality", "auto"),
        )
        m.fit(df)
        future = pd.DataFrame({"ds": freq[:steps]})
        forecast = m.predict(future)
        return forecast["yhat"].values.astype(float)

    if name == "NaiveForecaster":
        return np.repeat(y_train.iloc[-1], steps).astype(float)

    if name == "ThetaForecaster":
        # Theta simples: extrapola tendência linear recente
        x = np.arange(len(y_train))
        coef = np.polyfit(x[-min(50, len(x)):], y_train.values[-min(50, len(x)):], 1)
        return np.polyval(coef, np.arange(len(y_train), len(y_train) + steps))

    if name == "PolynomialTrendForecaster":
        degree = params.get("degree", 1)
        x = np.arange(len(y_train))
        coef = np.polyfit(x, y_train.values, degree)
        return np.polyval(coef, np.arange(len(y_train), len(y_train) + steps))

    raise ValueError(f"Modelo nativo desconhecido: {name}")


def ts_pre_analysis(y_series):
    """Pré-análise específica de série temporal."""
    info = {"warnings": [], "notes": []}
    y = pd.Series(y_series).dropna().astype(float)
    n = len(y)
    info["n_obs"] = int(n)
    info["mean"] = float(y.mean()) if n else 0
    info["std"] = float(y.std()) if n else 0

    if n < 20:
        info["warnings"].append("Série muito curta (< 20 obs) — modelos serão instáveis.")
        return info

    # Teste ADF de estacionariedade
    if HAS_STATSMODELS:
        try:
            adf = adfuller(y.values, autolag="AIC")
            info["adf_stat"] = float(adf[0])
            info["adf_pvalue"] = float(adf[1])
            if adf[1] > 0.05:
                info["warnings"].append(
                    f"ADF p-value = {adf[1]:.4f} > 0.05 — série provavelmente NÃO estacionária. "
                    "Considere diferenciação (ARIMA com d=1).")
            else:
                info["notes"].append(
                    f"ADF p-value = {adf[1]:.4f} ≤ 0.05 — série aparentemente estacionária.")
        except Exception:
            pass

    # Autocorrelação lag-1
    try:
        if n > 2:
            ac1 = float(pd.Series(y.values).autocorr(lag=1))
            info["autocorr_lag1"] = ac1
            if ac1 is not None and ac1 > 0.7:
                info["notes"].append(
                    f"Autocorrelação lag-1 = {ac1:.3f} — série altamente persistente.")
    except Exception:
        pass

    # Tendência (regressão linear simples)
    try:
        x = np.arange(n)
        coef = np.polyfit(x, y.values, 1)
        info["trend_slope"] = float(coef[0])
    except Exception:
        pass

    return info


# ============================================================
# SÉRIE TEMPORAL — GRÁFICOS
# ============================================================

def ts_forecast_plot(dates_train, y_train, dates_test, y_test, y_pred,
                     title="Previsão"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode="lines",
                              name="Treino", line=dict(color="#888")))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode="lines",
                              name="Real (teste)", line=dict(color="#1e88e5")))
    fig.add_trace(go.Scatter(x=dates_test, y=y_pred, mode="lines",
                              name="Previsto", line=dict(color="#e53935", dash="dash")))
    fig.update_layout(title=title, height=420, hovermode="x unified")
    return fig


def ts_residuals_plot(dates_test, y_test, y_pred, title="Resíduos"):
    resid = np.asarray(y_test) - np.asarray(y_pred)
    fig = px.scatter(x=dates_test, y=resid, title=title)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=380, xaxis_title="Data", yaxis_title="Resíduo")
    return fig


def ts_decomposition_plot(y_series, period=None):
    if not HAS_STATSMODELS:
        return None
    try:
        y = pd.Series(y_series).dropna().astype(float)
        if period is None:
            period = max(2, min(len(y) // 4, 12))
        if period < 2 or len(y) < 2 * period:
            return None
        res = seasonal_decompose(y, model="additive", period=period)
        fig = make_subplots_ts(res)
        return fig
    except Exception:
        return None


def make_subplots_ts(decomp):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Original", "Tendência",
                                        "Sazonalidade", "Resíduo"))
    for i, (name, series) in enumerate([
        ("Original", decomp.observed),
        ("Tendência", decomp.trend),
        ("Sazonalidade", decomp.seasonal),
        ("Resíduo", decomp.resid),
    ], start=1):
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", name=name),
            row=i, col=1,
        )
    fig.update_layout(height=700, showlegend=False,
                       title="Decomposição da série temporal")
    return fig


def ts_comparison_dashboard(y_test, dates_test, valid, selected_models,
                             max_facet_models=6):
    filtered = {k: v for k, v in valid.items() if k in selected_models}
    if not filtered:
        return None

    rows_bar = []
    for name, res in filtered.items():
        rows_bar.append({
            "Modelo": name,
            "MAE": res.get("test_mae"),
            "RMSE": res.get("test_rmse"),
            "MAPE (%)": res.get("test_mape"),
            "R²": res.get("test_r2"),
        })
    df_bar = pd.DataFrame(rows_bar)

    fig_metrics = None
    if not df_bar.empty:
        df_long = df_bar.melt(id_vars="Modelo",
                              value_vars=["MAE", "RMSE"],
                              var_name="Métrica", value_name="Valor")
        fig_metrics = px.bar(df_long, x="Modelo", y="Valor", color="Métrica",
                             barmode="group",
                             title="📊 Erros no teste (MAE / RMSE)")
        fig_metrics.update_layout(height=420, xaxis_tickangle=-30)

    fig_r2 = None
    df_r2 = df_bar.dropna(subset=["R²"])
    if not df_r2.empty:
        fig_r2 = px.bar(df_r2.sort_values("R²", ascending=False),
                        x="Modelo", y="R²", color="Modelo", text="R²",
                        title="📊 R² no teste")
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(height=420, showlegend=False, xaxis_tickangle=-30)

    fig_cv = None
    rows_cv = []
    for name, res in filtered.items():
        cv = res.get("cv_scores")
        if cv is None:
            continue
        cv = np.asarray(cv).ravel()
        if cv.size == 0:
            continue
        for k, v in enumerate(cv, start=1):
            rows_cv.append({"Modelo": name, "Fold": k, "R²": float(v)})
    if rows_cv:
        df_cv = pd.DataFrame(rows_cv)
        fig_cv = px.box(df_cv, x="Modelo", y="R²", color="Modelo",
                        points="all", title="📦 R² por fold (walk-forward)")
        fig_cv.update_layout(height=520, showlegend=False, xaxis_tickangle=-30)

    # Previsões
    n = len(filtered)
    fig_pred_facet = None
    preds_por_modelo = []

    if n <= max_facet_models:
        rows = []
        for name, res in filtered.items():
            yp = np.asarray(res["y_pred"]).ravel()
            dt = pd.to_datetime(res["dates_te"])
            for d, yp_v in zip(dt, yp):
                rows.append({"Modelo": name, "Data": d, "Previsto": float(yp_v)})
        df_pred = pd.DataFrame(rows)
        if not df_pred.empty:
            df_pred["Real"] = np.tile(np.asarray(y_test).ravel(), n)
            n_cols = 1
            n_rows = n
            v_spacing = min(0.05, 0.5 / max(n_rows - 1, 1))
            fig_pred_facet = go.Figure()
            for i, name in enumerate(filtered.keys()):
                sub = df_pred[df_pred["Modelo"] == name]
                fig_pred_facet.add_trace(go.Scatter(
                    x=sub["Data"], y=sub["Real"], mode="lines",
                    name=f"{name} — Real",
                    line=dict(color="#1e88e5"),
                    legendgroup=name,
                    showlegend=(i == 0),
                ))
                fig_pred_facet.add_trace(go.Scatter(
                    x=sub["Data"], y=sub["Previsto"], mode="lines",
                    name=f"{name} — Previsto",
                    line=dict(color="#e53935", dash="dash"),
                    legendgroup=name,
                    showlegend=(i == 0),
                ))
            fig_pred_facet.update_layout(
                title="🎯 Real × Previsto (todos os modelos)",
                height=max(400, 250 * n),
                hovermode="x unified",
            )
    else:
        for name, res in filtered.items():
            yp = np.asarray(res["y_pred"]).ravel()
            dt = pd.to_datetime(res["dates_te"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dt, y=np.asarray(y_test).ravel(),
                                     mode="lines", name="Real",
                                     line=dict(color="#1e88e5")))
            fig.add_trace(go.Scatter(x=dt, y=yp, mode="lines",
                                     name="Previsto",
                                     line=dict(color="#e53935", dash="dash")))
            fig.update_layout(title=f"Previsão — {name}", height=400,
                              hovermode="x unified")
            preds_por_modelo.append((name, fig))

    # Resíduos
    fig_resid = None
    rows_res = []
    for name, res in filtered.items():
        yp = np.asarray(res["y_pred"]).ravel()
        dt = pd.to_datetime(res["dates_te"])
        res_val = np.asarray(y_test).ravel()[:len(yp)] - yp
        for d, r in zip(dt, res_val):
            rows_res.append({"Modelo": name, "Data": d, "Resíduo": float(r)})
    if rows_res:
        df_res = pd.DataFrame(rows_res)
        fig_resid = px.line(df_res, x="Data", y="Resíduo", color="Modelo",
                            title="📉 Resíduos por modelo")
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(height=420, hovermode="x unified")

    return {
        "metrics_bar": fig_metrics,
        "r2_bar": fig_r2,
        "cv_box": fig_cv,
        "pred_facet": fig_pred_facet,
        "pred_por_modelo": preds_por_modelo,
        "resid": fig_resid,
        "df_bar": df_bar,
    }


# ============================================================
# CODIFICAÇÃO
# ============================================================

def apply_categorical_encoding_preview(df, encoding_map, ordinal_orders, manual_maps):
    out = df.copy()
    report = []
    for col, method in encoding_map.items():
        if col not in out.columns:
            continue
        if method == "drop":
            out = out.drop(columns=[col])
            report.append((col, "drop", "coluna removida")); continue
        s = out[col].astype("string").fillna("missing")
        if method == "onehot":
            enc = make_onehot_encoder()
            arr = enc.fit_transform(s.to_frame())
            names = [f"{col}__{cat}" for cat in enc.categories_[0]]
            out = pd.concat([out.drop(columns=[col]),
                             pd.DataFrame(arr, columns=names, index=out.index)], axis=1)
            report.append((col, "onehot", f"{len(names)} colunas"))
        elif method == "label":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "label", "1 coluna"))
        elif method == "ordinal":
            order = ordinal_orders.get(col, sorted(s.unique().tolist()))
            enc = OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "ordinal", "1 coluna"))
        elif method == "manual":
            mapping = manual_maps.get(col, {})
            out[col] = s.map(mapping).fillna(-1).astype(float)
            report.append((col, "manual", f"{len(mapping)} regras"))
    return out, report


def build_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode):
    categorical_columns, onehot_columns, label_columns = [], [], []
    ordinal_columns, manual_columns, dropped_columns = [], [], []
    for col, method in encoding_map.items():
        if col not in X.columns:
            continue
        if method == "drop":
            dropped_columns.append(col)
        elif method == "onehot":
            onehot_columns.append(col); categorical_columns.append(col)
        elif method == "label":
            label_columns.append(col); categorical_columns.append(col)
        elif method == "ordinal":
            ordinal_columns.append(col); categorical_columns.append(col)
        elif method == "manual":
            manual_columns.append(col); categorical_columns.append(col)
    numeric_columns = [c for c in X.columns if c not in categorical_columns and c not in dropped_columns]
    transformers = []
    if numeric_columns:
        numeric_pipe = Pipeline([("outliers", IQRTransformer(mode=outlier_mode)),
                                 ("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", numeric_pipe, numeric_columns))
    if onehot_columns:
        onehot_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", make_onehot_encoder())])
        transformers.append(("onehot", onehot_pipe, onehot_columns))
    if label_columns:
        label_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        transformers.append(("label", label_pipe, label_columns))
    for col in ordinal_columns:
        order = ordinal_orders.get(col)
        if not order:
            raise ValueError(f"A ordem ordinal da coluna '{col}' não foi informada.")
        ordinal_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OrdinalEncoder(categories=[order],
                                                            handle_unknown="use_encoded_value",
                                                            unknown_value=-1))])
        transformers.append((f"ordinal_{col}", ordinal_pipe, [col]))
    for col in manual_columns:
        mapping = manual_maps.get(col, {})
        manual_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                ("mapper", ManualMapper(mapping=mapping, unknown_value=-1.0))])
        transformers.append((f"manual_{col}", manual_pipe, [col]))
    if not transformers:
        raise ValueError("Nenhuma variável ficou disponível para o modelo.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


# ============================================================
# DESCOBERTA DE ESTIMADORES
# ============================================================

@st.cache_data(show_spinner=False)
def get_all_classifiers() -> list[str]:
    valid = []
    for name, klass in all_estimators(type_filter="classifier"):
        try:
            klass(); valid.append(name)
        except Exception:
            continue
    if xgb is not None and hasattr(xgb, "XGBClassifier"):
        valid.append("XGBClassifier")
    if lgb is not None and hasattr(lgb, "LGBMClassifier"):
        valid.append("LGBMClassifier")
    if cb is not None and hasattr(cb, "CatBoostClassifier"):
        valid.append("CatBoostClassifier")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_classifier_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="classifier")))
    except Exception:
        pass
    if xgb is not None:
        reg["XGBClassifier"] = xgb.XGBClassifier
    if lgb is not None:
        reg["LGBMClassifier"] = lgb.LGBMClassifier
    if cb is not None:
        reg["CatBoostClassifier"] = cb.CatBoostClassifier
    return reg


@st.cache_data(show_spinner=False)
def get_all_regressors() -> list[str]:
    valid = []
    for name, klass in all_estimators(type_filter="regressor"):
        try:
            klass(); valid.append(name)
        except Exception:
            continue
    if xgb is not None and hasattr(xgb, "XGBRegressor"):
        valid.append("XGBRegressor")
    if lgb is not None and hasattr(lgb, "LGBMRegressor"):
        valid.append("LGBMRegressor")
    if cb is not None and hasattr(cb, "CatBoostRegressor"):
        valid.append("CatBoostRegressor")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_regressor_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="regressor")))
    except Exception:
        pass
    if xgb is not None:
        reg["XGBRegressor"] = xgb.XGBRegressor
    if lgb is not None:
        reg["LGBMRegressor"] = lgb.LGBMRegressor
    if cb is not None:
        reg["CatBoostRegressor"] = cb.CatBoostRegressor
    return reg


@st.cache_data(show_spinner=False)
def get_all_clusterers() -> list[str]:
    valid = []
    try:
        for name, klass in all_estimators(type_filter="cluster"):
            try:
                klass(); valid.append(name)
            except Exception:
                continue
    except Exception:
        pass
    for name in CURATED_CLUSTERERS:
        if name not in valid:
            valid.append(name)
    if hdbscan_lib is not None:
        valid.append("HDBSCAN")
    if MiniSom is not None:
        valid.append("KohonenSOM")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_clusterer_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="cluster")))
    except Exception:
        pass
    if hdbscan_lib is not None:
        reg["HDBSCAN"] = hdbscan_lib.HDBSCAN
    if MiniSom is not None:
        reg["KohonenSOM"] = KohonenClusterer
    return reg


# ============================================================
# SEED
# ============================================================

def _set_random_state(estimator, random_state):
    if random_state is None:
        return estimator
    try:
        params = estimator.get_params()
    except Exception:
        return estimator
    if "random_state" in params:
        try:
            estimator.set_params(random_state=random_state)
        except Exception:
            pass
    return estimator


def resolve_random_state(choice, custom_value=None):
    if choice == "fixa":
        return DEFAULT_RANDOM_STATE
    if choice == "aleatoria":
        return None
    return int(custom_value)


# ============================================================
# PRÉ-ANÁLISE
# ============================================================

def analyze_dataset(df, features, target, task):
    X = df[features]
    n = len(X); p = len(features)
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in numeric_cols]
    analysis = {
        "n_samples": int(n), "n_features": int(p),
        "n_numeric": len(numeric_cols), "n_categorical": len(cat_cols),
        "numeric_features": numeric_cols, "categorical_features": cat_cols,
        "missing_pct": float(100 * X.isna().sum().sum() / (n * p)) if n and p else 0.0,
        "warnings": [], "notes": [],
    }
    if n < 100:
        analysis["warnings"].append("Poucas amostras (< 100). Prefira modelos simples.")
    if n > 100_000:
        analysis["notes"].append("Muitas amostras. Modelos escaláveis podem ser melhores.")
    if p == 0:
        analysis["warnings"].append("Nenhuma variável preditora selecionada.")
    if p > 0 and n > 0 and p > n:
        analysis["warnings"].append("p > n. Modelos regularizados são indicados.")
    if p > 50:
        analysis["warnings"].append("Alta dimensionalidade.")
    if analysis["missing_pct"] > 20:
        analysis["warnings"].append(f"{analysis['missing_pct']:.1f}% valores ausentes.")
    if cat_cols:
        analysis["notes"].append(f"{len(cat_cols)} variável(is) categórica(s).")

    if task == "regression" and target and target in df.columns:
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(y):
            analysis["target_mean"] = float(y.mean())
            analysis["target_std"] = float(y.std())
            try:
                analysis["target_skew"] = float(y.skew())
            except Exception:
                analysis["target_skew"] = 0.0
            if abs(analysis["target_skew"]) > 2:
                analysis["warnings"].append("Alvo assimétrico — considere transformação.")
            if numeric_cols and len(y) > 5:
                try:
                    X_num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
                    y_aligned = y.reindex(X_num.index).dropna()
                    if len(y_aligned) > 5:
                        corrs = X_num.loc[y_aligned.index].corrwith(y_aligned).abs().mean()
                        analysis["mean_abs_corr"] = float(corrs) if pd.notna(corrs) else 0.0
                        if analysis["mean_abs_corr"] < 0.15:
                            analysis["notes"].append("Correlação linear baixa.")
                except Exception:
                    pass
    elif task == "classification" and target and target in df.columns:
        y = df[target].astype("string").fillna("missing")
        counts = y.value_counts()
        analysis["n_classes"] = int(len(counts))
        analysis["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        analysis["is_binary"] = len(counts) == 2
        if len(counts) > 1 and counts.min() > 0:
            ratio = float(counts.max() / counts.min())
            analysis["imbalance_ratio"] = ratio
            if ratio > 5:
                analysis["warnings"].append(f"Classes desbalanceadas (razão {ratio:.1f}).")
        if len(counts) >= 2 and counts.min() < 5:
            analysis["warnings"].append("Alguma classe tem < 5 amostras.")
        if len(counts) > 10:
            analysis["warnings"].append(f"Muitas classes ({len(counts)}).")
    elif task == "clustering":
        if p > 20:
            analysis["notes"].append("Alta dimensionalidade tende a degradar o agrupamento.")
        try:
            if len(numeric_cols) > 1:
                corr = X[numeric_cols].corr().abs().values
                np.fill_diagonal(corr, 0)
                if corr.size:
                    analysis["max_feature_corr"] = float(np.nanmax(corr))
                    if analysis["max_feature_corr"] > 0.95:
                        analysis["notes"].append("Features altamente correlacionadas.")
        except Exception:
            pass
    return analysis


def recommend_algorithms(analysis, task):
    n = analysis["n_samples"]; p = analysis["n_features"]
    recs = []
    if task == "regression":
        recs = [("LinearRegression", "Baseline linear."),
                ("Ridge", "L2 robusto.")]
        if p >= 3 and n >= 30:
            recs += [("Lasso", "L1 com seleção."), ("ElasticNet", "L1+L2.")]
        if n <= 5000:
            recs += [("KNeighborsRegressor", "Amostras pequenas."), ("SVR", "Alta dimensão.")]
        recs.append(("DecisionTreeRegressor", "Interpretável."))
        if n >= 200:
            recs += [("RandomForestRegressor", "Ensemble robusto."),
                     ("ExtraTreesRegressor", "Ensemble aleatório."),
                     ("GradientBoostingRegressor", "Boosting acurácia.")]
        if n >= 1000:
            recs.append(("HistGradientBoostingRegressor", "Boosting rápido."))
        if p > n and n > 0:
            recs.insert(0, ("ElasticNet", "p > n."))
            recs.insert(0, ("Ridge", "p > n."))
    elif task == "classification":
        recs.append(("LogisticRegression", "Baseline linear."))
        if analysis.get("is_binary", False):
            recs += [("LinearDiscriminantAnalysis", "Binário gaussiano."),
                     ("RidgeClassifier", "L2 robusto.")]
        if n <= 5000:
            recs += [("KNeighborsClassifier", "Amostras pequenas."),
                     ("SVC", "SVM kernelizado."),
                     ("GaussianNB", "Baseline probabilístico.")]
        recs.append(("DecisionTreeClassifier", "Interpretável."))
        if n >= 200:
            recs += [("RandomForestClassifier", "Ensemble robusto."),
                     ("ExtraTreesClassifier", "Ensemble aleatório.")]
        if n >= 500:
            recs += [("GradientBoostingClassifier", "Boosting acurácia."),
                     ("HistGradientBoostingClassifier", "Boosting escalável."),
                     ("AdaBoostClassifier", "Boosting adaptativo.")]
        if analysis.get("imbalance_ratio", 1.0) > 5:
            recs += [("RandomForestClassifier", "class_weight='balanced'."),
                     ("LogisticRegression", "class_weight='balanced'.")]
        if p > 50:
            recs.insert(0, ("LogisticRegression", "Alta dimensão."))
    elif task == "clustering":
        recs = [("KMeans", "Clusters esféricos.")]
        if n >= 10000:
            recs += [("MiniBatchKMeans", "Escalável."), ("Birch", "Escalável.")]
        if n <= 10000:
            recs += [("AgglomerativeClustering", "Hierárquico."),
                     ("GaussianMixture", "Probabilístico.")]
        recs += [("DBSCAN", "Density-based."), ("MeanShift", "Sem k fixo."),
                 ("AffinityPropagation", "Sem k fixo.")]
        if n <= 5000:
            recs.append(("SpectralClustering", "Não convexos."))
        recs.append(("OPTICS", "Densidades variadas."))
        if hdbscan_lib is not None:
            recs.append(("HDBSCAN", "DBSCAN hierárquico."))
        if MiniSom is not None:
            recs.append(("KohonenSOM", "Mapa auto-organizável."))
    seen = set(); out = []
    for r in recs:
        if r[0] not in seen:
            seen.add(r[0]); out.append(r)
    return out


def _render_pre_analysis(analysis, recommendations):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", analysis["n_samples"])
    c2.metric("Features", analysis["n_features"])
    c3.metric("Numéricas", analysis["n_numeric"])
    c4.metric("Categóricas", analysis["n_categorical"])
    st.caption(f"Valores ausentes: {analysis['missing_pct']:.2f}%")
    if analysis.get("target_skew") is not None:
        st.caption(f"Alvo: média={analysis.get('target_mean'):.3f}, "
                   f"desvio={analysis.get('target_std'):.3f}, "
                   f"assimetria={analysis.get('target_skew'):.3f}")
    if analysis.get("n_classes"):
        st.caption(f"Classes: {analysis['n_classes']} "
                   f"(desbalanceamento ≈ {analysis.get('imbalance_ratio', 1.0):.2f})")
    for w in analysis["warnings"]:
        st.warning(w)
    for note in analysis["notes"]:
        st.info(note)
    if recommendations:
        st.markdown("#### ✅ Algoritmos recomendados")
        ui_dataframe(pd.DataFrame(recommendations,
                                  columns=["Algoritmo", "Justificativa"]),
                     hide_index=True)


# ============================================================
# GRADES DE HIPERPARÂMETROS
# ============================================================

def get_regressor_param_grid(name: str):
    grids = {
        "Ridge": {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"regressor__alpha": [0.001, 0.01, 0.1, 1.0]},
        "ElasticNet": {"regressor__alpha": [0.01, 0.1, 1.0],
                       "regressor__l1_ratio": [0.2, 0.5, 0.8]},
        "SVR": {"regressor__C": [0.1, 1.0, 10.0],
                "regressor__kernel": ["rbf", "linear"],
                "regressor__gamma": ["scale", "auto"],
                "regressor__epsilon": [0.01, 0.1, 0.5]},
        "KNeighborsRegressor": {"regressor__n_neighbors": [3, 5, 7, 9, 11],
                                "regressor__weights": ["uniform", "distance"],
                                "regressor__p": [1, 2]},
        "DecisionTreeRegressor": {"regressor__max_depth": [None, 3, 5, 10],
                                  "regressor__min_samples_split": [2, 5, 10],
                                  "regressor__min_samples_leaf": [1, 2, 4]},
        "RandomForestRegressor": {"regressor__n_estimators": [100, 200],
                                  "regressor__max_depth": [None, 5, 10],
                                  "regressor__min_samples_split": [2, 5]},
        "ExtraTreesRegressor": {"regressor__n_estimators": [100, 200],
                                "regressor__max_depth": [None, 5, 10]},
        "GradientBoostingRegressor": {"regressor__n_estimators": [100, 200],
                                      "regressor__learning_rate": [0.05, 0.1, 0.2],
                                      "regressor__max_depth": [2, 3]},
        "HistGradientBoostingRegressor": {"regressor__learning_rate": [0.05, 0.1, 0.2],
                                          "regressor__max_iter": [100, 200]},
        "AdaBoostRegressor": {"regressor__n_estimators": [50, 100, 200],
                              "regressor__learning_rate": [0.05, 0.1, 0.5]},
        "BayesianRidge": {"regressor__alpha_1": [1e-6, 1e-5, 1e-4],
                          "regressor__alpha_2": [1e-6, 1e-5, 1e-4]},
        "XGBRegressor": {"regressor__n_estimators": [100, 300],
                         "regressor__max_depth": [3, 6, 10],
                         "regressor__learning_rate": [0.05, 0.1, 0.2]},
        "LGBMRegressor": {"regressor__n_estimators": [100, 300],
                          "regressor__num_leaves": [15, 31, 63],
                          "regressor__learning_rate": [0.05, 0.1, 0.2]},
        "CatBoostRegressor": {"regressor__iterations": [200, 500],
                              "regressor__depth": [4, 6, 8],
                              "regressor__learning_rate": [0.05, 0.1, 0.2],
                              "regressor__verbose": [False]},
    }
    return grids.get(name)


def get_classifier_param_grid(name: str):
    grids = {
        "LogisticRegression": {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                               "classifier__max_iter": [200, 500]},
        "LinearDiscriminantAnalysis": [
            {"classifier__solver": ["svd"]},
            {"classifier__solver": ["lsqr"], "classifier__shrinkage": [None, "auto"]},
        ],
        "GaussianNB": {"classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
        "KNeighborsClassifier": {"classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
                                 "classifier__weights": ["uniform", "distance"],
                                 "classifier__p": [1, 2]},
        "SVC": {"classifier__C": [0.1, 1.0, 10.0, 100.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
        "DecisionTreeClassifier": {"classifier__criterion": ["gini", "entropy"],
                                   "classifier__max_depth": [None, 2, 3, 4, 5, 7, 10],
                                   "classifier__min_samples_split": [2, 5, 10],
                                   "classifier__min_samples_leaf": [1, 2, 4]},
        "RandomForestClassifier": {"classifier__n_estimators": [100, 200, 400],
                                   "classifier__max_depth": [None, 3, 5, 10],
                                   "classifier__min_samples_split": [2, 5, 10],
                                   "classifier__max_features": ["sqrt", "log2", None]},
        "ExtraTreesClassifier": {"classifier__n_estimators": [100, 200, 400],
                                 "classifier__max_depth": [None, 3, 5, 10],
                                 "classifier__min_samples_split": [2, 5, 10],
                                 "classifier__max_features": ["sqrt", "log2", None]},
        "GradientBoostingClassifier": {"classifier__n_estimators": [50, 100, 200],
                                       "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
                                       "classifier__max_depth": [1, 2, 3],
                                       "classifier__subsample": [0.8, 1.0]},
        "HistGradientBoostingClassifier": {"classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
                                           "classifier__max_iter": [100, 200, 300],
                                           "classifier__max_leaf_nodes": [15, 31, 63],
                                           "classifier__l2_regularization": [0.0, 0.1, 1.0]},
        "AdaBoostClassifier": {"classifier__n_estimators": [50, 100, 200, 400],
                               "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "RidgeClassifier": {"classifier__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                            "classifier__class_weight": [None, "balanced"]},
        "XGBClassifier": {"classifier__n_estimators": [100, 300],
                          "classifier__max_depth": [3, 6, 10],
                          "classifier__learning_rate": [0.05, 0.1, 0.2]},
        "LGBMClassifier": {"classifier__n_estimators": [100, 300],
                           "classifier__num_leaves": [15, 31, 63],
                           "classifier__learning_rate": [0.05, 0.1, 0.2]},
        "CatBoostClassifier": {"classifier__iterations": [200, 500],
                               "classifier__depth": [4, 6, 8],
                               "classifier__learning_rate": [0.05, 0.1, 0.2],
                               "classifier__verbose": [False]},
    }
    return grids.get(name)


def get_clusterer_param_grid(name: str):
    grids = {
        "KMeans": {"n_clusters": list(range(2, 11)), "init": ["k-means++", "random"],
                   "n_init": [10, 20]},
        "MiniBatchKMeans": {"n_clusters": list(range(2, 11)), "n_init": [10, 20]},
        "AgglomerativeClustering": {"n_clusters": list(range(2, 11)),
                                    "linkage": ["ward", "complete", "average"]},
        "DBSCAN": {"eps": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
                   "min_samples": [3, 5, 7, 10]},
        "GaussianMixture": {"n_components": [2, 3, 4, 5, 6],
                            "covariance_type": ["full", "diag"],
                            "reg_covar": [1e-6, 1e-3]},
        "Birch": {"n_clusters": list(range(2, 11)),
                  "threshold": [0.3, 0.5, 0.7, 1.0],
                  "branching_factor": [25, 50]},
        "SpectralClustering": {"n_clusters": list(range(2, 8)),
                               "affinity": ["rbf", "nearest_neighbors"]},
        "MeanShift": {"bandwidth": [None, 0.5, 1.0, 2.0]},
        "AffinityPropagation": {"damping": [0.5, 0.7, 0.9]},
        "OPTICS": {"min_samples": [3, 5, 10], "xi": [0.05, 0.1, 0.2]},
        "HDBSCAN": {"min_cluster_size": [5, 10, 20], "min_samples": [None, 3, 5, 10]},
        "KohonenSOM": {"som_x": [3, 5], "som_y": [3, 5],
                       "sigma": [1.0], "learning_rate": [0.5],
                       "num_iteration": [500, 1000]},
    }
    return grids.get(name)


def get_ts_param_grid(name: str):
    grids = {
        "ARIMA": [
            {"order": [(1, 0, 1)]},
            {"order": [(1, 1, 1)]},
            {"order": [(2, 1, 1)]},
            {"order": [(1, 1, 2)]},
            {"order": [(2, 1, 2)]},
        ],
        "SARIMAX": [
            {"order": [(1, 1, 1)], "seasonal_order": [(0, 0, 0, 0)]},
            {"order": [(1, 1, 1)], "seasonal_order": [(1, 1, 1, 12)]},
            {"order": [(1, 1, 1)], "seasonal_order": [(1, 1, 1, 7)]},
        ],
        "ExponentialSmoothing": [
            {"trend": ["add"], "seasonal": [None], "seasonal_periods": [None]},
            {"trend": ["add"], "seasonal": ["add"], "seasonal_periods": [12]},
            {"trend": ["add"], "seasonal": ["mul"], "seasonal_periods": [12]},
            {"trend": ["add"], "seasonal": ["add"], "seasonal_periods": [7]},
        ],
        "Prophet": [
            {"yearly_seasonality": [True, "auto"]},
            {"yearly_seasonality": [False, "auto"], "weekly_seasonality": [True]},
        ],
        "NaiveForecaster": {},
        "ThetaForecaster": {},
        "PolynomialTrendForecaster": {"degree": [1, 2]},
    }
    return grids.get(name)


# ============================================================
# AJUSTE MANUAL UNIVERSAL
# ============================================================

def parse_param_string(value_str: str):
    s = str(value_str).strip()
    low = s.lower()
    if low in {"none", "null"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if s == "":
        return s
    try:
        if "." not in s and "e" not in low:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def format_param_value(value) -> str:
    return "None" if value is None else str(value)


def extract_editable_params(estimator, prefix: str) -> list[str]:
    try:
        params = estimator.get_params(deep=True)
    except Exception:
        return []
    editable = []
    skip = {"n_jobs", "random_state", "verbose", "warm_start", "copy_X",
            "fit_intercept", "positive", "normalize", "precompute", "selection", "tol"}
    for key, value in params.items():
        if prefix and not key.startswith(prefix):
            continue
        short = key.split("__", 1)[-1]
        if short in skip:
            continue
        if isinstance(value, (int, float, bool, str, type(None))):
            editable.append(key)
    return sorted(editable)


def render_manual_params_universal(estimator, prefix, current_params, key_prefix):
    keys = extract_editable_params(estimator, prefix)
    if not keys and prefix:
        raw_keys = extract_editable_params(estimator, "")
        keys = [f"{prefix}{k}" for k in raw_keys]
    if not keys:
        st.info("Este modelo não expôs hiperparâmetros editáveis.")
        return {}
    edited: dict[str, str] = {}
    cols = st.columns(min(3, max(1, len(keys))))
    for i, key in enumerate(keys):
        short = key.split("__", 1)[-1] if "__" in key else key
        cur_val = current_params.get(key, "")
        with cols[i % len(cols)]:
            edited[key] = st.text_input(
                short,
                value=format_param_value(cur_val),
                key=f"{key_prefix}_{key}",
            )
    return edited


# ============================================================
# PIPELINES
# ============================================================

def make_regression_pipeline(estimator, X_train, encoding_map, ordinal_orders,
                              manual_maps, outlier_mode, threshold, max_features):
    pre = build_preprocessor(X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode)
    return Pipeline([
        ("preprocessor", pre),
        ("selector", CorrelationSelector(threshold, max_features)),
        ("scaler", StandardScaler()),
        ("regressor", estimator),
    ])


def make_classification_pipeline(estimator, X_train, encoding_map, ordinal_orders, manual_maps):
    pre = build_preprocessor(X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode="keep")
    return Pipeline([("preprocessor", pre), ("classifier", estimator)])


def make_clustering_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode="keep"):
    pre = build_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode=outlier_mode)
    return Pipeline([("preprocessor", pre), ("scaler", StandardScaler())])


# ============================================================
# AVALIAÇÃO — REGRESSÃO
# ============================================================

def randomized_iteration_limit(params, requested):
    if not params:
        return 1
    combos = 1
    for v in params.values():
        if isinstance(v, list):
            combos *= max(len(v), 1)
    return max(1, min(int(requested), combos))


def evaluate_regression(X_train, y_train, X_test, y_test, models, encoding_map,
                        ordinal_orders, manual_maps, outlier_mode, threshold=0.10,
                        max_features=None, cv=5, search_method="grid", n_iter=20,
                        random_state=42):
    results = {}
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    progress = st.progress(0)
    status = st.empty()
    items = list(models.items())
    for i, (name, cfg) in enumerate(items, start=1):
        status.info(f"Treinando **{name}** ({i}/{len(items)})...")
        try:
            pipe = make_regression_pipeline(clone(cfg["estimator"]), X_train,
                                            encoding_map, ordinal_orders, manual_maps,
                                            outlier_mode, threshold, max_features)
            cv_rows = None
            if cfg["params"]:
                if search_method == "grid":
                    search = GridSearchCV(pipe, cfg["params"], scoring="r2",
                                          cv=cv_strategy, n_jobs=-1,
                                          return_train_score=True, refit=True, error_score="raise")
                elif search_method == "random":
                    n_rand = randomized_iteration_limit(cfg["params"], n_iter)
                    search = RandomizedSearchCV(pipe, cfg["params"], n_iter=n_rand,
                                                scoring="r2", cv=cv_strategy, n_jobs=-1,
                                                return_train_score=True, refit=True,
                                                random_state=random_state, error_score="raise")
                else:
                    manual = {k: [v[0]] for k, v in cfg["params"].items()}
                    search = GridSearchCV(pipe, manual, scoring="r2", cv=cv_strategy,
                                          n_jobs=-1, return_train_score=True, refit=True,
                                          error_score="raise")
                search.fit(X_train, y_train)
                best = search.best_estimator_
                best_params = search.best_params_
                best_cv = float(search.best_score_)
                best_std = float(search.cv_results_["std_test_score"][search.best_index_])
                idx_best = search.best_index_
                cv_scores = np.asarray(
                    [search.cv_results_[f"split{k}_test_score"][idx_best] for k in range(cv)],
                    dtype=float)
                cv_rows = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15)
            else:
                cv_res = cross_validate(
                    pipe, X_train, y_train,
                    scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error",
                             "mae": "neg_mean_absolute_error"},
                    cv=cv_strategy, n_jobs=-1, error_score="raise")
                best = pipe.fit(X_train, y_train)
                best_params = {}
                best_cv = float(cv_res["test_r2"].mean())
                best_std = float(cv_res["test_r2"].std())
                cv_scores = np.asarray(cv_res["test_r2"], dtype=float)
                cv_rows = None
            pred = best.predict(X_test)
            mse = float(mean_squared_error(y_test, pred))
            results[name] = {
                "best_model": best, "best_params": best_params,
                "cv_mean": best_cv, "cv_std": best_std, "cv_scores": cv_scores,
                "test_r2": float(r2_score(y_test, pred)),
                "test_rmse": float(np.sqrt(mse)),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_mse": mse, "y_pred": pred, "search_table": cv_rows,
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(i / len(items))
    status.success("✅ Regressão concluída!")
    progress.empty()
    return results


# ============================================================
# AVALIAÇÃO — CLASSIFICAÇÃO
# ============================================================

def _format_best_params(params):
    if not params:
        return "—"
    return ", ".join(f"{k.replace('classifier__', '')}={v}" for k, v in params.items())


def evaluate_classifiers(X_train, y_train, X_test, y_test, classifier_names,
                         encoding_map, ordinal_orders, manual_maps, cv=5,
                         random_state=42, tune=False, search_method="grid",
                         n_iter=20, tuning_metric="f1_weighted"):
    registry = dict(get_classifier_discovery_registry())
    results = {}
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {"accuracy": "accuracy", "precision_weighted": "precision_weighted",
               "recall_weighted": "recall_weighted", "f1_weighted": "f1_weighted",
               "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    tuning_scorer = scoring[tuning_metric]
    progress = st.progress(0)
    status = st.empty()
    names = list(dict.fromkeys(classifier_names))
    for i, name in enumerate(names, start=1):
        status.info(f"Avaliando **{name}** ({i}/{len(names)})...")
        try:
            if name not in registry:
                raise ValueError(f"Classificador '{name}' não encontrado.")
            estimator = _set_random_state(registry[name](), random_state)
            pipe = make_classification_pipeline(estimator, X_train,
                                                encoding_map, ordinal_orders, manual_maps)
            param_grid = get_classifier_param_grid(name) if tune else None
            best_params = {}; tuning_table = None; tuned = False
            if tune and param_grid:
                if search_method == "grid":
                    search = GridSearchCV(pipe, param_grid, scoring=tuning_scorer,
                                          cv=cv_strategy, n_jobs=-1, refit=True, error_score="raise")
                else:
                    combos = list(ParameterGrid(param_grid))
                    n_draw = min(int(n_iter), len(combos))
                    search = RandomizedSearchCV(pipe, param_distributions=param_grid,
                                                n_iter=n_draw, scoring=tuning_scorer,
                                                cv=cv_strategy, n_jobs=-1, refit=True,
                                                random_state=random_state, error_score="raise")
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                tuned = True
                tuning_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15)
                cv_best = float(search.best_score_)
                cv_best_std = float(search.cv_results_["std_test_score"][search.best_index_])
            else:
                model = pipe.fit(X_train, y_train)
                cv_res = cross_validate(pipe, X_train, y_train, cv=cv_strategy,
                                        scoring=scoring, n_jobs=-1, error_score="raise")
                cv_best = float(cv_res[f"test_{tuning_metric}"].mean())
                cv_best_std = float(cv_res[f"test_{tuning_metric}"].std())
            if tuned:
                cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy,
                                        scoring=scoring, n_jobs=-1, error_score="raise")
            y_pred = model.predict(X_test)
            results[name] = {
                "best_model": model, "tuned": tuned,
                "tuning_method": search_method if tuned else "none",
                "tuning_metric": tuning_metric, "best_params": best_params,
                "best_params_text": _format_best_params(best_params),
                "tuning_table": tuning_table,
                "cv_tuning_mean": cv_best, "cv_tuning_std": cv_best_std,
                "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
                "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
                "cv_f1_weighted_mean": float(cv_res["test_f1_weighted"].mean()),
                "cv_f1_weighted_std": float(cv_res["test_f1_weighted"].std()),
                "cv_f1_macro_mean": float(cv_res["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(cv_res["test_f1_macro"].std()),
                "cv_balanced_accuracy_mean": float(cv_res["test_balanced_accuracy"].mean()),
                "cv_acc_scores": cv_res["test_accuracy"].tolist(),
                "cv_f1_scores": cv_res["test_f1_weighted"].tolist(),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "test_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "test_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "y_pred": y_pred,
            }
            try:
                if hasattr(model, "predict_proba"):
                    sv = model.predict_proba(X_test)
                    if sv.shape[1] == 2:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, sv[:, 1]))
                    else:
                        results[name]["test_roc_auc"] = float(roc_auc_score(
                            y_test, sv, multi_class="ovr", average="weighted"))
                elif hasattr(model, "decision_function"):
                    sv = model.decision_function(X_test)
                    if np.ndim(sv) == 1:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, sv))
                    else:
                        results[name]["test_roc_auc"] = float(roc_auc_score(
                            y_test, sv, multi_class="ovr", average="weighted"))
            except Exception:
                results[name]["test_roc_auc"] = None
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(i / len(names))
    status.success("✅ Classificação concluída!")
    progress.empty()
    return results


# ============================================================
# AVALIAÇÃO — AGRUPAMENTO
# ============================================================

def calculate_external_cluster_metrics(reference, labels):
    reference = pd.Series(reference).astype("string").fillna("missing").reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    if len(reference) != len(labels):
        return {"adjusted_rand": None, "normalized_mutual_info": None}
    try:
        ari = float(adjusted_rand_score(reference, labels))
    except Exception:
        ari = None
    try:
        nmi = float(normalized_mutual_info_score(reference, labels))
    except Exception:
        nmi = None
    return {"adjusted_rand": ari, "normalized_mutual_info": nmi}


def calculate_cluster_metrics(X_array, labels):
    X_array = np.asarray(X_array); labels = np.asarray(labels)
    noise = labels == -1; valid = ~noise
    X_valid = X_array[valid]; labels_valid = labels[valid]
    unique = np.unique(labels_valid)
    result = {"n_clusters": int(len(unique)), "n_noise": int(noise.sum()),
              "noise_pct": float(noise.mean() * 100) if len(labels) else 0.0,
              "silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    if len(unique) < 2 or len(labels_valid) <= len(unique):
        return result
    for key, fn in [("silhouette", silhouette_score),
                    ("calinski_harabasz", calinski_harabasz_score),
                    ("davies_bouldin", davies_bouldin_score)]:
        try:
            result[key] = float(fn(X_valid, labels_valid))
        except Exception:
            pass
    return result


def cluster_objective(metrics, metric):
    v = metrics.get(metric)
    if v is None or not np.isfinite(v):
        return None
    return float(v)


def make_cluster_estimator(name, params, random_state=None):
    registry = dict(get_clusterer_discovery_registry())
    registry.setdefault("KMeans", KMeans)
    registry.setdefault("MiniBatchKMeans", MiniBatchKMeans)
    registry.setdefault("AgglomerativeClustering", AgglomerativeClustering)
    registry.setdefault("DBSCAN", DBSCAN)
    registry.setdefault("GaussianMixture", GaussianMixture)
    registry.setdefault("Birch", Birch)
    registry.setdefault("MeanShift", MeanShift)
    registry.setdefault("SpectralClustering", SpectralClustering)
    registry.setdefault("AffinityPropagation", AffinityPropagation)
    registry.setdefault("OPTICS", OPTICS)
    if hdbscan_lib is not None:
        registry.setdefault("HDBSCAN", hdbscan_lib.HDBSCAN)
    if MiniSom is not None:
        registry.setdefault("KohonenSOM", KohonenClusterer)
    if name not in registry:
        raise ValueError(f"Algoritmo desconhecido: {name}")
    try:
        est = registry[name]()
    except Exception as exc:
        raise ValueError(f"Não foi possível instanciar {name}: {exc}")
    try:
        valid = set(est.get_params().keys())
    except Exception:
        valid = set(params.keys())
    filtered = {k: v for k, v in params.items() if k in valid}
    if random_state is not None and "random_state" in valid:
        filtered.setdefault("random_state", random_state)
    try:
        est.set_params(**filtered)
    except Exception as exc:
        raise ValueError(f"Erro ao configurar {name}: {exc}")
    return est


def tune_single_clustering_algorithm(name, X_transformed, tune=True,
                                     search_method="grid", n_iter=20,
                                     objective="silhouette", random_state=42,
                                     max_combinations=30,
                                     progress_cb=None):
    space = get_clusterer_param_grid(name)
    defaults = {
        "KMeans": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
        "MiniBatchKMeans": {"n_clusters": 3, "n_init": 10},
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
        "DBSCAN": {"eps": 0.5, "min_samples": 5},
        "GaussianMixture": {"n_components": 3, "covariance_type": "full",
                            "reg_covar": 1e-6},
        "Birch": {"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
        "MeanShift": {}, "SpectralClustering": {"n_clusters": 3},
        "AffinityPropagation": {"damping": 0.9}, "OPTICS": {"min_samples": 5},
        "HDBSCAN": {"min_cluster_size": 5},
        "KohonenSOM": {"som_x": 5, "som_y": 5, "sigma": 1.0,
                       "learning_rate": 0.5, "num_iteration": 1000},
    }

    if not tune or space is None:
        candidates = [defaults.get(name, {})]
    elif search_method == "grid":
        try:
            full = list(ParameterGrid(space))
        except Exception:
            full = [defaults.get(name, {})]
        if len(full) > max_combinations:
            candidates = list(ParameterSampler(
                space, n_iter=max_combinations, random_state=random_state))
        else:
            candidates = full
    else:
        try:
            total = max(1, len(list(ParameterGrid(space))))
        except Exception:
            total = 1
        candidates = list(ParameterSampler(
            space, n_iter=min(int(n_iter), total, max_combinations),
            random_state=random_state))

    rows = []
    best = None
    best_value = np.inf if objective == "davies_bouldin" else -np.inf
    n_total = len(candidates)

    for i, params in enumerate(candidates, start=1):
        if progress_cb is not None:
            try:
                progress_cb(name, i, n_total)
            except Exception:
                pass
        try:
            model = make_cluster_estimator(name, params, random_state)
            labels = model.fit_predict(X_transformed)
            metrics = calculate_cluster_metrics(X_transformed, labels)
            value = cluster_objective(metrics, objective)
            if value is None:
                continue
            row = dict(params)
            row.update({"n_clusters_real": metrics["n_clusters"],
                        "noise_pct": metrics["noise_pct"],
                        "silhouette": metrics["silhouette"],
                        "calinski_harabasz": metrics["calinski_harabasz"],
                        "davies_bouldin": metrics["davies_bouldin"],
                        "objective": value})
            rows.append(row)
            improved = value < best_value if objective == "davies_bouldin" else value > best_value
            if improved:
                best_value = value
                best = {"estimator": model, "labels": np.asarray(labels),
                        "metrics": metrics, "best_params": params}
        except Exception:
            continue

    if best is None:
        params = defaults.get(name, {})
        model = make_cluster_estimator(name, params, random_state)
        labels = model.fit_predict(X_transformed)
        best = {"estimator": model, "labels": np.asarray(labels),
                "metrics": calculate_cluster_metrics(X_transformed, labels),
                "best_params": params}

    tuning_table = pd.DataFrame(rows)
    if not tuning_table.empty:
        tuning_table = tuning_table.sort_values(
            "objective", ascending=(objective == "davies_bouldin")).head(20)

    best["tuned"] = bool(tune)
    best["tuning_method"] = search_method if tune else "none"
    best["tuning_metric"] = objective
    best["tuning_table"] = tuning_table
    return best


def fit_clustering_models(X, encoding_map, ordinal_orders, manual_maps,
                          selected_algorithms, n_clusters=3, eps=0.5, min_samples=5,
                          outlier_mode="keep", random_state=42, reference=None,
                          tune=False, search_method="grid", n_iter=20,
                          objective="silhouette", max_combinations=30):
    pipeline = make_clustering_preprocessor(X, encoding_map, ordinal_orders,
                                            manual_maps, outlier_mode)
    X_transformed = pipeline.fit_transform(X)

    results = {}
    n_algos = max(len(selected_algorithms), 1)
    overall = st.progress(0.0)
    status = st.empty()

    def _cb(name, i, total):
        try:
            base = list(selected_algorithms).index(name) / n_algos
        except ValueError:
            base = 0.0
        frac = (i / max(total, 1)) / n_algos
        status.info(f"🧩 **{name}** — combinação {i}/{total}")
        overall.progress(min(0.99, base + frac))

    for idx, name in enumerate(selected_algorithms, start=1):
        status.info(f"▶️ Iniciando **{name}** ({idx}/{n_algos})...")
        try:
            if tune:
                result = tune_single_clustering_algorithm(
                    name, X_transformed, tune=True,
                    search_method=search_method, n_iter=n_iter,
                    objective=objective, random_state=random_state,
                    max_combinations=max_combinations,
                    progress_cb=_cb)
            else:
                defaults = {
                    "KMeans": {"n_clusters": n_clusters, "init": "k-means++", "n_init": 10},
                    "MiniBatchKMeans": {"n_clusters": n_clusters, "n_init": 10},
                    "AgglomerativeClustering": {"n_clusters": n_clusters,
                                                "linkage": "ward"},
                    "DBSCAN": {"eps": eps, "min_samples": min_samples},
                    "GaussianMixture": {"n_components": n_clusters,
                                        "covariance_type": "full",
                                        "reg_covar": 1e-6},
                    "Birch": {"n_clusters": n_clusters, "threshold": 0.5,
                              "branching_factor": 50},
                    "MeanShift": {}, "SpectralClustering": {"n_clusters": n_clusters},
                    "AffinityPropagation": {"damping": 0.9},
                    "OPTICS": {"min_samples": min_samples},
                    "HDBSCAN": {"min_cluster_size": min_samples},
                    "KohonenSOM": {"som_x": 5, "som_y": 5, "sigma": 1.0,
                                   "learning_rate": 0.5, "num_iteration": 1000},
                }
                params = defaults.get(name, {})
                model = make_cluster_estimator(name, params, random_state)
                labels = model.fit_predict(X_transformed)
                result = {
                    "estimator": model, "labels": np.asarray(labels),
                    "metrics": calculate_cluster_metrics(X_transformed, labels),
                    "best_params": params, "tuned": False,
                    "tuning_method": "none", "tuning_metric": objective,
                    "tuning_table": None,
                }
            if reference is not None:
                result["metrics"].update(
                    calculate_external_cluster_metrics(reference, result["labels"]))
            results[name] = result
        except Exception as exc:
            results[name] = {"error": str(exc)}
        overall.progress(idx / n_algos)

    status.success("✅ Agrupamento concluído!")
    overall.empty()
    return X_transformed, results


# ============================================================
# DASHBOARDS DINÂMICOS DE COMPARAÇÃO
# ============================================================

def _render_dynamic_selector(all_models, key_prefix, default_top_n=None, help_text=""):
    if default_top_n is None:
        default = all_models
    else:
        default = all_models[: min(default_top_n, len(all_models))]
    col1, col2 = st.columns([3, 1])
    with col1:
        chosen = st.multiselect(
            "Modelos para comparar",
            all_models,
            default=default,
            key=f"{key_prefix}_models",
            help=help_text or "Escolha quais modelos exibir nos gráficos.",
        )
    with col2:
        max_facet = st.number_input(
            "Máx. em facetas",
            min_value=2, max_value=12, value=6, step=1,
            key=f"{key_prefix}_max_facet",
            help="Acima desse número, os gráficos vão para expanders (um por modelo).",
        )
    return chosen, int(max_facet)


def regression_comparison_dashboard(y_test, valid, selected_models, max_facet_models=6):
    filtered = {k: valid[k] for k in selected_models if k in valid}
    if not filtered:
        return None
    real = np.asarray(y_test)
    rows_bar = []
    for name, res in filtered.items():
        rows_bar.append({
            "Modelo": name,
            "R² teste": float(res.get("test_r2", np.nan)),
            "RMSE teste": float(res.get("test_rmse", np.nan)),
            "MAE teste": float(res.get("test_mae", np.nan)),
        })
    df_bar = pd.DataFrame(rows_bar)
    fig_bar = None
    if not df_bar.empty:
        df_long = df_bar.melt(id_vars="Modelo",
                              value_vars=["R² teste", "RMSE teste", "MAE teste"],
                              var_name="Métrica", value_name="Valor")
        fig_bar = px.bar(df_long, x="Modelo", y="Valor", color="Métrica",
                         barmode="group", title="📊 Métricas no conjunto de teste")
        fig_bar.update_layout(height=420, xaxis_tickangle=-30)
    rows_cv = []
    for name, res in filtered.items():
        cv = res.get("cv_scores")
        if cv is None:
            continue
        cv = np.asarray(cv).ravel()
        if cv.size == 0:
            continue
        for fold, score in enumerate(cv, start=1):
            rows_cv.append({"Modelo": name, "Fold": fold, "R²": float(score)})
    df_cv = pd.DataFrame(rows_cv)
    fig_box = None; fig_box_fb = None
    if not df_cv.empty:
        fig_box = px.box(df_cv, x="Modelo", y="R²", color="Modelo",
                         points="all", hover_data=["Fold"],
                         title="📦 R² por dobra da CV")
        fig_box.update_layout(height=520, showlegend=False, xaxis_tickangle=-30)
    elif not df_bar.empty:
        fig_box_fb = px.bar(df_bar.sort_values("R² teste", ascending=False),
                            x="Modelo", y="R² teste", color="Modelo",
                            title="📊 R² no teste (sem CV)", text="R² teste")
        fig_box_fb.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_box_fb.update_layout(height=520, showlegend=False, xaxis_tickangle=-30)
    fig_real = None; fig_resid = None; figuras_por_modelo = []
    n = len(filtered)
    if n <= max_facet_models:
        rows = []
        for name, res in filtered.items():
            pred = np.asarray(res["y_pred"])
            for r, p in zip(real, pred):
                rows.append({"Modelo": name, "Real": r, "Predito": p, "Resíduo": r - p})
        df_sc = pd.DataFrame(rows)
        n_cols = 2 if n > 1 else 1
        n_rows = int(np.ceil(n / n_cols))
        v_spacing = min(0.05, 0.5 / (n_rows - 1)) if n_rows > 1 else 0.0
        height = max(400, 240 * n_rows)
        fig_real = px.scatter(df_sc, x="Real", y="Predito", color="Modelo",
                              facet_col="Modelo", facet_col_wrap=n_cols,
                              facet_row_spacing=v_spacing,
                              hover_data=["Resíduo"], opacity=0.7,
                              title="🎯 Real × Predito")
        fig_real.update_layout(height=height)
        fig_real.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        for i, name in enumerate(filtered.keys()):
            sub = df_sc[df_sc["Modelo"] == name]
            if sub.empty:
                continue
            lo = float(min(sub["Real"].min(), sub["Predito"].min()))
            hi = float(max(sub["Real"].max(), sub["Predito"].max()))
            fig_real.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                               line=dict(dash="dash", color="red"),
                               row=(i // n_cols) + 1, col=(i % n_cols) + 1)
        fig_resid = px.scatter(df_sc, x="Predito", y="Resíduo", color="Modelo",
                               facet_col="Modelo", facet_col_wrap=n_cols,
                               facet_row_spacing=v_spacing, opacity=0.7,
                               title="📉 Resíduos")
        fig_resid.add_hline(y=0, line_dash="dash")
        fig_resid.update_layout(height=height)
        fig_resid.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        for name, res in filtered.items():
            pred = np.asarray(res["y_pred"])
            sub = pd.DataFrame({"Real": real, "Predito": pred, "Resíduo": real - pred})
            lo = float(min(sub["Real"].min(), sub["Predito"].min()))
            hi = float(max(sub["Real"].max(), sub["Predito"].max()))
            f1 = px.scatter(sub, x="Real", y="Predito", opacity=0.7,
                            title=f"Real × Predito — {name}")
            f1.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                         line=dict(dash="dash", color="red"))
            f1.update_layout(height=400)
            f2 = px.scatter(sub, x="Predito", y="Resíduo", opacity=0.7,
                            title=f"Resíduos — {name}")
            f2.add_hline(y=0, line_dash="dash", line_color="red")
            f2.update_layout(height=400)
            figuras_por_modelo.append((name, f1, f2))
    return {"bar_metrics": fig_bar, "box_cv": fig_box, "box_cv_fallback": fig_box_fb,
            "real_pred": fig_real, "resid": fig_resid, "por_modelo": figuras_por_modelo,
            "df_bar": df_bar, "df_cv": df_cv}


def classification_comparison_dashboard(y_test, valid, selected_models,
                                        classes, max_facet_models=6):
    filtered = {k: valid[k] for k in selected_models if k in valid}
    if not filtered:
        return None
    rows_bar = []
    for name, res in filtered.items():
        rows_bar.append({
            "Modelo": name,
            "Acurácia": float(res.get("test_accuracy", np.nan)),
            "F1 ponderado": float(res.get("test_f1", np.nan)),
            "F1 macro": float(res.get("test_f1_macro", np.nan)),
            "Balanced acc.": float(res.get("test_balanced_accuracy", np.nan)),
        })
    df_bar = pd.DataFrame(rows_bar)
    fig_bar = None
    if not df_bar.empty:
        df_long = df_bar.melt(id_vars="Modelo",
                              value_vars=["Acurácia", "F1 ponderado", "F1 macro",
                                          "Balanced acc."],
                              var_name="Métrica", value_name="Valor")
        fig_bar = px.bar(df_long, x="Modelo", y="Valor", color="Métrica",
                         barmode="group", title="📊 Métricas no teste")
        fig_bar.update_layout(height=420, xaxis_tickangle=-30, yaxis_range=[0, 1.05])
    rows_cv = []
    for name, res in filtered.items():
        cv = res.get("cv_f1_scores")
        if cv is None:
            continue
        cv = np.asarray(cv).ravel()
        if cv.size == 0:
            continue
        for fold, score in enumerate(cv, start=1):
            rows_cv.append({"Modelo": name, "Fold": fold, "F1 ponderado": float(score)})
    df_cv = pd.DataFrame(rows_cv)
    fig_box = None; fig_box_fb = None
    if not df_cv.empty:
        fig_box = px.box(df_cv, x="Modelo", y="F1 ponderado", color="Modelo",
                         points="all", hover_data=["Fold"],
                         title="📦 F1 ponderado por dobra da CV")
        fig_box.update_layout(height=520, showlegend=False, xaxis_tickangle=-30,
                              yaxis_range=[0, 1.05])
    elif not df_bar.empty:
        fig_box_fb = px.bar(df_bar.sort_values("F1 ponderado", ascending=False),
                            x="Modelo", y="F1 ponderado", color="Modelo",
                            title="📊 F1 ponderado no teste (sem CV)",
                            text="F1 ponderado")
        fig_box_fb.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_box_fb.update_layout(height=520, showlegend=False, xaxis_tickangle=-30,
                                 yaxis_range=[0, 1.05])
    n = len(filtered)
    fig_cm_facet = None
    cms_por_modelo = []
    if n <= max_facet_models:
        rows_cm = []
        for name, res in filtered.items():
            cm = confusion_matrix(y_test, res["y_pred"])
            for i, real_cls in enumerate(classes):
                for j, pred_cls in enumerate(classes):
                    rows_cm.append({"Modelo": name, "Real": real_cls,
                                    "Predito": pred_cls, "Contagem": int(cm[i, j])})
        df_cm = pd.DataFrame(rows_cm)
        if not df_cm.empty:
            n_cols = 2 if n > 1 else 1
            n_rows = int(np.ceil(n / n_cols))
            v_spacing = min(0.05, 0.5 / (n_rows - 1)) if n_rows > 1 else 0.0
            fig_cm_facet = px.density_heatmap(
                df_cm, x="Predito", y="Real", z="Contagem",
                facet_col="Modelo", facet_col_wrap=n_cols,
                facet_row_spacing=v_spacing,
                text_auto=True, color_continuous_scale="Blues",
                title="🧩 Matrizes de confusão")
            fig_cm_facet.update_layout(height=max(400, 260 * n_rows))
            fig_cm_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        for name, res in filtered.items():
            cm = confusion_matrix(y_test, res["y_pred"])
            f = px.imshow(cm, text_auto=True, x=classes, y=classes,
                          labels={"x": "Predito", "y": "Real", "color": "Qtd"},
                          title=f"Matriz de confusão — {name}",
                          aspect="auto", color_continuous_scale="Blues")
            f.update_layout(height=420)
            cms_por_modelo.append((name, f))
    fig_auc = None
    rows_auc = []
    for name, res in filtered.items():
        auc = res.get("test_roc_auc")
        if auc is None:
            continue
        rows_auc.append({"Modelo": name, "ROC-AUC": float(auc)})
    if rows_auc:
        df_auc = pd.DataFrame(rows_auc)
        fig_auc = px.bar(df_auc.sort_values("ROC-AUC", ascending=False),
                         x="Modelo", y="ROC-AUC", color="Modelo",
                         title="📈 ROC-AUC no teste", text="ROC-AUC")
        fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_auc.update_layout(height=420, showlegend=False, xaxis_tickangle=-30,
                              yaxis_range=[0, 1.05])
    return {"bar_metrics": fig_bar, "box_cv": fig_box, "box_cv_fallback": fig_box_fb,
            "cm_facet": fig_cm_facet, "cm_por_modelo": cms_por_modelo,
            "auc_bar": fig_auc, "df_bar": df_bar, "df_cv": df_cv}


def clustering_comparison_dashboard(X_transformed, valid, selected_models,
                                    max_facet_models=6):
    filtered = {k: valid[k] for k in selected_models if k in valid}
    if not filtered:
        return None
    rows_bar = []
    for name, res in filtered.items():
        m = res["metrics"]
        rows_bar.append({"Modelo": name, "Silhouette": m.get("silhouette"),
                         "Calinski-Harabasz": m.get("calinski_harabasz"),
                         "Davies-Bouldin": m.get("davies_bouldin"),
                         "Nº clusters": m.get("n_clusters"),
                         "Ruído (%)": m.get("noise_pct")})
    df_bar = pd.DataFrame(rows_bar)
    fig_sil = None
    df_sil = df_bar.dropna(subset=["Silhouette"])
    if not df_sil.empty:
        fig_sil = px.bar(df_sil.sort_values("Silhouette", ascending=False),
                         x="Modelo", y="Silhouette", color="Modelo",
                         title="📊 Silhouette (maior = melhor)", text="Silhouette")
        fig_sil.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_sil.update_layout(height=420, showlegend=False, xaxis_tickangle=-30)
    fig_db = None
    df_db = df_bar.dropna(subset=["Davies-Bouldin"])
    if not df_db.empty:
        fig_db = px.bar(df_db.sort_values("Davies-Bouldin"),
                        x="Modelo", y="Davies-Bouldin", color="Modelo",
                        title="📊 Davies-Bouldin (menor = melhor)",
                        text="Davies-Bouldin")
        fig_db.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_db.update_layout(height=420, showlegend=False, xaxis_tickangle=-30)
    n = len(filtered)
    fig_scatter_facet = None
    figuras_por_modelo = []
    X_arr = np.asarray(X_transformed)
    if X_arr.shape[1] >= 2:
        coords_2d = (PCA(n_components=2, random_state=0).fit_transform(X_arr)
                     if X_arr.shape[1] > 2 else X_arr[:, :2])
    else:
        coords_2d = np.column_stack([X_arr[:, 0], np.zeros(len(X_arr))])
    if n <= max_facet_models:
        rows = []
        for name, res in filtered.items():
            labs = np.asarray(res["labels"])
            for i, lab in enumerate(labs):
                rows.append({"Modelo": name,
                             "Componente 1": float(coords_2d[i, 0]),
                             "Componente 2": float(coords_2d[i, 1]),
                             "Cluster": str(lab)})
        df_sc = pd.DataFrame(rows)
        if not df_sc.empty:
            n_cols = 2 if n > 1 else 1
            n_rows = int(np.ceil(n / n_cols))
            v_spacing = min(0.05, 0.5 / (n_rows - 1)) if n_rows > 1 else 0.0
            fig_scatter_facet = px.scatter(
                df_sc, x="Componente 1", y="Componente 2",
                color="Cluster", facet_col="Modelo", facet_col_wrap=n_cols,
                facet_row_spacing=v_spacing, opacity=0.7,
                title="🧩 Clusters em PCA 2D")
            fig_scatter_facet.update_layout(height=max(400, 260 * n_rows))
            fig_scatter_facet.for_each_annotation(
                lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        for name, res in filtered.items():
            labs = np.asarray(res["labels"])
            sub = pd.DataFrame({"Componente 1": coords_2d[:, 0],
                                "Componente 2": coords_2d[:, 1],
                                "Cluster": labs.astype(str)})
            f = px.scatter(sub, x="Componente 1", y="Componente 2",
                           color="Cluster", opacity=0.7,
                           title=f"Clusters — {name}")
            f.update_layout(height=420)
            figuras_por_modelo.append((name, f, sub))
    fig_ext = None
    rows_ext = []
    for name, res in filtered.items():
        m = res["metrics"]
        ari = m.get("adjusted_rand"); nmi = m.get("normalized_mutual_info")
        if ari is not None or nmi is not None:
            rows_ext.append({"Modelo": name, "ARI": ari, "NMI": nmi})
    if rows_ext:
        df_ext = pd.DataFrame(rows_ext).melt(id_vars="Modelo",
                                              value_vars=["ARI", "NMI"],
                                              var_name="Métrica", value_name="Valor")
        fig_ext = px.bar(df_ext, x="Modelo", y="Valor", color="Métrica",
                         barmode="group", title="📈 ARI / NMI")
        fig_ext.update_layout(height=420, xaxis_tickangle=-30)
    return {"bar_sil": fig_sil, "bar_db": fig_db,
            "scatter_facet": fig_scatter_facet,
            "scatter_por_modelo": figuras_por_modelo,
            "ext_bar": fig_ext, "df_bar": df_bar}


# ============================================================
# INFERÊNCIA
# ============================================================

def predict_new_classification(model, new_df, features, classes):
    missing = [col for col in features if col not in new_df.columns]
    if missing:
        raise ValueError("Faltam variáveis: " + ", ".join(missing))
    X_new = new_df[features].copy()
    prediction = model.predict(X_new)
    labels = []
    for v in prediction:
        try:
            idx = int(v)
            labels.append(classes[idx] if 0 <= idx < len(classes) else str(v))
        except (TypeError, ValueError):
            labels.append(str(v))
    result = new_df.copy()
    result["Classe_Prevista"] = labels
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)
            result["Confianca"] = proba.max(axis=1)
    except Exception:
        pass
    return result


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_regression_target(y):
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().all():
        raise ValueError("A variável alvo para regressão não é numérica.")
    if y_num.isna().any():
        st.warning("Há valores inválidos no alvo.")
    y_num = y_num.dropna()
    if len(y_num) < 10:
        raise ValueError("A regressão requer pelo menos 10 observações.")
    return y_num


def validate_classification_target(y, cv):
    y_clean = y.astype("string").fillna("missing")
    counts = y_clean.value_counts()
    if len(counts) < 2:
        raise ValueError("A classificação requer pelo menos 2 classes.")
    if counts.min() < cv:
        raise ValueError(f"Classe com {int(counts.min())} obs < CV exigido ({cv}).")
    return y_clean


# ============================================================
# EXPORTAÇÃO DE FUNÇÕES MATEMÁTICAS
# ============================================================

def _fmt_coef(c):
    return f"+ {c:.6f}" if c >= 0 else f"- {abs(c):.6f}"


def _extract_linear_expr(estimator, feature_names):
    coefs = getattr(estimator, "coef_", None)
    intercept = getattr(estimator, "intercept_", 0.0)
    if coefs is None:
        return None, None
    coefs = np.asarray(coefs).ravel()
    if intercept is not None and np.ndim(intercept) > 0:
        intercept = float(np.asarray(intercept).ravel()[0])
    n = min(len(coefs), len(feature_names))
    coefs = coefs[:n]; names = list(feature_names)[:n]
    parts = [f"{float(intercept):.6f}"]
    for name, c in zip(names, coefs):
        parts.append(f"{_fmt_coef(float(c))} * {name}")
    expr = "y = " + " ".join(parts)
    tbl = pd.DataFrame({"Termo": ["(intercept)"] + names,
                        "Coeficiente": [float(intercept)] + [float(c) for c in coefs]})
    return expr, tbl


def extract_math_expression(pipeline_or_est, raw_features, task, classes=None, model_name=""):
    try:
        if task == "regression":
            estimator = pipeline_or_est.named_steps["regressor"]
            feature_names = raw_features
        elif task == "classification":
            estimator = pipeline_or_est.named_steps["classifier"]
            feature_names = raw_features
        elif task == "timeseries":
            estimator = pipeline_or_est.named_steps.get("regressor") \
                if hasattr(pipeline_or_est, "named_steps") else pipeline_or_est
            feature_names = raw_features
        else:
            estimator = pipeline_or_est
            feature_names = list(raw_features)
        cls_name = estimator.__class__.__name__
        if task in {"regression", "timeseries"}:
            if cls_name in {"LinearRegression", "Ridge", "Lasso", "ElasticNet",
                            "BayesianRidge", "HuberRegressor", "SGDRegressor"}:
                expr, tbl = _extract_linear_expr(estimator, feature_names)
                return {"expressao": expr or "—", "tabela": tbl, "tipo": "linear",
                        "complexidade": f"{len(feature_names)} termos"}
        if task == "classification":
            if cls_name == "LogisticRegression":
                coefs = np.atleast_2d(estimator.coef_)
                return {"expressao": "f(x) = softmax(Wx + b) — ver coeficientes",
                        "tabela": pd.DataFrame(coefs), "tipo": "logística",
                        "complexidade": f"{coefs.shape[0]} classes"}
        return {"expressao": f"Extração não implementada para {cls_name}.",
                "tabela": None, "tipo": cls_name, "complexidade": cls_name}
    except Exception as exc:
        return {"expressao": f"Erro ao extrair: {exc}", "tabela": None,
                "tipo": "erro", "complexidade": "—"}


def compare_math_functions(result_state):
    task = result_state["task"]
    results = result_state["results"]
    valid = {n: v for n, v in results.items() if "error" not in v}
    raw_features = result_state.get("features", [])
    classes = result_state.get("classes")
    rows = []; expressions = {}
    for name, res in valid.items():
        try:
            if task in {"regression", "classification", "timeseries"}:
                pipeline = res.get("best_model") or res.get("estimator")
            else:
                pipeline = res["estimator"]
            if pipeline is None:
                continue
            info = extract_math_expression(pipeline, raw_features, task, classes, name)
            expressions[name] = info
            if task == "regression":
                metrica = res.get("test_r2"); nome = "R² teste"
            elif task == "classification":
                metrica = res.get("test_f1"); nome = "F1 ponderado"
            elif task == "timeseries":
                metrica = res.get("test_r2"); nome = "R² teste"
            else:
                metrica = (res.get("metrics") or {}).get("silhouette")
                nome = "Silhouette"
            rows.append({"Modelo": name, "Tipo de função": info["tipo"],
                         "Complexidade": info["complexidade"],
                         nome: None if metrica is None else round(float(metrica), 4)})
        except Exception as exc:
            rows.append({"Modelo": name, "Tipo de função": "erro",
                         "Complexidade": str(exc)[:60]})
    return pd.DataFrame(rows), expressions


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0;">🧠 Análise de Inteligência Computacional — Completa</h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
            Regressão · Classificação · Agrupamento · <b>Série Temporal</b> ·
            Pré-análise · Todos os estimadores · Funções matemáticas
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuração")
    task_options = {
        "Regressão": "regression",
        "Classificação": "classification",
        "Agrupamento": "clustering",
        "Série Temporal": "timeseries",
    }
    task_label = st.radio("Tipo de tarefa", list(task_options.keys()), horizontal=False)
    st.session_state["task_type"] = task_options[task_label]
    st.divider()
    st.subheader("📁 Fonte de dados")
    source = st.radio("Origem", ["Arquivo Local", "UCI Repository"])
    uploaded_file = None; uci_id = None
    if source == "Arquivo Local":
        uploaded_file = st.file_uploader("Envie o arquivo",
                                          type=["csv", "xlsx", "xls", "json"])
    else:
        uci_id = st.number_input("ID do Dataset (UCI)", min_value=1, value=186, step=1)
    if ui_button("📥 Carregar Dados"):
        with st.spinner("Carregando dados..."):
            try:
                df = (load_dataframe(uploaded_file, "file")
                      if source == "Arquivo Local"
                      else load_dataframe(None, "UCI", int(uci_id)))
                if df.empty:
                    raise ValueError("O dataset carregado está vazio.")
                st.session_state["df_raw"] = df
                for k in ["encoding_config", "encoding_preview", "results",
                          "prediction_result", "clustering_result",
                          "pre_analysis_reg", "pre_analysis_clf",
                          "pre_analysis_cluster", "ts_results", "ts_config",
                          "ts_pre_analysis"]:
                    st.session_state[k] = None
                st.session_state["encoding_ready"] = False
                st.success(f"✅ Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas.")
            except Exception as exc:
                st.error(f"Erro ao carregar: {exc}")

if st.session_state["df_raw"] is None:
    st.info("👈 Use a barra lateral para carregar um conjunto de dados.")
    st.stop()

df_raw = st.session_state["df_raw"]

# Avise sobre dependências opcionais ausentes
if st.session_state["task_type"] == "timeseries":
    if not HAS_STATSMODELS and not HAS_PROPHET:
        st.warning(
            "⚠️ Nem `statsmodels` nem `prophet` estão instalados. "
            "Modelos nativos (ARIMA, SARIMAX, ETS, Prophet) não estarão disponíveis. "
            "Somente modelos tabulares com lag features poderão ser usados. "
            "Para instalar: `pip install statsmodels prophet`"
        )

tabs = st.tabs([
    "📊 Diagnóstico", "🏷️ Codificação", "🎯 Modelagem",
    "⏳ Série Temporal", "📈 Resultados", "📚 Explicações",
    "🧮 Funções Matemáticas",
])

# ========== TAB 1 — DIAGNÓSTICO ==========
with tabs[0]:
    st.header("📊 Descrição dos Dados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", df_raw.shape[0])
    c2.metric("Colunas", df_raw.shape[1])
    c3.metric("Numéricas", len(df_raw.select_dtypes(include=np.number).columns))
    c4.metric("Categóricas", len(detect_categorical_columns(df_raw)))
    info_rows = []
    for col in df_raw.columns:
        info_rows.append({
            "Coluna": col, "dtype": str(df_raw[col].dtype),
            "Não-nulos": int(df_raw[col].notna().sum()),
            "Nulos": int(df_raw[col].isna().sum()),
            "Únicos": int(df_raw[col].nunique(dropna=True)),
            "Amostra": " | ".join(
                df_raw[col].dropna().astype(str).head(3).tolist())[:80],
        })
    st.subheader("Tipos de dados por coluna")
    ui_dataframe(pd.DataFrame(info_rows), height=320)
    with st.expander("📋 Estatísticas descritivas", expanded=True):
        ui_dataframe(df_raw.describe(include="all").T)
    with st.expander("🔍 Outliers IQR — diagnóstico"):
        odf = iqr_outlier_summary(df_raw)
        if odf.empty:
            st.info("Sem colunas numéricas.")
        else:
            ui_dataframe(odf.round(4))
    st.subheader("📈 Visualizações")
    numeric_cols = list(df_raw.select_dtypes(include=np.number).columns)
    if numeric_cols:
        sel_h = st.multiselect("Colunas para histogramas", numeric_cols,
                                default=numeric_cols[:min(4, len(numeric_cols))])
        if sel_h:
            long = df_raw[sel_h].melt(var_name="Variável", value_name="Valor")
            fig = px.histogram(long, x="Valor", facet_col="Variável",
                               facet_col_wrap=2, marginal="box",
                               title="Distribuições numéricas")
            ui_plotly_chart(fig)
    if len(numeric_cols) >= 2:
        corr = df_raw[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=len(numeric_cols) <= 12,
                             aspect="auto", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Matriz de correlação")
        ui_plotly_chart(fig_corr)

# ========== TAB 2 — CODIFICAÇÃO ==========
with tabs[1]:
    st.header("🏷️ Configuração de Variáveis Categóricas")
    categorical_cols = detect_categorical_columns(df_raw)
    if not categorical_cols:
        st.success("Nenhuma coluna categórica detectada.")
        st.session_state["encoding_config"] = {"encoding_map": {}, "ordinal_orders": {},
                                                "manual_maps": {}}
        st.session_state["encoding_preview"] = df_raw.copy()
        st.session_state["encoding_ready"] = True
    else:
        encoding_map, ordinal_orders, manual_maps = {}, {}, {}
        st.write(f"**{len(categorical_cols)} coluna(s) categórica(s).**")
        for col in categorical_cols:
            with st.expander(f"🔸 {col}", expanded=False):
                vals = df_raw[col].dropna().astype(str).drop_duplicates().tolist()
                vals_sorted = sorted(vals)
                a, b, c = st.columns(3)
                a.metric("Únicos", len(vals_sorted))
                b.metric("Ausentes", int(df_raw[col].isna().sum()))
                c.metric("dtype", str(df_raw[col].dtype))
                method = st.radio("Método",
                                  ["onehot", "label", "ordinal", "manual", "drop"],
                                  format_func=lambda x: {
                                      "onehot": "One-Hot", "label": "Label",
                                      "ordinal": "Ordinal", "manual": "Manual",
                                      "drop": "Descartar",
                                  }[x], key=f"enc_{col}")
                encoding_map[col] = method
                if method == "ordinal":
                    txt = st.text_input("Ordem (por vírgula)",
                                        value=", ".join(vals_sorted), key=f"ord_{col}")
                    ordinal_orders[col] = [v.strip() for v in txt.split(",") if v.strip()]
                elif method == "manual":
                    sug = ", ".join(f"{v}={i}" for i, v in enumerate(vals_sorted))
                    mt = st.text_input("Mapeamento categoria=código", value=sug,
                                       key=f"man_{col}")
                    mapping, invalid = {}, []
                    for tok in mt.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        if "=" not in tok:
                            invalid.append(tok); continue
                        k, v = tok.split("=", 1)
                        try:
                            mapping[k.strip()] = float(v.strip())
                        except ValueError:
                            invalid.append(tok)
                    if invalid:
                        st.warning("Ignorados: " + ", ".join(invalid))
                    manual_maps[col] = mapping
        if ui_button("✅ Salvar configuração de codificação", type="primary"):
            try:
                preview, report = apply_categorical_encoding_preview(
                    df_raw, encoding_map, ordinal_orders, manual_maps)
                st.session_state["encoding_config"] = {
                    "encoding_map": encoding_map,
                    "ordinal_orders": ordinal_orders,
                    "manual_maps": manual_maps,
                }
                st.session_state["encoding_preview"] = preview
                st.session_state["encoding_ready"] = True
                st.success("✅ Salvo.")
                ui_dataframe(pd.DataFrame(report, columns=["Coluna", "Método", "Resultado"]))
            except Exception as exc:
                st.error(f"Erro: {exc}")

# ========== TAB 3 — MODELAGEM (Regressão / Classificação / Agrupamento) ==========
with tabs[2]:
    st.header("🎯 Configuração e Execução")
    if not st.session_state["encoding_ready"]:
        st.warning("Configure a codificação na aba anterior.")
    else:
        encoding_config = st.session_state["encoding_config"] or {}
        encoding_map = encoding_config.get("encoding_map", {})
        ordinal_orders = encoding_config.get("ordinal_orders", {})
        manual_maps = encoding_config.get("manual_maps", {})
        all_columns = list(df_raw.columns)

        # -------- REGRESSÃO --------
        if st.session_state["task_type"] == "regression":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox("🎯 Alvo", all_columns,
                                       index=len(all_columns) - 1, key="target_reg")
            with c2:
                features = st.multiselect(
                    "📊 Preditores",
                    [c for c in all_columns if c != target],
                    default=[c for c in all_columns if c != target][:10],
                    key="features_reg")
            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if ui_button("Analisar dados", key="analyze_reg"):
                        try:
                            a = analyze_dataset(df_raw, features, target, "regression")
                            r = recommend_algorithms(a, "regression")
                            st.session_state["pre_analysis_reg"] = {
                                "analysis": a, "recommendations": r}
                        except Exception as exc:
                            st.error(f"Erro: {exc}")
                    pre = st.session_state.get("pre_analysis_reg")
                    if pre:
                        _render_pre_analysis(pre["analysis"], pre["recommendations"])
                st.subheader("⚙️ Parâmetros")
                c1, c2, c3 = st.columns(3)
                with c1:
                    outlier_mode = st.selectbox(
                        "Tratamento de outliers",
                        ["keep", "winsorize", "median"],
                        format_func=lambda x: {"keep": "Manter",
                                                "winsorize": "Winsorizar",
                                                "median": "Mediana"}[x])
                with c2:
                    threshold = st.number_input("Threshold corr.", 0.0, 1.0, 0.10, 0.05)
                with c3:
                    max_feat_input = st.number_input("Máx. features (0=todas)", 0, 200, 0)
                c1, c2 = st.columns([1, 1])
                with c1:
                    cat_reg = st.selectbox(
                        "Catálogo",
                        ["Recomendados (pré-análise)", "Lista curada",
                         "Todos os regressores"], key="cat_reg")
                with c2:
                    tune_reg = st.checkbox("🔧 Otimizar", value=True, key="tune_reg")
                if cat_reg == "Recomendados (pré-análise)":
                    pre = st.session_state.get("pre_analysis_reg")
                    avail = [n for n, _ in pre["recommendations"]] if pre else CURATED_REGRESSORS
                elif cat_reg == "Todos os regressores":
                    avail = get_all_regressors()
                else:
                    avail = CURATED_REGRESSORS
                chosen_reg = st.multiselect("Regressores", avail,
                                            default=avail[:min(8, len(avail))],
                                            key="chosen_reg")
                c1, c2, c3 = st.columns(3)
                with c1:
                    search_method = st.selectbox("Busca", ["grid", "random", "manual"],
                        format_func=lambda x: {"grid": "Grid", "random": "Random",
                                                "manual": "Manual"}[x])
                    n_iter = st.number_input("n_iter", 2, 500, 20)
                    cv_reg = st.number_input("Folds CV", 3, 10, 5)
                with c2:
                    seed_choice = st.radio("Semente", ["fixa", "aleatoria", "custom"],
                        format_func=lambda x: {"fixa": "Fixa (42)",
                                                "aleatoria": "Aleatória",
                                                "custom": "Custom"}[x],
                        horizontal=True, key="seed_reg")
                    custom_seed = st.number_input("Valor", 0, 999999, 42,
                                                   disabled=seed_choice != "custom",
                                                   key="custom_seed_reg")
                if ui_button("▶️ Executar Regressão", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        if not chosen_reg:
                            raise ValueError("Selecione pelo menos um regressor.")
                        data = df_raw[features + [target]].copy()
                        y = pd.to_numeric(data[target], errors="coerce")
                        vm = y.notna()
                        X = data.loc[vm, features].copy(); y = y.loc[vm].copy()
                        if len(y) < 10:
                            raise ValueError("Poucas observações válidas.")
                        registry = get_regressor_discovery_registry()
                        models = {}
                        for n in chosen_reg:
                            if n not in registry:
                                continue
                            try:
                                est = registry[n]()
                            except Exception:
                                continue
                            est = _set_random_state(est, DEFAULT_RANDOM_STATE)
                            grid = get_regressor_param_grid(n) if tune_reg else None
                            models[n] = {"estimator": est, "params": grid or {}}
                        if not models:
                            raise ValueError("Nenhum regressor pôde ser instanciado.")
                        seed = resolve_random_state(seed_choice, int(custom_seed))
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=0.20, random_state=seed)
                        results = evaluate_regression(
                            X_tr, y_tr, X_te, y_te, models,
                            encoding_map, ordinal_orders, manual_maps, outlier_mode,
                            threshold=threshold,
                            max_features=int(max_feat_input) if max_feat_input > 0 else None,
                            cv=int(cv_reg), search_method=search_method,
                            n_iter=int(n_iter), random_state=seed)
                        st.session_state["results"] = {
                            "task": "regression", "results": results, "y_test": y_te,
                            "X_train": X_tr, "X_test": X_te, "y_train": y_tr,
                            "features": features, "target": target,
                            "train_n": len(X_tr), "test_n": len(X_te),
                            "seed": seed, "cv": int(cv_reg),
                            "encoding_config": encoding_config,
                            "outlier_mode": outlier_mode,
                            "threshold": float(threshold),
                            "max_features": int(max_feat_input) if max_feat_input > 0 else None,
                        }
                        st.success("🎉 Regressão concluída.")
                    except Exception as exc:
                        st.error(f"Erro: {exc}")

        # -------- CLASSIFICAÇÃO --------
        elif st.session_state["task_type"] == "classification":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox("🎯 Alvo", all_columns,
                                       index=len(all_columns) - 1, key="target_clf")
            with c2:
                features = st.multiselect(
                    "📊 Preditores",
                    [c for c in all_columns if c != target],
                    default=[c for c in all_columns if c != target][:10],
                    key="features_clf")
            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if ui_button("Analisar dados", key="analyze_clf"):
                        try:
                            a = analyze_dataset(df_raw, features, target, "classification")
                            r = recommend_algorithms(a, "classification")
                            st.session_state["pre_analysis_clf"] = {
                                "analysis": a, "recommendations": r}
                        except Exception as exc:
                            st.error(f"Erro: {exc}")
                    pre = st.session_state.get("pre_analysis_clf")
                    if pre:
                        _render_pre_analysis(pre["analysis"], pre["recommendations"])
                st.subheader("⚙️ Parâmetros")
                c1, c2, c3 = st.columns(3)
                with c1:
                    cv_clf = st.number_input("Folds CV", 3, 10, 5)
                with c2:
                    test_size_clf = st.number_input("Fração teste", 0.10, 0.40, 0.20, 0.05)
                with c3:
                    cat_clf = st.selectbox(
                        "Catálogo",
                        ["Recomendados (pré-análise)", "Lista recomendada",
                         "Todos os classificadores"], key="cat_clf")
                if cat_clf == "Recomendados (pré-análise)":
                    pre = st.session_state.get("pre_analysis_clf")
                    classifiers = [n for n, _ in pre["recommendations"]] if pre else CURATED_CLASSIFIERS
                elif cat_clf == "Lista recomendada":
                    classifiers = CURATED_CLASSIFIERS
                else:
                    classifiers = get_all_classifiers()
                h1, h2, h3 = st.columns(3)
                with h1:
                    tune_clf = st.checkbox("🔧 Otimizar", value=True, key="tune_clf")
                with h2:
                    tuning_method_clf = st.selectbox(
                        "Busca", ["grid", "random"],
                        format_func=lambda x: "Grid" if x == "grid" else "Random",
                        disabled=not tune_clf, key="tuning_method_clf")
                with h3:
                    n_iter_clf = st.number_input(
                        "n_iter", 2, 100, 20, 1,
                        disabled=(not tune_clf or tuning_method_clf != "random"),
                        key="n_iter_clf")
                tuning_metric_clf = st.selectbox(
                    "Métrica de seleção",
                    ["f1_weighted", "f1_macro", "accuracy", "balanced_accuracy"],
                    format_func=lambda x: {"f1_weighted": "F1 ponderado",
                                            "f1_macro": "F1 macro",
                                            "accuracy": "Acurácia",
                                            "balanced_accuracy": "Balanced acc."}[x],
                    disabled=not tune_clf, key="tuning_metric_clf")
                seed_choice = st.radio(
                    "Semente", ["fixa", "aleatoria", "custom"],
                    format_func=lambda x: {"fixa": "Fixa (42)",
                                            "aleatoria": "Aleatória",
                                            "custom": "Custom"}[x],
                    horizontal=True, key="seed_clf")
                custom_seed_clf = st.number_input(
                    "Custom", 0, 999999, 42, disabled=seed_choice != "custom",
                    key="custom_seed_clf")
                chosen_clf = st.multiselect(
                    "Classificadores", classifiers,
                    default=classifiers[:min(8, len(classifiers))],
                    key="chosen_clf")
                if ui_button("▶️ Executar Classificação", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        data = df_raw[features + [target]].copy()
                        y_raw = validate_classification_target(data[target], int(cv_clf))
                        data = data.loc[y_raw.index]
                        X, y = data[features], y_raw
                        if not chosen_clf:
                            raise ValueError("Selecione pelo menos um classificador.")
                        if y.value_counts().min() < 2:
                            raise ValueError("Cada classe precisa de pelo menos 2 obs.")
                        seed = resolve_random_state(seed_choice, int(custom_seed_clf))
                        X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(
                            X, y, test_size=float(test_size_clf),
                            random_state=seed, stratify=y)
                        classes = sorted(y_tr_raw.astype(str).unique().tolist())
                        if not set(y_te_raw.astype(str)).issubset(set(classes)):
                            raise ValueError("Classe no teste não aparece no treino.")
                        lbl = {c: i for i, c in enumerate(classes)}
                        y_tr = y_tr_raw.astype(str).map(lbl).to_numpy()
                        y_te = y_te_raw.astype(str).map(lbl).to_numpy()
                        results = evaluate_classifiers(
                            X_tr, y_tr, X_te, y_te, chosen_clf,
                            encoding_map, ordinal_orders, manual_maps,
                            cv=int(cv_clf), random_state=seed,
                            tune=bool(tune_clf), search_method=tuning_method_clf,
                            n_iter=int(n_iter_clf), tuning_metric=tuning_metric_clf)
                        st.session_state["results"] = {
                            "task": "classification", "results": results,
                            "y_test": y_te, "X_train": X_tr, "X_test": X_te,
                            "y_train": y_tr, "features": features, "target": target,
                            "classes": classes, "train_n": len(X_tr),
                            "test_n": len(X_te), "seed": seed, "cv": int(cv_clf),
                            "encoding_config": encoding_config,
                        }
                        st.session_state["prediction_result"] = None
                        st.success("🎉 Classificação concluída.")
                    except Exception as exc:
                        st.error(f"Erro: {exc}")

                # Inferência
                trained = st.session_state.get("results")
                if trained and trained.get("task") == "classification":
                    st.divider()
                    st.subheader("🔮 Classificar novos dados")
                    valid_models = {n: r for n, r in trained["results"].items()
                                    if "error" not in r}
                    if valid_models:
                        model_name = st.selectbox("Modelo",
                                                   list(valid_models.keys()),
                                                   key="pred_model")
                        new_file = st.file_uploader(
                            "Novo dataset", type=["csv", "xlsx", "xls", "json"],
                            key="new_file_pred")
                        if new_file is not None:
                            try:
                                nd = load_dataframe(new_file, "file")
                                missing = [c for c in trained["features"]
                                           if c not in nd.columns]
                                if missing:
                                    st.error("Faltam: " + ", ".join(missing))
                                else:
                                    ui_dataframe(nd[trained["features"]].head(20))
                                    if ui_button("🔮 Classificar", type="primary",
                                                  key="classify_new"):
                                        pred = predict_new_classification(
                                            valid_models[model_name]["best_model"],
                                            nd, trained["features"],
                                            trained["classes"])
                                        st.session_state["prediction_result"] = pred
                                        st.success(f"✅ {len(pred)} classificados.")
                            except Exception as exc:
                                st.error(f"Erro: {exc}")
                        if st.session_state.get("prediction_result") is not None:
                            ui_dataframe(st.session_state["prediction_result"],
                                         height=350)
                            ui_download_button(
                                "📥 Baixar previsões (.csv)",
                                st.session_state["prediction_result"].to_csv(
                                    index=False).encode("utf-8"),
                                file_name=f"previsoes_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                mime="text/csv", key="dl_pred")

        # -------- AGRUPAMENTO --------
        elif st.session_state["task_type"] == "clustering":
            st.subheader("🧩 Agrupamento não supervisionado")
            cluster_features = st.multiselect("📊 Variáveis", all_columns,
                                               default=all_columns, key="clu_feats")
            ref_opts = ["Nenhuma (sem rótulo)"] + all_columns
            ref_target = st.selectbox("🏷️ Rótulo de referência (opcional)",
                                       ref_opts, key="clu_ref")
            if cluster_features:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if ui_button("Analisar dados", key="analyze_clu"):
                        try:
                            a = analyze_dataset(df_raw, cluster_features, None, "clustering")
                            r = recommend_algorithms(a, "clustering")
                            st.session_state["pre_analysis_cluster"] = {
                                "analysis": a, "recommendations": r}
                        except Exception as exc:
                            st.error(f"Erro: {exc}")
                    pre = st.session_state.get("pre_analysis_cluster")
                    if pre:
                        _render_pre_analysis(pre["analysis"], pre["recommendations"])
            c1, c2, c3 = st.columns(3)
            with c1:
                n_clusters = st.number_input("Nº clusters", 2, 15, 3, 1, key="clu_n")
                cluster_outlier = st.selectbox(
                    "Outliers", ["keep", "winsorize", "median"],
                    format_func=lambda x: {"keep": "Manter",
                                            "winsorize": "Winsorizar",
                                            "median": "Mediana"}[x], key="clu_out")
            with c2:
                eps = st.number_input("DBSCAN eps", 0.01, 10.0, 0.50, 0.05, key="db_eps")
                min_samples = st.number_input("min_samples", 2, 50, 5, 1, key="db_ms")
            with c3:
                seed_choice = st.radio(
                    "Semente", ["fixa", "aleatoria", "custom"],
                    format_func=lambda x: {"fixa": "Fixa (42)",
                                            "aleatoria": "Aleatória",
                                            "custom": "Custom"}[x],
                    horizontal=True, key="clu_seed")
                custom_seed = st.number_input("Custom", 0, 999999, 42,
                                               disabled=seed_choice != "custom",
                                               key="clu_custom_seed")
            c1, c2 = st.columns([1, 1])
            with c1:
                cat_clu = st.selectbox(
                    "Catálogo",
                    ["Recomendados (pré-análise)", "Lista curada",
                     "Todos os clusterers"], key="cat_clu")
            with c2:
                tune_clu = st.checkbox("🔧 Otimizar", value=True, key="tune_clu")
            if cat_clu == "Recomendados (pré-análise)":
                pre = st.session_state.get("pre_analysis_cluster")
                avail_clu = [n for n, _ in pre["recommendations"]] if pre else CURATED_CLUSTERERS
            elif cat_clu == "Todos os clusterers":
                avail_clu = get_all_clusterers()
            else:
                avail_clu = CURATED_CLUSTERERS
            algorithms = st.multiselect("Algoritmos", avail_clu,
                                        default=avail_clu[:min(5, len(avail_clu))],
                                        key="chosen_clu")
            t1, t2, t3 = st.columns(3)
            with t1:
                tun_m_clu = st.selectbox("Busca", ["grid", "random"],
                    format_func=lambda x: "Grid" if x == "grid" else "Random",
                    disabled=not tune_clu, key="tun_m_clu")
            with t2:
                n_iter_clu = st.number_input("n_iter", 2, 100, 20, 1,
                    disabled=(not tune_clu or tun_m_clu != "random"),
                    key="n_iter_clu")
            with t3:
                tun_met_clu = st.selectbox(
                    "Métrica seleção",
                    ["silhouette", "calinski_harabasz", "davies_bouldin"],
                    format_func=lambda x: {"silhouette": "Silhouette (maior)",
                                            "calinski_harabasz": "CH (maior)",
                                            "davies_bouldin": "DB (menor)"}[x],
                    disabled=not tune_clu, key="tun_met_clu")
            if ui_button("▶️ Executar Agrupamento", type="primary", key="run_clu"):
                try:
                    if len(cluster_features) < 2:
                        raise ValueError("Selecione pelo menos duas variáveis.")
                    if not algorithms:
                        raise ValueError("Selecione pelo menos um algoritmo.")
                    ref = None if ref_target == "Nenhuma (sem rótulo)" else df_raw[ref_target]
                    seed = resolve_random_state(seed_choice, int(custom_seed))
                    X_clu = df_raw[cluster_features].copy()
                    X_t, clu_res = fit_clustering_models(
                        X_clu, encoding_map, ordinal_orders, manual_maps, algorithms,
                        n_clusters=int(n_clusters), eps=float(eps),
                        min_samples=int(min_samples),
                        outlier_mode=cluster_outlier,
                        random_state=seed, reference=ref, tune=bool(tune_clu),
                        search_method=tun_m_clu, n_iter=int(n_iter_clu),
                        objective=tun_met_clu, max_combinations=30)
                    state = {
                        "task": "clustering", "results": clu_res,
                        "features": cluster_features, "transformed": X_t,
                        "source_data": X_clu, "seed": seed,
                        "n_clusters": int(n_clusters),
                        "reference_target": None if ref_target == "Nenhuma (sem rótulo)" else ref_target,
                    }
                    st.session_state["clustering_result"] = state
                    st.session_state["results"] = state
                    st.success("🎉 Agrupamento concluído.")
                except Exception as exc:
                    st.error(f"Erro: {exc}")
            clu_state = st.session_state.get("clustering_result")
            if clu_state:
                valid_clu = {n: r for n, r in clu_state["results"].items()
                             if "error" not in r}
                if valid_clu:
                    rows = []
                    for n, r in valid_clu.items():
                        m = r["metrics"]
                        rows.append({"Algoritmo": n, "Clusters": m["n_clusters"],
                                     "Silhouette": m["silhouette"],
                                     "CH": m["calinski_harabasz"],
                                     "DB": m["davies_bouldin"],
                                     "Ruído (%)": m["noise_pct"],
                                     "ARI": m.get("adjusted_rand"),
                                     "NMI": m.get("normalized_mutual_info")})
                    st.subheader("📋 Métricas")
                    ui_dataframe(pd.DataFrame(rows).round(4), hide_index=True)

# ========== TAB 4 — SÉRIE TEMPORAL ==========
with tabs[3]:
    st.header("⏳ Análise de Série Temporal")
    st.caption(
        "Modelagem específica para dados ordenados no tempo. "
        "Use **modelos nativos** (ARIMA, SARIMAX, ETS, Prophet) e/ou "
        "**modelos tabulares com lag features** (Random Forest, XGBoost, etc.)."
    )

    time_cols = detect_datetime_columns(df_raw)
    all_columns = list(df_raw.columns)
    numeric_cols = list(df_raw.select_dtypes(include=np.number).columns)

    if not time_cols:
        st.error("Nenhuma coluna de data foi detectada. Converta uma coluna "
                 "para formato datetime no CSV antes de carregar.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            time_col = st.selectbox("🕐 Coluna de tempo", time_cols, key="ts_time")
        with c2:
            target_ts = st.selectbox("🎯 Variável alvo (série)",
                                      numeric_cols,
                                      index=len(numeric_cols) - 1 if numeric_cols else 0,
                                      key="ts_target")

        exog_cols = st.multiselect(
            "📊 Variáveis exógenas (opcional)",
            [c for c in all_columns if c not in [time_col, target_ts]],
            default=[], key="ts_exog")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_lags = st.number_input("Nº de lags", 1, 60, 5, 1, key="ts_lags",
                                      help="Quantas defasagens incluir como features.")
        with c2:
            test_size_ts = st.number_input("Fração de teste", 0.10, 0.40, 0.20, 0.05,
                                            key="ts_test_size")
        with c3:
            ts_cv_folds = st.number_input("Folds CV (walk-forward)", 2, 10, 5, 1,
                                           key="ts_cv_folds")

        # Pré-análise
        if ui_button("🔍 Pré-análise da série", key="ts_pre"):
            try:
                series_raw = pd.to_numeric(df_raw[target_ts], errors="coerce").dropna()
                info = ts_pre_analysis(series_raw)
                st.session_state["ts_pre_analysis"] = info
            except Exception as exc:
                st.error(f"Erro: {exc}")

        if st.session_state.get("ts_pre_analysis"):
            info = st.session_state["ts_pre_analysis"]
            st.subheader("📋 Pré-análise da série")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Observações", info.get("n_obs", 0))
            cc2.metric("Média", f"{info.get('mean', 0):.3f}")
            cc3.metric("Desvio padrão", f"{info.get('std', 0):.3f}")
            if info.get("adf_pvalue") is not None:
                st.caption(f"ADF p-value: {info['adf_pvalue']:.4f} "
                            f"(estatística: {info.get('adf_stat', 0):.4f})")
            if info.get("autocorr_lag1") is not None:
                st.caption(f"Autocorrelação lag-1: {info['autocorr_lag1']:.4f}")
            for w in info.get("warnings", []):
                st.warning(w)
            for n in info.get("notes", []):
                st.info(n)

            # Decomposição
            if HAS_STATSMODELS:
                with st.expander("📉 Decomposição (trend/seasonal/resid)"):
                    try:
                        s = pd.to_numeric(df_raw[target_ts], errors="coerce").dropna()
                        decomp_fig = ts_decomposition_plot(s)
                        if decomp_fig is not None:
                            ui_plotly_chart(decomp_fig)
                        else:
                            st.info("Série curta demais para decomposição.")
                    except Exception as exc:
                        st.warning(f"Decomposição falhou: {exc}")

        # Catálogos
        st.subheader("⚙️ Modelos")
        natives_avail = []
        if HAS_STATSMODELS:
            natives_avail = ["ARIMA", "SARIMAX", "ExponentialSmoothing",
                              "NaiveForecaster", "ThetaForecaster",
                              "PolynomialTrendForecaster"]
        if HAS_PROPHET:
            natives_avail.append("Prophet")

        tab1, tab2 = st.tabs(["Modelos nativos", "Modelos via lag features"])

        with tab1:
            if not natives_avail:
                st.warning("Nenhum modelo nativo disponível. "
                            "Instale `statsmodels` e/ou `prophet`.")
            else:
                chosen_native = st.multiselect(
                    "Modelos nativos",
                    natives_avail,
                    default=natives_avail[:min(3, len(natives_avail))],
                    key="ts_natives")
                tune_native = st.checkbox("🔧 Otimizar hiperparâmetros",
                                           value=True, key="ts_tune_native")

        with tab2:
            st.caption("Modelos tabulares aplicados via features de defasagem "
                        "(lags da série + exógenas).")
            chosen_lag_models = st.multiselect(
                "Modelos tabulares (via lags)",
                get_all_regressors(),
                default=[m for m in CURATED_TS_LAG_MODELS
                         if m in get_all_regressors()][:5],
                key="ts_lag_models")
            tune_lag = st.checkbox("🔧 Otimizar (lag models)",
                                    value=True, key="ts_tune_lag")

        c1, c2 = st.columns(2)
        with c1:
            search_method_ts = st.selectbox("Busca", ["grid", "random", "manual"],
                format_func=lambda x: {"grid": "Grid", "random": "Random",
                                        "manual": "Manual"}[x],
                key="ts_search")
        with c2:
            n_iter_ts = st.number_input("n_iter", 2, 200, 20, key="ts_n_iter")

        if ui_button("▶️ Executar Análise de Série Temporal", type="primary",
                      key="run_ts"):
            try:
                # Preparar dados temporais
                X_ts, y_ts, dates_ts = make_ts_dataframe(
                    df_raw, time_col, target_ts, exog_cols,
                    n_lags=int(n_lags))
                if len(y_ts) < 15:
                    raise ValueError(
                        f"Poucas observações válidas após lag features: {len(y_ts)}. "
                        "Reduza o número de lags ou verifique a série.")

                X_tr, X_te, y_tr, y_te, dates_tr, dates_te = temporal_split(
                    X_ts, y_ts, dates_ts, test_size=float(test_size_ts))

                st.info(f"Treino: {len(y_tr)} obs · Teste: {len(y_te)} obs · "
                        f"Features: {X_ts.shape[1]}")

                # Modelos nativos
                native_models = {}
                if 'chosen_native' in dir() and chosen_native:
                    for name in chosen_native:
                        grid = get_ts_param_grid(name) if tune_native else None
                        if grid:
                            if isinstance(grid, list):
                                params = grid[0]
                            else:
                                params = {k: v[0] for k, v in grid.items()}
                        else:
                            params = {}
                        native_models[name] = {"params": params}

                # Modelos tabulares
                lag_models = {}
                if 'chosen_lag_models' in dir() and chosen_lag_models:
                    reg = get_regressor_discovery_registry()
                    for name in chosen_lag_models:
                        if name not in reg:
                            continue
                        try:
                            est = reg[name]()
                        except Exception:
                            continue
                        est = _set_random_state(est, DEFAULT_RANDOM_STATE)
                        base_grid = get_regressor_param_grid(name) if tune_lag else None
                        # Renomear chaves de "regressor__" para "regressor__" (compatível)
                        params = base_grid or {}
                        lag_models[name] = {"estimator": est, "params": params}

                if not native_models and not lag_models:
                    raise ValueError("Selecione pelo menos um modelo.")

                # Rodar
                combined_results = {}
                if native_models:
                    with st.spinner("Ajustando modelos nativos..."):
                        native_results = evaluate_ts_native(
                            y_tr, y_te, dates_tr, dates_te,
                            native_models, n_splits=int(ts_cv_folds))
                        combined_results.update(native_results)

                if lag_models:
                    with st.spinner("Treinando modelos tabulares com lags..."):
                        lag_results = evaluate_ts_tabular(
                            X_tr, y_tr, X_te, y_te, dates_te,
                            lag_models, cv=int(ts_cv_folds),
                            search_method=search_method_ts,
                            n_iter=int(n_iter_ts),
                            random_state=DEFAULT_RANDOM_STATE)
                        combined_results.update(lag_results)

                st.session_state["ts_results"] = {
                    "task": "timeseries",
                    "results": combined_results,
                    "y_train": y_tr, "y_test": y_te,
                    "dates_train": dates_tr, "dates_test": dates_te,
                    "features": list(X_ts.columns),
                    "target": target_ts, "time_col": time_col,
                    "n_lags": int(n_lags),
                    "train_n": len(y_tr), "test_n": len(y_te),
                }
                st.session_state["results"] = st.session_state["ts_results"]
                st.success("🎉 Análise de série temporal concluída.")
            except Exception as exc:
                st.error(f"Erro: {exc}")

# ========== TAB 5 — RESULTADOS ==========
with tabs[4]:
    st.header("📈 Resultados e avaliação")
    result_state = st.session_state.get("results")
    if result_state is None:
        st.info("Execute uma análise na aba **Modelagem** ou **Série Temporal**.")
    else:
        task = result_state["task"]
        results = result_state["results"]
        valid = {n: v for n, v in results.items() if "error" not in v}
        errors = {n: v["error"] for n, v in results.items() if "error" in v}
        if errors:
            with st.expander("⚠️ Modelos que falharam"):
                ui_dataframe(pd.DataFrame([{"Modelo": n, "Erro": e}
                                            for n, e in errors.items()]))

        # -------- REGRESSÃO --------
        if task == "regression":
            if not valid:
                st.error("Nenhum modelo executado.")
            else:
                rows = []
                for n, r in valid.items():
                    params = ", ".join(f"{k.replace('regressor__', '')}={v}"
                                       for k, v in r["best_params"].items()) or "—"
                    cv_m = r.get("cv_mean")
                    rows.append({
                        "Modelo": n,
                        "CV R² médio": round(cv_m, 4) if cv_m is not None else None,
                        "R² teste": round(r["test_r2"], 4),
                        "RMSE teste": round(r["test_rmse"], 4),
                        "MAE teste": round(r["test_mae"], 4),
                        "Hiperparâmetros": params,
                        "_sort": cv_m if cv_m is not None else -np.inf,
                    })
                tbl = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
                       .drop(columns=["_sort"]).reset_index(drop=True))
                ui_dataframe(tbl, hide_index=True)

                st.divider()
                st.subheader("📊 Comparação dinâmica de modelos")
                all_models = list(valid.keys())
                chosen_cmp, max_facet = _render_dynamic_selector(
                    all_models, "reg_cmp", default_top_n=8)
                if chosen_cmp:
                    dash = regression_comparison_dashboard(
                        result_state["y_test"], valid, chosen_cmp,
                        max_facet_models=int(max_facet))
                    if dash:
                        if dash["bar_metrics"] is not None:
                            ui_plotly_chart(dash["bar_metrics"])
                        if dash["box_cv"] is not None:
                            ui_plotly_chart(dash["box_cv"])
                        if dash["real_pred"] is not None:
                            ui_plotly_chart(dash["real_pred"])
                            ui_plotly_chart(dash["resid"])
                        elif dash["por_modelo"]:
                            for name, f1, f2 in dash["por_modelo"]:
                                with st.expander(f"🔹 {name}", expanded=False):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        ui_plotly_chart(f1)
                                    with c2:
                                        ui_plotly_chart(f2)

        # -------- CLASSIFICAÇÃO --------
        elif task == "classification":
            if not valid:
                st.error("Nenhum classificador executado.")
            else:
                rows = []
                for n, r in valid.items():
                    cv_t = r.get("cv_tuning_mean")
                    rows.append({
                        "Classificador": n,
                        "Média CV (tuning)": round(cv_t, 4) if cv_t is not None else None,
                        "Hiperparâmetros": r.get("best_params_text", "—"),
                        "Acurácia teste": round(r["test_accuracy"], 4),
                        "F1 ponderado teste": round(r["test_f1"], 4),
                        "F1 macro teste": round(r["test_f1_macro"], 4),
                        "Balanced acc. teste": round(r["test_balanced_accuracy"], 4),
                        "ROC-AUC teste": None if r.get("test_roc_auc") is None
                                          else round(r["test_roc_auc"], 4),
                        "_sort": cv_t if cv_t is not None else -np.inf,
                    })
                tbl = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
                       .drop(columns=["_sort"]).reset_index(drop=True))
                ui_dataframe(tbl, hide_index=True, height=450)

                st.divider()
                st.subheader("📊 Comparação dinâmica de classificadores")
                all_models = list(valid.keys())
                chosen_cmp, max_facet = _render_dynamic_selector(
                    all_models, "clf_cmp", default_top_n=8)
                if chosen_cmp:
                    dash = classification_comparison_dashboard(
                        result_state["y_test"], valid, chosen_cmp,
                        result_state["classes"], max_facet_models=int(max_facet))
                    if dash:
                        if dash["bar_metrics"] is not None:
                            ui_plotly_chart(dash["bar_metrics"])
                        if dash["box_cv"] is not None:
                            ui_plotly_chart(dash["box_cv"])
                        if dash["auc_bar"] is not None:
                            ui_plotly_chart(dash["auc_bar"])
                        if dash["cm_facet"] is not None:
                            ui_plotly_chart(dash["cm_facet"])

        # -------- AGRUPAMENTO --------
        elif task == "clustering":
            if not valid:
                st.error("Nenhum algoritmo executado.")
            else:
                rows = []
                for n, r in valid.items():
                    m = r["metrics"]
                    rows.append({
                        "Algoritmo": n, "Nº clusters": m["n_clusters"],
                        "Silhouette": m["silhouette"],
                        "Calinski-Harabasz": m["calinski_harabasz"],
                        "Davies-Bouldin": m["davies_bouldin"],
                        "Ruído (%)": m["noise_pct"],
                        "ARI": m.get("adjusted_rand"),
                        "NMI": m.get("normalized_mutual_info"),
                    })
                ui_dataframe(pd.DataFrame(rows).round(4), hide_index=True)

                st.divider()
                st.subheader("📊 Comparação dinâmica de algoritmos")
                all_models = list(valid.keys())
                chosen_cmp, max_facet = _render_dynamic_selector(
                    all_models, "clu_cmp", default_top_n=6)
                if chosen_cmp:
                    dash = clustering_comparison_dashboard(
                        result_state["transformed"], valid, chosen_cmp,
                        max_facet_models=int(max_facet))
                    if dash:
                        if dash["bar_sil"] is not None:
                            ui_plotly_chart(dash["bar_sil"])
                        if dash["bar_db"] is not None:
                            ui_plotly_chart(dash["bar_db"])
                        if dash["scatter_facet"] is not None:
                            ui_plotly_chart(dash["scatter_facet"])

        # -------- SÉRIE TEMPORAL --------
        elif task == "timeseries":
            if not valid:
                st.error("Nenhum modelo de série temporal executado com sucesso.")
            else:
                rows = []
                for n, r in valid.items():
                    rows.append({
                        "Modelo": n,
                        "MAE": round(r.get("test_mae", np.nan), 4)
                                if r.get("test_mae") is not None else None,
                        "RMSE": round(r.get("test_rmse", np.nan), 4)
                                if r.get("test_rmse") is not None else None,
                        "MAPE (%)": round(r["test_mape"], 2)
                                    if r.get("test_mape") is not None else None,
                        "SMAPE (%)": round(r["test_smape"], 2)
                                     if r.get("test_smape") is not None else None,
                        "R²": round(r["test_r2"], 4)
                              if r.get("test_r2") is not None
                              and np.isfinite(r["test_r2"]) else None,
                        "CV R² (médio)": round(r["cv_mean"], 4)
                                         if r.get("cv_mean") is not None else None,
                    })
                ui_dataframe(pd.DataFrame(rows).round(4), hide_index=True)

                st.divider()
                st.subheader("📊 Comparação dinâmica de modelos temporais")
                all_models = list(valid.keys())
                chosen_cmp, max_facet = _render_dynamic_selector(
                    all_models, "ts_cmp", default_top_n=6,
                    help_text="Recomendado: 2 a 6 modelos.")
                if chosen_cmp:
                    dash = ts_comparison_dashboard(
                        result_state["y_test"], result_state["dates_test"],
                        valid, chosen_cmp, max_facet_models=int(max_facet))
                    if dash:
                        if dash["metrics_bar"] is not None:
                            ui_plotly_chart(dash["metrics_bar"])
                        if dash["r2_bar"] is not None:
                            ui_plotly_chart(dash["r2_bar"])
                        if dash["cv_box"] is not None:
                            ui_plotly_chart(dash["cv_box"])
                        if dash["pred_facet"] is not None:
                            ui_plotly_chart(dash["pred_facet"])
                        elif dash["pred_por_modelo"]:
                            for name, f in dash["pred_por_modelo"]:
                                with st.expander(f"🔹 {name}", expanded=False):
                                    ui_plotly_chart(f)
                        if dash["resid"] is not None:
                            ui_plotly_chart(dash["resid"])

        # Exportação geral
        st.divider()
        st.subheader("💾 Exportar resultados")
        if task == "clustering":
            export_rows = [{"algoritmo": n, **r["metrics"]} for n, r in valid.items()]
        else:
            export_rows = []
            for n, r in valid.items():
                clean = {}
                for k, v in r.items():
                    if k in {"best_model", "y_pred", "cv_scores", "cv_acc_scores",
                             "cv_f1_scores", "search_table", "tuning_table",
                             "dates_te", "estimator"}:
                        continue
                    clean[k] = v.item() if isinstance(v, (np.floating, np.integer)) else v
                clean["modelo"] = n
                export_rows.append(clean)
        csv_b = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        ui_download_button("📥 Baixar resultados (.csv)", csv_b,
                            file_name=f"resultados_{datetime.now():%Y%m%d_%H%M%S}.csv",
                            mime="text/csv", key="dl_res")
        report = {
            "tarefa": task, "features": result_state.get("features", []),
            "seed": result_state.get("seed"), "modelos": export_rows,
        }
        ui_download_button("📥 Baixar relatório (.json)",
                            json.dumps(report, indent=2, ensure_ascii=False,
                                        default=str).encode("utf-8"),
                            file_name=f"relatorio_{datetime.now():%Y%m%d_%H%M%S}.json",
                            mime="application/json", key="dl_rep")

# ========== TAB 6 — EXPLICAÇÕES ==========
with tabs[5]:
    st.header("📚 Explicações dos Algoritmos")
    st.caption("Descrições, funcionamento, quando usar e limitações de cada algoritmo.")
    task_filter = st.radio("Filtrar por tipo",
                            ["Todos", "Regressão", "Classificação", "Agrupamento",
                             "Série Temporal"],
                            horizontal=True, key="exp_filter")
    search_term = st.text_input("🔍 Buscar por nome", "", key="exp_search")
    items = sorted(ALGORITHM_EXPLANATIONS.items())
    filtered = []
    for name, info in items:
        if task_filter != "Todos" and info["tipo"] != task_filter:
            continue
        if search_term and search_term.lower() not in name.lower():
            continue
        filtered.append((name, info))
    if not filtered:
        st.info("Nenhum algoritmo corresponde aos filtros.")
    else:
        for name, info in filtered:
            with st.expander(f"**{name}** — {info['tipo']}", expanded=False):
                st.markdown(f"**O que faz:** {info['descricao']}")
                st.markdown(f"**Como funciona:** {info['como_funciona']}")
                st.markdown(f"**Quando usar:** {info['quando_usar']}")
                st.markdown(f"**Limitações:** {info['limitacoes']}")

# ========== TAB 7 — FUNÇÕES MATEMÁTICAS ==========
with tabs[6]:
    st.header("🧮 Funções matemáticas dos modelos")
    st.caption("Veja, exporte e compare as expressões matemáticas dos modelos treinados.")
    result_state = st.session_state.get("results")
    if result_state is None:
        st.info("Execute uma modelagem na aba **Modelagem** ou **Série Temporal**.")
    else:
        valid = {n: v for n, v in result_state["results"].items()
                 if "error" not in v}
        if not valid:
            st.warning("Nenhum modelo treinado com sucesso.")
        else:
            df_resumo, expressoes = compare_math_functions(result_state)
            st.subheader("📋 Resumo comparativo")
            ui_dataframe(df_resumo, hide_index=True,
                          height=min(400, 40 * (len(df_resumo) + 1)))

            st.subheader("🔍 Detalhamento das funções")
            sel = st.multiselect("Modelos para exibir",
                                  list(expressoes.keys()),
                                  default=list(expressoes.keys()),
                                  key="math_sel")
            if sel:
                for name in sel:
                    info = expressoes[name]
                    with st.expander(f"**{name}** · {info['tipo']}", expanded=False):
                        st.code(info["expressao"], language="text")
                        if info["tabela"] is not None and not info["tabela"].empty:
                            ui_dataframe(info["tabela"].round(6), hide_index=True)

st.divider()
st.caption(
    "🧠 Análise de Inteligência Computacional — Streamlit + Plotly | "
    "Regressão, Classificação, Agrupamento, Série Temporal (ARIMA/SARIMAX/ETS/Prophet + lag features). "
    "Pré-análise, recomendações, ajuste manual e exportação de funções."
)