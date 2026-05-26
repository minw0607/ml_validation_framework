"""
src/validation.py
ML Model Validation Framework — core ValidationFramework class.

All interactive widget logic (data loading, preprocessing, model training,
explainability, diagnostics) lives here; the demo notebook imports this class
and calls its methods step-by-step.
"""
import os
import pandas as pd
import numpy as np
import time
from math import sqrt
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import HBox, VBox, interactive
from IPython.display import display, clear_output, HTML, Markdown

from sklearn.datasets import (load_breast_cancer, load_wine, load_iris,
                               load_diabetes, fetch_california_housing)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    LabelEncoder, OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               IsolationForest)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import tree
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              mean_squared_error, mean_absolute_error, r2_score,
                              precision_score, recall_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import trange, tqdm

import graphviz
import eli5
from eli5.sklearn import PermutationImportance
import shap
import lime
import lime.lime_tabular

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class ValidationFramework:
  def __init__(self, data_dir=None):
    self.data = pd.DataFrame()  # data used in the validation
    self.data_copy = self.data  # a copy of the data (always save a copy when the data in use is changed)
    self.task = 'classification'   # task type: regression, classification (default) or outlier detection
    self.models = dict()           # dictionary of models to train the data
    self.metrics = dict()          # dictionary of metrics to measure model performance
    self.register = dict()         # dictionary of registered model(s)
    self.results = pd.DataFrame()  # training results
    self.target = ''            # target variable (for supervised learning)
    self.test_ratio = 0.2       # test ratio (default = 0.2)
    self.random = 0             # random state (default = 0)
    self.data_dir = self._resolve_data_dir(data_dir)

  @staticmethod
  def _section(icon, title, border_color, bg_color):
    """Return an HTML section-header string (replaces widgets.Tab)."""
    return (
      f'<div style="background:{bg_color};padding:10px 14px;'
      f'border-left:4px solid {border_color};border-radius:4px;'
      f'margin:16px 0 6px;font-size:15px;font-weight:bold">'
      f'{icon} {title}</div>'
    )

  def _resolve_data_dir(self, data_dir):
    if data_dir is not None and os.path.isdir(data_dir):
      return data_dir
    for candidate in ['data', '../data', 'ml-model-validation/data']:
      if os.path.isdir(candidate):
        return candidate
    return 'data'

  # upload data by user
  def data_uploader(self):
   uploaded = colab_files.upload() if IN_COLAB else None

  #use pre-loaded data

  def data_loader(self):
    """Interactive data loader: built-in sklearn datasets, credit CSV datasets, or upload."""
    DATASETS = {
      'Select Dataset': None,
      # ── sklearn built-ins ──────────────────────────────────────────
      'Breast Cancer  (Binary Classification)': {
        'key': 'breast_cancer', 'task': 'classification', 'target': 'target',
        'rows': 569,  'features': 30,
        'desc': 'Predict malignant vs benign tumors from 30 cell-nucleus measurements.',
        'source': 'sklearn'
      },
      'Wine Quality   (Multi-class Classification)': {
        'key': 'wine', 'task': 'classification', 'target': 'target',
        'rows': 178, 'features': 13,
        'desc': 'Classify wines by cultivar from 13 chemical measurements.',
        'source': 'sklearn'
      },
      'Iris           (Multi-class Classification)': {
        'key': 'iris', 'task': 'classification', 'target': 'target',
        'rows': 150, 'features': 4,
        'desc': 'Identify iris species from sepal and petal dimensions.',
        'source': 'sklearn'
      },
      'Diabetes       (Regression)': {
        'key': 'diabetes', 'task': 'regression', 'target': 'target',
        'rows': 442, 'features': 10,
        'desc': 'Predict diabetes disease progression from 10 baseline variables.',
        'source': 'sklearn'
      },
      'California Housing (Regression)': {
        'key': 'california_housing', 'task': 'regression', 'target': 'target',
        'rows': 20640, 'features': 8,
        'desc': 'Predict median house values from California census block data.',
        'source': 'sklearn'
      },
      # ── CSV datasets ──────────────────────────────────────────────
      'GMSC Credit Risk (Binary Classification)': {
        'key': 'gmsc', 'task': 'classification', 'target': 'SeriousDlqin2yrs',
        'rows': 30000, 'features': 10,
        'desc': ('Give Me Some Credit: predict financial distress in the next 2 years. '
                 'Real-world credit-scoring dataset with class imbalance (7% positive rate). '
                 'Sampled to 30 k rows for speed.'),
        'source': 'data/cs-training.csv'
      },
      'Taiwan Credit Default (Binary Classification)': {
        'key': 'taiwan_credit', 'task': 'classification',
        'target': 'default.payment.next.month',
        'rows': 30000, 'features': 23,
        'desc': 'Predict credit card default among Taiwanese customers (Oct 2005). '
                'Includes demographic, payment history, and billing features.',
        'source': 'data/UCI_Credit_Card.csv'
      },
      # ── user upload ───────────────────────────────────────────────
      'Upload CSV': {'key': 'upload', 'task': None, 'target': None,
                     'rows': None, 'features': None, 'desc': '', 'source': ''},
    }

    SKLEARN_LOADERS = {
      'breast_cancer':      load_breast_cancer,
      'wine':               load_wine,
      'iris':               load_iris,
      'diabetes':           load_diabetes,
      'california_housing': fetch_california_housing,
    }

    dropdown = widgets.Dropdown(options=list(DATASETS.keys()), description='Dataset:',
                                layout=widgets.Layout(width='60%'))
    info_out = widgets.Output()
    data_out = widgets.Output()

    def _info_card(meta):
      """Render a compact HTML info card for the selected dataset."""
      if meta is None or meta.get('key') in ('upload', None):
        return ''
      rows_str = f"{meta['rows']:,}" if meta['rows'] else '—'
      feat_str = str(meta['features']) if meta['features'] else '—'
      badge = ('<span style="background:#4CAF50;color:white;padding:2px 8px;'
               'border-radius:4px;font-size:12px">Classification</span>'
               if meta['task'] == 'classification' else
               '<span style="background:#2196F3;color:white;padding:2px 8px;'
               'border-radius:4px;font-size:12px">Regression</span>')
      return (f'<div style="border:1px solid #ddd;border-radius:6px;padding:12px;'
              f'margin:8px 0;background:#fafafa">'
              f'<b>Task:</b> {badge} &nbsp; '
              f'<b>Target:</b> <code>{meta["target"]}</code> &nbsp; '
              f'<b>Rows:</b> {rows_str} &nbsp; '
              f'<b>Features:</b> {feat_str}<br>'
              f'<span style="color:#555;font-size:13px">{meta["desc"]}</span>'
              f'</div>')

    def on_select(change):
      info_out.clear_output()
      data_out.clear_output()
      meta = DATASETS.get(change.new)
      if meta is None:
        return
      key = meta['key']

      with info_out:
        display(HTML(_info_card(meta)))

      with data_out:
        if key == 'upload':
          try:
            from google.colab import files as _colab_files
            uploaded = _colab_files.upload()
            fname = list(uploaded.keys())[0]
            df = pd.read_csv(fname)
          except ImportError:
            path_input = widgets.Text(placeholder='Full path to CSV file...',
                                      description='File path:',
                                      layout=widgets.Layout(width='60%'))
            load_btn = widgets.Button(description='Load', button_style='success')
            load_out = widgets.Output()
            def do_load(b):
              with load_out:
                load_out.clear_output()
                try:
                  df = pd.read_csv(path_input.value)
                  self.data = df
                  display(Markdown(f"**Loaded** `{path_input.value}`: "
                                   f"{len(df):,} rows × {len(df.columns)} cols."))
                  display(df.head())
                except Exception as e:
                  print(f"Error: {e}")
            load_btn.on_click(do_load)
            display(VBox([path_input, load_btn, load_out]))
            return

        elif key in SKLEARN_LOADERS:
          data = SKLEARN_LOADERS[key]()
          df = pd.DataFrame(data.data, columns=data.feature_names)
          df['target'] = data.target

        elif key == 'gmsc':
          fpath = os.path.join(self.data_dir, 'cs-training.csv')
          df = pd.read_csv(fpath).drop(columns=['Unnamed: 0'], errors='ignore')
          # Sample to 30 k for speed; stratify on target to preserve imbalance
          df = df.dropna().sample(n=min(30000, len(df)), random_state=42)
          df = df.reset_index(drop=True)

        elif key == 'taiwan_credit':
          fpath = os.path.join(self.data_dir, 'UCI_Credit_Card.csv')
          df = pd.read_csv(fpath).drop(columns=['ID'], errors='ignore')

        else:
          print(f"Unknown dataset key: {key}")
          return

        self.data = df
        # Auto-configure task and target for known datasets
        if meta['task']:
          self.task   = meta['task']
          self.target = meta['target']
          display(Markdown(
            f"**Loaded:** {len(df):,} rows × {len(df.columns)} cols.  "
            f"Auto-configured → task: `{self.task}`, target: `{self.target}`"
          ))
        else:
          display(Markdown(f"**Loaded:** {len(df):,} rows × {len(df.columns)} cols."))
        display(df.head())

    dropdown.observe(on_select, names='value')
    display(HTML("<h3>Select a dataset:"))
    display(dropdown)
    display(info_out)
    display(data_out)


  def data_summary(self):
    df = self.data
    # create widgets for selecting variable and data type
    var_selector = widgets.Dropdown(options=list(df.columns), description='Variable:')
    type_selector = widgets.Dropdown(options=['Numerical', 'Categorical'], description='Data Type:')

    # create button to confirm conversion
    convert_button = widgets.Button(description='Convert',
                                    button_style = 'Success')

    # create output areas for each summary section
    num_tab  = widgets.Output()
    cat_tab  = widgets.Output()
    null_tab = widgets.Output()
    dist_tab = widgets.Output()

    # display features with missing value status
    null_tab.clear_output()
    with null_tab:
      message_missing = HTML(f"<h3>Missing value (True / False):")
      display(message_missing)
      display(df.isnull().any())

    dist_tab.clear_output()
    with dist_tab:
      # Load a built-in dataset from seaborn library
      df = self.data

      # Create dropdowns for variable selection
      var1_select = widgets.Dropdown(
          options=df.columns,
          description='Variable 1:',
          value=df.columns[0]  # Pre-selected value
      )

      var2_select = widgets.Dropdown(
          options=df.columns,
          description='Variable 2:',
          value=df.columns[1]  # Pre-selected value
      )

      dist_var_select = widgets.Dropdown(
          options=df.columns,
          description='Variable:',
          value=df.columns[0]  # Pre-selected value
      )

      # Create output widgets
      dist_out = widgets.Output()
      scatter_out = widgets.Output()
      heatmap_out = widgets.Output()

      # Function to create and update plots
      def update_dist_plot(dist_var):
          dist_out.clear_output(wait=True)  # clear the old plot
          with dist_out:
              plt.figure(figsize=(6, 4))
              if df[dist_var].dtype == 'object':
                  sns.countplot(data=df, x=dist_var)
              else:
                  sns.histplot(data=df, x=dist_var)
              plt.title('Univariate Plot')
              plt.tight_layout()
              plt.show()

      def update_scatter_plot(var1, var2):
          scatter_out.clear_output(wait=True)  # clear the old plot
          with scatter_out:
              plt.figure(figsize=(6, 4))
              sns.scatterplot(data=df, x=var1, y=var2)
              plt.title('Bivariate Plot')
              plt.tight_layout()
              plt.show()

      def update_heatmap():
          heatmap_out.clear_output(wait=True)  # clear the old plot
          with heatmap_out:
              plt.figure(figsize=(6, 4))
              corr = df.select_dtypes(include=[np.number]).corr()  # calculate correlation of numerical columns
              sns.heatmap(corr, annot=False, cmap='coolwarm', cbar=True)  # set annot=False
              plt.title('Correlation Heatmap')
              plt.tight_layout()
              plt.show()

      # Update plots when dropdown selection changes
      dist_var_select.observe(lambda change: update_dist_plot(change.new), names='value')
      var1_select.observe(lambda change: update_scatter_plot(change.new, var2_select.value), names='value')
      var2_select.observe(lambda change: update_scatter_plot(var1_select.value, change.new), names='value')

      # Arrange widgets and plot
      interactive_dist_plot = widgets.VBox([dist_out, dist_var_select])
      interactive_scatter_plot = widgets.VBox([scatter_out, var1_select, var2_select])

      # Display widgets and plot
      display(widgets.HBox([interactive_dist_plot, interactive_scatter_plot, heatmap_out]))

      # Call the functions initially to display the plots
      update_dist_plot(dist_var_select.value)
      update_scatter_plot(var1_select.value, var2_select.value)
      update_heatmap()


    # define function to convert data type of selected variable
    def convert_data_type(var_name, new_type):
        if new_type == 'Numerical':
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
        elif new_type == 'Categorical':
            df[var_name] = df[var_name].astype('category')

        # update statistics summaries
        update_stats()

    # define function to update statistics summaries
    def update_stats():
        # get numerical variables
        num_vars = df.select_dtypes(include=['float64', 'int64']).columns

        # get categorical variables
        cat_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns

        # display statistics summary for numerical variables
        with num_tab:
            num_tab.clear_output()
            if num_vars.empty:
              print("There is no numerical feature in the dataset.")
            else:
              numerical_summary = df[num_vars].describe().transpose()
              display(numerical_summary)

        # display statistics summary for categorical variables
        with cat_tab:
            cat_tab.clear_output()
            if cat_vars.empty:
              print("There is no categorical feature in the dataset.")
            else:
              categorical_summary = df[cat_vars].describe().transpose()
              display(categorical_summary)

    # define function to handle button click event
    def on_convert_button_click(b):
        var_name = var_selector.value
        new_type = type_selector.value
        convert_data_type(var_name, new_type)
        self.data = df # update THE dataset attribute of the class

    # attach event handler to button click
    convert_button.on_click(on_convert_button_click)

    # display all sections directly (widgets.Tab unreliable in Colab)
    S = self._section
    display(HTML(S('📊', 'Numerical Features Statistics',   '#1a73e8', '#e8f0fe')))
    display(num_tab)
    display(HTML(S('🏷️', 'Categorical Features Statistics', '#7b1fa2', '#f3e5f5')))
    display(cat_tab)
    display(HTML(S('❓', 'Missing Value Analysis',           '#e65100', '#fff3e0')))
    display(null_tab)
    display(HTML(S('📈', 'Correlation Analysis',             '#34a853', '#e6f4ea')))
    display(dist_tab)
    display(widgets.HBox([var_selector, type_selector, convert_button]))

    update_stats()


 # data info (feature names, data types, counts, etc.)
  def data_info(self):
    df = self.data
    if df.empty:
      print('Please upload or load data first.')
    else:
      df.info()


  # data preprocessing
  # def data_preprocssing(self):
  #   df = self.data
  #   if df.empty:
  #     print('Please upload or load data first.')
  #   else:
  #     # missing data treatment
  #     missing_vars = df.columns[df.isnull().any()]

  #     # Create a dropdown widget for selecting the variable to impute
  #     var_null_dropdown = widgets.Dropdown(
  #         options=list(missing_vars),
  #         description='Feature with Null',
  #         disabled=False
  #     )

  #     # Create a dropdown widget for selecting the imputation method
  #     method_null_dropdown = widgets.Dropdown(
  #         options=['mean', 'median', 'mode', 'constant'],
  #         description='Imputation Method:',
  #         disabled=False
  #     )

  #     # Create a function to handle the change in the variable dropdown
  #     def var_null_dropdown_eventhandler(change):
  #         variable = change.new

  #         # Select the data for the selected variable
  #         variable_data = df[[variable]]

  #         # Determine the data type of the selected variable
  #         variable_dtype = df[variable].dtype

  #         # Determine the applicable imputation methods based on the data type
  #         if variable_dtype == 'object':
  #             imputation_methods = ['most_frequent', 'constant']
  #         else:
  #             imputation_methods = ['mean', 'median', 'most_frequent', 'constant']

  #         # Update the options in the method dropdown
  #         method_null_dropdown.options = imputation_methods

  #         # Create a SimpleImputer object with the selected imputation method
  #         imputer = SimpleImputer(strategy=method_null_dropdown.value)

  #         # Fit and transform the imputer on the variable data
  #         variable_data_imputed = imputer.fit_transform(variable_data)

  #         # Replace the original variable data with the imputed data in the dataframe
  #         df[variable] = variable_data_imputed

  #         # Display the first 10 rows of the normalized dataframe
  #         display(df.head(10))

  #     # Register the function as an event handler for the variable dropdown
  #     var_null_dropdown.observe(var_null_dropdown_eventhandler, names='value')

  #     # Display the variable dropdown
  #     #display(var_null_dropdown)
  #     #display(method_null_dropdown)

  #     # data normalization component
  #     # Get the numerical variables
  #     numerical_vars = list(df.select_dtypes(include=['float64', 'int64']).columns)

  #     # Create a dropdown widget for selecting the variable
  #     var_num_dropdown = widgets.Dropdown(options=numerical_vars, description='Features')

  #     # Create a dropdown widget for selecting the normalization method
  #     norm_dropdown = widgets.Dropdown(options=['min-max', 'z-score'], description='Normalization method')

  #     # Define a function to handle the change in the variable dropdown
  #     def var_num_dropdown_eventhandler(change):
  #         variable = change.new

  #         # Select the data for the selected variable
  #         variable_data = df[variable]

  #         # Determine the selected normalization method
  #         norm_method = norm_dropdown.value

  #         # Normalize the data based on the selected normalization method
  #         if norm_method == 'min-max':
  #             variable_data_normalized = \
  #             (variable_data - variable_data.min()) / (variable_data.max() - variable_data.min())
  #         elif norm_method == 'z-score':
  #             variable_data_normalized = \
  #             (variable_data - variable_data.mean()) / variable_data.std()

  #         # Update the variable data in the dataframe with the normalized data
  #         df[variable] = variable_data_normalized

  #         # Display the first 10 rows of the normalized dataframe
  #         display(df.head(10))

  #     # Register the function as an event handler for the variable dropdown
  #     var_num_dropdown.observe(var_num_dropdown_eventhandler, names='value')

  #     # Display the variable and normalization method dropdowns
  #     #display(var_num_dropdown)
  #     #display(norm_dropdown)

  #     # Create a tab widget with two tabs
  #     tab = widgets.Tab()
  #     tab_contents = ['Missing Data Imputation', 'Numerical Feature Normalization']
  #     tab.children = [widgets.VBox([var_null_dropdown, method_null_dropdown]), \
  #                     widgets.VBox([var_num_dropdown, norm_dropdown])]
  #     for i in range(len(tab_contents)):
  #         tab.set_title(i, tab_contents[i])

  #     # Display the tab widget
  #     display(tab)

  def data_preprocess(self):
    df = self.data
    if df.empty:
      display(HTML(
        '<div style="padding:12px;background:#fff3cd;border:1px solid #ffc107;'
        'border-radius:6px;margin:8px 0">'
        '⚠️ <b>No data loaded.</b> Please complete <b>Step 1 — Load Data</b> first.'
        '</div>'
      ))
      return

    try:
      # Exclude the target column from preprocessing dropdowns
      target = self.target
      feature_cols    = [c for c in df.columns if c != target]
      numerical_feats = [c for c in df.select_dtypes(
                            include=['float64', 'int64']).columns if c != target]

      # ── Imputation section ──────────────────────────────────────────
      var_dropdown = widgets.Dropdown(
        options=['— select variable —'] + feature_cols,
        description='Variable:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
      )

      impute_dropdown = widgets.Dropdown(
        options=['— select method —'],
        description='Method:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
      )

      def on_var_change(change):
        var = change['new']
        if var == '— select variable —':
          impute_dropdown.options = ['— select method —']
        elif df[var].dtype == object:
          impute_dropdown.options = ['— select method —', 'most_frequent', 'constant']
        else:
          impute_dropdown.options = ['— select method —', 'mean', 'median',
                                     'most_frequent', 'constant']
        impute_dropdown.value = '— select method —'   # always reset on variable change

      var_dropdown.observe(on_var_change, names='value')

      confirm_btn = widgets.Button(description='Apply', button_style='success')
      impute_out = widgets.Output()

      def on_impute_click(b):
        with impute_out:
          impute_out.clear_output()
          var    = var_dropdown.value
          method = impute_dropdown.value
          if var == '— select variable —' or method == '— select method —':
            print('Please select both a variable and an imputation method.')
            return
          n_missing = df[var].isnull().sum()
          if n_missing == 0:
            print(f'✔ No missing values in "{var}" — nothing to impute.')
            return
          imputer = SimpleImputer(strategy=method)
          df[var] = imputer.fit_transform(df[[var]])
          self.data = df
          print(f'✔ Imputed {n_missing} missing values in "{var}" using {method}.')
          display(df.head(10))

      confirm_btn.on_click(on_impute_click)

      impute_widget = widgets.VBox([
        widgets.HTML('<b>Fill missing values</b> — select a feature column and an imputation strategy.'),
        var_dropdown,
        impute_dropdown,
        confirm_btn,
        impute_out,
      ])

      # ── Normalization section ───────────────────────────────────────
      var_dropdown2 = widgets.Dropdown(
        options=['— select variable —'] + numerical_feats,
        description='Variable:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
      )

      norm_dropdown = widgets.Dropdown(
        options=['— select method —', 'standard', 'minmax'],
        description='Method:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
      )

      confirm_btn2 = widgets.Button(description='Apply', button_style='success')
      norm_out = widgets.Output()

      def on_norm_click(b):
        with norm_out:
          norm_out.clear_output()
          var    = var_dropdown2.value
          method = norm_dropdown.value
          if var == '— select variable —' or method == '— select method —':
            print('Please select both a variable and a normalization method.')
            return
          scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
          df[var] = scaler.fit_transform(df[[var]])
          self.data = df
          print(f'✔ Applied {method} normalization to "{var}".')
          display(df.head(10))

      confirm_btn2.on_click(on_norm_click)

      norm_widget = widgets.VBox([
        widgets.HTML('<b>Scale numerical features</b> — standard (z-score) or min-max normalization.'),
        var_dropdown2,
        norm_dropdown,
        confirm_btn2,
        norm_out,
      ])

      # ── Display sections (widgets.Tab is unreliable in Colab) ────────
      S = self._section
      display(HTML(S('🔧', 'Data Imputation',   '#1a73e8', '#e8f0fe')))
      display(impute_widget)
      display(HTML(S('📐', 'Data Normalization', '#34a853', '#e6f4ea')))
      display(norm_widget)

    except Exception as _e:
      import traceback as _tb
      print(f"ERROR in data_preprocess: {_e}")
      _tb.print_exc()



  def model_prepare(self):
    df = self.data
    # Create a dropdown widget for choosing the target variable
    target_widget = widgets.Dropdown(
        options=['make selection']+ list(df.columns) + ["N/A"],
        description='Target feature:',
        value = 'make selection')

    # Create a dropdown widget for choosing the task
    task_widget = widgets.Dropdown(
        options=['make selection', 'classification', 'regression', 'anomaly detection'],
        description='Task:',
        value = 'make selection')

    # Create a slider widget for choosing the train-test split ratio
    split_widget = widgets.FloatSlider(
        min=0.1,
        max=0.9,
        step=0.1,
        value=0.2,  # default test size
        description='Train-test split ratio:'
    )

    # Create the slider widget to select the random state with default value 0
    random_widget = widgets.IntSlider(
        value=0,   # Default value
        min=0,     # Minimum value
        max=10,    # Maximum value
        step=1,    # Step size
        description='Random state:',
        orientation='horizontal',
        readout=True,  # Display the selected value
        readout_format='d'  # Format the selected value as an integer
    )

    # Define a function to train and evaluate the model
    def train_setup(target_variable, task, train_test_split_ratio, random_state):
        # Split the data into train and test sets
        if target_variable == 'make selection' or task == 'make selection':
          print("Please make a selection.")
          return False
        else:
          #X = df.drop(target_variable, axis=1)
          #y = df[target_variable]
          #X_train, X_test, y_train, y_test = \
          #train_test_split(X, y, test_size=train_test_split_ratio, random_state=0)

          self.target = target_variable
          self.task = task
          self.test_ratio = train_test_split_ratio
          self.random = random_state

          return True
        # Train the model
        # if task == 'classification':
        #     model = LogisticRegression()
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        #     score = accuracy_score(y_test, y_pred)
        #     print(f'Accuracy score: {score:.2f}')
        # elif task == 'regression':
        #     model = LinearRegression()
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        #     mse = mean_squared_error(y_test, y_pred)
        #     print(f'Mean squared error: {mse:.2f}')

          #return X_train, X_test, y_train, y_test

    # Create a button widget to train and evaluate the model
    button = widgets.Button(description='Confirm',
                            button_style = 'success')
    output = widgets.Output()

    # Define a function to handle the button click event
    def on_button_click(b):
        with output:
            output.clear_output()
            if train_setup(target_widget.value, task_widget.value,
                           split_widget.value, random_widget.value):

              display(Markdown('Data preparation is complete.'))

    # Attach the button click event to the button widget
    button.on_click(on_button_click)

    # Display the widgets
    display(widgets.VBox([
        target_widget,
        task_widget,
        split_widget,
        button,
        output
    ]))


  def feature_select(self):
    df = self.data
    test_ratio = self.test_ratio
    random_state = self.random
    task = self.task
    target_var = self.target

    # create output areas for each feature selection section
    pearson_corr_tab  = widgets.Output()
    spearman_corr_tab = widgets.Output()
    importance_tab    = widgets.Output()

    S = self._section
    display(HTML(S('📐', 'Pearson Correlation',   '#1a73e8', '#e8f0fe')))
    display(pearson_corr_tab)
    display(HTML(S('📏', 'Spearman Correlation',  '#7b1fa2', '#f3e5f5')))
    display(spearman_corr_tab)
    display(HTML(S('🏆', 'Feature Importance',    '#34a853', '#e6f4ea')))
    display(importance_tab)

######################### TAB ###########################
    # display pearson correlation
    pearson_corr_tab.clear_output()
    with pearson_corr_tab:
      # Create the bar plot of the Pearson correlation coefficients
      def create_corr_plot(threshold):
        # Compute the Pearson correlation coefficients between the variables and the target variable
        df = self.data

        # Select only numberical variables for pearson correlation
        num_vars = df.drop(columns=[target_var]).select_dtypes(include=[np.number]).columns
        corr = df[num_vars].apply(lambda x: x.corr(df[target_var]))

        corr_filtered = corr[abs(corr) >= threshold]
        corr_filtered_less = corr[abs(corr) < threshold]

        plt.clf() # clear the existing plot if any
        plt.figure(figsize=(10,6))
        plt.bar(x=corr_filtered_less.index, height=corr_filtered_less.values, color='grey')
        plt.bar(x=corr_filtered.index, height=corr_filtered.values, color='green')
        plt.xticks(rotation=90)
        plt.axhline(y=threshold, color='r', linestyle='--')
        plt.axhline(y=-threshold, color='r', linestyle='--')
        plt.title('Pearson Correlation Coefficients with {}'.format(target_var))
        plt.show()

      # Create the widget to select the threshold
      threshold_slider = widgets.FloatSlider(
          value=0.1,
          min=0,
          max=1,
          step=0.01,
          description='Threshold:',
          orientation='horizontal'
      )

      # threshold_output = widgets.Output()

      # def threshold_slider_eventhandler(change):
      #   pearson_corr_tab.clear_output()
      #   threshold = change.new
      #   with threshold_output:
      #     create_corr_plot(threshold)
      #     threshold_slider.value = threshold
      #     display(widgets.HBox([threshold_slider, button_remove, button_revert]))

      # # Register the function as an event handler for the threshold slider
      # threshold_slider.observe(threshold_slider_eventhandler, names='value')

      # create a button to confirm the threshold and re-generate the plot
      button_threshold = widgets.Button(description='Confirm Threshold',
                        layout=widgets.Layout(width='25%', height='30px'),
                        button_style = 'success')

      # Create the button widget to confirm the selection of the threshold
      button_remove = widgets.Button(description='Remove features',
                                     button_style = 'success')
      button_revert = widgets.Button(description='Revert',
                                    icon='reply',
                                    button_style='warning')

      # create the output widgets
      output_remove = widgets.Output()
      output_revert = widgets.Output()
      # create a dictionary to map buttons to their corresponding output widgets
      output_dict = {button_remove: output_remove, button_revert: output_revert}




      # Define the output widget to display the plot and confirmation message
      pearson_out = widgets.Output()

      def on_button_threshold_clicked(b):
        threshold = threshold_slider.value

        with pearson_corr_tab:
          clear_output() # clear all existing outputs in the current tab

          threshold_slider.value = threshold
          display(widgets.HBox([threshold_slider, button_threshold]))
          display(widgets.HBox([button_remove, button_revert]))
          create_corr_plot(threshold)

          print("The threshold of Pearson Correlation is set to be {}.".format(threshold))

        #pearson_out.clear_output()
        # with pearson_out:
        #   clear_output()
        #   plt.clf()
        #   create_corr_plot(threshold)
        #   print("The threshold of Pearson Correlation is set to be {}.".format(threshold))
        #   #display(pearson_out)

      button_threshold.on_click(on_button_threshold_clicked)

      # Define a function to remove the variables with correlation less than the threshold
      def remove_variables(threshold):
          # Compute the Pearson correlation coefficients between the variables and the target variable
          df = self.data
          corr = df.drop(target_var, axis=1).apply(lambda x: x.corr(df[target_var]))
          corr_vars = corr[abs(corr) < threshold].index.tolist()
          df_corr = df.drop(columns = corr_vars)
          return df_corr


      # Define a function to handle the button click event
      def on_button_remove_clicked(b):
        threshold = threshold_slider.value
        #create_corr_plot(threshold)
        self.data_copy = self.data
        self.data = remove_variables(threshold) # update the dataset attribute of the class
        #  clear_output()
        with output_dict[b]:
          #with output_dict[button_revert]:
          display('Features with correlations less than {} have been removed.'.format(threshold))

          display(self.data)
        with pearson_corr_tab:
         display(output_dict[b])

      def on_button_revert_clicked(b):
          self.data = self.data_copy
          # clear the output generated by button1
          with output_dict[button_remove]:
            clear_output()
          #with output_dict[b]:
          display('The removed features are added back to the dataset.')

          display(self.data)
          #display(output_dict[b])

      # Attach the button click event to the button widgets
      button_remove.on_click(on_button_remove_clicked)
      button_revert.on_click(on_button_revert_clicked)


      display(widgets.HBox([threshold_slider, button_threshold]))
      display(widgets.HBox([button_remove, button_revert]))
      create_corr_plot(threshold_slider.value)

      #display(output_remove)
      #display(output_revert)

      # Create an output widget for printing the confirmation message and updated dataset
      #pearson_out = widgets.Output()

      # Display the correlation and widgets

      #pearson_widget = widgets.HBox([threshold_slider, button_threshold])
      #feature_widget = widgets.HBox([button_remove, button_revert])
      #display(widgets.HBox([button_remove, button_revert]))
      #display(widgets.VBox([pearson_widget, feature_widget]))
      #create_corr_plot(threshold_slider.value)
      #display(pearson_out)


######################### TAB #################################
    # display spearman correlation and correlation heatmap
    # Handling Multicollinear Features
    spearman_corr_tab.clear_output()
    with spearman_corr_tab:
      df = self.data
    target_variable = self.target
    num_vars = df.drop(columns=[target_variable]).select_dtypes(include=[np.number]).columns

    spearman_out = widgets.Output()

    threshold_slider_sp = widgets.FloatSlider(
        value=0.1, min=0, max=1, step=0.01, description='Threshold:')
    btn_sp = widgets.Button(description='Apply Threshold', button_style='success')
    btn_remove_sp = widgets.Button(description='Remove Features', button_style='success')
    btn_revert_sp = widgets.Button(description='Revert', icon='reply', button_style='warning')

    def draw_spearman(threshold=0.1):
      corr = df[num_vars].apply(lambda x: x.corr(df[target_variable], method='spearman'))
      pos = corr[abs(corr) >= threshold]
      neg = corr[abs(corr) < threshold]
      plt.clf()
      plt.figure(figsize=(10, 6))
      plt.bar(neg.index, neg.values, color='grey')
      plt.bar(pos.index, pos.values, color='steelblue')
      plt.xticks(rotation=90)
      plt.axhline(threshold, color='r', linestyle='--', label=f'+{threshold}')
      plt.axhline(-threshold, color='r', linestyle='--', label=f'-{threshold}')
      plt.title(f'Spearman Correlation with {target_variable}')
      plt.tight_layout()
      plt.show()

    def on_apply_sp(b):
      with spearman_corr_tab:
        clear_output()
        display(widgets.HBox([threshold_slider_sp, btn_sp]))
        display(widgets.HBox([btn_remove_sp, btn_revert_sp]))
        draw_spearman(threshold_slider_sp.value)
        display(spearman_out)

    def on_remove_sp(b):
      threshold = threshold_slider_sp.value
      corr = df.drop(target_variable, axis=1).apply(
          lambda x: x.corr(df[target_variable], method='spearman'))
      to_drop = corr[abs(corr) < threshold].index.tolist()
      self.data_copy = self.data
      self.data = self.data.drop(columns=to_drop)
      with spearman_out:
        spearman_out.clear_output()
        display(Markdown(f"Removed {len(to_drop)} feature(s): `{', '.join(to_drop)}`"))
        display(self.data)

    def on_revert_sp(b):
      self.data = self.data_copy
      with spearman_out:
        spearman_out.clear_output()
        display(Markdown("Reverted — all features restored."))

    btn_sp.on_click(on_apply_sp)
    btn_remove_sp.on_click(on_remove_sp)
    btn_revert_sp.on_click(on_revert_sp)

    display(widgets.HBox([threshold_slider_sp, btn_sp]))
    display(widgets.HBox([btn_remove_sp, btn_revert_sp]))
    draw_spearman(threshold_slider_sp.value)
    display(spearman_out)

######################### TAB #################################
    # display permutation based feature importance
    importance_tab.clear_output()
    with importance_tab:
      # take attributes from the main class
      df = self.data
      target_variable = self.target
      test_ratio = self.test_ratio
      random_state = self.random
      # declare the feature importance which will be used by multiple functions
      global feature_importance

      # creat an output to display the results within the tab
      output_importance = widgets.Output()

      def perm_importance(df, fitter):
        # First drop any rows with missing values. Note: this will be refined later
        #df = self.data
        df = df.dropna()
        # get numerical variables
        num_vars = df.select_dtypes(include=['float64', 'int64']).columns
        # get categorical variables
        cat_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns

        if num_vars.empty or cat_vars.empty:
          # With no numerical or categorical variable at all, no need to encode
          df_encoded = df
        else:
        # apply label encoding to the categorical variables
          le = LabelEncoder()
          df_cat_encoded = df[cat_vars].apply(le.fit_transform)
          # create a new dataset with the original numerical variables and encoded categorical variables
          df_encoded = pd.concat([df[num_vars], df_cat_encoded], axis=1)

        X = df_encoded.drop(target_variable, axis=1)
        y = df_encoded[target_variable]
        # split the data into training/validation and test sets
        # Using the global test size ratio and random state to ensure consistency
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y,
                                                            test_size= test_ratio,
                                                            random_state=random_state)
        # split the training/validation set into training and validation sets
        # test ratio set to be 0.25
        # the importance feature is tested on the validation dataset
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                          test_size=0.25,
                                                          random_state=42)

        # train the fitter with the training data
        fitter.fit(X_train, y_train)
        # calculate permuation importance using the fitted model based on validation dataset
        #for i in tqdm(range(100), desc='Processing...', unit='%'):
        #print("Calculating permuation based feature importance...")
        perm_importance = permutation_importance(fitter, X_val, y_val, n_jobs=-1, random_state=42)

        #print("done.")

        return perm_importance

      #display('check')
      def perm_importance_plot(perm_importance, threshold=0.1):
        # display the importance scores in a bar chart
        feature_importance = perm_importance.importances_mean

        X = df.drop(target_variable, axis=1)
        feature_names = X.columns
        sorted_idx = feature_importance.argsort()

        with output_importance:
          plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
          plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
          plt.axvline(x=threshold, color='r', linestyle='--')
          plt.title('Permutation Importance based on {}'.format(fitter_dropdown.value))
          plt.xlabel('Feature Importance Score')
          plt.show()

      # Create a dropdown widget for choosing the fitter
      fitter_list = ['logistic regression',
                    'random forest',
                    'gradient boosting machine',
                    'isolation forest']
      fitter_dropdown = widgets.Dropdown(
        options=['select a fitter']+ fitter_list,
        description='Fitter:',
        value = 'select a fitter')

      # Create a button widget to confirm the fitter
      fitter_confirm_button = widgets.Button(description='Confirm',
                                      button_style='success')

      # create a progress bar for calculating feature importance
      progress_bar = widgets.IntProgress(
                        value=1,
                        min=0,
                        max=10,
                        step=1,
                        description='Calculating:',
                        bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                        orientation='horizontal')

      # Define a function to handle the button click event
      def on_button_fitter_click(b):
        #global fitter
        if fitter_dropdown.value == 'choose a fitter':
          print('Please choose a fitter.')
          return
        elif fitter_dropdown.value == 'logistic regresssion':
          fitter = LogisticRegression()
        elif fitter_dropdown.value == 'random forest':
          fitter = RandomForestRegressor(n_estimators=100, max_samples = 0.2,
                                        random_state=42)
        elif fitter_dropdown.value == 'gradient boosting machine':
          fitter = GradientBoostingClassifier(n_estimators = 100, learning_rate=0.1,
                                              max_depth = 7,
                                              random_state = 42)
        elif fitter_dropdown.value == 'isolation forest':
          fitter = IsolationForest(n_estimators= 100, max_samples= 0.2,
                                  random_state=42)

        with output_importance:
          display(Markdown('The fitter is chosen to be {}.'.format(fitter_dropdown.value)))
          # set the width of the description
          #int_progress.layout.width = '500px'

          display(progress_bar)
          # start with a new line (otherwise it won't display properly)
          output_importance.append_stdout('\n')

          for i in range(9):
            if i==0:
              global importance # delcare as global variable to be used by other functions

              importance = perm_importance(self.data, fitter)

            # Update the progress bar as the program runs
            time.sleep(0.2)
            progress_bar.value = i+2

          # plot the feature importance bar chart
          # start with a new line (otherwise it won't display properly)
          #output_importance.append_stdout('\n')
          perm_importance_plot(perm_importance = importance)

      # Attach the button click event to the button widget
      fitter_confirm_button.on_click(on_button_fitter_click)


      # Create the widget to select the threshold
      threshold_importance_slider = widgets.FloatSlider(
          value=0.1,
          min=0,
          max=1,
          step=0.01,
          description='Threshold:',
          orientation='horizontal')

      # create a button widget to increase slider value
      increase_button = widgets.Button(description='+',
                                      button_style='primary',
                                      layout=widgets.Layout(width='auto'))

      # create a button widget to decrease slider value
      decrease_button = widgets.Button(description='-',
                                      button_style='primary',
                                      layout=widgets.Layout(width='auto'))


      # define functions to increase and decrease slider value
      def increase_slider_value(b):
          threshold_importance_slider.value += 0.01

      def decrease_slider_value(b):
          threshold_importance_slider.value -= 0.01

      # link button widgets to functions
      increase_button.on_click(increase_slider_value)
      decrease_button.on_click(decrease_slider_value)


      # create a button to confirm the threshold and re-generate the plot
      threshold_importance_button = widgets.Button(description='Confirm Threshold',
                          #layout=widgets.Layout(width='25%', height='30px')
                          button_style='success')

      # define action when click the button
      def on_threshold_importance_button_click(b):
        if fitter_dropdown.value =='choose a fitter':
          print('Please choose a fitter.')
          return
        else:
          threshold_importance = threshold_importance_slider.value

          # clear the previous output
          output_importance.clear_output()

          with output_importance:
            perm_importance_plot(importance, threshold = threshold_importance)

      # attached the action to the button
      threshold_importance_button.on_click(on_threshold_importance_button_click)

      # create a horizontal box to hold the slider and buttons
      threshold_slider_box = widgets.HBox([decrease_button,
                                           threshold_importance_slider,
                                           increase_button,
                                           ],
                                layout=widgets.Layout(justify_content='center', align_items='center'))

      # create a button to remove the variables with low feature importance
      feature_remove_button = widgets.Button(description='Remove Features',
                          #layout=widgets.Layout(width='25%', height='30px')
                          button_style = 'success')

      feature_revert_button = widgets.Button(description='Revert',
                                    icon='reply',
                                    button_style='warning')

      # create the output widgets
      output_remove = widgets.Output()
      output_revert = widgets.Output()
      # create a dictionary to map buttons to their corresponding output widgets
      output_dict = {button_remove: output_remove, button_revert: output_revert}

      # Define a function to remove the variables with importance less than the threshold
      def remove_variables(threshold):
        df = self.data
        #make a copy of the current dataset
        self.data_copy = self.data

        # Take the importance score from the global variable
        importance_score = importance.importances_mean

        X = df.drop(target_variable, axis=1)
        #y = df[target_variable]

        cols_to_keep = importance_score >= threshold
        cols_to_remove = X.columns[~cols_to_keep]
        updated_df = df.drop(cols_to_remove, axis=1)
        #cols_to_remove = importance_score < threshold
        #df = df.drop(columns = cols_to_remove)

        return updated_df

      # Define a function to handle the button click event
      def on_button_remove_clicked(b):
        threshold = threshold_importance_slider.value

        # update the dataset attribute of the class
        self.data = remove_variables(threshold)

        #  clear_output()
        with output_remove:
          #with output_dict[button_revert]:
          print('Features with Importance Score less than {} have been removed.'.format(threshold))

          display(self.data)

        with importance_tab:
         display(output_remove)

      def on_button_revert_clicked(b):
          self.data = self.data_copy
          # clear the output generated by the remove button
          with output_remove:
            clear_output()

          with output_revert:
            print('The removed features are added back to the dataset.')

            display(self.data)

          with importance_tab:
            display(output_revert)

      # Attach the button click event to the button widgets
      feature_remove_button.on_click(on_button_remove_clicked)
      feature_revert_button.on_click(on_button_revert_clicked)

      # create widget boxes
      fitter_widget = widgets.HBox([fitter_dropdown, fitter_confirm_button])
      threshold_widget = widgets.HBox([threshold_slider_box, threshold_importance_button])
      feature_remove_widget = widgets.HBox([feature_remove_button, feature_revert_button])
      # display widgets
      display(widgets.VBox([fitter_widget,
                          threshold_widget,
                          feature_remove_widget]))

      display(output_importance)

  def model_training(self):
    df = self.data
    test_ratio = self.test_ratio
    random_state = self.random
    task = self.task
    target_var = self.target

    X = df.drop(target_var, axis=1)
    y = df[target_var]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_ratio,
                                                        random_state=random_state)

    # Set default hyperparameters
    params_log = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'class_weight': 'balanced',  # handles class imbalance (prevents all-zero predictions)
        'max_iter': 1000,
    }
    params_rf = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42,
    }
    params_gbm = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42,
    }
    params_lgbm = {
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': 42,
    }
    params_xgb = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'random_state': 42,
    }
    params_dt = {
        'max_depth': 5,
        'min_samples_split': 20,
        'random_state': 42,
    }
    param_mlp = {
        'hidden_layer_sizes': (100,),
        'max_iter': 200,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
    }

    if task == 'classification':
      models = {
        'Logistic Regression':  LogisticRegression(**params_log),
        'Decision Tree':        DecisionTreeClassifier(**params_dt, criterion='gini'),
        'Random Forest':        RandomForestClassifier(**params_rf),
        'GBM':                  GradientBoostingClassifier(**params_gbm),
        **({'LightGBM':  LGBMClassifier(**params_lgbm)} if LGBM_AVAILABLE else {}),
        **({'XGBoost':   XGBClassifier(**params_xgb, eval_metric='logloss',
                                       verbosity=0)} if XGB_AVAILABLE else {}),
        'MLP':                  MLPClassifier(**param_mlp),
      }
      metrics = {
        'AUC':       roc_auc_score,
        'Accuracy':  accuracy_score,
        'Precision': lambda y, yh: precision_score(y, yh, zero_division=0),
        'Recall':    lambda y, yh: recall_score(y, yh, zero_division=0),
        'F1 Score':  lambda y, yh: f1_score(y, yh, zero_division=0),
      }
    elif task == 'regression':
      models = {
        'Linear Regression':    LinearRegression(),
        'Decision Tree':        DecisionTreeRegressor(**params_dt, criterion='squared_error'),
        'Random Forest':        RandomForestRegressor(**params_rf),
        'GBM':                  GradientBoostingRegressor(**params_gbm),
        **({'LightGBM':  LGBMRegressor(**params_lgbm)} if LGBM_AVAILABLE else {}),
        **({'XGBoost':   XGBRegressor(**params_xgb, verbosity=0)} if XGB_AVAILABLE else {}),
        'MLP':                  MLPRegressor(**param_mlp),
      }
      metrics = {
        'MSE':       mean_squared_error,
        'RMSE':      lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':       mean_absolute_error,
        'R-Squared': r2_score,
      }

    output = widgets.Output()

    # Define function to train selected models
    def train_models(selected_models, selected_metrics, hyperparameters):
        # save the selected model to Main
        self.models = {key: models[key] for key in selected_models}

        results = pd.DataFrame(columns=['Model'] + list(selected_metrics))

        progress_bar = widgets.IntProgress(
                value=1,
                min=1,
                max=len(selected_models)-1,
                step=1,
                description='Calculating',
                bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal')
        with output:
          display(progress_bar)

        for name, model in models.items():
            if name in selected_models:
              param = {key: item.value for key, item in hyperparameters[name].items()}
              if name == 'MLP':
                def convert_to_tuple(text_input):
                  return tuple(int(s) for s in text_input.split(',') if s.strip())
                param['hidden_layer_sizes'] = convert_to_tuple(param['hidden_layer_sizes'])

              model.set_params(**param)
              model.fit(X_train, y_train)

              # update model in Model model dictionary
              self.models[name] = model

              # Predict on test set
              if 'AUC' in selected_metrics:
                y_preds = model.predict_proba(X_test)[:, 1]
              else:
                y_preds = model.predict(X_test)
              #y_pred = model.predict_proba(X_test)[:, 1]
              #auc = roc_auc_score(y_test, y_pred)
              results_row = {'Model': name}
              #new_row = {'Model': name, 'AUC': auc}
              for metric_name in selected_metrics:
                # Compute selected metric
                if task == 'classification':
                  score = metrics[metric_name](y_test, y_preds if metric_name == 'AUC' else y_preds.round())
                elif task == 'regression':
                  score = metrics[metric_name](y_test, y_preds)

                # Add score to results row
                results_row[metric_name] = score

              # Add results row to results DataFrame
              #results = results.append(results_row, ignore_index=True)

              results = pd.concat([results, pd.DataFrame(results_row, index=[0])],
                                        ignore_index=True)
              #results = results.append({'Model': name, 'AUC': auc}, ignore_index=True)
              progress_bar.value +=1

        # Close the progress bar
        progress_bar.close()

        # save training results to Main
        self.results = results

        # Display results
        with output:
          output.clear_output()
          message = HTML(f"<h3>The selected models are:</h3>{', '.join(selected_models)}")
          #table = pd.DataFrame(results).set_index('Model')
          #table = HTML(results.to_html())
          table = results.style.hide(axis="index")
          display(message, table)

    # Create checkboxes and configuration buttons for models
    checkboxes = {}
    config_buttons = {}
    for name in models.keys():
        checkbox = widgets.Checkbox(description=name, value=False,
                                    layout=widgets.Layout(width='auto'))
        checkboxes[name] = checkbox
        config_button = widgets.Button(description='Configure',
                                      button_sytle = 'primary', #'info',
                                      layout=widgets.Layout(width='auto'))
        config_buttons[name] = config_button

    # Create checkboxes for metrics
    metricboxes = {}
    for name in metrics.keys():
        metricbox = widgets.Checkbox(description=name, value=False,
                                    layout=widgets.Layout(width='auto'))
        metricboxes[name] = metricbox

    # Define function to handle configuration button click
    def on_config_button_click(model_name):
        #print(model_name + ":")
        #display(*hyperparameters_inputs[model_name].values())
        config = hyperparameters_inputs[model_name]

        if model_name == 'MLP':
          config_section.children = [widgets.Label(value=model_name + ':')] + \
            [widgets.Label('Hidden layers: comma-separated sizes, e.g. 100,50 for two layers')] + \
            list(config.values())
        else:
          config_section.children = [widgets.Label(value=model_name + ':')] +\
                                  list(config.values())

    # Attach button click handler
    for name, button in config_buttons.items():
        button.on_click(lambda b, model_name=name: on_config_button_click(model_name))
    #hyperparameter_button.on_click(on_hyperparameter_button_click)

    # Create hyperparameters input fields
    hyperparameters_inputs = {}
    for name in models.keys():
        # Define hyperparameter fields for each model
        if name == 'Logistic Regression':
            hyperparameters_fields = {
                'C':            widgets.FloatText(description='C', value=1.0),
                'penalty':      widgets.Dropdown(description='Penalty',
                                    options=[None, 'l1', 'l2'], value='l2'),
                'solver':       widgets.Dropdown(description='Solver',
                                    options=['lbfgs', 'liblinear', 'saga'], value='lbfgs'),
                'class_weight': widgets.Dropdown(description='class_weight',
                                    options=[None, 'balanced'], value='balanced'),
                'max_iter':     widgets.IntText(description='max_iter', value=1000),
            }
        elif name == 'Decision Tree':
            hyperparameters_fields = {
                'max_depth':        widgets.IntText(description='max_depth', value=5),
                'min_samples_split':widgets.IntText(description='min_samples_split', value=20),
            }
        elif name == 'Linear Regression':
            hyperparameters_fields = {
                'fit_intercept': widgets.Checkbox(description='fit_intercept', value=True),
            }
        elif name == 'Random Forest':
            hyperparameters_fields = {
                'n_estimators': widgets.IntText(description='n_estimators', value=100),
                'max_depth': widgets.IntText(description='max_depth', value=5)
            }
        elif name == 'GBM':
            hyperparameters_fields = {
              'n_estimators': widgets.IntText(description='n_estimators', value=100),
              'max_depth': widgets.IntText(description='max_depth', value=5),
              'learning_rate': widgets.FloatText(description='learning_rate', value=0.1)
            }
        elif name == 'LightGBM':
            hyperparameters_fields = {
              'n_estimators':  widgets.IntText(description='n_estimators', value=100),
              'num_leaves':    widgets.IntText(description='num_leaves', value=31),
              'max_depth':     widgets.IntText(description='max_depth', value=-1),
              'learning_rate': widgets.FloatText(description='learning_rate', value=0.1),
              'reg_alpha':     widgets.FloatText(description='reg_alpha', value=0.0),
              'reg_lambda':    widgets.FloatText(description='reg_lambda', value=0.0),
            }
        elif name == 'XGBoost':
            hyperparameters_fields = {
              'n_estimators':     widgets.IntText(description='n_estimators', value=100),
              'max_depth':        widgets.IntText(description='max_depth', value=6),
              'learning_rate':    widgets.FloatText(description='learning_rate', value=0.1),
              'subsample':        widgets.FloatText(description='subsample', value=0.8),
              'colsample_bytree': widgets.FloatText(description='colsample_bytree', value=0.8),
              'reg_alpha':        widgets.FloatText(description='reg_alpha', value=0.0),
              'reg_lambda':       widgets.FloatText(description='reg_lambda', value=1.0),
            }
        elif name == 'MLP':
            hyperparameters_fields = {
              'hidden_layer_sizes': widgets.Text(value='100,', description='Hidden layers:',
                                                 continuous_update=False),
              'activation':    widgets.Dropdown(options=['relu', 'tanh', 'logistic', 'identity'],
                                                value='relu', description='Activation:'),
              'solver':        widgets.Dropdown(options=['adam', 'sgd', 'lbfgs'],
                                                value='adam', description='Solver:'),
              'learning_rate': widgets.Dropdown(options=['constant', 'invscaling', 'adaptive'],
                                                value='constant', description='LR schedule:'),
              'alpha':         widgets.FloatText(description='alpha (L2)', value=0.0001),
            }
        else:
            hyperparameters_fields = {}

        hyperparameters_inputs[name] = hyperparameters_fields

    # Create confirmation button
    confirmation_button = widgets.Button(description='Train Model(s)', \
                                        button_style = 'success')

    # Define function to handle button click
    def on_confirmation_button_click(button):
        selected_models = [name for name, checkbox in checkboxes.items() if checkbox.value]
        selected_metrics = [name for name, checkbox in metricboxes.items() if checkbox.value]

        output.clear_output()
        train_models(selected_models, selected_metrics, hyperparameters_inputs)

    # Attach button click handler
    confirmation_button.on_click(on_confirmation_button_click)


    # Display checkboxes and hyperparameter confirmation button
    message_model = HTML(f"<h3>Select model(s):")
    message_metric = HTML(f"<h3>Select metric(s):")
    display(message_model)

    # Display checkboxes and button
    # Create layout for checkboxes and configuration buttons
    checkboxes_model = widgets.VBox([checkboxes[name] for name in models.keys()])
    checkboxes_metric = widgets.VBox([metricboxes[name] for name in metrics.keys()])
    config_buttons_box = widgets.VBox([config_buttons[name] for name in models.keys()])
    confirmation_box = widgets.VBox([confirmation_button])

    # Create layout for hyperparameter configuration section
    config_section = widgets.VBox([])

    boxes_layout = widgets.HBox([checkboxes_model, config_buttons_box, config_section])
    layout = widgets.VBox([checkboxes_metric,
                          confirmation_box])

    display(boxes_layout)
    display(message_metric)
    display(layout)
    display(output)

  def model_register(self):
    models_selected = self.models
    results = self.results

    checkboxes_register = [widgets.Checkbox(value=False, description=model_name) for model_name in models_selected.keys()]

    register_button = widgets.Button(
        description='Register model(s)',\
        button_style = 'success')
    # Change button color and font weight
    #register_button.style.button_color = 'green'
    register_button.style.font_weight = 'bold'

    output = widgets.Output()
    def on_register_clicked(b):
      output.clear_output()
      #self.register = dict()  # clear the list of registered models in main
      models_registered = dict()
      for checkbox in checkboxes_register:
        model_name = checkbox.description
        if checkbox.value:
          models_registered[model_name] = models_selected[model_name]
          #print(f"Model '{model_name}' has been successfully registered.")

      # select the rows corresponding to the registered models
      results_registered = results[results.iloc[:, 0].isin(models_registered.keys())]
      # save registered models to main
      self.register = models_registered

      with output:
        message = HTML(f"<h3>The registered models are:</h3>{', '.join(models_registered.keys())}")
        table = results_registered.style.hide(axis="index")
        display(message, table)

    register_button.on_click(on_register_clicked)

    # Display the widgets
    message1 = HTML(f"<h3>Register models:")
    display(message1)
    display(widgets.VBox(checkboxes_register + [register_button]))
    display(output)

  def model_explain(self):
    df = self.data
    test_ratio = self.test_ratio
    random_state = self.random
    #task = self.task
    target_var = self.target

    X = df.drop(target_var, axis=1)
    y = df[target_var]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_ratio,
                                                        random_state=random_state)

    feature_names = np.array(X_train.columns) # get feature names

    models = self.register  # load the dictionary of registered model(s)

    if not models:
      print("There is no model registered.")
      return


    # define a main output
    output_main = widgets.Output()

    # Widget to select model
    model_select = widgets.Dropdown(options= list(models.keys()),
                                    description='Registered models:',
                                    )

    # define an output to take messages from shap values (but not display)
    output = widgets.Output()

    model = models[model_select.value]
    explainer = shap.Explainer(model, X_train)

    #with output: # not to show the percentage progress from shap value
    #shap_values = explainer.shap_values(X_test, check_additivity = True)
    with output:
      shap_values = explainer(X_test, check_additivity=True)


    def on_change_global(change):
      if change['name'] == 'value' and (change['new'] != change['old']):
          #clear_output(wait=True)
          #display(model_select)
          #with global_tab:
          global_tab.clear_output()
          output_local.clear_output()
          pdp_tab.clear_output()

          model = models[change['new']]

          explainer = shap.Explainer(model, X_train)

          with output: # not show the percentage progress from shap value
            shap_values = explainer(X_test, check_additivity = True)

          progress_bar = widgets.IntProgress(
              value=1,
              min=1,
              max=10,
              step=1,
              description='Calculating',
              bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
              #orientation='horizontal'
              )

          with output_main:
            display(progress_bar)

          with global_tab:
            plot_feature_importance(change['new'], X_test, y_test)

          # Update the progress bar
          progress_bar.value +=1

          with global_tab:
            plot_shap_global(shap_values)

          # Update the progress bar
          progress_bar.value +=1

          with local_tab:
            output_local.clear_output()
            with output_local:
              plot_shap_local(instance_selector.value)
              plot_lime(instance_selector.value)

          # Update the progress bar
          progress_bar.value +=1

          with pdp_tab:
            output_pdp.clear_output()
            with output_pdp:
              output_pdp.clear_output()
              widgets.HBox([output1, output2])

          # Update the progress bar
          progress_bar.value +=1

          # Close the progress bar
          progress_bar.close()

    model_select.observe(on_change_global)

    display(HTML(f"<h3>Select a model:"))
    display(model_select)
    display(output_main)
    #display(progress_bar)

    # create output areas for each explainability section
    global_tab = widgets.Output()  # global feature importance
    local_tab  = widgets.Output()  # local feature importance
    pdp_tab    = widgets.Output()  # partial dependence plot

    S = self._section
    display(HTML(S('🌐', 'Global Explainability',     '#1a73e8', '#e8f0fe')))
    display(global_tab)
    display(HTML(S('🔍', 'Local Explainability',      '#7b1fa2', '#f3e5f5')))
    display(local_tab)
    display(HTML(S('📉', 'Partial Dependence Plot',   '#34a853', '#e6f4ea')))
    display(pdp_tab)

    ############ Global Explainability Tab ############

    def plot_feature_importance(model_name, X, y):
      model = models[model_name]

      result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
      sorted_idx = result.importances_mean.argsort()

      #with global_tab:
        # Create boxplot
      fig, ax = plt.subplots()
      ax.boxplot(result.importances[sorted_idx].T,
                vert=False, widths=0.7)

      # Set y-tick labels and positions manually
      ax.set_yticks(np.arange(len(sorted_idx)) + 1)
      ax.set_yticklabels(X.columns[sorted_idx])

      ax.set_xlabel("Decrease in accuracy score")
      plt.title("Permutation Feature Importance")

      plt.show()

      ''' use eli5 feature importance
      # We calculate the permutation importance using eli5 library
      perm = PermutationImportance(model, random_state=1).fit(X, y)
      importance_weights = perm.feature_importances_
      importance_std = perm.feature_importances_std_

      # Sorting the features based on importance
      indices = np.argsort(importance_weights)[::-1]

      # Rearranging feature names so they match the sorted feature importances
      names = [X.columns[i] for i in indices]

      # Creating plot
      plt.figure()
      # Create plot title
      plt.title("Permutation Feature Importance")
      # Add bars
      plt.bar(range(X_test.shape[1]), importance_weights[indices])
      # Add error bars
      plt.errorbar(range(X_test.shape[1]), importance_weights[indices], yerr=importance_std[indices], fmt='o', color='r')
      # Add feature names as x-axis labels
      plt.xticks(range(X_test.shape[1]), names, rotation=90)

      # Show plot
      with global_tab:
        plt.show()
      '''

    def plot_shap_global(shap_values):
      #model = models[model_name]

      #explainer = shap.Explainer(model, X_train)
      #with output: # not to show the percentage progress from shap value
      #  shap_values = explainer.shap_values(X, check_additivity = True)

      with global_tab:

        if model_select.value in ('Random Forest', 'GBM', 'LightGBM', 'XGBoost', 'Decision Tree'):
          '''
          Random Forest models treat binary classification as a multi-class problem with two classes.
          When you compute SHAP values for Random Forests on binary classification tasks,
          you get a three-dimensional array with the shape (num_samples, num_features, num_classes),
          with the last dimension denoting the SHAP values for each class.

          If your dataset has imbalanced classes,
          you might want to compute a weighted average where the weights are the proportions of each class.
          '''
          # Assuming y_train is your training set labels
          class_proportions = y_train.value_counts(normalize=True).sort_index().values

          # Weighted average SHAP values based on class proportions
          # Note that shap_values.values gives the actual SHAP values from the Explanation object
          weighted_shap = np.abs(shap_values.values).mean(axis=0).sum(axis=1)

          # Plotting the global feature importance
          features = X_test.columns
          sorted_idx = np.argsort(weighted_shap)[::-1]
          plt.barh(features[sorted_idx], weighted_shap[sorted_idx])
          plt.xlabel("Weighted SHAP Value")
          plt.title("SHAP (Average) Feature Importance")
          plt.show()

        else:
          '''
          For Gradient Boosting Machines and other models (like logistic regression),
          when you compute SHAP values for binary classification tasks,
          the output is usually a matrix with the shape (num_samples, num_features).
          This is because the SHAP values are typically calculated with respect to the log-odds space,
          which is a single continuous output.
          In effect, you're only seeing the SHAP values related to the positive class since it's the only class that's explicitly modeled.
          '''
          fig, ax = plt.subplots(figsize=(3,2))
          # Generate a SHAP summary plot
          shap.summary_plot(shap_values, X_test, plot_size=0.15,
                            plot_type = 'bar', # disabling this will show a scattered heated plot
                            show = False)
          #'show = False' is necessary to ensure that the plot does not get shown immediately, allowing you to add a title
          ax.set_title("SHAP (Average) Feature Importance")
          plt.show()

    # Initial displays
    with global_tab:
      plot_feature_importance(model_select.value, X_test, y_test)
      plot_shap_global(shap_values)


    ############# Local Explainability Tab ###############

    # Create a widget for instance selection
    instance_selector = widgets.IntText(
        value=0,
        description='Instance:',
        disabled=False
    )

    def plot_shap_local(instance_index):
      # Clear previous output
      #display.clear_output(wait=True)
      # Create a Tree Explainer object
      #model = models[model_name]

      # Create a SHAP explainer
      #explainer = shap.Explainer(model, X_train)

      # Calculate SHAP values
      #shap_values = explainer(X_test)

      # Plot the SHAP values for the selected instance in the test set
      fig, ax = plt.subplots(figsize=(3,2))

      if model_select.value == 'Random Forest' or 'logistic Regression':
        # Predict the class for the instance:
        predicted_class = model.predict(X_test.iloc[instance_index].values.reshape(1, -1))[0]
        # Get the SHAP values for that class:
        class_specific_shap_values = shap_values[instance_index, :, predicted_class]
        # Get the expected value (base value) for the predicted class
        expected_value = explainer.expected_value[predicted_class]
        # Create an explanation object
        explanation = shap.Explanation(values=class_specific_shap_values,
                                      base_values=expected_value,
                                      data=X_test.iloc[instance_index])
        # Plot the waterfall chart
        shap.plots.waterfall(explanation, show=False)
        ax.set_title("Shapley Values for Instance " + str(instance_index))
        plt.show()

      else:
        shap.plots.waterfall(shap_values[instance_index], show = False)
        ax.set_title("Shapley Values for Instance " + str(instance_index))
        plt.show()

    def plot_lime(instance_index):
      feature_names = X.columns.tolist()  # Get the feature names
      #class_names = ['0', '1'] #['default', 'non-default']  # You may need to update this based on your actual classes
      # Create a Lime explainer object
      explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                        feature_names=feature_names,
                                                        #class_names=class_names,
                                                        discretize_continuous=True)
      # Extract the instance in the test set for which we want to explain the model's prediction
      X_instance = X_test.values[instance_index]
      #X_instance = pd.DataFrame(X_instance.reshape(1, -1), columns=feature_names)

      with output: # not showing the warning
        # Explain a prediction (for example, the first instance from the test set)
        exp = explainer.explain_instance(X_instance, model.predict_proba, num_features=len(feature_names), top_labels=3)

        # 1. Predict the class for the instance
        predicted_class = model.predict(X_instance.reshape(1, -1))[0]

      # 2. Get the class index
      predicted_class_index = list(model.classes_).index(predicted_class)
      # Plot the explanation (there are two methods)
      #fig = exp.show_in_notebook(show_table=True, show_all=False)
      fig = exp.as_pyplot_figure(label=predicted_class_index)
      plt.title("LIME for Instance {} (Predicted Class = {})".format(instance_index, predicted_class_index))
      plt.show()


    with local_tab:
      display(HTML(f"<h3>Select an instance:"))
      display(instance_selector)

      # define output for the local explainablity session
      output_local = widgets.Output()

      with output_local:
        plot_shap_local(instance_selector.value)
        plot_lime(instance_selector.value)

      display(output_local)

    def on_change_local(change):
      if change['name'] == 'value' and (change['new'] != change['old']):
        instance_index = change['new']

        output_local.clear_output()
        with local_tab:
          output_local.clear_output()
          with output_local:
            plot_shap_local(instance_index)
            plot_lime(instance_index)

          #display(output_local)

    instance_selector.observe(on_change_local)


    ############ PDP Tab ############
    # Function to plot PDP for selected single feature
    def plot_single_feature_pdp(feature_name):
        feature_idx = list(feature_names).index(feature_name)
        _ = PartialDependenceDisplay.from_estimator(
            model, X_test, features=[feature_idx], feature_names=feature_names)
        plt.title("PDP for {}".format(feature_name))
        plt.show()

    # Function to plot PDP for selected pair of features
    def plot_pair_features_pdp(feature_name1, feature_name2):
        feature_idx1 = list(feature_names).index(feature_name1)
        feature_idx2 = list(feature_names).index(feature_name2)
        _ = PartialDependenceDisplay.from_estimator(
            model, X_test, features=[[feature_idx1, feature_idx2]], feature_names=feature_names)
        plt.title("Two-Way PDP: {} vs {}".format(feature_name1, feature_name2))
        plt.show()

    # Create dropdown menu for single feature selection
    single_feature_dropdown = widgets.Dropdown(
        options=feature_names,
        value=feature_names[0],
        description='Select Feature:'
    )

    # Create two dropdown menus for pair feature selection
    pair_feature_dropdown1 = widgets.Dropdown(
        options=feature_names,
        value=feature_names[0],
        description='Feature 1:'
    )

    pair_feature_dropdown2 = widgets.Dropdown(
        options=feature_names,
        value=feature_names[1],
        description='Feature 2:'
    )

    # Create the two interactive plots
    single_feature_plot = widgets.interactive(plot_single_feature_pdp, feature_name=single_feature_dropdown)
    pair_feature_plot = widgets.interactive(plot_pair_features_pdp, feature_name1=pair_feature_dropdown1, feature_name2=pair_feature_dropdown2)

    with pdp_tab:
      message1 = widgets.HTML(f"<h3>One-Way PDP</h3>")
      message2 = widgets.HTML(f"<h3>Two-Way PDP")
      output1 = widgets.VBox([message1, single_feature_plot])
      output2 = widgets.VBox([message2, pair_feature_plot])
      # Display the two plots side by side
      #widgets.HBox([single_feature_plot, pair_feature_plot])

      output_pdp = widgets.Output()

      with output_pdp:
        output_pdp.clear_output()

        display(widgets.HBox([output1, output2]))

      display(output_pdp)



  def model_diagnose(self):
    df = self.data
    test_ratio = self.test_ratio
    random_state = self.random
    task = self.task
    target_var = self.target

    X = df.drop(target_var, axis=1)
    y = df[target_var]

    # Split dataset into training (60%), calibration(20%), and test sets(20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    feature_names = np.array(X_train.columns) # get feature names

    models = self.register  # load the dictionary of registered model(s)

    if not models:
      print("There is no model registered.")
      return

    # define a main output
    output_main = widgets.Output()

    # Widget to select model
    model_select = widgets.Dropdown(options= list(models.keys()),
                                    description='Registered models:',
                                    )

    # selected model from registration
    model = models[model_select.value]

    # identify categorical columns of the dataset
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns

    # apply OneHotEndcoder when categortical feasures exist (need to improve this part later)
    if len(categorical_cols) > 0:
      preprocessor = ColumnTransformer(transformers=[
      ('cat', OneHotEncoder(), categorical_cols)  # Assume `categorical_columns` is defined
      ])

      # model pipeline
      model = Pipeline(steps=[
          ('preprocessor', preprocessor),
          ('model', model)
      ])


    # Re-fit the model with the new training data and testing data
    model.fit(X_train, y_train)


    # Measure strangeness on the calibration set
    if task == 'classification':
      cal_probas = model.predict_proba(X_cal)
      cal_strangeness = 1 - cal_probas[np.arange(len(y_cal)), y_cal]

      # Predict on the test set and measure strangeness
      test_probas = model.predict_proba(X_test)

      '''
      By selecting only the probabilities for class 1 ([:, 1]),
      we are specifically looking at how well our model is calibrated for detecting the positive class (e.g.,"default").
      This is often the class of greater interest in such medical scenarios,
      so we might only plot the reliability diagram for this class to see how well our probabilities align with the actual outcomes.
      '''
      test_probs_class1 = model.predict_proba(X_test)[:, 1]
      test_strangeness = 1 - test_probas.max(axis=1)

    if task == 'regression':
      # Get predicted values and residuals for the calibration set
      cal_pred = model.predict(X_cal)
      cal_residuals = np.abs(cal_pred - y_cal)

      # Get predicted values and residuals for the test set
      test_pred = model.predict(X_test)
      test_residuals = np.abs(test_pred - y_test)

    # diplay the widgets initially
    with output_main:
      display(HTML(f"<h3>Select a model:"))
      display(model_select)

    display(output_main)

    # create tabs
    reliability_tab = widgets.Output() # reliability
    accuracy_tab = widgets.Output()  # accuacy
    overfit_tab = widgets.Output()    # overfit/underfit
    weakspot_tab = widgets.Output()    # weakness spot
    resiliency_tab = widgets.Output()  # resiliency
    robust_tab = widgets.Output()      #robustness
    fairness_tab = widgets.Output()    # fairness
    S = self._section
    display(HTML(S('🎯', 'Accuracy',    '#1a73e8', '#e8f0fe')))
    display(accuracy_tab)
    display(HTML(S('📡', 'Reliability', '#7b1fa2', '#f3e5f5')))
    display(reliability_tab)
    display(HTML(S('⚖️', 'Overfit',     '#e65100', '#fff3e0')))
    display(overfit_tab)
    display(HTML(S('🔦', 'Weak Spot',   '#c62828', '#ffebee')))
    display(weakspot_tab)
    display(HTML(S('🛡️', 'Resiliency',  '#00796b', '#e0f2f1')))
    display(resiliency_tab)
    display(HTML(S('💪', 'Robustness',  '#4527a0', '#ede7f6')))
    display(robust_tab)
    display(HTML(S('⚖️', 'Fairness',    '#34a853', '#e6f4ea')))
    display(fairness_tab)

    def on_change_global(change):
      if change['name'] == 'value' and (change['new'] != change['old']):
        #clear_output(wait=True)
        #display(model_select)
        #with global_tab:
        #output_local.clear_output()
        #pdp_tab.clear_output()
        reliability_output.clear_output()

        progress_bar = widgets.IntProgress(
            value=1,
            min=1,
            max=10,
            step=1,
            description='Calculating',
            bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            #orientation='horizontal'
            )

        with output_main:
          display(progress_bar)

        # Get the selected model and confidence level
        model = models[change['new']]
        #model = models[model_select.value]

        # Re-fit the model with the new training data and testing data
        model.fit(X_train, y_train)

        # prepare the datasets for testing and calibration
        if task == 'classification':
          cal_probas = model.predict_proba(X_cal)
          cal_strangeness = 1 - cal_probas[np.arange(len(y_cal)), y_cal]

          # Predict on the test set and measure strangeness
          test_probas = model.predict_proba(X_test)

          '''
          By selecting only the probabilities for class 1 ([:, 1]),
          we are specifically looking at how well our model is calibrated for detecting the positive class (e.g.,"default").
          This is often the class of greater interest in such medical scenarios,
          so we might only plot the reliability diagram for this class to see how well our probabilities align with the actual outcomes.
          '''
          test_probs_class1 = model.predict_proba(X_test)[:, 1]
          test_strangeness = 1 - test_probas.max(axis=1)

        if task == 'regression':
          # Get predicted values and residuals for the calibration set
          cal_pred = model.predict(X_cal)
          cal_residuals = np.abs(cal_pred - y_cal)

          # Get predicted values and residuals for the test set
          test_pred = model.predict(X_test)
          test_residuals = np.abs(test_pred - y_test)

        # Update the progress bar
        progress_bar.value +=1

        with reliability_tab:
          reliability_output.clear_output()

          with reliability_output:
           reliability_plot()

          display(reliability_output)

        # Update the progress bar
        progress_bar.value +=1

        # Close the progress bar
        progress_bar.close()

    # Attach the update function to the widgets
    model_select.observe(on_change_global, names='value')
    #alpha_slider.observe(on_change_global, names='value')

    # define metrics
    metric_functions = {
        'ACC': accuracy_score,
        'AUC': roc_auc_score,
        'F1': f1_score,
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'R2': r2_score
    }

    def evaluate_score(y_true, y_pred, metric, threshold=0.5):
      # Convert probabilities to binary class labels based on the threshold
      if metric not in metric_functions:
          raise ValueError(f"Unsupported metric: {metric}")

      if task == 'classification' and metric == 'AUC':
          if len(np.unique(y_true)) == 2:
              return metric_functions[metric](y_true, y_pred) # AUC uses probabilities, not labels
          else:
              # Return a default score or np.nan if not both classes are present
              return 0.5  # a random guess

      if task == 'classification' and metric in ['ACC', 'F1']:
        y_pred_labels = (y_pred >= threshold).astype(int)

        if metric == 'F1':
              # Handle the case where there are no positive predictions or labels
              return metric_functions[metric](y_true, y_pred_labels, zero_division=0)
        else:
              return metric_functions[metric](y_true, y_pred_labels)
      else:
          return metric_functions[metric](y_true, y_pred)  # regression metric

################# Resiliency #####################

    # Dictionary to store the metric functions
    if task == 'classification':
      y_pred = model.predict_proba(X_test)[:, 1] # predicted probabilities
    elif task == 'regression':
      y_pred = model.predict(X_test) # predicted values

    # Calculate the PSI for a single variable
    def calculate_psi(expected_array, actual_array, buckettype='bins', buckets=10, axis=0):
      def scale_range (input, min, max):
          input += -(np.min(input))
          input /= np.max(input) / (max - min)
          input += min
          return input

      breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

      if buckettype == 'bins':
          breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
      elif buckettype == 'quantiles':
          breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

      expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
      actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

      def sub_psi(e_perc, a_perc):
          ''' Calculate the actual PSI value from comparing the values.
              Update the actual value to a very small number if equal to zero
          '''
          if a_perc == 0:
              a_perc = 0.0001
          if e_perc == 0:
              e_perc = 0.0001

          return (e_perc - a_perc) * np.log(e_perc / a_perc)

      psi_value = np.sum(np.fromiter((sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))), dtype=float))
      #psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

      return psi_value

    '''
    # define metrics
    metric_functions = {
        'ACC': accuracy_score,
        'AUC': roc_auc_score,
        'F1': f1_score,
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'R2': r2_score
    }

    def evaluate_score(y_true, y_pred, metric, threshold=0.5):
        # Convert probabilities to binary class labels based on the threshold
        if metric not in metric_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        if task == 'classification' and metric == 'AUC':
            if len(np.unique(y_true)) == 2:
                return metric_functions[metric](y_true, y_pred) # AUC uses probabilities, not labels
            else:
                # Return a default score or np.nan if not both classes are present
                return 0.5  # a random guess

        if task == 'classification' and metric in ['ACC', 'F1']:
          y_pred_labels = (y_pred >= threshold).astype(int)

          if metric == 'F1':
                # Handle the case where there are no positive predictions or labels
                return metric_functions[metric](y_true, y_pred_labels, zero_division=0)
          else:
                return metric_functions[metric](y_true, y_pred_labels)
        else:
            return metric_functions[metric](y_true, y_pred)  # regression metric
      '''

    def get_worst_cluster(n_clusters = 10, metric = 'AUC'):
        if metric not in metric_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)  # Keep the feature names

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_test_scaled)
        # Predict clusters for the test set
        clusters = kmeans.predict(X_test_scaled)
        # Evaluate the performance of the model on each cluster
        cluster_scores = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            score = evaluate_score(y_test.iloc[cluster_indices], y_pred[cluster_indices], metric)
            cluster_scores.append((cluster, score))

        # Sort clusters by score and select the worst performing cluster
        worst_cluster, worst_score = sorted(cluster_scores, key=lambda x: x[1])[0]
        #worst_cluster = sorted(cluster_scores, key=lambda x: x[1])[0][0]

        # Indices of the worst performing cluster
        worst_cluster_indices = np.where(clusters == worst_cluster)[0]

        return worst_cluster_indices, worst_score

    # Define the function for the worst cluster analysis
    def plot_worst_cluster(metric='AUC', max_k=10):
        if metric not in metric_functions:
          raise ValueError(f"Unsupported metric: {metric}")

        worst_scores = []
        k_values = list(range(2, max_k + 1))

        # Perform KMeans clustering and identify the worst cluster for each K
        for k in k_values:
            _, worst_score = get_worst_cluster(k, metric)
            worst_scores.append(worst_score)

        # Plot the worst cluster performance
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, worst_scores, marker='o')
        plt.axhline(evaluate_score(y_test, y_pred, metric), color='red', linestyle='--', label='Overall Test Performance')
        plt.xticks(k_values)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel(metric)
        plt.title(f'Worst Cluster Performance by Number of Clusters ({metric})')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_worst_samples(alpha, immu_feature=None):
        y_pred_copy = pd.Series(y_pred, index=X_test.index)
        residuals = np.abs(y_pred_copy - y_test)

        worst_indices = []

        if immu_feature is not None and immu_feature in X_test.columns:
            X_test_copy = X_test.copy()
            X_test_copy['bin'], bins = pd.qcut(X_test_copy[immu_feature], q=10, retbins=True, duplicates='drop')

            for bin_label in X_test_copy['bin'].unique():
                bin_indices = X_test_copy[X_test_copy['bin'] == bin_label].index

                # Calculate residuals for the current bin
                residuals_bin = residuals.loc[bin_indices]

                # Sort and select worst samples within the bin
                sorted_indices_bin = np.argsort(-residuals_bin)[:int(alpha * len(bin_indices))]
                worst_indices.extend(bin_indices[sorted_indices_bin])
        else:
            # For the case without an immutable feature
            sorted_indices = residuals.index[np.argsort(-residuals)[:int(alpha * len(y_test))]]
            worst_indices.extend(sorted_indices)
            #print(worst_indices)

        return worst_indices


    # Define the function for resilience analysis
    def resilience_score(metric='AUC', alpha=0.1, immu_feature=None):
        # Check if the metric is supported
        if metric not in metric_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        worst_indices = get_worst_samples(alpha, immu_feature)
        y_test_worst = y_test.loc[worst_indices]
        #y_pred_worst = y_pred[worst_indices]

        #y_test_worst = y_test[worst_indices]
        y_pred_copy = pd.Series(y_pred, index=X_test.index)
        y_pred_worst = y_pred_copy.loc[worst_indices]

        score = evaluate_score(y_test_worst, y_pred_worst, metric)

        return score

    def plot_resilience(metric='AUC', immu_feature=None):
        performance_scores = []
        alpha_values = np.arange(0.1, 1.1, 0.1)  # from 0.1 to 1.0

        # Calculate performance for varying worst sample ratios
        for alpha in alpha_values:
          worest_score = resilience_score(metric=metric, alpha=alpha, immu_feature = immu_feature)
          performance_scores.append(worest_score)

        # Plotting the performance scores
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, performance_scores, label='Worst Sample Performance')

        # Plot the performance on the entire test set
        overall_score = evaluate_score(y_test, y_pred, metric)

        plt.axhline(overall_score, color='red', linestyle='--', label='Overall Test Performance')

        plt.xlabel('Worst Sample Ratio')
        plt.ylabel(metric)
        plt.title(f'Model Performance under Worst Sample Scenario ({metric})')
        plt.legend()
        plt.show()

    def plot_resilience_distance(method='worst-sample', distance_metric='PSI', alpha=0.1, top_k = 10, immu_feature=None):
      # top_k: top k features to display
        if method == 'worst-sample':
            worst_indices = get_worst_samples(alpha, immu_feature)
            X_worst = X_test.loc[worst_indices]
        elif method == 'worst-cluster':
            worst_indices,_ = get_worst_cluster(n_clusters = 10, metric = 'AUC')
            X_worst = X_test.iloc[worst_indices]

        feature_distances = {}

        for feature in X_test.columns:
            expected = X_test[feature]
            actual = X_worst[feature]

            if distance_metric == 'PSI':
                distance = calculate_psi(expected, actual)
            elif distance_metric == 'WD1':
                distance = wasserstein_distance(expected, actual)
            elif distance_metric == 'KS':
                distance = ks_2samp(expected, actual)[0]  # Just the statistic, not the p-value
            else:
                raise ValueError("Unsupported distance_metric")

            feature_distances[feature] = distance

        # Sort features by distance and select top_k
        sorted_features = sorted(feature_distances.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Display the top features with their distances
        #print(f"Top {top_k} features with largest {distance_metric} distance under {method} method:")
        #for feature, distance in sorted_features:
        #   print(f"{feature}: {distance}")

        # Plot the top features with their distances
        features, distances = zip(*sorted_features)
        plt.figure(figsize=(10, 6))
        plt.barh(features, distances, color='skyblue')
        plt.xlabel('Distance Measure')
        plt.ylabel('Features')
        plt.title(f'Top {top_k} Features by {distance_metric} Distance')
        plt.gca().invert_yaxis()  # Invert y-axis to have the largest bar on top
        plt.show()


    # Define Widgets
    method_widget = widgets.Dropdown(
        options=['worst-sample',
                'worst-cluster'],  # Add more methods as they are implemented
        value='worst-sample',
        description='Method:',
        disabled=False,
    )

    metric_classification_widget = widgets.Dropdown(
        options=['ACC', 'AUC', 'F1'],
        value='ACC',
        description='Metric:',
        disabled=False,
    )

    metric_regression_widget = widgets.Dropdown(
        options=['MSE', 'MAE', 'R2'],
        value='MSE',
        description='Metric:',
        disabled=False,
    )

    worst_ratio_widget = widgets.FloatSlider(
        value=0.1,
        min=0,
        max=1.0,
        step=0.1,
        description='Worst Ratio:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    # Define the dropdown widget for the distance measure
    distance_measure_widget = widgets.Dropdown(
        options=['PSI', 'WD1', 'KS'],
        value='PSI',
        description='Distance Measure:',
        disabled=False,
    )

    # Container widget to dynamically display the correct metric options
    metric_widget = widgets.VBox([metric_classification_widget])

    # Create a dropdown for selecting the immutable feature
    immu_feature_widget = widgets.Dropdown(
        options=[None] + list(X.columns),
        value=None,
        description='Immutable Feature:',
        disabled=False,
    )

    # Function to update metric options based on task
    def update_metric_options():
        if task == 'classification':
            metric_widget.children = [metric_classification_widget]
        elif task == 'regression':
            metric_widget.children = [metric_regression_widget]

    # Button to run the analysis
    run_button = widgets.Button(description="Run Analysis",
                                button_style='success')

    # Function to handle button click event
    output_resilient = widgets.Output() # output for the resilient plots

    def on_run_button_clicked(b):
        output_resilient.clear_output()

        method = method_widget.value
        metric = metric_widget.children[0].value
        worst_ratio = worst_ratio_widget.value
        immu_feature = immu_feature_widget.value
        distance_metric = distance_measure_widget.value
        # Call your resilience_analysis function here with the selected options
        with output_resilient:
          if method == 'worst-sample':
            plot_resilience(metric, immu_feature)
            plot_resilience_distance(method, distance_metric, worst_ratio, top_k = 10, immu_feature=immu_feature)
          if method == 'worst-cluster':
            plot_worst_cluster(metric)
            plot_resilience_distance(method, distance_metric, worst_ratio, top_k = 10, immu_feature=immu_feature)

    run_button.on_click(on_run_button_clicked)

    # Display widgets
    widget_inputs_1 = widgets.HBox([method_widget, metric_widget, immu_feature_widget])
    widget_inputs_2 = widgets.HBox([distance_measure_widget, worst_ratio_widget])

    with resiliency_tab:
      display(widgets.VBox([widget_inputs_1,
                          widget_inputs_2,
                          run_button]))

      display(output_resilient)


################# Reliability #####################
    # Create widgets
    alpha_slider = widgets.FloatSlider(value=0.95, min=0, max=1.0, step=0.01, description='alpha:')
    plus_button = widgets.Button(description='+',
                                button_style='primary',
                                layout=widgets.Layout(width='25px'))

    minus_button = widgets.Button(description='-',
                                  button_style='primary',
                                  layout=widgets.Layout(width='25px'))

    ui = widgets.HBox([alpha_slider, minus_button, plus_button])

    # Update function for the buttons
    def increment(b):
        alpha_slider.value = min(alpha_slider.value + 0.01, 1.0)

    def decrement(b):
        alpha_slider.value = max(alpha_slider.value - 0.01, 0.0)

    plus_button.on_click(lambda b: increment(b))
    minus_button.on_click(lambda b: decrement(b))

    def on_change_reliability(change):
      if change['name'] == 'value' and (change['new'] != change['old']):
        #clear_output(wait=True)
        #display(model_select)
        #with global_tab:
        reliability_output.clear_output()
        #output_local.clear_output()
        #pdp_tab.clear_output()

        # Get the selected model and confidence level
        #model = models[change['new']]
        #model = models[model_select.value]

        # Re-fit the model with the new training data and testing data
        #model.fit(X_train, y_train)
        # prepare the datasets for testing and calibration
        #data_prepare()

        with reliability_tab:
          with reliability_output:
           reliability_plot()
          display(reliability_output)

    # Attach the update function to the widgets
    alpha_slider.observe(on_change_reliability, names='value')

    with reliability_tab:
      # Display the widgets
      message = widgets.HTML(f"<h3>Select Confidence Level:")
      #display(message)
      display(widgets.VBox([message, ui]))

    out_widget = widgets.Output() # an output to hold the plot
    def plot_conformal_classification(alpha=0.95):
      with out_widget:
        out_widget.clear_output(wait=True)
        plt.figure(figsize=(16, 6))

        # Calculate the cutoff based on alpha and calibration strangeness
        cutoff = np.percentile(cal_strangeness, 100 * (alpha))

        '''
        # Plot calibration set strangeness scores
        plt.subplot(1, 3, 1)
        plt.hist(strangeness, bins=30, label='Calibration strangeness', alpha=0.7)
        plt.axvline(x=cutoff, color='r', linestyle='--', label=f'Cutoff at alpha={alpha}')
        plt.xlabel('Strangeness')
        plt.ylabel('Count')
        plt.legend()
        '''

        # Plot strangeness distribution for the test set
        plt.subplot(1, 2, 1)
        plt.hist(test_strangeness, bins=30, label='Test set strangeness', alpha=0.7)
        plt.axvline(x=cutoff, color='r', linestyle='--', label=f'Cutoff at alpha={alpha:.2f}')
        plt.xlabel('Strangeness')
        plt.ylabel('Count')
        plt.title('Distribution of Strangeness for Test Set')
        plt.legend()

        # Scatter plot for test set
        plt.subplot(1, 2, 2)
        conformal_idx = test_strangeness <= cutoff  # Conformal points are below or equal to the cutoff
        non_conformal_idx = ~conformal_idx
        plt.scatter(np.arange(len(test_strangeness))[conformal_idx], test_strangeness[conformal_idx], color='g', label=f'Conformal ({conformal_idx.mean() * 100:.2f}%)')
        plt.scatter(np.arange(len(test_strangeness))[non_conformal_idx], test_strangeness[non_conformal_idx], color='r', label=f'Non-Conformal ({non_conformal_idx.mean() * 100:.2f}%)')
        plt.axhline(y=cutoff, color='b', linestyle='--', label='Cutoff')
        plt.xlabel('Test instances')
        plt.ylabel('Strangeness')
        plt.title('Confomarl vs. Non-conformal Points in Test Set')
        plt.legend()

        #plt.tight_layout()
        plt.show()

    #out2_widget = widgets.Output()
    def plot_conformal2_classification():
    #with out2_widget:
      #out2_widget.clear_output(wait=True)

      # Calculate p-values for test instances
      p_values = [(cal_strangeness >= t_s).mean() for t_s in test_strangeness]

      plt.figure(figsize=(16, 6))

      plt.subplot(1, 2, 1)
      # Visualize p-values distribution
      plt.hist(p_values, bins=10, edgecolor='k', alpha=0.7, density=True)
      plt.xlabel('P-value')
      plt.ylabel('Density')
      plt.title('Distribution of P-values for Test Set')

      # Calculate the normalized instance indices for the calibration and test sets
      cal_indices_norm = np.arange(len(cal_strangeness)) / len(cal_strangeness)
      test_indices_norm = np.arange(len(test_strangeness)) / len(test_strangeness)

      plt.subplot(1, 2, 2)
      # Visualize calibration set vs. test set strangeness
      plt.scatter(cal_indices_norm, np.sort(cal_strangeness), label='Calibration Strangeness', s=50)
      plt.scatter(test_indices_norm, np.sort(test_strangeness), label='Test Strangeness', s=50, alpha=0.6)
      plt.xlabel('Normalized Instances (sorted by strangeness)')
      plt.ylabel('Strangeness')
      plt.title('Strangeness of Calibration vs. Test Set')
      plt.legend()
      plt.show()

    def plot_reliability_classification(y_true, y_probs, n_bins=10, ax=None):
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)

        if ax is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax.plot(prob_pred, prob_true, 's-', label='Model')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Reliability Diagram')
        ax.legend()
        plt.show()

    def plot_conformal_regression(alpha=0.90):
        # Calculate the cutoff based on alpha and calibration residuals
        cutoff = np.percentile(cal_residuals, 100 * (alpha))

        plt.figure(figsize=(16, 6))

        # Identify which residuals are within the cutoff
        within_cutoff = test_residuals <= cutoff
        outside_cutoff = ~within_cutoff

        # Calculate the percentage of test points that are green
        percent_within_cutoff = np.mean(within_cutoff) * 100
        percent_outside_cutoff = 100- percent_within_cutoff

        # Scatter plot for predicted values (blue dots)
        plt.scatter(np.arange(len(test_pred)), test_pred, c='b', s=30, label='Prediction')

        # Scatter plot for true test values (green dots when within interval, red dots when outside)
        plt.scatter(np.arange(len(test_pred))[within_cutoff], y_test[within_cutoff],
                    c='g', s=30, zorder=5, label=f'Within interval ({percent_within_cutoff:.2f}%)')
        plt.scatter(np.arange(len(test_pred))[outside_cutoff], y_test[outside_cutoff],
                    c='r', s=30, zorder=5, label=f'Outside interval ({percent_outside_cutoff:.2f}%)')

        # Prediction intervals (blue bars)
        plt.errorbar(np.arange(len(test_pred)), test_pred, yerr=cutoff, fmt='none', ecolor='gray', elinewidth=1)

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Conformal Prediction (alpha={alpha*100:.2f}%)')
        plt.legend()

        # Display the percentage of test points that are green on the plot
        #plt.text(0.7 * len(test_pred), min(test_pred), f'{percent_within_cutoff:.2f}% within interval',
        #         fontsize=12, ha='center')

        plt.show()

    def plot_predicted_vs_actual(y_true, y_pred):
      # Create figure and a grid of subplots
      fig, axs = plt.subplots(1, 2, figsize=(16, 8))
      """
      Plots a scatter plot comparing the true and predicted values for regression.
      """
      ax = axs[0]
      ax.scatter(y_true, y_pred, alpha=0.5)
      ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
      ax.set_xlabel('True Values')
      ax.set_ylabel('Predicted Values')
      ax.set_title('Predicted vs Actual Values')

      """
      Plots a histogram of the residuals (true value - predicted value) for regression.
      """
      residuals = y_true - y_pred
      ax=axs[1]
      ax.hist(residuals, bins=30, alpha=0.5, edgecolor='k')
      ax.axvline(x=0, color='r', linestyle='--')
      ax.set_xlabel('Residuals')
      ax.set_ylabel('Frequency')
      ax.set_title('Distribution of Residuals')

      # Show plots
      plt.tight_layout()
      plt.show()

    # combine all reliability plots
    def reliability_plot():
      if task == "classification":
        # plot conformal prediction
        out = widgets.interactive_output(plot_conformal_classification, {'alpha': alpha_slider})
        display(out)
        display(out_widget)

        # plot conformal prediction (p-value)
        plot_conformal2_classification()

        # Plot reliability diagram
        plot_reliability_classification(y_test, test_probs_class1)
      elif task == "regression":
        # plot the conformal prediction plot
        plot_conformal_regression(alpha=alpha_slider.value)

        # plot the prediction vs. actual and residual diagrams
        plot_predicted_vs_actual(y_test, test_pred)
      else:
        print("Task is missing.")

    # Display the outputs
    reliability_output = widgets.Output()
    with reliability_tab:
      with reliability_output:
        reliability_plot()
      display(reliability_output)
############## Weak Spot ######################
    # Reset the indices of X_test and y_test to align them
    X_test1 = X_test.reset_index(drop=True)
    y_test1 = pd.Series(y_test).reset_index(drop=True)

    predictions = model.predict(X_test1)

    # Create 'accuracy_bool' column to indicate correct predictions
    X_test1['accuracy_bool'] = predictions == y_test1.values

    # Fit and visualize a decision tree
    def fit_and_visualize_dt(df, predictors, show = False):
        X = df[predictors]
        y = df['accuracy_bool']
        tree = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)
        tree.fit(X, y)

        '''
        viz = dtreeviz(
            model,
            X,
            y,
            target_name='accuracy_bool',
            feature_names=predictors,
            class_names=["True", "False"]
        )
        viz.view()  # This will open the decision tree visualization
        '''
        # Visualize the Decision Tree
        if show == True:
          feature1 = predictors[0]
          feature2 = predictors[1]
          plt.figure(figsize=(15, 10))
          plot_tree(tree, filled=True, feature_names=predictors, class_names=['True', 'False'], rounded=True)
          plt.title(f"Decision Tree Based on {feature1} & {feature2}")
          plt.show()

        return tree

    def find_weak_spots_and_leaves(model, df, predictors, threshold = 0.05):
        leaf_ids = model.apply(df[predictors])
        t = model.tree_
        baseline_acc = np.mean(df['accuracy_bool'])
        #print(baseline_acc)

        segments = defaultdict(list)
        weak_spots = {}

        for idx in set(leaf_ids):
            leaf_indices = np.where(leaf_ids == idx)[0]
            segments[idx] = leaf_indices

            if len(leaf_indices) < 5:
                continue

            leaf_acc = np.mean(df['accuracy_bool'].iloc[leaf_indices])
            if baseline_acc- leaf_acc > threshold:
                weak_spots[idx] = leaf_indices

        return weak_spots, segments

    def get_leaf_accuracies(model, df, predictors):
      leaf_ids = model.apply(df[predictors])
      t = model.tree_
      leaf_accuracies = {}

      for idx in set(leaf_ids):
          leaf_indices = np.where(leaf_ids == idx)[0]
          #if len(leaf_indices) < 5:
          #    continue
          leaf_accuracies[idx] = np.mean(df['accuracy_bool'].iloc[leaf_indices])

      return leaf_accuracies

    # Generate a colormap. For instance, let's use the 'Reds' colormap.
    cmap = plt.get_cmap('Reds')

    def plot_decision_tree_segment(model, df, predictors):
        leaf_ids = model.apply(df[predictors])
        df['segment'] = leaf_ids  # Add the segment column to df
        baseline_acc = np.mean(df['accuracy_bool'])

        segments = defaultdict(list)

        # Calculate the accuracy for each segment/leaf
        leaf_accuracies = {}
        for idx in set(leaf_ids):
            leaf_indices = np.where(leaf_ids == idx)[0]
            segments[idx] = leaf_indices

            #if len(leaf_indices) < 5:
            # continue
            leaf_acc = np.mean(df['accuracy_bool'].iloc[leaf_indices])
            leaf_accuracies[idx] = leaf_acc
            #print(leaf_acc)

        # Create a dataframe to map each segment to its accuracy
        df_leaf = df[predictors + ['segment']].copy()
        df_leaf['accuracy'] = df_leaf['segment'].map(leaf_accuracies)

        # Visualization for two features
        unique_segments = df_leaf['segment'].unique()
        plt.figure(figsize=(12, 10))

        for segment in unique_segments:
            segment_data = df_leaf[df_leaf['segment'] == segment]

            if len(segment_data) >= 5:  # Only plot segments with 5 or more data points
              plt.scatter(segment_data[predictors[0]], segment_data[predictors[1]],
                        label=f'Segment {segment} - Accuracy: {leaf_accuracies[segment]:.4f}')

        plt.xlabel(predictors[0])
        plt.ylabel(predictors[1])
        plt.legend()
        plt.title(f'Accuracy by {predictors[0]} and {predictors[1]} segments')
        plt.show()

    # plot one-way weak spot
    def plot_one_way_segments_and_weak_spots_heatmap(feature1):
        predictors = [feature1]
        model = fit_and_visualize_dt(X_test1, predictors)  # Assuming this function trains and returns a model
        weak_spots, segments = find_weak_spots_and_leaves(model, X_test1, predictors, threshold=0.1)  # Assuming this function identifies weak spots and leaves

        plt.figure(figsize=(10, 6))

        feature_data = X_test1[feature1]
        total_samples = len(feature_data)

        # Get accuracies for each leaf node
        leaf_accuracies = get_leaf_accuracies(model, X_test1, predictors)  # Assuming this function calculates and returns leaf accuracies

        cmap = cm.get_cmap('coolwarm')  # Get the colormap

        # Plot histogram-like bars based on the segments from the decision tree
        for leaf_idx, indices in segments.items():
            leaf_accuracy = leaf_accuracies.get(leaf_idx, 1)  # Default to accuracy=1 if not found
            color_value = cmap(leaf_accuracy)  # Use accuracy to map to colors

            # Here we assume `segments` provides indices; you can change this to actual value ranges if needed
            min_val = feature_data.iloc[indices].min()
            max_val = feature_data.iloc[indices].max()
            frequency = len(indices)
            frequency_percent = (frequency / total_samples) * 100  # Frequency percentage

            plt.bar(min_val, height=frequency, width=max_val-min_val, color=color_value, align='edge')

            # Add frequency percentage text above the bar
            plt.text(min_val, frequency, f"{frequency_percent:.2f}%", ha='left', va='bottom')

            # Mark the bar with leaf ID if it's a weak spot
            if leaf_idx in weak_spots:
              plt.text((min_val + max_val) / 2, frequency / 2, str(leaf_idx), ha='center', va='center', color='red')

        plt.title(f"One-Way Weak Spot for {feature1}")
        plt.xlabel(feature1)
        plt.ylabel("Frequency")
        plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap), label="Accuracy")

        plt.show()

        if weak_spots:
          dot_data = tree.export_graphviz(model, out_file=None,
                                          feature_names=predictors,
                                          class_names=['True', 'False'],
                                          filled=True, rounded=True,
                                          special_characters=True, node_ids=True,
                                        )
          # Add a label (title) to the DOT source code
          title = f"Decision Tree Based on {feature1}"
          #dot_data_with_title = f'digraph Tree {{\nsize="4,4"; \nlabel="{title}";\nlabelloc=t;\nlabeljust=l;\nfontsize=8;\nfontname=Helvetica;\n{dot_data[dot_data.index("{") + 1:]}'
          #dot_data_with_title = f'digraph Tree {{\ngraph [size="4,4"];\nnode [fontsize=8];\nlabel="{title}";\nlabelloc=t;\nlabeljust=l;\nfontsize=8;\nfontname=Helvetica;\n{dot_data[dot_data.index("{") + 1:]}'
          dot_data_with_title = f'digraph Tree {{\nlabel="{title}";\nlabelloc=t;\nnode [fontsize=8];\nlabeljust=c;\nfontsize=12;\nfontname=Helvetica;\n{dot_data[dot_data.index("{") + 1:]}'

          graph = graphviz.Source(dot_data_with_title)
          display(graph)

    # Function to plot two-way weak spots
    def plot_segments_and_weak_spots_heatmap(feature1, feature2):
        predictors = [feature1, feature2]
        model = fit_and_visualize_dt(X_test1, predictors)
        weak_spots, segments = find_weak_spots_and_leaves(model, X_test1, predictors, threshold = 0.1)

        plt.figure(figsize=(10, 10))

        feature1_min, feature1_max = X_test1[feature1].min(), X_test1[feature1].max()
        feature2_min, feature2_max = X_test1[feature2].min(), X_test1[feature2].max()

        # Get accuracies for each leaf node. This is a mock-up example; replace with your actual function.
        leaf_accuracies = get_leaf_accuracies(model, X_test1, predictors)

        for leaf_idx, indices in segments.items():
          leaf_accuracy = leaf_accuracies.get(leaf_idx)  # Default to accuracy=1 if not found
          color_value = cmap(1 - leaf_accuracy)  # 1 - accuracy to map lower accuracies to darker colors

          #feature1_range = get_range_of_feature(tree, X_test[feature1], leaf_idx, 0)
          #feature2_range = get_range_of_feature(tree, X_test[feature2], leaf_idx, 1)

          #plt.fill_betweenx(feature2_range, feature1_range[0], feature1_range[1], color=color_value, alpha=0.5)

          feature1_range = (X_test1[feature1].iloc[indices].min(), X_test1[feature1].iloc[indices].max())
          feature2_range = (X_test1[feature2].iloc[indices].min(), X_test1[feature2].iloc[indices].max())

          plt.fill_betweenx(feature2_range, feature1_range[0], feature1_range[1], color=color_value, alpha=1)

          if leaf_idx in weak_spots:
              plt.text(np.mean(feature1_range), np.mean(feature2_range), str(leaf_idx), fontsize=12, color='red')

        plt.xlim([feature1_min, feature1_max])
        plt.ylim([feature2_min, feature2_max])
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Two-Way Weak Spots for {feature1} & {feature2}")
        plt.show()

        # Conditionally show the decision tree plot if weak_spots are not empty
        if weak_spots:
          #print(weak_spots)
          plot_decision_tree_segment(model, X_test1, predictors)

          dot_data = tree.export_graphviz(model, out_file=None,
                              feature_names=predictors,
                              class_names=['True', 'False'],
                              filled=True, rounded=True,
                              special_characters=True, node_ids=True,
                              )
          # Add a label (title) to the DOT source code
          title = f"Decision Tree Based on {feature1} & {feature2}"
          dot_data_with_title = f'digraph Tree {{\nlabel="{title}"; \nlabelloc=t;\nnode [fontsize=8]; \nlabelloc=t;\nlabeljust=c;\nfontsize=12;\nfontname=Helvetica;\n{dot_data[dot_data.index("{") + 1:]}'

          graph = graphviz.Source(dot_data_with_title)
          display(graph)

    # Get feature names for dropdown
    feature_names = df.columns.tolist()

    # Create widgets
    feature_one_way_widget = widgets.Dropdown(
        options=feature_names,
        value=feature_names[1],
        description='Feature:',
        disabled=False,
    )

    feature1_widget = widgets.Dropdown(
        options=feature_names,
        value = feature_names[1],
        description='Feature 1:',
        disabled=False,
    )

    feature2_widget = widgets.Dropdown(
        options=feature_names,
        value = feature_names[2],
        description='Feature 2:',
        disabled=False,
    )

    # Create interactive widgets
    one_way_interact = interactive(plot_one_way_segments_and_weak_spots_heatmap,
                                  feature1=feature_one_way_widget)
    #one_way_interact.manual = True

    two_way_interact = interactive(plot_segments_and_weak_spots_heatmap,
                                  feature1=feature1_widget, feature2=feature2_widget)
    #two_way_interact.manual = True

    message11 = widgets.HTML(value="<h3>One-Way Weak Spots</h3>")
    message12 = widgets.HTML(value="<h3>Two-Way Weak Spots</h3>")
    with weakspot_tab:
      # Display widgets
      display(HBox([VBox([message11,one_way_interact]), VBox([message12, two_way_interact])]))


################### Robustness #########################
    def robust_analysis(X_test, y_test, perturb_method='raw', perturb_features= 'Select All', perturb_size=0.1, metric='AUC', iterations=10):

      perturbed_scores = []

      #if perturb_features == 'Select All':
      #  perturb_features = X.columns

      if 'Select All' in perturb_features:
        perturb_features = X.columns

      # Function to perturb categorical features
      def perturb_categorical(column, perturb_size):
          # Calculate category frequencies
          frequencies = column.value_counts(normalize=True)
          categories = frequencies.index
          probabilities = frequencies.values

          # Perturb with a probability of `perturb_size`
          mask = np.random.rand(len(column)) < perturb_size
          perturbed_values = np.random.choice(categories, size=mask.sum(), p=probabilities)
          column[mask] = perturbed_values
          return column

      # Perform perturbation and calculate scores
      for _ in range(iterations):  # Repeat 10 times (default)
          X_test_perturbed = X_test.copy()

          for feature in perturb_features:
              if  X_test[feature].dtype in [np.float64, np.int64]: # for numerical features
                if perturb_method == 'raw':
                  noise_std = perturb_size * X_test_perturbed[feature].var()
                  noise = np.random.normal(0, noise_std, X_test_perturbed[feature].shape)
                  X_test_perturbed[feature] += noise
                elif perturb_method == 'quantile':
                  # Convert to quantiles
                  quantiles = X_test_perturbed[feature].rank(pct=True)
                  # Add uniform noise to the quantiles
                  perturbed_quantiles = quantiles + np.random.uniform(-perturb_size, perturb_size, quantiles.shape)
                  perturbed_quantiles = np.clip(perturbed_quantiles, 0, 1)  # Ensure quantiles are within [0, 1]
                  # Convert the perturbed quantiles back to original feature space
                  X_test_perturbed[feature] = np.quantile(X_test[feature], perturbed_quantiles)
              elif X_test[feature].dtype == 'category': # for categorical features
                  X_test_perturbed[feature] = perturb_categorical(X_test_perturbed[feature], perturb_size)

          # Evaluate model performance on perturbed data
          if task == 'classification':
              y_pred_perturbed = model.predict_proba(X_test_perturbed)[:, 1]
          else:
              y_pred_perturbed = model.predict(X_test_perturbed)

          score = evaluate_score(y_test, y_pred_perturbed, metric)
          perturbed_scores.append(score)

      return perturbed_scores

    def robust_plot(X_test, y_test, selected_features = 'Select All', metric='AUC'):
      perturbation_sizes = np.arange(0, 0.6, 0.1)  # From 0 to 0.5 in 0.1 increments
      results = {}

      if 'Select All' in selected_features:
        selected_features = X.columns

      for size in perturbation_sizes:
        results[size] = robust_analysis(X_test, y_test, perturb_method='raw', perturb_features=selected_features, perturb_size=size, metric=metric, iterations = 10)


      plt.figure(figsize=(10, 6))
      plt.boxplot([results[size] for size in perturbation_sizes], labels=[str(round(size, 2)) for size in perturbation_sizes])
      plt.xlabel('Perturbation Size')
      plt.ylabel(metric)
      plt.title('Model Robustness: Perturbation on Selected Features')
      plt.grid(True)
      plt.show()

    # Output widget for robustness
    output_robust = widgets.Output()

    # Create a SelectMultiple widget for feature selection
    feature_selector = widgets.SelectMultiple(
        options= ['Select All']+ list(X_test.columns),
        value=['Select All'],
        description='Features',
        disabled=False
    )

    '''
    def update_options(change):
      # If 'Select All' is selected, select all features
      if 'Select All' in change['new']:
          feature_selector.value = list(X_test.columns)

    feature_selector.observe(update_options, names='value')
    '''

    # Button to run the analysis
    run_button = widgets.Button(description="Confirm", button_style='success')

    # Button click event handler
    def on_run_button_clicked(b):
        output_robust.clear_output()
        with output_robust:
            selected_features = list(feature_selector.value)
            robust_plot(X_test, y_test, selected_features, metric='AUC')

    run_button.on_click(on_run_button_clicked)

    with output_robust:
      selected_features = list(feature_selector.value)
      robust_plot(X_test, y_test, selected_features, metric='AUC')

    with robust_tab:
      # Display the widgets
      display(widgets.VBox([feature_selector,
                            run_button,
                            output_robust]))

############ Fairness #######################
    # define Precision Ratio function
    def calculate_pr(feature, protected_class):
        #Ensure predictions are added to the test set
        predictions = X_test.copy()
        predictions['pred'] = model.predict(X_test)
        predictions['true'] = y_test  # Add true labels
        #predictions['pred'] = (predictions['pred'] >= 0.5).astype(int)

        # Check if the feature is numerical and needs binning
        if X[feature].dtype in [np.float64, np.int64]:
            # Perform equal-quantile binning if not already done
            _, bins = pd.qcut(X[feature], q=10, retbins=True, duplicates='drop')
            bin_labels = range(1, len(bins))
            predictions['binned_feature'] = pd.cut(X[feature], bins=bins, labels=bin_labels)
            protected_mask = predictions['binned_feature'] == protected_class
        else:
            protected_mask = predictions[feature] == protected_class

        reference_mask = ~protected_mask  # Use the complement as the reference group


        # Assuming 'true' and 'pred' columns exist in predictions DataFrame
        TP_protected = ((predictions['pred'] == 1) & (predictions['true'] == 1) & protected_mask).sum()
        FP_protected = ((predictions['pred'] == 1) & (predictions['true'] == 0) & protected_mask).sum()

        TP_reference = ((predictions['pred'] == 1) & (predictions['true'] == 1) & reference_mask).sum()
        FP_reference = ((predictions['pred'] == 1) & (predictions['true'] == 0) & reference_mask).sum()

        PPV_protected = TP_protected / (TP_protected + FP_protected) if (TP_protected + FP_protected) > 0 else None
        PPV_reference = TP_reference / (TP_reference + FP_reference) if (TP_reference + FP_reference) > 0 else None

        pr = PPV_protected / PPV_reference if PPV_reference is not None and PPV_protected is not None else None

        # Cleanup if binned feature was added
        if 'binned_feature' in predictions.columns:
            predictions.drop(columns=['binned_feature'], inplace=True)

        return pr

    # define adverse impact ratio function
    def calculate_air(feature, protected_class):
        # Ensure predictions are added to the test set
        predictions = X_test.copy()
        predictions['pred'] = model.predict(X_test)
        #predictions['pred'] = (predictions['pred'] >= 0.5).astype(int)

        # Check if the feature is numerical and needs binning
        if X[feature].dtype in [np.float64, np.int64]:
            # Perform equal-quantile binning if not already done
            _, bins = pd.qcut(X[feature], q=10, retbins=True, duplicates='drop')
            bin_labels = range(1, len(bins))
            predictions['binned_feature'] = pd.cut(X[feature], bins=bins, labels=bin_labels)
            protected_mask = predictions['binned_feature'] == protected_class
        else:
            protected_mask = predictions[feature] == protected_class

        reference_mask = ~protected_mask  # Use the complement as the reference group

        TP_FP_protected = predictions.loc[protected_mask, 'pred'].sum()
        TP_FP_reference = predictions.loc[reference_mask, 'pred'].sum()

        n_protected = protected_mask.sum()
        n_reference = reference_mask.sum()

        # Calculate AIR, handle division by zero if necessary
        air = (TP_FP_protected / n_protected) / (TP_FP_reference / n_reference) if n_reference > 0 and n_protected > 0 else None

        # Remove temporary binned feature column if it was added
        if 'binned_feature' in predictions.columns:
            predictions.drop(columns=['binned_feature'], inplace=True)

        return air

    # define Standardized Mean Difference (SMD) function
    def calculate_SMD(feature, protected_class):
        predictions = X_test.copy()
        predictions['pred'] = model.predict(X_test)
        #predictions['pred'] = (predictions['pred'] >= 0.5).astype(int)

        # Check if the feature is numerical and needs binning
        if X[feature].dtype in [np.float64, np.int64]:
            # Perform equal-quantile binning if not already done
            _, bins = pd.qcut(X[feature], q=10, retbins=True, duplicates='drop')
            bin_labels = range(1, len(bins))
            predictions['binned_feature'] = pd.cut(X[feature], bins=bins, labels=bin_labels)
            protected_mask = predictions['binned_feature'] == protected_class
        else:
            protected_mask = predictions[feature] == protected_class

        reference_mask = ~protected_mask  # Use the complement as the reference group

        mean_protected = predictions[protected_mask]['pred'].mean()
        mean_reference = predictions[reference_mask]['pred'].mean()

        pooled_std = np.sqrt(((protected_mask.sum() - 1) * predictions[protected_mask]['pred'].std() ** 2 +
                              (reference_mask.sum() - 1) * predictions[reference_mask]['pred'].std() ** 2) /
                            (protected_mask.sum() + reference_mask.sum() - 2))

        smd = (mean_protected - mean_reference) / pooled_std
        return smd

    # Function to evaluate fairness and visualize AIR
    # Create separate output containers for AIR plot
    air_plot_output = widgets.Output()

    def plot_air(selected_pairs):
        # Initialize lists to store AIR values and labels for the plot
        air_values = []
        labels = []

        # Loop through each (feature, protected_class) pair
        for feature, protected_class in selected_pairs:
            # Calculate AIR for the current pair
            air = calculate_air(feature, protected_class)
            labels.append(f"{feature}: {protected_class}")
            air_values.append(air)

        with air_plot_output:
          clear_output(wait=True)
        # Adjust the plot size dynamically based on the number of pairs. Increase the vertical size factor as needed.
        # Inside plot_air
          air_values_filtered = [value if value is not None else -1 for value in air_values]  # Replace None with -1
          labels_filtered = [label for label, value in zip(labels, air_values) if value is not None]  # Keep corresponding labels

          plt.figure(figsize=(10, max(2, len(labels_filtered) * 1)))  # Adjust based on filtered labels
          plt.barh(labels_filtered, air_values_filtered, color='skyblue')
          plt.axvline(1, color='red', linestyle='--', label='Fairness (AIR=1)')
          plt.xlabel('Adverse Impact Ratio (AIR)')
          plt.ylabel('Protected Class')
          plt.title('Adverse Impact Ratio Analysis')
          plt.legend()
          plt.tight_layout()
          plt.show()

    pr_plot_output = widgets.Output()
    # Function to evaluate fairness and visualize AIR
    def plot_pr(selected_pairs):
        pr_values = []
        labels = []

        for feature, protected_class in selected_pairs:
            pr = calculate_pr(feature, protected_class)
            labels.append(f"{feature}: {protected_class}")
            pr_values.append(pr)

        # Plotting
        with pr_plot_output:
          clear_output(wait = True)
          plt.figure(figsize=(10, max(2, len(selected_pairs) * 0.5)))
          plt.barh(labels, pr_values, color='skyblue')
          plt.axvline(1, color='red', linestyle='--', label='Fairness (PR=1)')
          plt.xlabel('Precision Ratio (PR)')
          plt.ylabel('Protected Class')
          plt.title('Precision Ratio Analysis')
          plt.legend()
          plt.tight_layout()
          plt.show()

    smd_plot_output = widgets.Output()
    def plot_smd(selected_pairs):
        smd_values = []
        labels = []

        for feature, protected_class in selected_pairs:
          smd = calculate_SMD(feature, protected_class)
          labels.append(f"{feature}: {protected_class}")
          smd_values.append(smd)

        with smd_plot_output:
          plt.figure(figsize=(10, max(2, len(selected_pairs) * 0.5)))
          plt.barh(labels, smd_values, color='skyblue')
          plt.axvline(0, color='red', linestyle='--', label='No Difference')
          plt.xlabel('Standardized Mean Difference (SMD)')
          plt.ylabel('Comparison')
          plt.title('Standardized Mean Difference Analysis')
          plt.legend()
          plt.tight_layout()
          plt.show()

    # Widget for selecting sensitive attributes
    sensitive_attr_widget = widgets.Dropdown(
        options=X.columns,
        description='Feature:',
        disabled=False
    )

    # UI for specifying the protected class
    protected_class_widget = widgets.Dropdown(
        description='Protected Class:',
        disabled=False
    )

    def update_protected_class_options(*args):
        attr = sensitive_attr_widget.value
        if X[attr].dtype == 'object' or X[attr].dtype.name == 'category':
            # For categorical attributes
            options = [(str(v), v) for v in X[attr].unique()]
        else:
            # For numerical attributes, perform binning
            binned_values, bins = pd.cut(X[attr], bins=10, retbins=True, labels=range(1, 11))
            options = [(f"Bin {i}: {round(bins[i-1], 2)} to {round(bins[i], 2)}", i) for i in range(1, len(bins))]

        protected_class_widget.options = options
        # Resetting the value to the first option to avoid selection errors
        protected_class_widget.value = options[0][1]

    sensitive_attr_widget.observe(update_protected_class_options, 'value')
    update_protected_class_options()  # Call it once initially to set default options

    add_button = widgets.Button(description='Add Feature/Class',
                                button_style='success')
    generate_plot_button = widgets.Button(description='Generate Disparity Plot',
                                          button_style='info')

    # Containers
    global added_features_classes
    added_features_classes = []  # Clear the list of added feature/class pairs

    features_classes_container = widgets.Output()

    # Callbacks
    def add_feature_class(b):
        #global added_features_classes
        feature = sensitive_attr_widget.value
        protected_class = protected_class_widget.value
        added_features_classes.append((feature, protected_class))

        # Display the current list of added features/classes
        with features_classes_container:
            clear_output()
            print("Added Feature/Protected Classes:")
            for f, c in added_features_classes:
                print(f"{f}: {c}")

    # Adjusted Callback for Generating the AIR Plot
    def generate_fairness_plot(b):
        # Assuming plot_air is defined to take a list of (feature, protected_class) tuples
        if task == 'classification':
          plot_air(added_features_classes)
          plot_pr(added_features_classes)
        if task == 'regression':
          plot_smd(added_features_classes)

    # Link callbacks to buttons
    add_button.on_click(add_feature_class)
    # Link the corrected callback to the generate_plot_button
    #with plot_output_container:
    generate_plot_button.on_click(generate_fairness_plot)


    # Add a new button for clearing the selections and plot
    clear_button = widgets.Button(description='Clear Selections', button_style='warning')

    # Function to handle the clearing operation
    def clear_selections(b):
        # global added_features_classes
        # added_features_classes = []  # Clear the list of added feature/class pairs

        # Clear the display of added feature/class pairs
        with features_classes_container:
          clear_output()
          print("Added Feature/Protected Classes cleared.")

        # Optionally, clear the plot or reset any display related to the AIR analysis
        # This depends on how the AIR plot is being displayed.
        # For example, if using Output widget for the plot, clear it:
        with pr_plot_output:
          clear_output()
        with air_plot_output:
          clear_output()
        with smd_plot_output:
          clear_output()


    # Link the clear_button to the clear_selections function
    clear_button.on_click(clear_selections)

    # Now display the UI elements
    with fairness_tab:
      display(widgets.VBox([sensitive_attr_widget,
              protected_class_widget,
              widgets.HBox([add_button, clear_button]),
              generate_plot_button,
              features_classes_container,
              air_plot_output,
              pr_plot_output,
              smd_plot_output]))
