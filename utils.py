import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

import pandas as pd
import numpy as np

def visualize(df, isFlattened = False, showHistogram=True, showScatter = True, showBoxplot = True, showHeatMap = True):
    sns.set_style("whitegrid")

    if showHistogram == True:
        if isFlattened:
            df.drop(['ID', 'Output Measured Value'], axis=1).hist(bins=15, figsize=(15, 10), layout=(7, 10))
        else : 
            df.drop(['ID', 'Output Measured Value', 'Position'], axis=1).hist(bins=15, figsize=(15, 10), layout=(7, 10))
        plt.tight_layout()
        plt.show()

    d1_columns = [col for col in df.columns if 'D1' in col]
    d2_columns = [col for col in df.columns if 'D2' in col]
    assert len(d1_columns) == len(d2_columns), "The number of D1 and D2 features do not match."

    
    if showScatter:
        num_plots = len(d1_columns)
        num_rows = num_plots // 3 + (num_plots % 3 > 0)
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))

        axes = axes.flatten()

        for i, (d1_col, d2_col) in enumerate(zip(d1_columns, d2_columns)):
            sns.scatterplot(data=df, x=d1_col, y='Output Measured Value', ax=axes[i], color='blue', label=d1_col)
            sns.scatterplot(data=df, x=d2_col, y='Output Measured Value', ax=axes[i], color='orange', label=d2_col)
            axes[i].legend()
            axes[i].set_title(f'Scatter plot of {d1_col} and {d2_col} vs. Output Measured Value')

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    
    if showBoxplot == True:
        plt.figure(figsize=(15, 10))
        if isFlattened:
            sns.boxplot(data=df.drop(['ID', 'Output Measured Value'], axis=1))
        else:
            sns.boxplot(data=df.drop(['ID', 'Output Measured Value', 'Position'], axis=1))
        plt.xticks(rotation=90)
        plt.title('Boxplot of features')
        plt.show()

    
    if showHeatMap == True:
        if isFlattened:
            correlation_matrix = df.drop(['ID', 'Strength_2', 'Strength_3'], axis=1).corr()
        else:
            correlation_matrix = df.drop(['ID', 'Position', 'Strength_2', 'Strength_3'], axis=1).corr()

        plt.figure(figsize=(df.shape[1]/2, df.shape[1]/2))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
        
def plot_actual_vs_predicted(plot_df):
    f, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=plot_df.index, y='Actual', data=plot_df, color="b", label='Actual')
    sns.lineplot(x=plot_df.index, y='Predicted', data=plot_df, color="r", label='Predicted')

    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel('Value')
    ax.set_xlabel('Sample Index')
    ax.set_title('Actual vs Predicted Values with Differences')
    plt.legend()

    plt.show()
    
def plot_actual_vs_predicted_clusters(plot_dfs, rmses, r2s, num_of_clusters, overall_title):
    num_plots = len(plot_dfs)
    rows = (num_plots + 2) // 3  # 3 plots per row

    f, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten()
    max_r2_index = r2s.index(max(r2s))

    for i, (df, rmse, r2, num_cluster) in enumerate(zip(plot_dfs, rmses, r2s, num_of_clusters)):
        ax = axes[i]
        sns.lineplot(x=df.index, y='Actual', data=df, color="b", label='Actual', ax=ax)
        sns.lineplot(x=df.index, y='Predicted', data=df, color="r", label='Predicted', ax=ax)
        ax.set_ylabel('Value')
        ax.set_xlabel('Sample Index')
        ax.set_title(f'Model {i+1}: RMSE: {rmse:.3f}, R²: {r2:.3f}, Clusters: {num_cluster}', color = 'red' if i == max_r2_index else 'black')
        ax.legend()

    plt.tight_layout()
    f.suptitle(overall_title, fontsize=16, y=1.02)
    plt.show()

def LinearRegressionModel(df, target = 'Output Measured Value', features=[], print_logs=True):
    X = df[features] if features else df.drop(['ID', 'Output Measured Value', 'Strength_2', 'Strength_3'], axis=1)
    y = df[target]
    lr_model = LinearRegression()

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    y_pred_cv = cross_val_predict(lr_model, X, y, cv=cv)
    rmse_cv = mean_squared_error(y, y_pred_cv, squared=False)
    r2_cv = r2_score(y, y_pred_cv)
    
    if print_logs:
        print(f'Cross-Validated LinearRegressionModel RMSE: {rmse_cv}')
        print(f'Cross-Validated LinearRegressionModel R-squared: {r2_cv}')

    results_df_cv = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv}).reset_index(drop=True)
    return results_df_cv, rmse_cv, r2_cv

def RandomForestRegressionModel(df, target = 'Output Measured Value', features=[], print_logs=True):
    X = df[features] if features else df.drop(['ID', 'Output Measured Value', 'Strength_2', 'Strength_3'], axis=1)
    y = df[target]

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rfr = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rfr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    if print_logs:
        print(f"Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred_cv = cross_val_predict(best_model, X, y, cv=5)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    r2_cv = r2_score(y, y_pred_cv)
    
    if print_logs:
        print(f'Cross-Validated RandomForestRegressionModel RMSE: {rmse_cv}')
        print(f'Cross-Validated RandomForestRegressionModel R-squared: {r2_cv}')

    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv})
    return results_df, rmse_cv, r2_cv

def RandomForestClassifierModel(df, num_of_classes, cv=5, print_logs=True):
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df.drop(['ID', 'Output Measured Value', 'Strength_2', 'Strength_3'], axis=1))
    y = df['Strength_2'] if num_of_classes == 2 else df['Strength_3']

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    if print_logs:
        print(f"Best parameters: {grid_search.best_params_}")

    cv_accuracy = np.mean(cross_val_score(best_model, X, y, cv=cv, scoring='accuracy'))
    if print_logs:
        print(f'Cross-Validated RandomForestClassifierModel Accuracy for {num_of_classes} classes: {round(cv_accuracy * 100, 1)}%')

    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv})
    return results_df, cv_accuracy

def plot_confusion_matrix(df):
    conf_matrix = confusion_matrix(df['Actual'], df['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    

def plot_roc_auc(df):
    y_true = df['Actual']
    y_pred = df['Predicted']

    classes = np.unique(y_true)

    # Check if it is a binary classification
    is_binary_class = len(classes) == 2

    # Binarize the output
    y_true_binarized = label_binarize(y_true, classes=classes) if not is_binary_class else (y_true == classes[1]).astype(int)
    y_pred_binarized = label_binarize(y_pred, classes=classes) if not is_binary_class else (y_pred == classes[1]).astype(int)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if is_binary_class:
        fpr[0], tpr[0], _ = roc_curve(y_true_binarized, y_pred_binarized)
        roc_auc[0] = auc(fpr[0], tpr[0])
        classes = [classes[1]]  # Use positive class for binary classification
    else:
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    for i, color in zip(range(len(classes)), colors):
        label = f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})' if not is_binary_class else 'ROC curve (area = {0:0.2f})'.format(roc_auc[0])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
