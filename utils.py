import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

def visualize(df, isFlattened = False, showHistogram=True, showScatter = True, showBoxplot = True, showHeatMap = True):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Histograms for features
    if showHistogram == True:
        if isFlattened:
            df.drop(['ID', 'Output Measured Value'], axis=1).hist(bins=15, figsize=(15, 10), layout=(7, 10))
        else : 
            df.drop(['ID', 'Output Measured Value', 'Position'], axis=1).hist(bins=15, figsize=(15, 10), layout=(7, 10))
        plt.tight_layout()
        plt.show()

    # Assuming df is your DataFrame and the columns are named as provided
    # Extracting the column names for D1 and D2
    d1_columns = [col for col in df.columns if 'D1' in col]
    d2_columns = [col for col in df.columns if 'D2' in col]

    # Ensure that d1_columns and d2_columns have the same length
    assert len(d1_columns) == len(d2_columns), "The number of D1 and D2 features do not match."

    
    if showScatter:
        # Define the grid size
        num_plots = len(d1_columns)
        num_rows = num_plots // 3 + (num_plots % 3 > 0)
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Iterate over the columns and plot each pair
        for i, (d1_col, d2_col) in enumerate(zip(d1_columns, d2_columns)):
            sns.scatterplot(data=df, x=d1_col, y='Output Measured Value', ax=axes[i], color='blue', label=d1_col)
            sns.scatterplot(data=df, x=d2_col, y='Output Measured Value', ax=axes[i], color='orange', label=d2_col)
            axes[i].legend()
            axes[i].set_title(f'Scatter plot of {d1_col} and {d2_col} vs. Output Measured Value')

        # Hide any unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        # Adjust layout for better visibility
        plt.tight_layout()
        plt.show()

    
    if showBoxplot == True:
        # Boxplots for features to check for outliers
        plt.figure(figsize=(15, 10))
        if isFlattened:
            sns.boxplot(data=df.drop(['ID', 'Output Measured Value'], axis=1))
        else:
            sns.boxplot(data=df.drop(['ID', 'Output Measured Value', 'Position'], axis=1))
        plt.xticks(rotation=90)
        plt.title('Boxplot of features')
        plt.show()

    
    if showHeatMap == True:
        # Compute the correlation matrix
        if isFlattened:
            correlation_matrix = df.drop(['ID', 'Strength'], axis=1).corr()
        else:
            correlation_matrix = df.drop(['ID', 'Position', 'Strength'], axis=1).corr()

        # Heatmap of the correlation matrix
        plt.figure(figsize=(df.shape[1]/2, df.shape[1]/2))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
        
def plot_actual_vs_predicted(plot_df):
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 6))

    # Plot the actual values
    sns.lineplot(x=plot_df.index, y='Actual', data=plot_df, color="b", label='Actual')

    # Plot the predicted values
    sns.lineplot(x=plot_df.index, y='Predicted', data=plot_df, color="r", label='Predicted')

    # Add some formatting and titles
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel('Value')
    ax.set_xlabel('Sample Index')
    ax.set_title('Actual vs Predicted Values with Differences')
    plt.legend()

    # Show the plot
    plt.show()
    
def plot_actual_vs_predicted_clusters(plot_dfs, rmses, r2s, num_of_clusters, overall_title):
    num_plots = len(plot_dfs)
    rows = (num_plots + 2) // 3  # 3 plots per row

    # Set up the matplotlib figure
    f, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Find the index of the highest R²
    max_r2_index = r2s.index(max(r2s))

    # Iterate over each set of predictions and corresponding metrics
    for i, (df, rmse, r2, num_cluster) in enumerate(zip(plot_dfs, rmses, r2s, num_of_clusters)):
        ax = axes[i]

        # Plot the actual values
        sns.lineplot(x=df.index, y='Actual', data=df, color="b", label='Actual', ax=ax)

        # Plot the predicted values
        sns.lineplot(x=df.index, y='Predicted', data=df, color="r", label='Predicted', ax=ax)

        # Add some formatting and titles
        ax.set_ylabel('Value')
        ax.set_xlabel('Sample Index')
        ax.set_title(f'Model {i+1}: RMSE: {rmse:.3f}, R²: {r2:.3f}, Clusters: {num_cluster}', color = 'red' if i == max_r2_index else 'black')

        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Add an overall title
    f.suptitle(overall_title, fontsize=16, y=1.02)

    # Show the plot
    plt.show()

def LinearRegressionModel(df, print_logs=True):
    # Preparing the feature matrix and target vector
    X = df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1)
    y = df['Output Measured Value']
    lr_model = LinearRegression()

    # Cross-validation approach
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    y_pred_cv = cross_val_predict(lr_model, X, y, cv=cv)
    rmse_cv = mean_squared_error(y, y_pred_cv, squared=False)
    r2_cv = r2_score(y, y_pred_cv)
    
    if print_logs:
        print(f'Cross-Validated LinearRegressionModel RMSE: {rmse_cv}')
        print(f'Cross-Validated LinearRegressionModel R-squared: {r2_cv}')

    # Creating DataFrames with actual and predicted values
    results_df_cv = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv}).reset_index(drop=True)

    return results_df_cv, rmse_cv, r2_cv


def RandomForestRegressionModel(df, print_logs=True):
    # Preparing the feature matrix and target vector
    X = df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1)
    y = df['Output Measured Value']

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Random Forest Regressor with Grid Search
    rfr = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rfr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    # Best parameters
    if print_logs:
        print(f"Best parameters: {grid_search.best_params_}")

    # Best model
    best_model = grid_search.best_estimator_
    
    # Cross-validated predictions
    y_pred_cv = cross_val_predict(best_model, X, y, cv=5)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    r2_cv = r2_score(y, y_pred_cv)
    
    if print_logs:
        print(f'Cross-Validated RandomForestRegressionModel RMSE: {rmse_cv}')
        print(f'Cross-Validated RandomForestRegressionModel R-squared: {r2_cv}')

    # Results DataFrame
    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv})

    return results_df, rmse_cv, r2_cv

def RandomForestClassifierModel(df, cv=5, print_logs=True):
    """
    Train a RandomForestClassifier model using GridSearchCV.

    Parameters:
    df: DataFrame containing the dataset.
    cv: Number of cross-validation folds.
    print_logs: Boolean indicating whether to print log messages.

    Returns:
    A tuple containing the results DataFrame and cross-validated accuracy.
    """

    # Preparing the feature matrix and target vector
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1))
    y = df['Strength']

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Random Forest Classifier with Grid Search
    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    # Best parameters and model
    best_model = grid_search.best_estimator_
    if print_logs:
        print(f"Best parameters: {grid_search.best_params_}")

    # Evaluating the model using cross-validation
    cv_accuracy = np.mean(cross_val_score(best_model, X, y, cv=cv, scoring='accuracy'))
    if print_logs:
        print(f'Cross-Validated RandomForestClassifierModel Accuracy: {cv_accuracy}')

    # Results DataFrame
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
    results_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv})

    return results_df, cv_accuracy

def plot_confusion_matrix(df):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(df['Actual'], df['Predicted'])

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.show()