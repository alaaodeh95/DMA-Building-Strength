import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

def visualize(df, isFlattened = False):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Histograms for features
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

    # Boxplots for features to check for outliers
    plt.figure(figsize=(15, 10))
    if isFlattened:
        sns.boxplot(data=df.drop(['ID', 'Output Measured Value'], axis=1))
    else:
        sns.boxplot(data=df.drop(['ID', 'Output Measured Value', 'Position'], axis=1))
    plt.xticks(rotation=90)
    plt.title('Boxplot of features')
    plt.show()
    
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
    
    
def LinearRegressionModel(df):
    # Preparing the feature matrix and target vector
    X = df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1)
    y = df['Output Measured Value']

    # Splitting the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creating a Linear Regression model as a baseline
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predicting and evaluating the model
    y_pred = lr_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f'Baseline Model RMSE: {rmse}')
    print(f'Baseline Model R-squared: {r2}')

    # Creating a DataFrame with actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Optionally, reset index to make it more readable
    results_df = results_df.reset_index(drop=True)

    # Display the DataFrame
    print(results_df)
    return results_df

def plot_actual_vs_predicted(df):
    # Create a new DataFrame for plotting
    plot_df = df.copy()
    plot_df['Difference'] = plot_df['Actual'] - plot_df['Predicted']
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 6))

    # Plot the differences
    sns.barplot(x=plot_df.index, y='Difference', data=plot_df, palette="vlag", ax=ax)

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
    
    
def RandomForestRegressionModel(df):
    # Preparing the feature matrix and target vector
    X = df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1)
    y = df['Output Measured Value']

    # Setting up the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    # Creating the Grid Search with Cross-Validation
    rfr = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluating the model using cross-validation
    best_model = grid_search.best_estimator_
    cv_rmse = np.sqrt(-cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error'))
    print(f'Cross-Validated RMSE: {np.mean(cv_rmse)}')

    
def RandomForestRegressionModel(df):
    # Preparing the feature matrix and target vector
    X = df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1)
    y = df['Output Measured Value']

    # Splitting the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Setting up the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    # Creating the Grid Search
    rfr = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Predicting and evaluating the model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Optimized Model RMSE: {rmse}')

    # Creating a DataFrame with actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
    print(results_df)
    
    
def RandomForestClassifierModel(df):
    # Preparing the feature matrix and target vector
    imputer = SimpleImputer(strategy='median')

    X = imputer.fit_transform(df.drop(['ID', 'Output Measured Value', 'Strength'], axis=1))
    y = df['Strength']

    # Setting up the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Creating the Grid Search with Cross-Validation
    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluating the model using cross-validation
    best_model = grid_search.best_estimator_
    cv_accuracy = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f'Cross-Validated Accuracy: {np.mean(cv_accuracy)}')
    