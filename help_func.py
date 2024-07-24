import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.stats import skew
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm


def cycle_transform(value):

    sin = np.sin(2 * np.pi *
                                   value / 53)
    cos = np.cos(2 * np.pi *
                                   value / 53)
    

    return sin, cos


def skewness(df, threshold=0.5):
    sk = []
    for column in df.columns:
        # Calculate skewness
        skewness = skew(df[column].dropna() + 1)  # Drop NA values for skew calculation
        # print the skewness
        print(f"Skewness of {column}: {skewness}")
        # Check if skewness is beyond a certain threshold and values are positive
        if abs(skewness) > threshold and all(df[column] + 1 > 0):
            print(f"Column {column} with skewness {skewness} is a candidate for Box-Cox transformation.")
            sk.append(column)

    if len(sk) == 0:
        print("No columns with skewness beyond threshold found.")
            
    return sk


def apply_boxcox(x, lmbda=None):
	"""
	Apply the Box-Cox transformation to a dataset.
	
	Parameters:
	- x: 1D array-like data to transform.
	- lmbda: Optional lambda value for transformation. If None, the lambda is estimated.
	
	Returns:
	- Transformed data and the lambda used for the transformation.
	"""
	x_transformed, lmbda_optimal = boxcox(x, lmbda=lmbda)
	return x_transformed, lmbda_optimal


def apply_inverse_boxcox(y, lmbda):
	"""
	Apply the inverse Box-Cox transformation to a dataset.
	
	Parameters:
	- y: 1D array-like data to inverse transform.
	- lmbda: Lambda value used for the Box-Cox transformation.
	
	Returns:
	- Inverse transformed data.
	"""
	return inv_boxcox(y, lmbda)


def boxcox_transform(df, columns):
    """
    Apply the Box-Cox transformation to a dataset.
    
    Parameters:
    - df: DataFrame to transform.
    - columns: Columns to transform
    """
    lmbdas = {}
    for column in columns:
        df[column], lam = apply_boxcox(df[column] + 1)  # Add 1 to avoid non-positive values
        lmbdas[column] = lam

    return df, lmbdas


def inverse_boxcox_transform(df, lmbdas):
    """
    Apply the inverse Box-Cox transformation to a dataset.
    
    Parameters:
    - df: DataFrame to transform.
    - columns: Columns to transform
    - lmbdas: Lambda values used for the Box-Cox transformations.
    """
    for column in lmbdas.keys():
        df[column] = apply_inverse_boxcox(df[column], lmbdas[column]) - 1  # Subtract 1 to revert the previous addition

    return df


def moving_average(data, window):
    ma = data.rolling(window=window, min_periods=1).mean()
    return ma

def shift_data(data, lag=0):  
    return data.shift(lag).fillna(data.shift(lag).iloc[0])

def correlation(data1, data2):
    return data1.corr(data2)

def grid_search(data, window_range, lag_range, target_column):
    best_corr = 0
    best_window = None
    best_lag = None
    target_data = target_column
    feature_data = data.drop(columns=[target_column])

    for window in window_range:
        for lag in lag_range:
            # Apply moving average
            ma_data = moving_average(feature_data , window)
            
            # Apply data shift
            shifted_data = shift_data(ma_data, lag)
            
            # Calculate correlation
            corr = correlation(shifted_data, target_data)
            abs_corr = np.abs(corr)
            
            # Update best correlation and parameters
            if abs_corr > best_corr:
                best_corr = corr
                best_window = window
                best_lag = lag
    
    if window is None:
        print("No best window parameter found.")
        window = 1
    
    if lag is None:
        print("No best lag parameter found.")
        lag = 0

    return best_window, best_lag, best_corr

def grid_search_wrapper(df, target_column, window_range, lag_range, features= 'all'):
    best_params = {}

    if features == 'all':
        features = df.columns.tolist()
        features.remove(target_column)

    else:
        features = features
    
    for feature in features:
        data = df[feature]
        
        best_window, best_lag, best_corr = grid_search(data, window_range, lag_range, target_column=df[target_column])
        if best_window is None:
            print(f"No best parameters found for feature {feature}.")
            best_window = 1

        if best_lag is None:
            print(f"No best parameters found for feature {feature}.")
            best_lag = 0


        best_params[feature] = {'window': best_window, 'lag': best_lag, 'best_correlation':best_corr}
    
    return best_params



def compare_correlations(original_corr, best_params):
    corr_diff = {}
    for feature in best_params.keys():
        corr_diff[feature] = abs(best_params[feature]['best_correlation']) - abs(original_corr.loc[feature])
        #print(corr_diff)
        print(f"Feature: {feature}, Correlation Difference: {corr_diff[feature]}")
    
    return corr_diff


def create_extra_features(df, features, drorp_original = False):
    """
    Create extra features by calculating moving averages for each feature in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the original features.
    - features (dict): A dictionary containing the window and lag values for each feature.
    - sj (bool): A boolean value indicating whether the data is for San Juan (sj=True) or Iquitos (sj=False).

    Returns:
    - df (pandas.DataFrame): The DataFrame with the additional moving average features added.
    """
    tot_cas = None

    if 'total_cases' in df.columns:
        tot_cas = df['total_cases']

    df = df[features.keys()]

    for feature in features.keys():

        print(f'feature: {feature}')

        window = features[feature]['window']
        if window == 0 or window is None:
            print(f"Invalid window value for feature {feature}: {window}, window set to 1.")
            window == 1



        lag = features[feature]['lag']
        if lag is None:
            print(f"Invalid lag value for feature {feature}: {lag}, lag set to 0.")
            lag == 0

        
        print(f'before ma {df.shape}')
        df[f'{feature}_ma'] = moving_average(df[feature], window)
        print(f'after ma {df.shape}')
        df[f'{feature}_ma'] = shift_data(df[f'{feature}_ma'], lag)
        print(f'after shift {df.shape}')

        if drorp_original:
            df.drop(feature, axis=1, inplace=True)

    if tot_cas is not None:

        df['total_cases'] = tot_cas

        
    # df.dropna(inplace=True)

    return df




def generate_model_formula(features, with_ma = False):
    formula = "total_cases ~ 1 + " + ' + '.join(features.columns[:-1])
    if with_ma:
        formula += ' + ' + ' + '.join([f'{feature}_ma' for feature in features.columns[:-1]])
    return formula



def get_best_model(train, test, model_formula):
    # Step 1: specify the form of the model

    model_formula = model_formula

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
