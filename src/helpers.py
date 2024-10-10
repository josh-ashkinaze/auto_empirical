import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import random 
import logging
import re
import os
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
import chardet


random.seed(42)
np.random.seed(42)


#####################################################################
# DATA WRANGLING
#####################################################################

def clean_vars(s, how='title'):
    """
    Simple function to clean titles for plots

    Params
    - s (str): The string to clean
    - how (str, default='title'): How to return string. Can be either ['title', 'lowercase', 'uppercase']

    Returns
    - cleaned string
    """
    assert how in ['title', 'lowercase', 'uppercase'], "Bad option!! see docs"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    s = s.replace('_', ' ')
    if how == 'title':
        return s.title()
    elif how=='lower':
        return s.lower()
    elif how=='upper':
        return s.upper()


def read_csv_robust(file_path, sep=",", num_bytes=10000):
    """
    A function to robustly read in CSVs when they may contain different kinds of encoding errors

    Params:
        file_path (str): The file path
        sep (str): The string seperator
        num_bytes(int, default=10000): Reads in this sample to get encoding 

    Returns
        pandas df if success else None 
    """
    # Detect the file encoding
    def detect_encoding(file_path, num_bytes):
        with open(file_path, 'rb') as file:
            rawdata = file.read(num_bytes)
            result = chardet.detect(rawdata)
            return result['encoding']

    encoding_detected = detect_encoding(file_path, num_bytes)

    # Try reading the file with the detected encoding
    try:
        df = pd.read_csv(file_path, encoding=encoding_detected, on_bad_lines='skip', sep=sep)
        print(f"File read successfully with encoding: {encoding_detected}")
        return df
    except Exception as e:
        print(f"Failed to read with detected encoding {encoding_detected}. Error: {e}")

        # Fallback to UTF-8
        try:
            print("Attempting to read with UTF-8...")
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=sep)
            print("File read successfully with UTF-8.")
            return df
        except Exception as e:
            print(f"Failed to read with UTF-8. Error: {e}")

            # Second fallback to ISO-8859-1
            try:
                print("Attempting to read with ISO-8859-1...")
                df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', sep=sep)
                print("File read successfully with ISO-8859-1.")
                return df
            except Exception as e:
                print(f"Failed to read with ISO-8859-1. Error: {e}")
                raise ValueError("All attempts failed. Please check the file for issues beyond encoding.")



#####################################################################
# LATEX 
#####################################################################

def statsmodels2latex(model, beta_digits=2, se_digits=2, p_digits=3, ci_digits=2, print_sci_not=False):
    """
    This function summarizes the results from a fitted statistical model,
    printing a LaTeX formatted string for each parameter in the model that includes the beta coefficient,
    standard error, p-value, and 95% CI.
    
    Parameters:
    - model: A fitted statistical model with methods to extract parameters, standard errors,
             p-values, and confidence intervals.
    - beta_digits (default = 2): Number of decimal places for beta coefficients.
    - se_digits (default = 2): Number of decimal places for standard errors.
    - p_digits (default = 3): Number of decimal places for p-values.
    - ci_digits (default = 2): Number of decimal places for confidence intervals.
    - print_sci_not: Boolean to print very small p-values (p<0.001) in scientific notation or just write 'p<0.001'
    
    """
    
    summary_strs = []
    # Check if the necessary methods are available in the model
    if not all(hasattr(model, attr) for attr in ['params', 'bse', 'pvalues', 'conf_int']):
        raise ValueError("Model does not have the required methods (params, bse, pvalues, conf_int).")
    
    # Retrieve parameter estimates, standard errors, p-values, and confidence intervals
    params = model.params
    errors = model.bse
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    for param_name, beta in params.items():
        # Escape LaTeX special characters in parameter names
        safe_param_name = param_name.replace('_', '\\_')
        
        se = errors[param_name]
        p = pvalues[param_name]
        ci_lower, ci_upper = conf_int.loc[param_name]

        if p < 0.001:
            if print_sci_not:
                p_formatted = f"= {p:.2e}"
            else:
                p_formatted = f"<0.001"
        else:
            p_formatted = f"= {p:.{p_digits}f}"

        summary = (f"{safe_param_name}: $\\beta = {beta:.{beta_digits}f}$, "
                   f"$SE = {se:.{se_digits}f}$, $p {p_formatted}$, "
                   f"$95\\% CI = [{ci_lower:.{ci_digits}f}, {ci_upper:.{ci_digits}f}]$")
        print(summary)



def stargazer2latex(star, filename, add_ci=True, display_mod=False):
    """
    Function to process the Stargazer object and save the LaTeX output to a file.

    Params:
    - star: Stargazer object
    - filename: str, the path to save the LaTeX output
    - add_ci: bool, whether to add 95% CIs to the LaTeX output
    - display_mod: bool, whether to display the Stargazer object before saving the LaTeX output


    Example:
        add_words = smf.ols('ai_add_wc ~ IsConstrained*PromptType', data=df).fit()
        remove_words = smf.ols('ai_remove_wc ~ IsConstrained*PromptType', data=df).fit()
        star = Stargazer([add_words, remove_words]) 
        star.custom_columns(["Count of Added Words", "Count of Removed Words"])
        star.title("OLS regressions of additions and removals.") 
        stargazer2latex(star, "../tables/reg_auto_add_rem.tex")
    """
    
    print(f"Starting to process the Stargazer object for {filename}")
    if display_mod:
        print(star)
    base_title = star.title_text if star.title_text else "OLS regressions of additions and removals."
    
    
    # Handle CI and if so append this to title
    if add_ci:
        star.show_confidence_intervals(True)
    ci_string = " 95\% CIs in parentheses." if add_ci else ""
    star.title(base_title + ci_string)
    
    # Set table lable based on filename
    table_label = filename.split("/")[-1].replace(".tex", "")
    star.table_label = table_label
    
    # Stargazer adds "T." to factor variables which looks ugly so I remove these
    # Also, latex does not like underscores unless you're in math mode so remove too
    latex_content = star.render_latex().replace("_", "")
    latex_content = latex_content.replace("T.", "")
    
    with open(filename, "w") as tex_file:
        tex_file.write(latex_content)
    
    print(f"Processed LaTeX saved to {filename}")
    return star




#####################################################################
# PLOTTING
#####################################################################

def make_aesthetic(hex_color_list=None, with_gridlines=False, bold_title=False, save_transparent=False, font_scale=2):
    """Make Seaborn look clean and add space between title and plot"""
    
    # Note: To make some parts of title bold and others not:
    # plt.title(r$'\bf{bolded title}$\nAnd a non-bold subtitle')
    
    sns.set(style='white', context='paper', font_scale=font_scale)
    if not hex_color_list:
        hex_color_list = [
        "#89DAFF", # Pale azure
        "#D41876", # Telemagenta
        "#00A896", # Persian green,
        "#826AED", # Medium slate blue
        "#F7B2AD", # Melon
        "#342E37", # Dark grayish-purple
        "#7DCD85", # Emerald
        "#E87461", # Medium-bright orange
        "#E3B505", # Saffron
        "#2C3531", # Dark charcoal gray with a green undertone
        "#D4B2D8", # Pink lavender
        "#7E6551", # Coyote
        "#F45B69", # Vibrant pinkish-red
        "#020887", # Phthalo Blue
        "#F18805"  # Tangerine
        ]
    sns.set_palette(sns.color_palette(hex_color_list))
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        pass
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    if bold_title:
        plt.rcParams['axes.titleweight'] = 'bold'
    else:
        plt.rcParams['axes.titleweight'] = 'regular'
    plt.rcParams['axes.grid'] = with_gridlines
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['savefig.transparent'] = save_transparent
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['axes.titlepad'] = 20*(font_scale/1)
    return hex_color_list

    


#####################################################################
# STATISTICS
#####################################################################


def pretty_print_desc_stats(data, n_bootstrap=10000, ci=False, ci_level=0.95, n_digits=2, seed=42):
    """
    Calculate descriptive statistics and print a LaTeX string in APA format.

    Args:
        data (array-like): Array of data to calculate statistics on.
        n_bootstrap (int, optional): Number of bootstrap samples. Default is 10000.
        ci (bool, optional): Whether to include confidence intervals. Default is False.
        ci_level (float, optional): Confidence interval level if ci is True. Default is 0.95.
        n_digits (int, optional): Number of digits to round the values to. Default is 2.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        str: A formatted LaTeX string with the mean, median, and standard deviation,
             and optionally the confidence interval.
    """
    data = np.array(data)
    
    mean = np.mean(data)
    median = np.median(data)
    sd = np.std(data, ddof=1)
    
    if ci:
        bootstrap_results = bootstrap(
            (data,),
            np.mean,
            n_resamples=n_bootstrap,
            random_state=seed,
            confidence_level=ci_level
        )
        ci_lower, ci_upper = bootstrap_results.confidence_interval
        ci_str = f", {int(ci_level*100)}\\% \\text{{CI}} = [{ci_lower:.{n_digits}f}, {ci_upper:.{n_digits}f}]"
    else:
        ci_str = ""
    
    mean_str = f"{mean:.{n_digits}f}"
    median_str = f"{median:.{n_digits}f}"
    sd_str = f"{sd:.{n_digits}f}"

    latex_string = f"$M = {mean_str}, Mdn = {median_str}, SD = {sd_str}{ci_str}$"

    return latex_string


def bootstrap_mean(data, n_bootstrap=10000, ci=95, seed=42):
    """
    Generate bootstrap confidence interval for the mean of the input data using scipy.stats.bootstrap.
    
    Args:
        data: The input data. Can be a Pandas Series, list, or numpy array.
        n_bootstrap: The number of bootstrap samples to generate. Default is 10000.
        ci: The confidence interval percentage. Default is 95%.
        seed: The random seed for reproducibility. Default is 42.
    
    Returns:
        A dict with keys 'mean', 'lower', and 'upper'
    
    Raises:
        ValueError: If the input data is not a recognized type or is empty.
    """
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass  # Already a numpy array
    else:
        raise ValueError("Input data must be a Pandas Series, list, or numpy array")

    if data.size == 0:
        raise ValueError("Input data is empty")

    data = data.ravel()

    data_mean = np.mean(data)

    result = bootstrap(
        (data,),
        np.mean,
        n_resamples=n_bootstrap,
        random_state=seed,
        confidence_level=ci/100,
        method='percentile'
    )

    return {
        'mean': data_mean,
        'lower': result.confidence_interval.low,
        'upper': result.confidence_interval.high
    }
