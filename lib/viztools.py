import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import statsmodels 

def fit_posterior_sample(w_as, qx_as, degree=10, resolution=500):

    X = w_as.reshape(-1, 1)
    y = qx_as

    polyreg_scaled = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    _ = polyreg_scaled.fit(X, y)

    X_vals = np.linspace( np.min(X), np.max(X), resolution).reshape(-1, 1)
    y_hat = polyreg_scaled.predict(X_vals)

    return (X_vals[:,0], y_hat)


def create_joint_posterior_plot(w_as, qx_as, mode_precision=250, true_params=None):

    # fit the joint distribution mean
    (w_as_fit, qx_as_fit) = fit_posterior_sample(w_as, qx_as)

    # fit the kde
    post_kde = statsmodels.api.nonparametric.KDEMultivariate(\
        data=[w_as, qx_as], var_type='cc', bw='normal_reference')

    # search the kde for the posterior mode
    grid_x = np.linspace(np.min(w_as), np.max(w_as), mode_precision)
    grid_y = np.linspace(np.min(qx_as), np.max(qx_as), mode_precision)
    
    search_grid = np.meshgrid(grid_x, grid_y)
    search_grid = np.array([
        search_grid[0].flatten(),
        search_grid[1].flatten()
    ]).T

    # evaluate the kde (to find the mode)
    mode_search = post_kde.pdf(search_grid)
    mode_idx = np.argmax(mode_search)
    mode_approx_est = search_grid[mode_idx]
    approx_w_as, approx_qx_as = (
        round(mode_approx_est[0], 3), 
        round(mode_approx_est[1], 3))
    
    # create the figure
    _ = plt.figure(1)
    fig, ax = plt.subplots(figsize=(10,5))    

    # contour plot
    kde_plot_df = pd.DataFrame({
        "w_as": w_as,
        "qx_as": qx_as
    })
    _ = sns.kdeplot(kde_plot_df, x="w_as", y="qx_as",\
        color = (0.1, 0.3, 0.5, 0.5))

    # line plot (joint distribution)
    _ = sns.lineplot(x = w_as_fit, y=qx_as_fit,\
        label="joint distribution fit", color="orange")
    
    # mode point
    df_approx_mode = pd.DataFrame({
        "w_as": [approx_w_as],
        "qx_as": [approx_qx_as]
    })    
    mode_pt_label = f"mode: w_as={approx_w_as}, qx_as={approx_qx_as}"    
    _ = sns.scatterplot(df_approx_mode, x="w_as", y="qx_as",\
        color="black", s=150, marker='P', label=mode_pt_label)

    # true value point
    if true_params is not None:
        true_w_as, true_qx_as = true_params
        df_true = pd.DataFrame({
            "w_as": [true_w_as],
            "qx_as": [true_qx_as]})
        
        _ = sns.scatterplot(df_true, x="w_as", y="qx_as",\
                label="true_values: w_as=" + str(round(true_w_as, 3))+\
                " qx_as=" + str(round(true_qx_as, 3)), color="green", s=150, marker='P')

    _ = plt.legend()
    _ = plt.show()

    return {
        "joint_fit": (w_as_fit, qx_as_fit),
        "joint_kde": post_kde,
        "mode": (approx_w_as, approx_qx_as)
    }
    


    


