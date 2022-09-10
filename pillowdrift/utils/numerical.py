import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp


def ksmirnovtest(values_reference, values_current, threshold=0.95):
    _, pvalue = ks_2samp(values_reference, values_current)
    if pvalue < 1 - threshold:
        verdict = 'detected'
    else:
        verdict = 'not detected'

    return verdict, round(pvalue, 5)

def continuous_data(data_reference, data_current, columns, config):
    continuous = config['model']['variables']['numerical']
    num_elements = []
    for val_ref, val_cur, col in zip(data_reference, data_current, columns):
        if col in continuous:
            element = (col, val_ref, val_cur, 'continuous')
            num_elements.append(element)

    return num_elements


def estimate_density(vector, u):
    scipy_kernel = gaussian_kde(vector)
    v = scipy_kernel.evaluate(u)
    return list(v)


def numerical_distribution_sampler(numerical_elements):
    new_numerical_elements = []
    for element in numerical_elements:
        name = element[0]
        val_ref = element[1]
        val_cur = element[2]

        # Compute the KS test, retrieve the p-value and the verdict
        verdict, pvalue = ksmirnovtest(val_ref, val_cur)
        
        # Compute the data distributions
        u = np.linspace(min(val_ref), max(val_ref), 1000)
        val_ref = estimate_density(np.array(val_ref), u)
        val_cur = estimate_density(np.array(val_cur), u)
        u = [round(element, 3) for element in u]
        
        name = 'Variable: {} <br> Drift: {} <br> P-value: {}'.format(
            name, verdict, pvalue)

        new_numerical_elements.append(
            (name, u, val_ref, val_cur, 'continuous'))
    return new_numerical_elements
