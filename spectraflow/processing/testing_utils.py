import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


# Plot similarity metrics
# Evaluation processes derived from pDeep2 (cum_plot.py)
def plot_metrics(metric_list, metrics, thresholds, bin=200, output_path=None, plot_display=True):
    fig, ax = plt.subplots()
    results = {}

    # Reverse the order of metric_list and metrics
    metric_list = metric_list[::-1]
    metrics = metrics[::-1]

    for metric_values, metrics in zip(metric_list, metrics):

        pcc = np.sort(metric_values)
        cum = np.array([(pcc > j / bin).mean() * 100 for j in range(bin + 1)])
        df = pd.DataFrame({'Spectral Similarity': np.linspace(0, 1, bin + 1), metrics: cum})
        results[f"{metrics}-AUC"] = cum.sum() / (bin + 1)
        results.update(similarity_statistics(metric_values, thresholds, metrics))

        if plot_display:
            df.plot(ax=ax, x='Spectral Similarity', y=metrics)

    # Save .eps file to output folder
    if output_path:
        ax.set_xlim([0, 1.05])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim([0, 105])
        ax.set_ylabel('Spectral Similarity > Thresholds (%)')
        plt.savefig(output_path)

    # Display in environment
    if plot_display:
        ax.set_xlim([0, 1.05])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim([0, 105])
        ax.set_ylabel('Spectral Similarity > Thresholds (%)')
        plt.show()

    return results


def similarity_statistics(metric_values, val_list, metrics=None):
    results = {}

    if isinstance(metric_values, (list, np.ndarray)):
        for val in val_list:
            if isinstance(metric_values, np.ndarray) and metric_values.dtype.kind == 'b':
                percent = (metric_values.astype(int) > val).mean()
            else:
                percent = (np.array(metric_values) > val).mean()
            results[f"{metrics}{int(100 * val)}"] = percent

            if metrics:
                print(f"{100 * percent:.1f}% {metrics}s > {val:.2f}")

        median = np.median(metric_values)
        results[metrics, 'Med'] = median

        if metrics:
            print(f"{median:.3f} median {metrics}")
    else:
        # Validation measures
        print("Invalid similarity data")

    return results


def model_evaluation(predict_groups, real_groups):
    pccs = []
    spcs = []
    coses = []
    kdts = []
    SAs = []

    for key, value in predict_groups.items():
        predict = value[-1]
        predict[predict < 1e-4] = 0
        real = real_groups[key][-1]
        ypred_seq = np.reshape(predict, (predict.shape[0], predict.shape[1] * predict.shape[2]), order='C')
        ytest_seq = np.reshape(real, (real.shape[0], real.shape[1] * real.shape[2]), order='C')

        current_pcc = []
        current_cos = []
        current_spc = []
        current_kdt = []
        current_sa = []

        for i in range(len(predict)):
            ypred = ypred_seq[i]
            ytest = ytest_seq[i]

            if np.all(ypred == ypred[0]) or np.all(ytest == ytest[0]):
                pcc = 0
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    pcc = np.corrcoef(ypred, ytest)[0, 1]

            cos = cosine_similarity(ypred, ytest)
            spc = spearmanr(ypred, ytest)[0]
            kdt = kendalltau(ypred, ytest)[0]
            sa = np.mean(np.abs(ypred - ytest))

            if not np.isnan(pcc):
                current_pcc.append(pcc)
            if not np.isnan(cos):
                current_cos.append(cos)
            if not np.isnan(spc):
                current_spc.append(spc)
            if not np.isnan(kdt):
                current_kdt.append(kdt)
            if not np.isnan(sa):
                current_sa.append(sa)

        if len(current_pcc) > 0:
            pccs.append(np.median(current_pcc))
        if len(current_cos) > 0:
            coses.append(np.median(current_cos))
        if len(current_spc) > 0:
            spcs.append(np.median(current_spc))
        if len(current_kdt) > 0:
            kdts.append(np.median(current_kdt))
        if len(current_sa) > 0:
            SAs.append(np.median(current_sa))

        print(
            'Peptide length: {}, Size: {}, Median values: pcc = {:.3f}, cos = {:.3f}, spc = {:.3f}, kdt = {:.3f}, SA = {:.3f}'.format(
                key, len(predict), np.median(current_pcc), np.median(current_cos), np.median(current_spc),
                np.median(current_kdt),
                np.median(current_sa)))

    pccs, coses, spcs, kdts, SAs = np.array(pccs), np.array(coses), np.array(spcs), np.array(kdts), np.array(SAs)
    out_median = "[Medians] Spec. Angle: {:.3f}, Cosine: {:.3f}, Pearson: {:.3f}, Spearman: {:.3f}, Kendall Tau: {:.3f}".format(
        np.nanmedian(SAs), np.nanmedian(coses), np.nanmedian(pccs), np.nanmedian(spcs), np.nanmedian(kdts))
    out_mean = "[Means] Spec. Angle: {:.3f}, Cosine: {:.3f}, Pearson: {:.3f}, Spearman: {:.3f}, Kendall Tau: {:.3f}".format(
        np.nanmean(SAs), np.nanmean(coses), np.nanmean(pccs), np.nanmean(spcs), np.nanmean(kdts))

    print(out_median)
    print(out_mean)
    return pccs, coses, spcs, kdts, SAs


# Calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))

    # Handle zero division by checking if norms are zero
    if norm_a == 0 or norm_b == 0:
        return 0

    similarity = dot_product / (norm_a * norm_b)
    return similarity


# Filter out warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
