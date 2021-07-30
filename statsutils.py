
import numpy
import itertools
import random
import pandas

from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats
from collections import defaultdict

class Combinations:
    """
    Creates a `Combinations` class. This class is required if the user want to
    limit the possible combinations
    """
    def __init__(self, samples, r=2, possible_combinations=[]):
        """
        Instantiates the `Combinations` class

        :param samples: A `list` of samples
        :param r: An `int` of the length of the tuple that is returned
        :param possible_combinations: A `list` of possible combinations
        """
        self.samples = samples
        if possible_combinations:
            self.possible_combinations = possible_combinations
        else:
            self.possible_combinations = list(itertools.combinations(range(len(samples)), r=r))
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.current >= len(self.possible_combinations):
            raise StopIteration
        self.current += 1
        return tuple(self.samples[c] for c in self.possible_combinations[self.current - 1])

    def __len__(self):
        return len(self.possible_combinations)

def bootstrap(values, reps=10000, method=None):
    """
    Bootstraps an array of values

    :param values: A `numpy.ndarray` with shape (N, ) OR (N, M)
    :param reps: An `int` of the number of repetions to sample
    :param method: A `function` used to compute the statistics. Defaults to `numpy.mean`

    :returns : A `numpy.ndarray` of the bootstrapped samples
    """
    choices = numpy.random.choice(len(values), size=(reps, len(values)), replace=True)
    if isinstance(method, type(None)):
        method = numpy.mean
    values = method(values[choices], axis=1)
    return values

def permute(samples, group_indexes):
    """
    Permutes a raveled sampled array and returns the new smaples

    :param samples: A `numpy.ndarray` with shape (N, )
    :param group_indexes: A `list` of group indexes

    :returns : A `numpy.ndarray` of the permuted samples
    """
    numpy.random.shuffle(samples)
    return numpy.array([samples[index] for index in group_indexes])

def resampling_F(samples, raveled_samples, group_indexes, permutations=10000):
    """
    Computes the F statistics using a resampling of samples

    :param samples: A `list` of sample
    :param raveled_samples: A `list` of all available samples
    :param group_indexes: A `list` of associated groups of raveled samples
    :param permutations: An `int` of the number of permutations

    :returns : A `float` of the calculated p value
    """
    gt_fstat, _ = stats.f_oneway(*samples)
    p_fstat = []
    for _ in range(permutations):
        tmp_samples = permute(raveled_samples, group_indexes)
        statistic, _  = stats.f_oneway(*tmp_samples)
        p_fstat.append(statistic)
    p_fstat = numpy.array(p_fstat)
    p_value = numpy.sum(p_fstat >= gt_fstat, axis=0) / permutations
    return p_value

def resampling_stats(samples, labels, raveled_samples=None, group_indexes=None, possible_combinations=[],
                     permutations=10000, show_ci=True, seed=42):
    """
    Computes the pair-wise comparisons of each sample in the list using a resampling
    statistical test

    :param samples: A `list` of sample
    :param labels: A `list` of associated labels of each samples
    :param raveled_samples: (Optional) A `list` of all available samples
    :param group_indexes: (Optional) A `list` of associated groups
    :param possible_combinations: (Optional) A `list` of possible combinations to restrict
                                  the post-hoc analysis.
    :param permutations: An `int` of the number of permutations
    :param show_ci: A `bool` wheter to plot the confidence interval
    :param seed: An `int` of the random seed

    :returns : A `list` of p-values for each comparisons
               A `float` of the F_p_value (None if len(samples) == 2)
    """
    # Sets the random seed
    random.seed(seed)
    numpy.random.seed(seed)

    # Calculates the raveled_samples and group_indexes
    if isinstance(raveled_samples, type(None)):
        raveled_samples, group_indexes = [], []
        current_count = 0
        for i, samp in enumerate(samples):
            raveled_samples.extend(samp)
            group_indexes.append(current_count + numpy.arange(len(samp)))
            current_count += len(samp)
    samples, raveled_samples, group_indexes = map(numpy.array, (samples, raveled_samples, group_indexes))

    # Resampled anova
    F_p_value = None
    if len(samples) > 2:
        F_p_value = resampling_F(samples, raveled_samples, group_indexes, permutations=permutations)
        if numpy.all(F_p_value > 0.05):
            print("Resampling F (pvalue : {})".format(F_p_value))
            return numpy.array([1. for _ in range(len(Combinations(labels, r=2)))]),\
                    F_p_value
        else:
            pass

    p_values = []

    if show_ci:
        possible_treatments = []
        for t1, t2 in Combinations(labels, r=2, possible_combinations=possible_combinations):
            possible_treatments.append([t1, t2])
        possible_treatments = set(map(tuple, possible_treatments))

        plot_info = {
            key : {
                "figax" : pyplot.subplots(tight_layout=True, figsize=(12, 3)),
                "current_count" : 0,
                "treatments" : []
            } for key in possible_treatments
        }
    for j, ((sample1, sample2), (treatment1, treatment2)) in enumerate(zip(tqdm(Combinations(samples, r=2, possible_combinations=possible_combinations)),\
                                                                    Combinations(labels, r=2, possible_combinations=possible_combinations))):

        gt_abs_diff = numpy.abs(numpy.mean(sample1, axis=0) - numpy.mean(sample2, axis=0))
        concatenated = numpy.concatenate((sample1, sample2), axis=0)
        p_abs_diff = []
        for _ in range(permutations):
            numpy.random.shuffle(concatenated)
            p_abs_diff.append(numpy.abs(numpy.mean(concatenated[:len(sample1)], axis=0) - numpy.mean(concatenated[len(sample1):], axis=0)))
        p_abs_diff = numpy.array(p_abs_diff)
        p_value = numpy.sum(p_abs_diff >= gt_abs_diff, axis=0) / permutations
        p_values.append(p_value)

        if show_ci:
            key = tuple([treatment1, treatment2])
            fig, ax = plot_info[key]["figax"]
            i = plot_info[key]["current_count"]
            plot_info[key]["treatments"].append(f"{treatment1}, {treatment2}")
            print('CI')
            print(plot_info[key]["treatments"], numpy.quantile(p_abs_diff, q=0.05),numpy.quantile(p_abs_diff, q=0.95))
            ax.bar(i, numpy.quantile(p_abs_diff, q=0.95), color="grey", alpha=0.7, width=1, zorder=3)
            ax.bar(i, gt_abs_diff, alpha=0.7, color="tab:blue", width=1, zorder=1)
            ax.set_xticks(numpy.arange(len(plot_info[key]["treatments"])))
            ax.set_xticklabels(plot_info[key]["treatments"], rotation=45, horizontalalignment="right")
            ax.set_ylim(0, 0.2)

            plot_info[key]["current_count"] += 1

    return p_values, F_p_value

def create_latex_table(pvalues, samples, formatted_labels, output_file=None):
    """
    Creates a latex table by using pandas as a backend

    :param pvalues: A `list` of pvalues
    :param samples: A `list` of all samples
    :param formatted_labels: A `list` of formatted labesl
    :param output_file: (Optional) A `str` path to the output_file

    :returns : A `str` of the created latex table
    """
    def formatter(x):
        try:
            x = float(x)
            if MINVAL == abs(x):
                formatted_x = "<\\SI{1.0000e-4}{}"
            else:
                formatted_x = "\\SI{{{:0.4e}}}{{}}".format(abs(x))
            if x < 0:
                if (x > -0.05):
                    return "\\textcolor[rgb]{{0.93,0.26,0.18}}{{{}}}".format(formatted_x)
                else :
                    return formatted_x
            else:
                if (x < 0.05):
                    return "\\textcolor[rgb]{{0.39,0.68,0.75}}{{{}}}".format(formatted_x)
                else:
                    return formatted_x
        except ValueError:
            return "-"
    MINVAL = 1e-9
    df = pandas.DataFrame(index=formatted_labels, columns=formatted_labels)
    for (row, col), (s1, s2), val in zip(Combinations(formatted_labels, r=2), Combinations(samples, r=2), pvalues):
        is_smaller = (numpy.mean(s1) - numpy.mean(s2)) < 0
        if is_smaller:
            df.loc[row, col] = "{}".format(-1 * max(val, MINVAL))
        else:
            df.loc[row, col] = "{}".format(max(val, MINVAL))

        is_smaller = (numpy.mean(s2) - numpy.mean(s1)) < 0
        if is_smaller:
            df.loc[[col], row] = "{}".format(-1 * max(val, MINVAL))
        else:
            df.loc[col, row] = "{}".format(max(val, MINVAL))

    with pandas.option_context("max_colwidth", 1000):
        out = df.to_latex(open(output_file, "w") if isinstance(output_file, str) else output_file,
                          formatters=[formatter] * len(formatted_labels),
                          na_rep="-", column_format="c" * (len(formatted_labels) + 1),
                          escape=False)
    return out

def cumming_plot(ctrl, samples, labels=None,looklist=None):
    """
    Creates a Cumming Plot from the ctrl and samples

    :param ctrl: A `numpy.ndarray` of ctrl data
    :param samples: A `list` of samples
    :param labels: (optional) A `list` of associated labels

    :returns : A `matplotlib.Figure` of the Cumming plot
               A `matplotlib.Axes` of the Cumming plot
    """
    if isinstance(ctrl, (list, tuple)):
        ctrl = numpy.array(ctrl)
    if isinstance(samples, (list, tuple)):
        samples = numpy.array(samples, dtype="object")

    bstrap_ctrl = bootstrap(ctrl)

    fig, ax = pyplot.subplots()
    
    data = []
    print(len(samples))
    for i, sample in enumerate(samples):
        
        diff = bootstrap(sample) - bstrap_ctrl
        parts=ax.violinplot(diff.astype(float), positions=[i], widths=0.9)
        
        if looklist != None:
            for pc in parts['bodies']:

                pc.set_facecolor(looklist[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.8)
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor('black')
                #vp.set_linelength(0.5)
                vp.set_linewidth(1)
        ax.scatter([i, i], numpy.quantile(diff, [0.05, 0.95]), marker="_",c='red',s=300,linewidths=2)
        ax.scatter(i*numpy.ones((1,len(sample))),sample-1,c=looklist[i],edgecolors='black',s=25)
    ax.axhline(y=0, color="black", linestyle="dashed")

    if isinstance(labels, (list, tuple)):
        ax.set(
            xticks=numpy.arange(len(labels)), xticklabels=labels
        )

    return fig, ax,parts


if __name__ == "__main__":

    # Example using resampling_stats
    labels = ["A", "B", "C"]
    samples = [numpy.random.normal(loc=i, scale=1, size=(15,)) for i in range(3)]
    p_value, F_p_value = resampling_stats(samples, labels, show_ci=False)
    print(p_value)
    table = create_latex_table(p_value, samples, labels)
    print(table)

    # Example using cumming plot
    ctrl = samples[0]
    samples = samples[1:]
    fig, ax = cumming_plot(ctrl, samples, labels=labels[1:])
    pyplot.show()
