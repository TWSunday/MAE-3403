#Chat.gpt was used as a resource to help create this code
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_normal_distribution_and_probability(mean, std_dev, x_range, probability_condition, label):
    """
    This function takes five arguments and plots a probability curve relating to the arguments while shading in the
    portion under the curve that correlates to the probability of a random variable occurring within a given range.
    :param mean: Mean of the set which is normally distributed.
    :param std_dev: Standard deviation of the same set.
    :param x_range: Range of x values that we want to plot.
    :param probability_condition: Function defining the probability condition.
    :param label: Label for the distribution in the legend.
    :return: matlib plot with shaded functions created by the given arguments.
    """
    distribution = stats.norm(mean, std_dev)

    # Generate x values based on arguments for plotting
    x_values = np.linspace(x_range[0], x_range[1], 100)

    # Calculate the probability condition based on the given range, mean, and standard deviation
    probability = probability_condition(x_values, mean, std_dev)

    # Plot the normal distribution over the range given as argument
    plt.plot(x_values, distribution.pdf(x_values), label=f'{label}')

    # Fill the area under the curve based on the probability condition
    plt.fill_between(x_values, distribution.pdf(x_values), where=probability, alpha=1, label=f'P({probability_condition.__name__})')

# Initialize the plot
plt.figure(figsize=(10, 10))

# Plot the first normal distribution (N(0, 1))
plt.subplot(1,2,1)
plot_normal_distribution_and_probability(mean=0, std_dev=1, x_range=(-5, 5),
                                         probability_condition=lambda x_values, mean, std_dev: (x_values < 1),
                                         label='N(0, 1)')
# Create title, labels, annotations, etc.
plt.title('Normal Distributions and Probabilities')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
plt.legend()
plt.grid(True)
plt.text(-5, .35, r'$\mu=0,\ \sigma=1$')
plt.annotate('Probability x<1', xy=(-1, .05), xytext=(-5, .15),
             arrowprops=dict(facecolor='blue'),
             )


# Plot the second normal distribution (N(175, 3))
plt.subplot(1,2,2)
plot_normal_distribution_and_probability(mean=175, std_dev=3, x_range=(150,200),
                                         probability_condition=lambda x_values, mean, std_dev: (x_values > mean + 2 * std_dev),
                                         label='N(175, 3)')
# Create title, labels, annotations, etc.
plt.title('Normal Distributions and Probabilities')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
plt.legend()
plt.grid(True)
plt.text(150, .115, r'$\mu=175,\ \sigma=3$')
plt.annotate('Probability x>$\mu+2\sigma$', xy=(182, .006), xytext=(180, .04),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             )

# Show the plot
plt.show()


