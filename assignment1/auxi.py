#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
import cifar10_utils
cifar10 = cifar10_utils.get_cifar10('data/')
#cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
 #                                               return_numpy=True)


#%%
dir(cifar10["test"])
#%%
#%%
x = np.arange(10*10).reshape(10,10)
#%%
plt.style.use('seaborn-v0_8')

fig = plt.figure()
ax = plt.gca()
im = ax.matshow(x, cmap='cividis')
fig.colorbar(im)
ax.set_ylabel("True Classes")
ax.set_xlabel("Predicted Classes")
ax.set_xticks(np.arange(10))
ax.set_xticklabels(cifar10["test"].classes, rotation = 90)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(cifar10["test"].classes)
ax.set_title("Confusion matrix")
#ax.tight_layout()
ax.xaxis.set_ticks_position('bottom')
plt.grid(False)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        plt.text(j, i, str(x[i, j]), va='center', ha='center', color='white')
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Generating three lists of 10 values
list1 = np.random.randint(1, 10, 10)
list2 = np.random.randint(1, 10, 10)
list3 = np.random.randint(1, 10, 10)

# Setting up the figure and axis
fig, ax = plt.subplots()

# Number of categories
categories = 10

# Bar width
bar_width = 0.2

# X-axis positions for each category
x_positions = np.arange(categories)

# Plotting bars for each list
ax.bar(x_positions - bar_width, list1, width=bar_width, label='F-beta = 0.1')
ax.bar(x_positions, list2, width=bar_width, label='F-beta = 1')
ax.bar(x_positions + bar_width, list3, width=bar_width, label='F-beta = 10')

# Adding labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Plot with 30 Bars (3 for Each Category)')

# Adding legend
ax.legend()

# Show the plot
plt.show()

# %%

# %%
