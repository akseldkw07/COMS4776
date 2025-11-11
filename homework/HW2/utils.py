import numpy as np
import matplotlib.pyplot as plt


def save_loss_comparison_by_dataset(results):
    """Plot comparison of training and validation loss curves from all four
    runs in Part 3, comparing by dataset while holding hidden size constant.

    Arguments:
        results: Dictionary with keys (hidden_size, dataset_name) and values
                containing 'train_loss' and 'val_loss' lists.
    """
    # Extract the configurations
    configs = list(results.keys())
    
    # Group by hidden size
    hidden_sizes = sorted(set(h for h, _ in configs))
    
    # For each hidden size, get the two datasets
    groups = {}
    for h in hidden_sizes:
        groups[h] = [(h, d) for h, d in configs if h == hidden_sizes[0] or h == hidden_sizes[1]]
    
    # Get the four models: (h1, d1), (h1, d2), (h2, d1), (h2, d2)
    h1, h2 = hidden_sizes[0], hidden_sizes[1]
    datasets = sorted(set(d for _, d in configs))
    d1, d2 = datasets[0], datasets[1]
    
    l1 = results[(h1, d1)]
    l2 = results[(h1, d2)]
    l3 = results[(h2, d1)]
    l4 = results[(h2, d2)]

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].plot(range(len(l1['train_loss'])), l1['train_loss'], label="ds=" + d1)
    ax[0][0].plot(range(len(l2['train_loss'])), l2['train_loss'], label="ds=" + d2)
    ax[0][0].title.set_text("Train Loss | Model Hidden Size = {}".format(h1))

    ax[0][1].plot(range(len(l1['val_loss'])), l1['val_loss'], label="ds=" + d1)
    ax[0][1].plot(range(len(l2['val_loss'])), l2['val_loss'], label="ds=" + d2)
    ax[0][1].title.set_text("Val Loss | Model Hidden Size = {}".format(h1))

    ax[1][0].plot(range(len(l3['train_loss'])), l3['train_loss'], label="ds=" + d1)
    ax[1][0].plot(range(len(l4['train_loss'])), l4['train_loss'], label="ds=" + d2)
    ax[1][0].title.set_text("Train Loss | Model Hidden Size = {}".format(h2))

    ax[1][1].plot(range(len(l3['val_loss'])), l3['val_loss'], label="ds=" + d1)
    ax[1][1].plot(range(len(l4['val_loss'])), l4['val_loss'], label="ds=" + d2)
    ax[1][1].title.set_text("Val Loss | Model Hidden Size = {}".format(h2))

    for i in range(2):
        ax[i][0].set_xlabel("Epochs", fontsize=10)
        ax[i][0].set_ylabel("Loss", fontsize=10)
        ax[i][1].set_xlabel("Epochs", fontsize=10)
        ax[i][1].set_ylabel("Loss", fontsize=10)
        ax[i][0].legend(loc="upper right")
        ax[i][1].legend(loc="upper right")

    fig.suptitle("Performance by Dataset Size", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig("./loss_plot_by_dataset.pdf")
    plt.show()


def save_loss_comparison_by_hidden(results):
    """Plot comparison of training and validation loss curves from all four
    runs in Part 3, comparing by hidden size while holding dataset constant.

    Arguments:
        results: Dictionary with keys (hidden_size, dataset_name) and values
                containing 'train_loss' and 'val_loss' lists.
    """
    # Extract the configurations
    configs = list(results.keys())
    
    # Get unique hidden sizes and datasets
    hidden_sizes = sorted(set(h for h, _ in configs))
    datasets = sorted(set(d for _, d in configs))
    
    h1, h2 = hidden_sizes[0], hidden_sizes[1]
    d1, d2 = datasets[0], datasets[1]
    
    l1 = results[(h1, d1)]
    l2 = results[(h2, d1)]
    l3 = results[(h1, d2)]
    l4 = results[(h2, d2)]

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].plot(range(len(l1['train_loss'])), l1['train_loss'], label="hid_size=" + str(h1))
    ax[0][0].plot(range(len(l2['train_loss'])), l2['train_loss'], label="hid_size=" + str(h2))
    ax[0][0].title.set_text("Train Loss | Dataset = " + d1)

    ax[0][1].plot(range(len(l1['val_loss'])), l1['val_loss'], label="hid_size=" + str(h1))
    ax[0][1].plot(range(len(l2['val_loss'])), l2['val_loss'], label="hid_size=" + str(h2))
    ax[0][1].title.set_text("Val Loss | Dataset = " + d1)

    ax[1][0].plot(range(len(l3['train_loss'])), l3['train_loss'], label="hid_size=" + str(h1))
    ax[1][0].plot(range(len(l4['train_loss'])), l4['train_loss'], label="hid_size=" + str(h2))
    ax[1][0].title.set_text("Train Loss | Dataset = " + d2)

    ax[1][1].plot(range(len(l3['val_loss'])), l3['val_loss'], label="hid_size=" + str(h1))
    ax[1][1].plot(range(len(l4['val_loss'])), l4['val_loss'], label="hid_size=" + str(h2))
    ax[1][1].title.set_text("Val Loss | Dataset = " + d2)

    for i in range(2):
        ax[i][0].set_xlabel("Epochs", fontsize=10)
        ax[i][0].set_ylabel("Loss", fontsize=10)
        ax[i][1].set_xlabel("Epochs", fontsize=10)
        ax[i][1].set_ylabel("Loss", fontsize=10)
        ax[i][0].legend(loc="upper right")
        ax[i][1].legend(loc="upper right")

    fig.suptitle("Performance by Hidden State Size", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig("./loss_plot_by_hidden.pdf")
    plt.show()



def plot_scaling_law(all_results):
    """Plot the scaling law showing validation loss vs FLOPs."""
    
    plt.figure(figsize=(8, 6))
    
    for parameters, results in all_results.items():
        flops = results['flops']
        val_loss = results['val_loss']
        
        # Plot the trajectory
        plt.plot(flops, val_loss, marker='o', label=f'params={parameters}', linewidth=2.5)
        
        # Mark the final point
        plt.scatter(flops[-1], val_loss[-1], s=100, zorder=5)
    
    plt.xlabel('FLOPs', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Scaling Law: Validation Loss vs Compute (FLOPs)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./scaling_law.pdf')
    plt.show()


def plot_scaling_law_poly(all_results):
    """Plot only the polynomial fit (degree 5) for validation loss vs FLOPs."""
    
    plt.figure(figsize=(8, 6))
    
    for parameters, results in all_results.items():
        flops = np.array(results['flops'])
        val_loss = np.array(results['val_loss'])
        poly_coeffs = np.polyfit(np.log10(flops), np.log10(val_loss), 5)
        poly_func = np.poly1d(poly_coeffs)
        flops_smooth = np.linspace(flops.min(), flops.max(), 500)
        val_loss_smooth = 10 ** poly_func(np.log10(flops_smooth))
        plt.plot(flops_smooth, val_loss_smooth, label=f'params={parameters}', linewidth=2.5)
    
    plt.xlabel('FLOPs', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Scaling Law: Validation Loss vs Compute (FLOPs)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./scaling_law_polyfit.pdf')
    plt.show()



def interpolate(loss, flops, target, deg=5):
  flops = np.array(flops)
  target = np.array(target)
  if target >= flops.min() and target <= flops.max():
        f = np.polyfit(flops, loss, deg=deg)
        return np.polyval(f, target)

  return None

def plot_isoflop(target_flops, colors, params_vs_loss, optimal_params):

  fig, ax = plt.subplots(figsize=(8, 6))


  for idx, (target, (parameters, val_loss), opt) in enumerate(zip(target_flops, params_vs_loss, optimal_params)):
      if len(parameters) > 2: 
        

        x = np.array(parameters)
        y = np.array(val_loss)

        ax.scatter(x, y, 
                  label=f'{target/1e9:.0f}0 GFlops',
                  color=colors[idx], 
                  s=80, 
                  zorder=3)
        
        p = np.polyfit(np.log10(x), y, 2)
        
        
        px = np.logspace(np.log10(x[0]), np.log10(x[-1]), 100)
        py = np.polyval(p, np.log10(px))
        ax.plot(px, py, color=colors[idx], linewidth=2, zorder=2)
        
        ax.axvline(x=opt, color=colors[idx], linestyle='--', alpha=0.5, zorder=1)
        
        print(f"{target/1e9:.2f} GFlops: Optimal params = {opt:,.0f}")

  ax.legend(loc="upper right", fontsize=10)
  ax.set_xscale('log')
  ax.set_xlabel('# Parameters', fontsize=12)
  ax.set_ylabel('Validation Loss', fontsize=12)
  ax.set_title('IsoFLOP Analysis: Optimal Model Size vs Compute Budget', fontsize=14)
  ax.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('./isoflop_analysis.pdf')
  plt.show()



def fit_linear_log(x, y):
    m, c = np.polyfit(np.log10(x), np.log10(y) , 1)
    return m, c
    
def plot_flops_params(target_flops, params):

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(target_flops, params)
    m, c = fit_linear_log(target_flops, params)
    print(f"y = {m}x + {c}")

    lx = np.logspace(np.log10(1e10), np.log10(1e15), 100)
    ly = 10**(np.log10(lx) * m + c)
    ax.plot(lx, ly, color = '#E41A1C')

    ax.legend(loc = "lower left")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FLOPs')
    plt.ylabel('# Parameters')
    plt.xlim()
    plt.title('Compute Optimal Models')
    plt.grid()


def plot_flops_tokens(target_flops, optimal_params, all_results):

  optimal_tokens = []
  for target, opt in zip(target_flops, optimal_params):
    parameters = []
    tokens = []

    for params, results in all_results.items():
      loss = interpolate(results['val_loss'], results['flops'], target, deg=4)
      toks = interpolate(results['tokens'], results['flops'], target, deg=4)

      if loss != None and tokens != None:
        parameters.append(params)
        tokens.append(toks)

    sorted_indices = np.argsort(np.array(parameters))
    parameters = np.array(parameters)[sorted_indices].tolist()
    tokens = np.array(tokens)[sorted_indices].tolist()
    
    optimal_tokens.append(np.interp(opt, parameters, tokens))

  fig, ax = plt.subplots(figsize=(6, 5))
  ax.scatter(target_flops, optimal_tokens)

  m, c = fit_linear_log(target_flops, optimal_tokens)
  print(f"y = {m}x + {c}")

  lx = np.logspace(np.log10(1e10), np.log10(1e15), 100)
  ly = 10**(np.log10(lx) * m + c)
  ax.plot(lx, ly, color = '#E41A1C')

  ax.legend(loc = "lower left")
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('FLOPs')
  plt.ylabel('# Tokens')
  plt.title('Compute Optimal Models')
  plt.grid()
