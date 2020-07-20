# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def load_nll(filename):
    m1s, m2s = [], []
    with open(filename) as f:
        for line in f:
            m1, m2 = line.strip().split('\t')
            m1s.append(-float(m1))
            m2s.append(-float(m2))
    return m1s, m2s

fig = plt.figure()

# beta-VAE one std

ax2 = fig.add_subplot("221")
m1, m2 = load_nll("plot/nll_1.txt")
ax2.hist(m1, 50, color="tab:green", alpha=0.5)
ax2.hist(m2, 50, color="tab:red", alpha=0.5)

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:red"]]
labels = [r"$\beta$-VAE", r"$\pm\sigma$"]
ax2.legend(handles, labels, loc='upper right')

ax2.set_ylabel('Number of Samples')
ax2.set_title('(A)', x=0.0, fontsize=12)

# beta-VAE two std

ax4 = fig.add_subplot("222")
m1, m2 = load_nll("plot/nll_2.txt")
ax4.hist(m1, 50, color="tab:green", alpha=0.5)
ax4.hist(m2, 50, color="tab:purple", alpha=0.5)

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:purple"]]
labels = [r"$\beta$-VAE", "$\pm2*\sigma$"]
ax4.legend(handles, labels, loc='upper right')

ax4.set_title('(B)', x=0.0, fontsize=12)

# beta-VAE extreme

ax6 = fig.add_subplot("223")
m1, m2 = load_nll("plot/nll_3.txt")
ax6.hist(m1, 50, color="tab:green", alpha=0.5)
ax6.hist(m2, 50, color="tab:pink", alpha=0.5)

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:pink"]]
labels = [r"$\beta$-VAE", "extremum"]
ax6.legend(handles, labels, loc='upper right')

ax6.set_xlabel('NLL of the Latent Codes')
ax6.set_ylabel('Number of Samples')
ax6.set_title('(C)', x=0.0, fontsize=12)

# CP-VAE

ax8 = fig.add_subplot("224")
m1, m2 = load_nll("plot/ours_nll.txt")
ax8.hist(m1, 50, color="tab:cyan", alpha=0.5)
ax8.hist(m2, 50, color="tab:orange", alpha=0.5)

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:cyan", "tab:orange"]]
labels = ["CP-VAE", "vertices"]
ax8.legend(handles, labels, loc='upper right')

ax8.set_xlabel('NLL of the Latent Codes')
ax8.set_title('(D)', x=0.0, fontsize=12)

plt.savefig('plot/nll_comparison.png')
