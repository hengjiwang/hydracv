import pickle
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

FILENAME = "Pre_Bisect_40x_4fps_ex3"
figx = pickle.load(open('../data/figures/sep_fluo_' + FILENAME + '.fig.pickle', 'rb'))
plt.show()