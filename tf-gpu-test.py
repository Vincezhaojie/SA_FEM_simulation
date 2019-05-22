import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_excel('W_Form_Elementsize.xlsx')
# up = 50
# df = df[df['maxDisp(mm)'] < up]
# x = df['maxDisp(mm)']
# y1 = df['maxDisp(1508)']
# y2 = df['maxDisp(1005)']
# y3 = df['maxDisp(2508)']
# y4 = df['maxDisp(3008)']
#
# fig = plt.figure(figsize=(9, 8))
# E1005 = plt.plot(x.values, y2.values, 'o', alpha=0.4, label='Elementsgröße: 0,10mm')
# E1508 = plt.plot(x.values, y1.values, 'o', alpha=0.4, label='Elementsgröße: 0,15mm')
# E2010 = plt.plot([0, up], [0, up], '--', c=(0, 0, 0), label='angenommene richtige Werte (0,20mm)')
# E2508 = plt.plot(x.values, y3.values, 'o', alpha=0.4, label='Elementsgröße: 0,25mm')
# E3008 = plt.plot(x.values, y4.values, 'o', alpha=0.4, label='Elementsgröße: 0,30mm')
# #plt.plot([y_test.min(), y_test.max()], [y_test.min() + 0.05, y_test.max() + 0.05], '--', c=(0, 0, 0))
# #plt.plot([y_test.min(), y_test.max()], [y_test.min() - 0.05, y_test.max() - 0.05], '--', c=(0, 0, 0))
#
#
# plt.legend(loc='upper left')
# plt.xlabel('Werte bei Elementsgöße: 0.2mm')
# plt.ylabel('Werte bei gegebenen Elementsgrößen')
# plt.title('Einfluss der Elementsgröße')
# plt.show()

from pandas.plotting import scatter_matrix
df1 = pd.read_excel('W_Form_simulationDaten_1553758068644_clean.xlsx')
df2 = pd.read_excel('W_Form_simulationDaten_1553765907570_clean.xlsx')
df3 = pd.read_excel('W_Form_simulationDaten_1553765909540_clean.xlsx')
df4 = pd.read_excel('W_Form_simulationDaten_1553902548685_double_clean.xlsx')
df5 = pd.read_excel('W_Form_big_data_complecated_part1.xlsx')
df = pd.concat([df1, df2, df3, df4, df5])
df4 = df4.drop(columns=['maxDisp(mm)', 'maxStress(MPa)'])
df4.corr().to_excel('korrelation_koeffizient.xlsx', index=True)
#scatter_matrix(df4, alpha=0.2, figsize=(16, 16), diagonal='kde')
#plt.show()
