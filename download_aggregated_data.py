# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:43:16 2020

@author: Aristarchus
"""

import requests
import time
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H=".", **kwargs):
    

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",figsize=(12, 9),
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))
    axe.set_xlim(0.25, len(labels) + 0.75)
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel('Number of instances')
    axe.set_xlabel('Region')
    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    axe.set_xlim(-0.5, 5)
    return axe

#def adjacent_values(vals, q1, q3):
#    upper_adjacent_value = q3 + (q3 - q1) * 1.5
#    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
#
#    lower_adjacent_value = q1 - (q3 - q1) * 1.5
#    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
#    return lower_adjacent_value, upper_adjacent_value
#
#
#def set_axis_style(ax, labels):
#    ax.get_xaxis().set_tick_params(direction='out')
#    ax.xaxis.set_ticks_position('bottom')
#    ax.set_xticks(np.arange(1, len(labels) + 1))
#    ax.set_xticklabels(labels)
#    ax.set_xlim(0.25, len(labels) + 0.75)
#    ax.set_xlabel('Sample name')

myheader={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0)' + \
          ' Gecko/20100101 Firefox/73.0', \
          'referer': 'https://aemo.com.au/aemo/apps/visualisation/index.html'}

pathstr = ".\\scrapped_energy_market_new\\"
basestr = 'PRICE_AND_DEMAND_'
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
years = ['2014', '2015', '2016', '2017', '2018', '2019']
areas = ['QLD', 'VIC', 'NSW', 'TAS', 'SA']

#for year in years:
#    for month in months:
#        for area in areas:
#            time.sleep(5)
#            filestr = basestr + year + month + '_' + area + '.csv'
#            url = 'https://aemo.com.au/aemo/data/nem/priceanddemand/' \
#            + filestr
#            print(url)
#            myfile = requests.get(url, headers=myheader)
#            open(pathstr+filestr, 'wb').write(myfile.content)
#            
def createAreaDict():
    return {'QLD':[], 'VIC':[], 'NSW':[], 'TAS':[], 'SA':[]}
    
    


data_per_area = {'QLD':[], 'VIC':[], 'NSW':[], 'TAS':[], 'SA':[]}

data_per_year_per_area = {'2014':createAreaDict(), '2015':createAreaDict(), 
                          '2016':createAreaDict(), '2017':createAreaDict(),
                          '2018':createAreaDict(), '2019':createAreaDict()}
data_per_month_per_area = {'JAN':createAreaDict(), 'FEB':createAreaDict(),
                           'MAR':createAreaDict(), 'APR':createAreaDict(),
                           'MAY':createAreaDict(), 'JUN':createAreaDict(), 
                           'JUL':createAreaDict(), 'AUG':createAreaDict(), 
                           'SEP':createAreaDict(), 'OCT':createAreaDict(),
                           'NOV':createAreaDict(), 'DEC':createAreaDict()}

data_per_month_per_area2019 = {'JAN':createAreaDict(), 'FEB':createAreaDict(),
                           'MAR':createAreaDict(), 'APR':createAreaDict(),
                           'MAY':createAreaDict(), 'JUN':createAreaDict(), 
                           'JUL':createAreaDict(), 'AUG':createAreaDict(), 
                           'SEP':createAreaDict(), 'OCT':createAreaDict(),
                           'NOV':createAreaDict(), 'DEC':createAreaDict()}

data_per_half_hour_2019_JUNE = {'00:00':createAreaDict(), '00:30':createAreaDict(),
                           '01:00':createAreaDict(), '01:30':createAreaDict(),
                           '02:00':createAreaDict(), '02:30':createAreaDict(),
                           '03:00':createAreaDict(), '03:30':createAreaDict(), 
                           '04:00':createAreaDict(), '04:30':createAreaDict(),
                           '05:00':createAreaDict(), '05:30':createAreaDict(),
                           '06:00':createAreaDict(), '06:30':createAreaDict(),
                           '07:00':createAreaDict(), '07:30':createAreaDict(),
                           '08:00':createAreaDict(), '08:30':createAreaDict(),
                           '09:00':createAreaDict(), '09:30':createAreaDict(), 
                           '10:00':createAreaDict(), '10:30':createAreaDict(),
                           '11:00':createAreaDict(), '11:30':createAreaDict(),
                           '12:00':createAreaDict(), '12:30':createAreaDict(),
                           '13:00':createAreaDict(), '13:30':createAreaDict(),
                           '14:00':createAreaDict(), '14:30':createAreaDict(),
                           '15:00':createAreaDict(), '15:30':createAreaDict(), 
                           '16:00':createAreaDict(), '16:30':createAreaDict(),
                           '17:00':createAreaDict(), '17:30':createAreaDict(),
                           '18:00':createAreaDict(), '18:30':createAreaDict(),
                           '19:00':createAreaDict(), '19:30':createAreaDict(),
                           '20:00':createAreaDict(), '20:30':createAreaDict(),
                           '21:00':createAreaDict(), '21:30':createAreaDict(), 
                           '22:00':createAreaDict(), '22:30':createAreaDict(),
                           '23:00':createAreaDict(), '23:30':createAreaDict()}

solar_gen = {'00:00':[], '00:30':[],
           '01:00':[], '01:30':[],
           '02:00':[], '02:30':[],
           '03:00':[], '03:30':[], 
           '04:00':[], '04:30':[],
           '05:00':[], '05:30':[],
           '06:00':[], '06:30':[],
           '07:00':[], '07:30':[],
           '08:00':[], '08:30':[],
           '09:00':[], '09:30':[], 
           '10:00':[], '10:30':[],
           '11:00':[], '11:30':[],
           '12:00':[], '12:30':[],
           '13:00':[], '13:30':[],
           '14:00':[], '14:30':[],
           '15:00':[], '15:30':[], 
           '16:00':[], '16:30':[],
           '17:00':[], '17:30':[],
           '18:00':[], '18:30':[],
           '19:00':[], '19:30':[],
           '20:00':[], '20:30':[],
           '21:00':[], '21:30':[], 
           '22:00':[], '22:30':[],
           '23:00':[], '23:30':[]}

month_num_to_str = dict(zip(months, data_per_month_per_area.keys()))

for year in years:
    for month in months:
        for area in areas:
            with open(pathstr + basestr + year + month + '_' + area + '1.csv', newline='') as f:
                f_reader = csv.DictReader(f)
                curr_list = list(f_reader)
                rrp_list = [float(d['RRP']) for d in curr_list]
                data_per_year_per_area[year][area].extend(rrp_list)
                data_per_month_per_area[month_num_to_str[month]][area].extend(rrp_list)
                if year == '2019':
                    data_per_month_per_area2019[month_num_to_str[month]][area].extend(rrp_list)
                if year == '2019' and month == '06':
                    for key in data_per_half_hour_2019_JUNE.keys():
                        for d in curr_list:
                            if d['SETTLEMENTDATE'].find(key) != -1:
                                data_per_half_hour_2019_JUNE[key][area].append(float(d['RRP']))



with open(pathstr+'PSOLAR.csv', newline='') as f:
    f_reader = csv.DictReader(f, delimiter=';')
    sol_list = list(f_reader)
    
for key in solar_gen.keys():
    for d in sol_list:
        if d['DateTime'].find(key) != -1:
            solar_gen[key].append(float(d['Measurement']))
                
barWidth = 0.0625

barlist = []
r_list = []

fig = plt.figure()
ax = plt.subplot(111)

for month in months:
    bar = [np.mean(data_per_month_per_area2019[month_num_to_str[month]][area]) for area in areas]
    barlist.append(bar)
    
r_list.append(np.arange(len(barlist[0])))
for i in np.arange(len(barlist)-1):
    r_list.append(np.array([x + barWidth for x in r_list[i]]))
    
colorlist = ['#d7b776', '#d160bc', '#adaecc', '#040b35', '#f8b23b', '#185735', 
             '#12c5dc', '#eb1e55', '#02180b', '#de99a4', '#099d62', '#abbfd5']

for i in np.arange(len(months)):
    plt.bar(r_list[i], barlist[i], color=colorlist[i], width=barWidth, edgecolor='white', label=month_num_to_str[months[i]])

plt.title('Average spot price per month per area for 2019')    
plt.xlabel('Region')
plt.ylabel('Spot Price ($)')
plt.xticks([r + barWidth for r in range(len(barlist[0]))], areas)

chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
fig = plt.gcf()
fig.set_size_inches(15, 9)
plt.show()

price_up_5000_per_year_per_area = {key : {k:[float(price) for price in value if float(price) > 5000] \
                                              for (k, value) in data_per_year_per_area[key].items()} for key in years}
price_500_5000_per_year_per_area = {key : \
                    {k:[float(price) for price in value if float(price) > 500 and \
                    float(price) < 5000] for (k, value) in data_per_year_per_area[key].items()} for key in years}
price_250_500_per_year_per_area = {key : \
                    {k:[float(price) for price in value if float(price) > 250 and \
                    float(price) < 500] for (k, value) in data_per_year_per_area[key].items()} for key in years}

df_list = []

for year in years:    
    arr = np.asarray([[len(mylist) for mylist in price_250_500_per_year_per_area[year].values()],
    [len(mylist) for mylist in price_500_5000_per_year_per_area[year].values()],
    [len(mylist) for mylist in price_up_5000_per_year_per_area[year].values()]])
    arr = arr.T;
    df = pd.DataFrame(arr, index=areas, columns=["250\$ - 500\$", "500\$ - 5000\$", "more than 5000$"])
    df_list.append(df)

plot_clustered_stacked(df_list,years, title="Number of high-prices events per year per area")

#fig = plt.figure()
#ax = plt.subplot(111)
fig, ax1 = plt.subplots()
plt.title('AVERAGE SPOT PRICE and PV GENERATION vs TIME OF DAY')    

x = list(np.arange(1,49))
    
y = [np.mean(data_per_half_hour_2019_JUNE[k]['QLD']) for k in data_per_half_hour_2019_JUNE.keys()]

y2 = [np.mean(solar_gen[k]) for k in solar_gen.keys()]

err = [np.std(data_per_half_hour_2019_JUNE[k]['QLD']) for k in data_per_half_hour_2019_JUNE.keys()]

labels = list(data_per_half_hour_2019_JUNE.keys())

plt.xticks(x, labels, rotation='vertical')

color = 'tab:red'

ax1.set_xlabel('Time (HH:MM)')
ax1.set_ylabel('SPOT PRICE ($)', color=color)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('PV Generation', color=color) 
ax2.plot(x, y2, color=color)

ax2.tick_params(axis='y', labelcolor=color)

#plt.plot(x, y)
## You can specify a rotation for the tick labels in degrees or with keywords.
#plt.xticks(x, labels, rotation='vertical')
## Pad margins so that markers don't get clipped by the axes
#plt.margins(0.02)
## Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
fig = plt.gcf()
fig.set_size_inches(15, 9)
plt.show()
#violin_data = [np.array(mylist) for mylist in data_per_area.values()]
#
#viol_data = [sorted([float(in_str) for in_str in in_list]) for in_list in violin_data]
#
#fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharey=True)
#
##
#
#ax2.set_title('Customized violin plot')
#parts = ax2.violinplot(viol_data, bw_method='silverman',
#        showmeans=False, showmedians=False,
#        showextrema=False)
#
#for pc in parts['bodies']:
#    pc.set_facecolor('#545FDA')
#    pc.set_edgecolor('black')
#    pc.set_alpha(1)
#
#quartile1, medians, quartile3 = np.percentile(viol_data, [25, 50, 75], axis=1)
#whiskers = np.array([
#    adjacent_values(sorted_array, q1, q3)
#    for sorted_array, q1, q3 in zip(viol_data, quartile1, quartile3)])
#whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
#
#inds = np.arange(1, len(medians) + 1)
#ax2.scatter(inds, medians, marker='.', color='black', s=30, zorder=3)
#ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
##ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
#
## set style for the axes
#labels = data_per_area.keys()
##for ax in [ax1, ax2]:
##ax2.set_ylim([-100,500])
#set_axis_style(ax2, labels)
#
#plt.subplots_adjust(bottom=0.15, wspace=0.05)
#plt.show()                    