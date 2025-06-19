#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:43:37 2021

author: Alexandra John supervised by Ulrike Horn

tkinter_GUI_class

"""

import math
import tkinter as tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class interpolate_blinks_GUI:

    def __init__(self, samps_np, raw_data, interpolated, interval_size, sampling_frequency, save_path):
        self.root = tk.Tk()
        self.idx = 0
        self.interval_start = 0
        self.samps_np = samps_np
        self.raw_data = raw_data
        self.interpolated = interpolated
        self.manually_interpolated = np.zeros(len(self.samps_np))
        self.interval_size = interval_size
        self.sampling_frequency = sampling_frequency
        self.interval_end = self.interval_start + self.interval_size
        self.save_path = save_path
        self.idx_max = math.ceil((len(self.samps_np[:])) / self.interval_size) - 1
        self.blink_collector = []

        self.root.title("blink detection")
        self.root.geometry("1000x600")

        self.label1 = tk.Label(self.root, text="click on blinks and press interpolate")
        self.label1.pack()

        # matplotlib plot graph
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        x = np.arange(self.interval_start, self.interval_end)
        xticks = np.divide(x, self.sampling_frequency)  # to convert sampling rate in sec
        self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                     color='lightcoral')
        self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
        self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1),
                                        'upper right'))
        [self.ylim_min, self.ylim_max] = self.ax.get_ylim()

        # connect matplotlib figure to tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # connect mouse_click to fig
        self.canvas.mpl_connect("button_press_event", self.mouse_click)

        # button close
        self.button1 = tk.Button(self.root, text="close", command=lambda: self.closing())
        self.button1.pack(side="bottom")

        # button interpolate
        self.button2 = tk.Button(self.root, text="interpolate", command=lambda: self.interpolate_blink())
        self.button2.pack(side="top")

        # button next
        self.button3 = tk.Button(self.root, text="next", command=lambda: self.next_timeframe())
        self.button3.pack(side="right")

        # button back
        self.button4 = tk.Button(self.root, text="back", command=lambda: self.previous_timeframe())
        self.button4.pack(side="left")

        self.root.mainloop()

    def closing(self):
        combined_bool = np.logical_or(self.interpolated, self.manually_interpolated)
        df = pd.DataFrame({'interpolated_data': self.samps_np, 'raw_data': self.raw_data,
                           'interpolated_bool': combined_bool})
        df.to_csv(self.save_path)
        self.root.destroy()

    def mouse_click(self, event):

        global x_coord
        x_coord = event.xdata
        print("selected x coordinate:", x_coord.round(1))

        # collecting data points -> if more than 2 data points it only keeps the last two
        self.blink_collector.append(x_coord)
        if len(self.blink_collector) > 2:
            del self.blink_collector[0]

        # update plot
        self.fig.clear()
        x = np.arange(self.interval_start, self.interval_end)  # to make ongoing x axis
        xticks = np.divide(x, self.sampling_frequency)  # to convert in sec
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                     color='lightcoral')
        self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
        self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1), 'upper right'))
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        self.canvas.draw()

        for i, idx in enumerate(self.blink_collector[:]):
            # plot a red dot
            self.ax.scatter(self.blink_collector[i],
                            self.samps_np[int(self.blink_collector[i] * self.sampling_frequency)],
                            c='green', marker='.')
            self.ax.axvline(x=self.blink_collector[i], c='green', alpha=0.3)
            self.canvas.draw()

    def interpolate_blink(self):

        blink_start = int(self.blink_collector[0] * self.sampling_frequency)
        blink_end = int(self.blink_collector[1] * self.sampling_frequency)

        # interpolate between start and end-index
        blink_range = np.arange(blink_start, blink_end)
        interp = np.interp(x=blink_range, xp=[blink_start, blink_end],
                           fp=[self.samps_np[blink_start], self.samps_np[blink_end]])
        # print("interpolated:", interp)

        # replace blink range with interpolated values
        self.samps_np[blink_start:blink_end] = interp
        # and track which indices have been corrected
        self.manually_interpolated[blink_start:blink_end] = 1

        # update plot
        self.fig.clear()
        x = np.arange(self.interval_start, self.interval_end)  # to make ongoing x axis
        xticks = np.divide(x, self.sampling_frequency)  # to convert in sec
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                     color='lightcoral')
        self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
        self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1),
                                        'upper right'))
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        self.canvas.draw()

        self.blink_collector = []

        return self.samps_np

    def next_timeframe(self):
        # increase self.idx index and adapt interval_start and interval_end
        if self.idx < self.idx_max - 1:
            self.idx = self.idx + 1
            self.interval_start = self.idx * self.interval_size
            self.interval_end = self.interval_start + self.interval_size
            # plot next time interval
            self.fig.clf()
            x = np.arange(self.interval_start, self.interval_end)  # to make ongoing x axis
            xticks = np.divide(x, self.sampling_frequency)  # to convert in sec
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                         color='lightcoral')
            self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
            self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1), 'upper right'))
            [self.ylim_min, self.ylim_max] = self.ax.get_ylim()
            self.canvas.draw()
        elif self.idx < self.idx_max:  # last interval does not fit
            self.idx = self.idx + 1
            self.interval_start = self.idx * self.interval_size
            self.interval_end = len(self.samps_np[:])
            self.fig.clf()
            x = np.arange(self.interval_start, self.interval_end)  # to make ongoing x axis
            xticks = np.divide(x, self.sampling_frequency)  # to convert in sec
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                         color='lightcoral')
            self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
            self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1), 'upper right'))
            [self.ylim_min, self.ylim_max] = self.ax.get_ylim()
            self.canvas.draw()

        self.blink_collector = []

    def previous_timeframe(self):
        # decrease self.idx index and adapt interval_start and interval_end
        if self.idx > 0:  # only go back if possible
            self.idx = self.idx - 1
            self.interval_start = self.idx * self.interval_size
            self.interval_end = self.interval_start + self.interval_size
            # plot previous time interval
            self.fig.clf()
            x = np.arange(self.interval_start, self.interval_end)
            xticks = np.divide(x, self.sampling_frequency)  # to convert in sec
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(xticks, self.raw_data[self.interval_start:self.interval_end], linestyle='dashed',
                         color='lightcoral')
            self.ax.plot(xticks, self.samps_np[self.interval_start:self.interval_end], color='royalblue')
            self.ax.add_artist(AnchoredText("[{}/{}]".format(self.idx + 1, self.idx_max + 1), 'upper right'))
            [self.ylim_min, self.ylim_max] = self.ax.get_ylim()
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.blink_collector = []
