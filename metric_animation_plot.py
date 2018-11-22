# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/11/6
@description: 用于绘制实时图,并进行T分布检验（双侧）
"""
import matplotlib.pyplot as plt
import  matplotlib.animation as animation
from scipy import stats
import numpy as np


class metric_animation_plot(object):

    def __init__(self):
        self.fig = plt.figure()
        self.px = self.fig.add_subplot(3,1,1)
        self.rx = self.fig.add_subplot(3,1,2)
        self.fx = self.fig.add_subplot(3,1,3)

    def animate(self,i):
        x = []
        p = []
        r = []
        f = []
        with open('resource/result/metric.txt','r',encoding='utf-8')as fr:
            data = fr.readlines()
        for d in data:
            x.append(float(d.split()[0]))
            p.append(float(d.split()[1]))
            r.append(float(d.split()[2]))
            f.append(float(d.split()[3]))
        interval_p, interval_r, interval_f = 0.0,0.0,0.0
        if len(p) > 10:
            interval_p, interval_r, interval_f = self.T_test(p,r,f)
        self.px.clear()
        self.rx.clear()
        self.fx.clear()
        self.px.plot(x,p)
        self.rx.plot(x,r)
        self.fx.plot(x,f)
        self.px.set_title('Precision    Confidence Interval:{:.2f}%-{:.2f}%'.format(interval_p[0]*100,interval_p[1]*100))
        self.rx.set_title('Recall    Confidence Interval:{:.2f}%-{:.2f}%'.format(interval_r[0]*100,interval_r[1]*100))
        self.fx.set_title('F1-Measure    Confidence Interval:{:.2f}%-{:.2f}%'.format(interval_f[0]*100,interval_f[1]*100))
        plt.savefig('resource/result/metric.jpg')

    def T_test(self,p,r,f):
        p_test = np.array(p[-10:])
        mean_p = p_test.mean()
        std_p = p_test.std()
        interval_p = stats.t.interval(0.95,len(p_test)-1,mean_p,std_p)

        r_test = np.array(r[-10:])
        mean_r = r_test.mean()
        std_r = r_test.std()
        interval_r = stats.t.interval(0.95,len(r_test)-1,mean_r,std_r)

        f_test = np.array(f[-10:])
        mean_f = f_test.mean()
        std_f = f_test.std()
        interval_f = stats.t.interval(0.95,len(f_test)-1,mean_f,std_f)

        return interval_p,interval_r,interval_f

    def plot(self):
        ani = animation.FuncAnimation(self.fig,self.animate,interval=5000)
        plt.show()

if __name__ == '__main__':
    fig = metric_animation_plot()
    fig.plot()