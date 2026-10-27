import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab1'))
from local_evidence import axes_for
from local_research import HOLDOUT


class LocalPlotTests(unittest.TestCase):
    def test_y_limits_follow_measured_values_before_zero_floor(self):
        fig,axes=axes_for('Test','Error, %')
        try:
            axes[0].plot([10,20,30],[10,70,125])
            axes[0].set_ylim(bottom=0)
            self.assertGreaterEqual(axes[0].get_ylim()[1],125)
        finally:
            plt.close(fig)

    def test_holdout_domain_contains_every_observed_size(self):
        fig,axes=axes_for('Test','Error, %',grid=HOLDOUT)
        try:
            for ax,sizes in zip(axes,HOLDOUT.values(),strict=True):
                lo,hi=ax.get_xlim()
                self.assertLessEqual(lo,min(sizes))
                self.assertGreaterEqual(hi,max(sizes))
        finally:
            plt.close(fig)

class UnifiedChartTests(unittest.TestCase):
    def test_all_three_classes_share_one_axis(self):
        from local_evidence import unified_axes
        fig,ax=unified_axes('Error, %')
        try:
            self.assertEqual(len(fig.axes),1)
            self.assertEqual(len(ax.patches),3)
            self.assertLess(ax.get_xlim()[0],0)
            self.assertGreater(ax.get_xlim()[1],3)
        finally:
            plt.close(fig)

    def test_smooth_connector_passes_through_real_values_without_overshoot(self):
        import numpy as np
        from local_evidence import smooth_points
        x=np.array([.1,.7,1.2,1.8,2.2,3.])
        y=np.array([0.,25.,30.,20.,7.,2.])
        xx,yy=smooth_points(x,y)
        self.assertGreater(len(xx),len(x))
        for actual_x,actual_y in zip(x,y,strict=True):
            self.assertAlmostEqual(float(yy[np.argmin(abs(xx-actual_x))]),actual_y)
        self.assertGreaterEqual(float(min(yy)),float(min(y)))
        self.assertLessEqual(float(max(yy)),float(max(y)))
