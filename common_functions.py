import numpy as np
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math
from scipy.integrate import quad

# Math font for every figure; set here because every figure script imports this module.
# The "custom" fontset lets the three faces be picked independently: italic-bold STIX
# for variables, DejaVu Sans for numerals and upright symbols so they match the axis
# labels, and DejaVu Sans Bold for $\bf{...}$ so mathtext panel letters sit on the same
# glyphs as the titles set with fontweight="bold".
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.it"] = "STIXGeneral:italic:bold"
plt.rcParams["mathtext.rm"] = "DejaVu Sans"
plt.rcParams["mathtext.bf"] = "DejaVu Sans:bold"
plt.rcParams["mathtext.sf"] = "DejaVu Sans"
# unused, but the fontset's defaults point at uninstalled families and log a
# findfont warning on every render
plt.rcParams["mathtext.cal"] = "STIXGeneral:italic"
plt.rcParams["mathtext.tt"] = "DejaVu Sans Mono"

colorA = np.array([0.8,0.8,0.8])
colorA_face = colorA/2+1/2
colorA_edge = colorA

colorB = np.array([0.6,0.6,0.6])
colorB_face = colorB/2+1/2
colorB_edge = colorB

color1 = np.array([187, 221, 255])/256
color1_face = color1/2+1/2
color1_edge = color1/2

color2 = np.array([255, 229, 187])/256
color2_face = color2/2+1/2
color2_edge = color2/2

color3 = np.array([206, 255, 187])/256
color3_face = color3/2+1/2
color3_edge = color3/2

def color_by_both(sigma2,shift):
    color='k'
    if np.isclose(sigma2,80,rtol=0.1):
        light = True
    else:
        light = False
    
    if np.isclose(shift,50,rtol=0.1):
        if light:
            color = 'lightcoral'
        else:
            color = 'firebrick'

    elif np.isclose(shift,80,rtol=0.1):
        if light:
            color = 'purple'
        else:
            color = 'cornflowerblue'

    return color

def fixation_probability(x, a,shift,Va):
    change_in_x = shift*a*x*(1-x)/Va
    x_new = x + change_in_x
    return fixation_probability_steady_state(x=x,a=a)
    
def fixation_probability_steady_state(x, a):
    return (erf(a/2)-erf(a/2*(1-2*x)))/(2*erf(a/2))


def make_figure_square(fig_function, fig_width=6., fig_height=6., *args, **kwargs):
    fig, fig_width, fig_height, width_inches, height_inches = fig_function(fig_width=fig_width,fig_height=fig_height,*args, **kwargs)
    
    if not np.isclose(width_inches, height_inches, rtol=0.05):
        
        new_fig_height = width_inches / height_inches * fig_height
        plt.close(fig)
        return make_figure_square(fig_function=fig_function, fig_width=fig_width, fig_height=new_fig_height, *args, **kwargs)
    else:
        return fig, fig_width, fig_height, width_inches, height_inches
    
def make_figure_set_width(fig, filename, target_width_inches=6.5, dpi=400, max_iter=10,save=True):
    """Save an already-built `fig` rescaled so its tight bbox is `target_width_inches` wide.

    The whole figure is scaled uniformly (preserving aspect), so every figure that goes
    through this comes out at the same width. Text/legend sizes are in points and do not
    scale with the canvas, so the tight width is not exactly linear in the figure size;
    a few iterations converge. Use in place of fig.savefig(..., bbox_inches="tight").
    """
    for _ in range(max_iter):
        fig.canvas.draw()
        tight_width_inches = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if np.isclose(tight_width_inches, target_width_inches, rtol=0.005):
            break
        scale = target_width_inches / tight_width_inches
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scale, h * scale)
    if save:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    else:
        return fig

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        self.__Characters = []
        for c in text:
            if c == ' ':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    # overloaded so the characters follow on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """Draw nothing; just update the characters' positions and angles."""
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """Update positions and rotations of the individual text elements."""
        # aspect ratio, per https://stackoverflow.com/a/42014041/2454357
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        figW, figH = self.axes.get_figure().get_size_inches()
        _, _, w, h = self.axes.get_position().bounds
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            if ir == il:
                ir += 1

            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            # character position, interpolated between the two points
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            # offset for the vertical alignment, in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            drp = np.dot(dr,rot_mat)

            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            rel_pos += w-used



