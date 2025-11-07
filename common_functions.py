import numpy as np
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math
from scipy.integrate import quad

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

def fixation_probability(x, a,shift,Va):
    change_in_x = shift*a*x*(1-x)/Va
    x_new = x + change_in_x
    return fixation_probability_steady_state(x=x,a=a)
    
def fixation_probability_steady_state(x, a):
    return (erf(a/2)-erf(a/2*(1-2*x)))/(2*erf(a/2))

def variance_star(a,x):
    return 2*a**2*x*(1-x)

def folded_sojourn_time(a,x,N):
    value = 2*np.exp(-variance_star(a,x)/2)/(1-x)
    if x < 1/(2*N):
        value *= 2*N
    else:
        value *= 1/x
    return value

def establishment_prob(S,shift,x,sign=1,N=5000):
    a = np.sqrt(S)*sign
    Sx = lambda x: a*(shift-a*(1/2-x))/(2*N)
    if Sx(x) < 0: return 0
    establishment_prob_l = lambda x: (1 - np.exp(-4*N*Sx(x)*x))/(1 - np.exp(-4*N*Sx(x)))
    return establishment_prob_l(x)

def make_figure_square(fig_function, fig_width=6., fig_height=6., *args, **kwargs):
    fig, fig_width, fig_height, width_inches, height_inches = fig_function(fig_width=fig_width,fig_height=fig_height,*args, **kwargs)
    
    # Check if width_inches and height_inches are close to each other
    if not np.isclose(width_inches, height_inches, rtol=0.05):
        
        # Convert back to figure height in inches
        new_fig_height = width_inches / height_inches * fig_height
        plt.close(fig)
        return make_figure_square(fig_function=fig_function, fig_width=fig_width, fig_height=new_fig_height, *args, **kwargs)
    else:
        return fig, fig_width, fig_height, width_inches, height_inches
    
def make_figure_set_width(fig_function, filename, target_width_inches=6., fig_width=6., fig_height=6., *args, **kwargs):
    fig, trash, fig_height, width_inches, height_inches = make_figure_square(fig_function=fig_function, fig_width=fig_width, fig_height=fig_height, *args, **kwargs)
    
    # Get the renderer for the figure canvas
    renderer = fig.canvas.get_renderer()

    # Get the tight bounding box in pixels
    tight_width_inches = fig.get_tightbbox(renderer).width

    if np.isclose(tight_width_inches,target_width_inches,rtol=0.01):
        plt.savefig(filename,dpi=300,bbox_inches='tight')
    else:
        plt.close(fig)
        new_fig_width = fig_width * target_width_inches / tight_width_inches
        print(f'Adjusting figure width from {fig_width:.2f} to {new_fig_width:.2f} inches to achieve target width of {target_width_inches:.2f} inches.')
        make_figure_set_width(fig_function=fig_function, filename=filename, target_width_inches=target_width_inches, fig_width=new_fig_width, fig_height=fig_height, *args, **kwargs)
        

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
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

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used



####### These are related to calculations of number of fixations and adaptation #######

def expected_change_in_frequency(a,N,D,x):
    return a/(2*N)*(D-a*(1/2-x))*x*(1-x)

def recursion_deterministic(a2,N,D,sigma2,x0):
    a = np.sqrt(a2)
    while x0 < 1/2:
        dx = expected_change_in_frequency(a=a,N=N,D=D,x=x0)
        if dx < 0:
            return 0
        else:
            x0 += dx
            D -= 2*a*dx+D*sigma2/(2*N)
    if x0 >= 1/2:
        return 1
    else:
        return 0
    
def calculate_min_freq_fix_seg(a2,N,shift,sigma2,min_guess,max_guess):
    
    current_guess = (max_guess+min_guess)/2
    if max_guess-min_guess < 1e-6:
        return current_guess
    else:
        fixed = recursion_deterministic(a2=a2,N=N,D=shift,sigma2=sigma2,x0=current_guess)
        if fixed:
            max_guess = current_guess
        else:
            min_guess = current_guess
        return calculate_min_freq_fix_seg(a2,N,shift,sigma2,min_guess,max_guess)

def calculate_min_shift_fix_new(a2,N,sigma2,min_guess,max_guess):
    
    current_guess = (max_guess+min_guess)/2
    if max_guess-min_guess < 1e-1:
        return current_guess
    else:
        fixed = recursion_deterministic(a2=a2,N=N,D=current_guess,sigma2=sigma2,x0=1/(2*N))
        if fixed:
            max_guess = current_guess
        else:
            min_guess = current_guess
        return calculate_min_shift_fix_new(a2,N,sigma2,min_guess,max_guess)

def n_fix_seg_given_a(a2,N,shift,sigma2,mode):
    a = np.sqrt(a2)
    if mode == 'fixation':
        g = lambda a,x: 1
    else:
        g = lambda a,x: 2*a*(1-x)
    x_c = calculate_min_freq_fix_seg(a2,N,shift,sigma2,min_guess=0,max_guess=1/2)
    n_fix = quad(lambda x: g(a=a,x=x)*folded_sojourn_time(a=np.sqrt(a),N=N,x=x)*establishment_prob(S=a2,shift=shift,x=x,sign=1,N=N),x_c,1/2,points=[1/(2*N)])[0]
    return n_fix

def n_fix_new_given_a(a2,N,shift,sigma2,mode):
    a = np.sqrt(a2)
    if mode == 'fixation':
        g = lambda a,x: 1
    else:
        g = lambda a,x: 2*a*(1-x)
    lambda_c = calculate_min_shift_fix_new(a2,N,sigma2,min_guess=0,max_guess=100)
    t_c = np.log(shift/lambda_c)*(2*N)/sigma2
    if t_c <= 0:
        return 0 
    else:
        return quad(lambda t: g(a=a,x=1/(2*N))*establishment_prob(S=a2,shift=shift*np.exp(-sigma2/(2*N)*t),x=1/(2*N),sign=1,N=N),0,t_c)[0]

