# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:28:07 2019

@author: tpv
"""

from WoodProblemDefinition import Stock, Order1, Order2, Order3
from shapely.geometry import Point
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
import math
import shapely
from descartes import PolygonPatch
from shapely.ops import cascaded_union
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

from DEGL import DEGL

# Simple helper class for getting matplotlib patches from shapely polygons with different face colors
class PlotPatchHelper:
    # a colormap with 41 colors
    CMapColors = np.array([
            [0,0.447,0.741,1],
            [0.85,0.325,0.098,1],
            [0.929,0.694,0.125,1],
            [0.494,0.184,0.556,1],
            [0.466,0.674,0.188,1],
            [0.301,0.745,0.933,1],
            [0.635,0.078,0.184,1],
            [0.333333333,0.333333333,0,1],
            [0.333333333,0.666666667,0,1],
            [0.666666667,0.333333333,0,1],
            [0.666666667,0.666666667,0,1],
            [1,0.333333333,0,1],
            [1,0.666666667,0,1],
            [0,0.333333333,0.5,1],
            [0,0.666666667,0.5,1],
            [0,1,0.5,1],
            [0.333333333,0,0.5,1],
            [0.333333333,0.333333333,0.5,1],
            [0.333333333,0.666666667,0.5,1],
            [0.333333333,1,0.5,1],
            [0.666666667,0,0.5,1],
            [0.666666667,0.333333333,0.5,1],
            [0.666666667,0.666666667,0.5,1],
            [1,0,0.5,1],
            [1,0.333333333,0.5,1],
            [1,0.666666667,0.5,1],
            [1,1,0.5,1],
            [0,0.333333333,1,1],
            [0,0.666666667,1,1],
            [0,1,1,1],
            [0.333333333,0,1,1],
            [0.333333333,0.333333333,1,1],
            [0.333333333,0.666666667,1,1],
            [0.333333333,1,1,1],
            [0.666666667,0,1,1],
            [0.666666667,0.333333333,1,1],
            [0.666666667,0.666666667,1,1],
            [0.666666667,1,1,1],
            [1,0,1,1],
            [1,0.333333333,1,1],
            [1,0.666666667,1,1]
            ])
    
    
    # Alpha controls the opaqueness, Gamma how darker the edge line will be and LineWidth its weight
    def __init__(self, Gamma=1.3, Alpha=0.9, LineWidth=2.0):
        self.Counter = 0
        self.Gamma = Gamma          # darker edge color if Gamma>1 -> faceColor ** Gamma; use np.inf for black
        self.Alpha = Alpha          # opaqueness level (1-transparency)
        self.LineWidth = LineWidth  # edge weight
    
    # circles through the colormap and returns the FaceColor and the EdgeColor (as FaceColor^Gamma)
    def nextcolor(self):
        col = self.CMapColors[self.Counter,:].copy()
        self.Counter = (self.Counter+1) % self.CMapColors.shape[0]
        return (col, col**self.Gamma)
    
    # returns a list of matplotlib.patches.PathPatch from the provided shapely polygons, using descartes; a list is 
    # returned even for a single polygon for common handling
    def get_patches(self, poly):
        if not isinstance(poly, list): # single polygon, make it a one element list for common handling
            poly = [poly]
        patchList = []
        for p in poly:
            fCol, eCol = self.nextcolor()
            patchList.append(PolygonPatch(p, alpha=self.Alpha, FaceColor=fCol, EdgeColor=eCol, 
                                          LineWidth=self.LineWidth))        
        return patchList


# Plots one or more shapely polygons in the provided axes ax. The named parameter values **kwargs are passed into
# PlotPatchHelper's constructor, e.g. you can write plotShapelyPoly(ax, poly, LineWidth=3, Alpha=1.0). Returns a list
# with the drawn patches objects even for a single polygon, for common handling
def plotShapelyPoly(ax, poly, **kwargs):
    return [ax.add_patch(p) for p in PlotPatchHelper(**kwargs).get_patches(poly)]



def ObjectiveFcn(particle,nVars,Stock,Order):
    """ MATLAB's peaks function -> objective (fitness function) """

    res=0
    newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),particle[j*3+2], origin='centroid') for j in range(len(Order))]
    remaining = Stock    
    #for i in range(0,len(w1)):
    #for p in newOrder:
    #    remaining = remaining.difference(p)
        #if(remaining.is_valid==False):
        #    print(remaining.is_valid)
        #    remaining.buffer(0)
        #if(remaining.is_empty==True):
        #    print("empty=%d"%remaining.is_empty)
            #remaining.buffer(0)
        #remaining = remaining.difference(p)
    
    unionNewOrder=shapely.ops.cascaded_union(newOrder)
    
    remaining = Stock.difference(unionNewOrder)
    # prevent errors in case that difference results in invalid or empty polygons
    if(remaining.is_valid==False):
        print(remaining.is_valid)
        remaining.buffer(0)
    if(remaining.is_empty==True):
        print("empty=%d"%remaining.is_empty)
        remaining.buffer(0)
            
    outOfStock=unionNewOrder.difference(Stock) # take newOrder out of stock - inverse of remaining
    
    areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
    overlapArea = areaSum-unionNewOrder.area
    
    dist_from_zero = sum([newOrder[i].area*(newOrder[i].centroid.x+newOrder[i].centroid.y) for i in range(0,len(newOrder))])
    ch= (remaining.convex_hull)
    lamda = (ch.area)/(remaining.area)-1
    alpha = 1.11
    fsm = 1/(1+alpha*lamda)

    res = outOfStock.area*1000 + overlapArea* 1000  +dist_from_zero*1 + 300*fsm

    return res


class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DEGL object and updates all elements in the figure.
        
        The figure has a top row of 1 subplots. This shows the best-so-far global finess value .
        The bottom row shows the global best-so-far solution achieved by the algorithm and the remaining current stock after placement.
    """
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution,, global fitness line,etc) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
         
        # figure
        self.fig = plt.figure()
        self.ax=[1,2,3]

        self.ax[0] = plt.subplot(211)

        self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
        self.lineBestFit, = self.ax[0].plot([], [])
        
        # auto-arrange subplots to avoid overlappings and show the plot
        # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
        self.ax[1] = plt.subplot(223)
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2] = plt.subplot(224)
        self.ax[2].set_title('Remaining after set difference')
        self.fig.tight_layout()

    
    def update(self, deglObject):
        """ Updates the figure in each iteration provided a DEGL object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        
        #Changes in plot in order to plot transformed order and remainings on every best fitness update
        if deglObject.Iteration == -1:
            xdata = np.arange(deglObject.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
       
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
        
        newOrder= deglObject.newOrder
        remaining = deglObject.remaining
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2].set_title('Remaining after set difference')
        pp = plotShapelyPoly(self.ax[1], [deglObject.Stock]+newOrder)
        pp[0].set_facecolor([1,1,1,1])
        plotShapelyPoly(self.ax[2], remaining)
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.ax[2].set_xlim(self.ax[1].get_xlim())
        self.ax[2].set_ylim(self.ax[1].get_ylim())
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def OutputFcn(deglObject, figObj):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        
        Always returns False (== don't stop the iterative process)
    """
    if deglObject.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.5f}'.format(deglObject.Iteration, deglObject.GlobalBestFitness))
    
    figObj.update(deglObject)
    
    return False



if __name__ == "__main__":
    # in case someone tries to run it from the command-line...
    plt.ion()
    #np.random.seed(113)
    start = time.time()
    # View the first stock together with pieces of the first order:
    # ------- PUT ALL ORDERS HERE------------------
    orderList = [Order1, Order2, Order3]
    numPolygons=sum([len(Order1), len(Order2), len(Order3)])
    
    finalList=[] # save order parts with result positions
    finalListPerStock=[] # save stock Index for the previous
    notFittedList=[] # list with polygons weren't fitted
    #orderList = [Order2, Order3]
    orderN = len(orderList)
    remaining = Stock.copy()
    remainingN= len(remaining)
    counter=0
    polygonsFitted=0
    iterationsList=[]
    while (orderList):
        fitted = 0 # flag indicates if order was placed 
        counter=counter+1
        currentOrder = orderList[0] # define current Order (it may be a part of order that was split)
        # save sum of areas of current order's parts
        currentOrderArea= sum([currentOrder[w].area for w in range(0,len(currentOrder))])
        #currentOrderArea = cascaded_union(currentOrder).area
        # save area of each remaining
        remainingsArea = np.array([remaining[k].area for k in range(0,len(remaining))])
        # [x,y,theta] for each part so 3* len(currentOrder)
        nVars = len(currentOrder) * 3

        # find which stocks (or remainings) have bigger area than the order's area calculated
        # and keep them as possible solutions
        # try to fit order in them starting by the smallest one
        bigEnough=(np.where(remainingsArea>currentOrderArea))[0]
        realIndexes = np.argsort(remainingsArea[bigEnough])
        bigEnough=bigEnough[realIndexes]
        print(bigEnough)
        for stockIndex in bigEnough:
            print("Try Stockindex=%d   -> OrderIndex=%d"% (stockIndex,counter))
            # set currentStock for degl the stocks-remainings from the local list
            currentStock = remaining[stockIndex]
            # Set lower and upper bounds for the 3 variables for each chromosome
            # as the bounds of stocks
            (minx, miny, maxx, maxy) = currentStock.bounds
            LowerBounds = np.ones(nVars)
            w1= [b for b in range(0,nVars,3)]
            w2= [b for b in range(1,nVars,3)]
            w3= [b for b in range(2,nVars,3)]
            LowerBounds[w1]= minx 
            LowerBounds[w2]= miny
            LowerBounds[w3]= 0
            
            UpperBounds = np.ones(nVars)
            UpperBounds[w1] = maxx
            UpperBounds[w2] = maxy
            UpperBounds[w3] = 90*4 # it can also work with 2 discrete values 0,90 in range {0,90}
            ##np.random.seed(13)
            
            figObj = FigureObjects(minx, maxx) # no need           
            outFun = lambda x: OutputFcn(x, figObj)
            
            deglObject = DEGL(ObjectiveFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                         OutputFcn=outFun, UseParallel=False, MaxStallIterations=15,Stock=currentStock,Order=currentOrder,remaining=currentStock,newOrder=currentOrder)
    
            deglObject.optimize()
            # get result and temporary apply the transformations
            resultPositions = deglObject.GlobalBestPosition
            newOrder = [ shapely.affinity.rotate( 
                    shapely.affinity.translate(currentOrder[k], xoff=resultPositions[k*3], yoff=resultPositions[k*3+1]), 
                    resultPositions[k*3+2], origin='centroid') for k in range(len(currentOrder))]
            iterationsList.append(deglObject.Iteration)
            #if (xwrese )
            # find if the current order was placed inside the currentStock
            # if area of difference of currentStock from union of order is bigger than a tolerance
            # go to next choice
            unionNewOrder=shapely.ops.cascaded_union(newOrder)
            difUnionNewOrder=unionNewOrder.difference(currentStock) # take newOrder out of stock - inverse of remaining
            if difUnionNewOrder.area >0.0001:
                continue
            
            # check if there is overlap and skip
            # overlap area is equal with sumOfArea - areaOfUnion
            areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
            difArea = areaSum-unionNewOrder.area
            if difArea > 0.0001:
                continue
            # Previous way to check overlap and out of stock area
            """isOut = 0
            existOverlap = 0
            check = [(newOrder[w].area-(currentStock.area -currentStock.difference(newOrder[w]).area)) <0.0001 for w in range(0,len(newOrder))]
            if False in check:
                isOut = 1
            
            if (isOut):
               continue 
            for i in range(0,len(newOrder)-1):
                check2 = [newOrder[i].intersection(newOrder[w]).area>0.0001 for w in range(i+1,len(newOrder))]
                if True in check2:
                    existOverlap=1
                    break
            if(existOverlap):
                continue
            """
            # this part of code is executed only if there is no overlap and no polygons out of stock and then it breaks the inner loop 
            fitted=1
            for p in newOrder:
                remaining[stockIndex] = remaining[stockIndex].difference(p)
            
            break
        #if polygons don't fit then split order in 2 parts
        if (fitted==0):

            if(int((len(currentOrder)/2))!=0):
                temp1=(currentOrder[0:int((len(currentOrder)/2))])
                temp2=(currentOrder[int((len(currentOrder)/2)):len(currentOrder)])
                orderList = [temp1]+[temp2] + orderList[1:]
            else:
                # if order contains only one polygon and cannot be fitted, it will add it to notFittedList and it will go to next order    
                notFittedList.append(orderList[0])
                orderList.remove(currentOrder)
                
        else:
            # if polygons of current order is fitted, then increase the number of fitted polygons, append the parts of order in finalList, append the stockIndex and remove the fitted order
            polygonsFitted=polygonsFitted+len(currentOrder)
            finalList.append( newOrder)
            finalListPerStock.append(stockIndex)
            orderList.remove(currentOrder)
            print("Fitted Stockindex=%d   -> OrderIndex=%d"% (stockIndex,counter))

    end = time.time()
    print("Time elapsed = %f"%(end - start))
    print("Polygons fitted=%d from %d polygon"%(polygonsFitted,numPolygons))
    print("\nNumber of Iterations (mean,min,max) = (%f,%f,%f)"%(np.mean(iterationsList),np.min(iterationsList),np.max(iterationsList)))
    # NOTE: the above operation is perhaps faster if we perform a cascade union first as below, check it on your code:
    #remaining = Stock[6].difference(shapely.ops.cascaded_union(newOrder))
    #Plot remainings
    ind=0
    fig, ax = plt.subplots(ncols=4,nrows=2, figsize=(16,9))
    fig.canvas.set_window_title('Remainings- Polygons fitted=%d from %d polygons'%(polygonsFitted,numPolygons))
    for i in range(0,len(Stock)):
        if i>=4:
            ind=1
        plotShapelyPoly(ax[ind][i%4], remaining[i])
        (minx, miny, maxx, maxy) = Stock[i].bounds
        ax[ind][i%4].set_ylim(bottom=miny,top=maxy)
        ax[ind][i%4].set_xlim(left=minx ,right=maxx)
        ax[ind][i%4].set_title('Remaining of Stock[%d]'%i)
        #ax[ind][i%4].relim()
        #ax[ind][i%4].autoscale_view()

    #Save figure with remainings
    import os
    file_name = "DEGLfull.png"
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".png")[0] + str(expand) + ".png"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
            
    fig.savefig(file_name)

    
    
