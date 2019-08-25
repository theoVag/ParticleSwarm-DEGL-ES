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

import shapely
from descartes import PolygonPatch
from shapely.ops import cascaded_union


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
    # prevent some errors of empty or invalid result
    remaining = Stock.difference(unionNewOrder)
    if(remaining.is_valid==False):
        print(remaining.is_valid)
        remaining.buffer(0)
    if(remaining.is_empty==True):
        remaining.buffer(0)
            
    if(unionNewOrder.is_valid==False):
        unionNewOrder.buffer(0)
    if(unionNewOrder.is_empty==True):
        unionNewOrder.buffer(0)
        
    outOfStock=unionNewOrder.difference(Stock) # take newOrder out of stock - inverse of remaining
    if(outOfStock.is_valid==False):
        print(outOfStock.is_valid)
        outOfStock.buffer(0)
    if(outOfStock.is_empty==True):
        outOfStock.buffer(0)
    
    areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
    overlapArea = areaSum-unionNewOrder.area
    
    dist_from_zero = sum([newOrder[i].area*(newOrder[i].centroid.x+newOrder[i].centroid.y) for i in range(0,len(newOrder))])
    ch= (remaining.convex_hull)
    lamda = (ch.area)/(remaining.area)-1
    alpha = 1.11
    fsm = 1/(1+alpha*lamda)

    res = outOfStock.area*1000 + overlapArea* 1000  +dist_from_zero*1 + 300*fsm #+ dist_sum*5

    return res


if __name__ == "__main__":
    # in case someone tries to run it from the command-line...
    plt.ion()
    from scipy.optimize import minimize, rosen, rosen_der
    from noisyopt import minimizeCompass
    ##np.random.seed(13)
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
        
        #remainingsArea = np.sort(remainingsArea)
        # find which stocks (or remainings) have bigger area than the order's area calculated
        # and keep them as possible solutions
        # try to fit order in them starting by the smallest one
        bigEnough=(np.where(remainingsArea>currentOrderArea))[0]
        realIndexes = np.argsort(remainingsArea[bigEnough])
        bigEnough=bigEnough[realIndexes]
        print(bigEnough)
        for stockIndex in bigEnough:
            print("Try Stockindex=%d   -> OrderIndex=%d"% (stockIndex,counter))
            # set currentStock for algorithm the stocks-remainings from the local list
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
            LowerBounds[w3]= 0 # mmporei na thelei mirror tis gwnies mesa stis 360
            
            UpperBounds = np.ones(nVars)
            UpperBounds[w1] = maxx
            UpperBounds[w2] = maxy
            UpperBounds[w3] = 90*4 # it can work also with 2 discrete values 0,90 in range {0,90}
            ##np.random.seed(13)
            
            # Initialization with acceptable values
            lbMatrix = LowerBounds
            ubMatrix = UpperBounds
            bRangeMatrix = ubMatrix - lbMatrix
            x0 = lbMatrix + np.random.rand(1,nVars) * bRangeMatrix #[0] isws    
            args2  = [nVars,currentStock,currentOrder]#Nelder-Mead      L-BFGS-B         SLSQP
            
            # Open the second comment to use noisyopt package
            res = minimize(ObjectiveFcn, x0,args=(nVars,currentStock,currentOrder),method='Nelder-Mead', tol=1e-6,options={'maxiter': 10000,'disp': True})
            #res = minimizeCompass(ObjectiveFcn, x0=x0[0], deltatol=0.1, paired=False,args=args2,errorcontrol=False)
            resultPositions = res.x
            
            newOrder = [ shapely.affinity.rotate( 
                    shapely.affinity.translate(currentOrder[k], xoff=resultPositions[k*3], yoff=resultPositions[k*3+1]), 
                    resultPositions[k*3+2], origin='centroid') for k in range(len(currentOrder))]
            
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
            
            # this part of code is executed only if there is no overlap and no polygons out of stock and then it breaks the inner loop 
            fitted=1
            for p in newOrder:
                remaining[stockIndex] = remaining[stockIndex].difference(p)
                if(remaining[stockIndex].is_valid==False):
                    print(remaining[stockIndex].is_valid)
                    remaining[stockIndex].buffer(0)
                if(remaining[stockIndex].is_empty==True):
                    print("empty=%d"%remaining[stockIndex].is_empty)
                    remaining[stockIndex].buffer(0)
            
            
                
            break
        #if polygons don't fit then split order in 2 parts
        if (fitted==0):
            #orderList.remove(currentOrder)
            
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
        ax[ind][i%4].set_title('Remaining of Stock[%d]'%i)
        ax[ind][i%4].relim()
        ax[ind][i%4].autoscale_view()

    
    #Save figure with remainings
    import os
    file_name = "nelderfull.png"
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

    
    
