# -*- coding: utf-8 -*-
"""
Particle swarm minimization algorithm with dynamic random neighborhood topology.

The DynNeighborPSO class implements a (somewhat) simplified version of the particleswarm algorithm from MATLAB's 
Global Optimization Toolbox.
"""
from random import randint, random
import numpy as np
import warnings
import shapely
import math
try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False


def extract_obj( degl ):
        
        chromosome = degl.GlobalBestPosition
        
        newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(degl.Order[j], xoff=chromosome[j*3], yoff=chromosome[j*3+1]),chromosome[j*3+2], origin='centroid') for j in range(len(degl.Order))]
        """remaining = pso.Stock    
        for i in range(0,len(newOrder)):
            for p in newOrder:
                remaining = remaining.difference(p)"""
        unionNewOrder=shapely.ops.cascaded_union(newOrder)
         
        remaining = (degl.Stock).difference(unionNewOrder)
        # prevent errors in case that difference results in invalid or empty polygons
        if(remaining.is_valid==False):
            #print(remaining.is_valid)
            remaining.buffer(0)
        if(remaining.is_empty==True):
            #print("empty=%d"%remaining.is_empty)
            remaining.buffer(0)
                
        if(unionNewOrder.is_valid==False):
            #print(unionNewOrder.is_valid)
            unionNewOrder.buffer(0)
        if(unionNewOrder.is_empty==True):
            #print("empty=%d"%difUnionNewOrder.is_empty)
            unionNewOrder.buffer(0)
            
        difUnionNewOrder=unionNewOrder.difference(degl.Stock) # take newOrder out of stock - inverse of remaining
        if(difUnionNewOrder.is_valid==False):
            #print(difUnionNewOrder.is_valid)
            difUnionNewOrder.buffer(0)
        if(difUnionNewOrder.is_empty==True):
            #print("empty=%d"%difUnionNewOrder.is_empty)
            difUnionNewOrder.buffer(0)        
        return [newOrder,remaining]
        

class DEGL:
    """         
        deglObject = DEGL(ObjectiveFcn, nVars, ...) creates the DEGL object stored in variable deglObject and 
            performs all initialization tasks (including calling the output function once, if provided).
        
        deglObject.optimize() subsequently runs the whole iterative process.
        
        After initialization, the deglObject object has the following properties that can be queried (also during the 
            iterative process through the output function):
            o All the arguments passed during deglObject (e.g., deglObject.MaxIterations, deglObject.ObjectiveFcn,  deglObject.LowerBounds, 
                etc.). See the documentation of the __init__ member below for supported options and their defaults.
            o Iteration: the current iteration. Its value is -1 after initialization 0 or greater during the iterative
                process.
            o Velocity: the current velocity vectors (nParticles x nVars)
            o CurrentGenFitness: the current population's fitnesses for all chromosomes (D x 1)
            o PreviousBestPosition: the best-so-far positions found for each individual (D x nVars)
            o PreviousBestFitness: the fitnesses of the best-so-far individuals (D x 1)
            o GlobalBestFitness: the overall best fitness attained found from the beginning of the iterative process
            o GlobalBestPosition: the overall best position found from the beginning of the iterative process
            o k: the current neighborhood size
            o StallCounter: the stall counter value (for updating inertia)
            o StopReason: string with the stopping reason (only available at the end, when the algorithm stops)
            o GlobalBestSoFarFitnesses: a numpy vector that stores the global best-so-far fitness in each iteration. 
                Its size is MaxIterations+1, with the first element (GlobalBestSoFarFitnesses[0]) reserved for the best
                fitness value of the initial swarm. Accordingly, pso.GlobalBestSoFarFitnesses[pso.Iteration+1] stores 
                the global best fitness at iteration pso.Iteration. Since the global best-so-far is updated only if 
                lower that the previously stored, this is a non-strictly decreasing function. It is initialized with 
                NaN values and therefore is useful for plotting it, as the ydata of the matplotlib line object (NaN 
                values are just not plotted). In the latter case, the xdata would have to be set to 
                np.arange(pso.MaxIterations+1)-1, so that the X axis starts from -1.
    """
    
    
    def __init__( self
                , ObjectiveFcn
                , nVars
                , LowerBounds = None
                , UpperBounds = None
                , D = None
                , Nf=0.1
                , alpha = 0.8
                , beta = 0.8
                , wmin= 0.4
                , wmax = 0.8
                , FunctionTolerance = 1.0e-3 # -6------------------------------------------------------------------------------------------------------------------
                , MaxIterations = None
                , MaxStallIterations = 15
                , OutputFcn = None
                , UseParallel = False
                , Stock = None
                , Order = None
                ,remaining = None
                ,newOrder = None
                ,u=None
                ):
        """ The object is initialized with two mandatory positional arguments:
                o ObjectiveFcn: function object that accepts a vector (the chromosome) and returns the scalar fitness 
                                value, i.e., FitnessValue = ObjectiveFcn(chromosome)
                o nVars: the number of problem variables
            The algorithm tries to minimize the ObjectiveFcn.
            
            The arguments LowerBounds & UpperBounds lets you define the lower and upper bounds for each variable. They 
            must be either scalars or vectors/lists with nVars elements. If not provided, LowerBound is set to -1000 
            and UpperBound is set to 1000 for all variables. If vectors are provided and some of its elements are not 
            finite (NaN or +-Inf), those elements are also replaced with +-1000 respectively.
            
            The rest of the arguments are the algorithm's options:
                o D (default:  min(200,10*nVars)): Number of chromosomes in the population, an integer greater than 1.
                o Nf (default: 0.1): Neighborhood size fraction
                o alpha (default: 0.8): Scale factor.
                o beta (default: 0.8): Scale factor.
                o wmin (default: 0.4): Minimum weight.
                o wmax (default: 0.8): Maximum weight.
                o MinNeighborsFraction (default: 0.25): Minimum adaptive neighborhood size, a scalar in [0, 1].
                o FunctionTolerance (default: 1e-6): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o MaxIterations (default: 200*nVars): Maximum number of iterations.
                o MaxStallIterations (default: 20): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o OutputFcn (default: None): Output function, which is called at the end of each iteration with the 
                    iterative data and they can stop the solver. The output function must have the signature 
                    stop = fun(), returning True if the iterative process must be terminated. degl is the 
                    deglObject object (self here). The output function is also called after population initialization 
                    (i.e., within this member function).
                o UseParallel (default: False): Compute objective function in parallel when True. The latter requires
                    package joblib to be installed (i.e., pip install joplib or conda install joblib).
                o Stock: Stock useful for fitness function calculation.
                o Order: Order useful for fitness function calculation.
                o Remaining: Remaining useful for fitness function calculation and plots.
                o newOrder: The order with the transformation according to the current solution useful for fitness function calculation and plots.
                o u: New temporary solution used in fitness calculation for u vector.

        """
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        self.Order=Order
        self.Stock = Stock
        self.remaining = remaining
        self.newOrder = newOrder
        
        self.alpha = alpha
        self.beta = beta
        self.wmin = wmin
        self.wmax = wmax
        self.Nf=Nf
        
        # assert options validity (simple checks only) & store them in the object
        if D is None:
            self.D = min(200, 10*nVars)
        else:
            assert np.isscalar(D) and D > 1, \
                "The D option must be a scalar integer greater than 1."
            self.D = max(2, int(round(self.D)))
        

        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        if MaxIterations is None:
            self.MaxIterations = 100*nVars
        else:
            assert np.isscalar(MaxIterations), "The MaxIterations option must be a scalar integer greater than 0."
            self.MaxIterations = max(1, int(round(MaxIterations)))
        assert np.isscalar(MaxStallIterations), \
            "The MaxStallIterations option must be a scalar integer greater than 0."
        self.MaxStallIterations = max(1, int(round(MaxStallIterations)))
        
        self.OutputFcn = OutputFcn
        assert np.isscalar(UseParallel) and (isinstance(UseParallel,bool) or isinstance(UseParallel,np.bool_)), \
            "The UseParallel option must be a scalar boolean value."
        self.UseParallel = UseParallel
        
        # lower bounds
        if LowerBounds is None:
            self.LowerBounds = -1000.0 * np.ones(nVars)
        elif np.isscalar(LowerBounds):
            self.LowerBounds = LowerBounds * np.ones(nVars)
        else:
            self.LowerBounds = np.array(LowerBounds, dtype=float)
        self.LowerBounds[~np.isfinite(self.LowerBounds)] = -1000.0
        assert len(self.LowerBounds) == nVars, \
            "When providing a vector for LowerBounds its number of element must equal the number of problem variables."
        # upper bounds
        if UpperBounds is None:
            self.UpperBounds = 1000.0 * np.ones(nVars)
        elif np.isscalar(UpperBounds):
            self.UpperBounds = UpperBounds * np.ones(nVars)
        else:
            self.UpperBounds = np.array(UpperBounds, dtype=float)
        self.UpperBounds[~np.isfinite(self.UpperBounds)] = 1000.0
        assert len(self.UpperBounds) == nVars, \
            "When providing a vector for UpperBounds its number of element must equal the number of problem variables."
        
        assert np.all(self.LowerBounds <= self.UpperBounds), \
            "Upper bounds must be greater or equal to lower bounds for all variables."
        
        
        # check that we have joblib if UseParallel is True
        if self.UseParallel and not HaveJoblib:
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; swarm objective values will be computed in serial mode instead.""")
            self.UseParallel = False
        
        #DEGL
        
        # Initialization
        lbMatrix = np.tile(self.LowerBounds, (self.D, 1))
        ubMatrix = np.tile(self.UpperBounds, (self.D, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        self.z = lbMatrix + np.random.rand(self.D,nVars) * bRangeMatrix
        #print(self.z.shape)
        #self.L =np.zeros([self.D,nVars])
        #self.y =np.zeros([self.D,nVars])
        #self.g =np.zeros([self.D,nVars])
        #self.u =np.zeros([self.D,nVars])
        
        # calculate neighborhood radius
        k = math.floor(self.D*self.Nf)
        
        # Initial fitness calculation
        self.CurrentGenFitness = np.zeros(self.D)
        self.__evaluateGenInit()
        
        # Initial best-so-far individuals and global best
        self.PreviousBestPosition = self.z.copy()
        self.PreviousBestFitness = self.CurrentGenFitness.copy()
        
        #bInd = self.CurrentGenFitness.argmin()
        # ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR IF EMPTY
        if not(self.CurrentGenFitness.size):
            print("-------ep")
            print(self.z.shape)
        
        # find and save global best fitness and its position
        bInd = self.CurrentGenFitness.argmin()
        self.GlobalBestFitness = self.CurrentGenFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
        
        # iteration counter starts at -1, meaning initial population
        self.Iteration = -1;
        
        self.StallCounter = 0;
        
        # Keep the global best of each iteration as an array initialized with NaNs. First element is for initial swarm,
        # so it has self.MaxIterations+1 elements. Useful for output functions, but is also used for the insignificant
        # improvement stopping criterion.
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
        
        # call output function, but neglect the returned stop flag
        if self.OutputFcn:
            self.OutputFcn(self)
    
    # function for calculating fitness for initialization (vector z)
    def __evaluateGenInit(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentGenFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.z[i,:],self.nVars,self.Stock,self.Order) for i in range(self.D) )
        else:
            self.CurrentGenFitness[:] = [self.ObjectiveFcn(self.z[i,:],self.nVars,self.Stock,self.Order) for i in range(self.D)]
    
    # function for calculating fitness for u vector. Kept 2 function only for reducing changes in code        
    def __evaluateGen(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentGenFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.u[i,:],self.nVars,self.Stock,self.Order) for i in range(self.D) )
        else:
            self.CurrentGenFitness[:] = [self.ObjectiveFcn(self.u[i,:],self.nVars,self.Stock,self.Order) for i in range(self.D)]
        
    def optimize( self ):
        """ Runs the iterative process on the initialized population. """
        nVars = self.nVars

        k = math.floor(self.D*self.Nf)
        
        L =np.zeros([self.D,nVars])
        y =np.zeros([self.D,nVars])
        g =np.zeros([self.D,nVars])
        self.u =np.zeros([self.D,nVars])
        
        #u=self.u
        # start the iteration
        doStop = False
        
        while not doStop:
            self.Iteration += 1

            weight=self.wmin+(self.wmax-self.wmin)*((self.Iteration-1)/(self.MaxIterations-1)) #+0.15 #+1

            for p in range(self.D):
                
                #MUTATION
                #neighbors calculation
                nn_vector=np.array([w for w in range((p-k),(p+k+1))])
                nn_vector[nn_vector<0] = nn_vector[nn_vector<0] + self.D
                nn_vector[nn_vector>(self.D-1)]=nn_vector[nn_vector>(self.D-1)] -self.D 
                
                #neighbors = self.z[nn_vector]
                neighbors_rand_ind = np.random.choice( nn_vector, size=2+1, replace=False)
                neighbors_rand_ind[neighbors_rand_ind==p] = neighbors_rand_ind[2]
                ind_p = neighbors_rand_ind[0]
                ind_q = neighbors_rand_ind[1]
                
                bInd = self.PreviousBestFitness[nn_vector].argmin()
                bestNeighbor = nn_vector[bInd]
                zbest_neighbor = self.z[bestNeighbor]
                
                #calc z
                L[p,:]=self.z[p,:] + self.alpha * (zbest_neighbor-self.z[p,:])+self.beta*(self.z[ind_p,:]-self.z[ind_q,:])
                
                #calc glob
                rand_ind = np.random.choice( self.D, size=2+1, replace=False)
                rand_ind[rand_ind==p] = rand_ind[2]
                ind_r1 = rand_ind[0]
                ind_r2 = rand_ind[1]
                
                bInd = self.PreviousBestFitness.argmin()
                zbest = self.z[bInd]
                
                g[p,:]=self.z[p,:] + self.alpha * (zbest-self.z[p,:])+self.beta*(self.z[ind_r1,:]-self.z[ind_r2,:])
                
                
                y[p,:]=weight*g[p,:] + (1-weight)*L[p,:]
                
                #CROSSOVER    #SATURATE
                Cr=0.8
                jrand= randint(0,self.D)
                
                for l in range(0,self.nVars):
                    if(random()<Cr or l==jrand):
                        self.u[p,l]=y[p,l]
                        
                    else:
                        self.u[p,l]=self.z[p,l]
                        
                
                # check bounds violation
                #self.z=u.copy()

                posInvalid = self.u[p,:] < self.LowerBounds
                self.u[p,posInvalid] = self.LowerBounds[posInvalid]
                
                posInvalid = self.u[p,:] > self.UpperBounds
                self.u[p,posInvalid] = self.UpperBounds[posInvalid]
                
                # used in case that we have discrete theta with values 0,90
                """w3= [b for b in range(2,self.nVars,3)]
                for m in w3:
                    if u[p,m] <45:
                        u[p,m] = 0
                    else:
                        u[p,m]=90"""
                
           
            #calculate fitness
            self.__evaluateGen()
            # find chromosomes tha has been improved and replace the old values with the new
            genProgressed = self.CurrentGenFitness < self.PreviousBestFitness
            self.PreviousBestPosition[genProgressed, :] = self.u[genProgressed, :]
            self.z[genProgressed, :] = self.u[genProgressed, :]
            self.PreviousBestFitness[genProgressed] = self.CurrentGenFitness[genProgressed]
            
            # update global best, adaptive neighborhood size and stall counter
            newBestInd = self.CurrentGenFitness.argmin()
            newBestFit = self.CurrentGenFitness[newBestInd]
            
            if newBestFit < self.GlobalBestFitness:
                self.GlobalBestFitness = newBestFit
                self.GlobalBestPosition = self.z[newBestInd, :].copy()
                
                self.StallCounter = max(0, self.StallCounter-1)
                # calculate remaining only once when fitness is improved to save some time
                # useful for the plots created
                [self.newOrder,self.remaining] = extract_obj(self)
            else:
                self.StallCounter += 1
                
            # first element of self.GlobalBestSoFarFitnesses is for self.Iteration == -1
            self.GlobalBestSoFarFitnesses[self.Iteration+1] = self.GlobalBestFitness
            
            # run output function and stop if necessary
            if self.OutputFcn and self.OutputFcn(self):
                self.StopReason = 'OutputFcn requested to stop.'
                doStop = True
                continue
            
            # stop if max iterations
            if self.Iteration >= self.MaxIterations-1:
                self.StopReason = 'MaxIterations reached.'
                doStop = True
                continue
            
            # stop if insignificant improvement
            if self.Iteration > self.MaxStallIterations:
                # The minimum global best fitness is the one stored in self.GlobalBestSoFarFitnesses[self.Iteration+1]
                # (only updated if newBestFit is less than the previously stored). The maximum (may be equal to the 
                # current) is the one  in self.GlobalBestSoFarFitnesses MaxStallIterations before.
                minBestFitness = self.GlobalBestSoFarFitnesses[self.Iteration+1]
                maxPastBestFit = self.GlobalBestSoFarFitnesses[self.Iteration+1-self.MaxStallIterations]
                if (maxPastBestFit == 0.0) and (minBestFitness < maxPastBestFit):
                    windowProgress = np.inf  # don't stop
                elif (maxPastBestFit == 0.0) and (minBestFitness == 0.0):
                    windowProgress = 0.0  # not progressed
                else:
                    windowProgress = abs(minBestFitness - maxPastBestFit) / abs(maxPastBestFit)
                if windowProgress <= self.FunctionTolerance:
                    self.StopReason = 'Population did not improve significantly the last MaxStallIterations.'
                    doStop = True
            
        
        # print stop message
        print('Algorithm stopped after {} iterations. Best fitness attained: {}'.format(
                self.Iteration+1,self.GlobalBestFitness))
        print(f'Stop reason: {self.StopReason}')
        
            
