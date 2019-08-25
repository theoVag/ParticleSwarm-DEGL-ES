# ParticleSwarm-DEGL-ES
Optimized manangement of wood stock of carpentry factories via Particles Swarm and Differential evolution DEGL


Use of evolution algorithms for industrial processes optimization

The purpose of this project is the optimized manangement of wood stock of carpentry factories. For illustration purpose our data was 3 orders consisting of a list of polygons and 8 stocks in which our implementation should fit in the most appropriate way. The target is to fit the whole number of orders and beyond that to place the polygons of orders in stock such that the remainings of stocks can be exploited in new orders. This practical means that the remainings of stocks should as compact as it can. The implemented fitness function is the core of the solution and it is presented below.
for Particles fitness function:
res = outOfStock.area ∗ 10000 + overlapArea ∗ 10000 + dist_from_zero/10 ∗ 10 + 100 ∗ fsm
where:
dist from zero: Sum of edited_Order_area * (centroid.x + centroid.y) for each part of the order 
outOfStock.area : Sum of edited_Order area that lies outside the current stock
overlapArea: Sum of edited_Order area that is occupied by more than one polygon.
#Measure of compactness fsm
ch = (remaining.convexhull)
lamda = (ch.area)=(remaining.area) − 1
alpha = 1.1
fsm = 1/(1 + alpha ∗ lambda)
for DEGL fitness function: 
res = outOfStock.area ∗ 1000 + overlapArea ∗ 1000 + dist_from_zero ∗ 1 + 300 ∗ fsm

This fitness function was used in different variants in Particle swarm algorithm and in DEGL (Differential Evolution with global and local neighborhood topologies). Each particle has x,y,theta for all the polygons of the order. There was also a comparison with the functions Nelder-Mead, L-BFGSB, SLSQP from python scipy.optimize and the patternsearch − noisyopt.

Some execution examples images and details can be found in the report. The full report is available only in greek.


References for the algorithms used:
[1] S. K. Mylonas, D. G. Stavrakoudis, J. B. Theocharis, and P. A. Mastorocostas, “A Region-Based
GeneSIS Segmentation Algorithm for the Classification of Remotely Sensed Images,” Remote
Sensing, vol. 7, no. 3, pp. 2474–2508, Mar. 2015. Online link: http://www.mdpi.com/2072-4292/
7/3/2474 (p. 1)

[2] S. Das, A. Abraham, U. Chakraborty, and A. Konar, “Differential Evolution Using a Neighborhood-Based Mutation Operator,” IEEE Transactions on Evolutionary Computation, vol. 13, no. 3,
pp. 526–553, Jun. 2009. (p. 3)

[3] S. Das and S. Sil, “Kernel-induced fuzzy clustering of image pixels with an improved differential
evolution algorithm,” Information Sciences, vol. 180, no. 8, pp. 1237–1256, Apr. 2010. Online link:
http://www.sciencedirect.com/science/article/pii/S0020025509005192 (p. 3)

[4] M. Dorigo and T. Stützle, Ant Colony Optimization, ser. Bradford Books. Cambridge, MA, USA:
MIT Press, Jun. 2004. (p. 6)

extra libraries:
https://github.com/Toblerity/Shapely

https://github.com/andim/noisyopt
