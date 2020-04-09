# Invariants-App

## Contributors
* Marcelo Manzo
* Rodrigo Zhou

Application developed in undergraduate research about topological invariants. Project supervised by mathematician Paul Alexander Schweitzer(Princeton University)

The algorithms were tested comparing the obtained results with the table of knots provided by Indiana University (https://knotinfo.math.indiana.edu/). Currently, it is computing Jones Polynomial and Kauffman Polynomial X correctly for knots with at most 6 crossing numbers. Unfortunately, there is a bug in the algorithm for computing the Kauffman Polynomial F (not known yet).

The number of Thurston-Bennequin and Maslov (Rotation) is computed using the projection of a topological knot. The algorithm implements the necessary changes (not visible in the application) to create a Legendrian knot, and then computes the aforementioned numbers.
