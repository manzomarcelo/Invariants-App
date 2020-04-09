# Invariants-App

## Contributors
* Marcelo Manzo
* Rodrigo Zhou

Application developed in undergraduate research about topological invariants. Project supervised by mathematician Paul Alexander Schweitzer(Princeton University)

Currently, it is computing Jones Polynomial and Kauffman Polynomial X correctly. Unfortunately, there is a bug in the algorithm for computing the Kauffman Polynomial F (not known yet).

Possible future improvement: re-code the algorithm for Kauffman Polynomial X using bit mask. It will probably improve its performance.

The number of Thurston-Bennequin and Maslov (Rotation) is computed using the projection of a topological knot. The algorithm makes the changes (not visible in the application) to approximate to a Legendrian knot, and then, compute these numbers.
