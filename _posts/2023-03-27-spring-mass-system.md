# Spring Mass System Simulation

You may be asking: a spring mass what? Well, imagine you wanted to simulate the motion of a piece of elastic fabric hung up on a wall, or the motion of atoms in molecules (sometimes you want to, believe me). Such systems can be described by spring mass systems. These are systems that contain multiple bodies with mass that are connected by springs. Of course, the connections can be arbitrary and the "power" of the springs (the spring constant) can be different for every spring. There are also one or more fixtures we can tether the masses on, so tthey won't fall down.
</br>
</br>
In this post we'll be looking at a simulation of a system of fixtures, masses, and springs in two dimensions. Fixtures and masses can be set up with initial coordinates and initial velocities (not for fixtures, of course). After setting the desired temporal length and the number of time steps, the simulation is run and a plot is created with the fixtures, masses, and the masses' trajectories. There's also an option to save the trajectories to a file.
</br>

## Physical background

A system of springs and masses can be described by a set of differential equations:


In principle, these equations can be solved analytically. However, that's more of a maths problem than a software development problem. For the sake of simplicity, we're gonna solve the equations numerically with the help of the (forward) Euler method.

## Examples

Let's see what's possible with a basic simulation tool like this. First, we try to simulate a vertically oscillating mass.

image

Next, let's look at a nonelastic pendulum.

image

So far, so good. There's nothing special going on here.

Now, let's see what happens if we build an elastic pendulum.

image

Nice! Now we're gonna take a look at a double pendulum.

image

Double pendulums are actually quite fascinating because of their so-called chaotic behavior. This means that for slight variations in the initial conditions, the resulting trajectories can be significantly different from one another. Such systems can be described with the help of chaos theory. But now, let's create a bit more chaos, shall we? Here's an elastic double pendulum.

image

Isn't it amazing what a basic algorithm for solving differential equations is capable of?
