# Spring Mass System Simulator

Today we're gonna take a look at solving the equations of motion for a spring mass system. Now, you may be asking: solving a what? Well, imagine you wanted to simulate the motion of a piece of elastic fabric hung up on a wall, or the motion of atoms in molecules (and believe me, sometimes you $do$ want to). Such systems can be described by spring mass systems. These are systems that contain multiple bodies with mass that are connected by springs. Of course, the connections can be arbitrary and the "stiffness" of the springs (the spring constant) can be different for every spring. There are also one or more fixtures we can tether the masses on, so they won't fall down.
You can find the complete code in my GitHub repository at https://github.com/sdomoszlai13/spring-mass-system-simulator.
<br>
<br>
The program we'll get to know runs a simulation of a system of fixtures, masses, and springs in two dimensions. Fixtures and masses can be set up with initial coordinates and initial velocities (not for fixtures, of course). After setting the desired temporal length and the number of time steps, the simulation is run and a plot is created with the fixtures, masses, and the masses' trajectories so we can see which way they went. There's also an option to save the trajectories to a file.
<br>

{% include info.html text="The following paragraph is a bit technical. Feel free to dive into it but be warned: I'm sure you've never experienced it, but let me tell you that maths can cause some frustration." %}


## Physical Background

Consider a mass $m$ attached to a spring with a spring constant $k$ moving in 1D. Then, the equation of motion of the mass is:

$$m\ddot{x}(t) = -kx + mg,$$

where $x(t)$ is the position of the mass and $g$ is the gravitational acceleration. If two masses are coupled in series, the system can be described by a set of equations (omitting time dependence notation for simplicity):

$$m_1\ddot{x_1} = -k_1x_1 + k_2(x_2-x_1) + m_1g,$$

$$m_2\ddot{x_2} = k_2(x_2-x_1) + m_2g.$$

If we allow the masses to move in 2D, the equations become

$$m_1\ddot{\vec{x_1}} = -k_1(||\vec{x_1}|| - l_1)\frac{\vec{x_1}}{||\vec{x_1}||} + k_2(||\vec{x_2} - \vec{x_1}|| - l_2)\frac{\vec{x_2} - \vec{x_1}}{||\vec{x_2} - \vec{x_1}||} + m_1\vec{g},$$

$$m_2\ddot{\vec{x_2}} = -k_2(||\vec{x_2} - \vec{x_1}|| - l_2)\frac{\vec{x_2} - \vec{x_1}}{||\vec{x_2} - \vec{x_1}||} + m_2\vec{g}.$$

A general system of springs and masses can be described by a similar set of equations. By solving the equations for $x(t)$, we can determine the trajectories of the masses. In principle, these equations can be solved analytically. However, that's more of a maths problem than a software development problem. For the sake of simplicity, we're gonna solve the equations numerically with the help of the (forward) Euler method. This is a quite simple method to solve differential equations and works as follows. <br>

Let's suppose we know the position $x(t)$ and the velocity $v(t)$ of a mass at the time $t$, as well as the force $F(t)$ acting on it. A small time $\Delta t$ later, at the time $t' = t + \Delta t$, we can say that approximately

$$x(t') = x(t) + v(t)\Delta t,$$

$$v(t') = v(t) + a(t)\Delta t = v(t) + \frac{F(t)}{m}\Delta t,$$

$$F(t') = \sum_{j} F_{spring_j} + mg.$$


To make use of this algorithm, we have to divide our simulation time into $n$ discrete time steps which a time difference of $\Delta t$.


## Examples

Let's see what's possible with a basic simulation tool like this. First, we try to simulate a vertically oscillating mass, also known as a harmonic oscillator.

![](/images/spring-mass-system-simulator/harm-osc.png "Mass attached to vertical spring (harmonic oscillator)")

Next, let's look at a non-elastic pendulum.

![](/images/spring-mass-system-simulator/swing.png "Mass attached to a rigid, swinging, initially vertical spring")

So far, so good. There's nothing special going on here.

Now, let's see what happens if we build an elastic pendulum.

![](/images/spring-mass-system-simulator/el-pend.png "Mass attached to an elastic, swinging, initially vertical spring")

Nice! Now we're gonna take a look at a double pendulum.

![](/images/spring-mass-system-simulator/doub-pend-rig.png "Two coupled masses attached to a rigid, swinging, initially vertical spring (inelastic double pendulum)")

Double pendulums are actually quite fascinating because of their so-called chaotic behavior. This means that for slight variations in the initial conditions, the resulting trajectories can be significantly different from one another. Such systems can be described with the help of chaos theory. But now, let's create a bit more chaos, shall we? Here's an elastic double pendulum.

![](/images/spring-mass-system-simulator/doub-pend-el.png "Two coupled masses attached to a rigid, swinging, initially vertical spring (elastic double pendulum)")

Isn't it amazing what a basic algorithm for solving differential equations is capable of?

Of course, there are some pitfalls you have to avoid to get realistic results. Simulation time and the number of time steps is critical. There should be at least 1000 time steps for each second of simulation time, depending on expected velocities of the masses. Similarly, if the spring constant is increased, the number of time steps should also be increased. This is important because of higher velocities and accelerations. As the time steps are distributed equally on the time scale, calculated points on the trajectory are more spaced out if the velocity of a mass is higher. This can be seen in the image below.

![](/images/spring-mass-system-simulator/zoom1.png "Points are more spaced out when a body is moving with a higher velocity")


You can find the complete code in my GitHub repository at https://github.com/sdomoszlai13/spring-mass-system-simulator.
