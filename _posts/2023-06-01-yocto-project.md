# The Yocto Project: It's Linux, stupid!

The spectrum of software complexity and abstraction level nowadays is reeeally wide: It ranges from training super complicated AI stuff on supercomputers through powerful modern laptop PCs down to ticket machines or microwave ovens and all the way down to flipping single bits in a microcontroller. This range of applications often requires completely different techniques and tools and usually different mindsets too. An average person might think 'A computer is a computer, no matter the size! Those geeks just look at their black screens with green letters flashing on them. Using a little PC? You just need to write less code!'. But we are software developers and know better, *right*? There's no single piece of software or framework that's used in that truly diverse spectrum of applications. If you look close enough though, you will see an old friend rising on the hoizon as a truly versatile tool for the future: Linux.

Though some say Moore's law is already dead, this doesn't mean the rapid development of computer technology and with it miniaturization doesn't continue at an extraordinary speed. In recent years, this has lead to unprecedented possibilities in the world of embedded systems. In 2007, the STM32 platform was launched, and in 2012, the Raspberry Pi was born. Since then, credit-card-sized computers have sprang up like mushrooms. The minimum system requirements for running Linux are nowadays satisfied by almost every microchip that cost more than 10 USD. Linux has been the go-to operating system for supercomputer architects for years. Linux has therefore become a software tool that is used from IoT devices and routers through laptop PCs and servers all the way up to supercomputers.
It's Linux, stupid!

![](/images/yocto-project/tux.png "Tux, the Linux penguin")

## 1. Embedded Linux

In this article, we'll be focusing on the lower end of the software complexity spectrum: Embedded systems. The software run by these devices can be divided in the following three categories:

* Bare Metal:
  - little to no software overhead
  - strict timing
  - high control of hardware
  - very little power consumption
* Real-Time Operating System (RTOS):
  - scheduler overhead
  - multithreading
  - high control of hardware
  - libraries
* General Purpose Operating System (GPOS):
  - less control of hardware
  - singificant overhead (background tasks, memory management...)
  - usually portable


Linux is a GPOS. Thus, only the categories 'bare metal' and 'RTOS' aren't covered by Linux (yet?). A simple way to run a Linux distro on a single-board computer (SBC) is to flash an image of the distro onto an SD card and boot the OS from it. However, in this case there are usually a number of system components that aren't needed for the specific application and can be considered bloatware. These can be removed manually once the system is booted. If you now think 'there must be a better way to achieve this', you're absolutely right. Imagine a software that lets you select exactly which system components and even which drivers you'd like to use and *creates* this customized Linux distro for you. Look no further: This is the Yocto Project.

## 2. Yocto Project Basics

Before getting started, it's important to be familiar with the below concepts.

* Recipe: instructions to build one or more packages
* Layer: related set of instructions for the OpenEmbedded System Build. They can override previous instructions, settings, or even layers and are used to logically separate build components
* Metadata: folders containing auxiliary files like configuration files or user-supplied recipe files
* Board Support Package (BSP): layer with board-specific info on how to build image
* BitBake: component used by the OpenEmbedded build system to build images
* Image: output of the build process, bootable
* Poky: reference distribution, serves as an example

In short, the Yocto Project allows us to gather our chosen recipes into layers and build our custom image for a specific board with the help of the corresponding board support package. For the build process, we use BitBake. One huge advantage of the Yocto Project compared to similar software is that if an image has been built and you want to rebuild it after a modification, only the modified files need to be rebuilt which makes the process really fast. To see how a finished image looks like, we can take a look at Poky (not part of this tutorial).

## 3. Target Devices

Embedded Linux distros are usually run SBCs like a Raspberry Pi or an STM32. There are a lot of devices that support slim Linux distros like the well-known Raspberr Pis, but also the Swiss-made Toradex Apalis, Verdin or Colibri family of SBCs or a whole palette of 96boards SBC editions. Usually, these devices come with 500 MB-4 GB of RAM, 1-16 GB flash memory and a clock speed of 200 MHz-2 GHz, depending on the intended use. Though a wide variety of these boards is available, it's good to consider popularity as this usually correlates with the amount of helpful information available online.

In this tutorial, we'll be using an STM32MP157C to run the image we're gonna build.


[comment]: <> (## 4. Example: Custom Linux distribution for the STM32157C)
[comment]: <> (Every now and then come over ( anyone?))
