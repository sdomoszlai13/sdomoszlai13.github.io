# Show Me Your ADS-B signature and I'll Tell You Who You Are - DIY Flight Tracking

Ever wondered where a plane is heading when you looked up in the sky and saw a tiny, distant airplane zipping in the sky? Think about it. On every commercial flight up there, there are sitting 100+ people from probably a different country then yours, all with different families, jobs, backgrounds, goals... Most of them never met. But they came together for that single flights, unified in their goal for a few hours. They all paid to get to their distant destination, some visiting family members, others hoping to close a deal, apply for a job or just relax on the beach. So, where are they heading?

As with nearly everyhting in the 21st century, turns out there's also an app as well as a website for that (multiple, to be precise). Here, take a look:

* [FlightRadar24](https://flightradar24.com)
* [planefinder](https://planefinder.net/)
* [FlightAware](https://www.flightaware.com/live/)

You can even click on any aircraft to see some details: What type of aircraft is flying there? Where is it heading to? How long does its flight take? If you pay a subscription fee, you can even see live meteorological data, the aircraft's age etc. If you now look up in the sky, determine the rough heading of an aircraft you see and find it based on the heading in a flight tracker, you can tell where the flight started and where it will end. Try it!

![](/images/flight-tracker/miami-a380.png "Miami airport with an approaching A380 from London")

Pretty cool. But how does this work?

## The ID card of aircraft: ADS-B

ADS-B was introduced in the early 2000s to be able to identify and track flights. It specifies what data an aircraft should sent and in which format it should do it and what frequency (1090 MHz) the messages should be sent on.

The important thing to know here is that this data broadcasted by every plane is not only sent to airports or ATC but broadcast through the ether. This means that any enthusiast (nerd?) can receive these signals with a proper antenna, a USB dongle that acts as an SDR (Software-Defined Radio) and a PC. Some people around the globe are doing this 24/7 and feed their data to sites like those mentioned above. These sites aggregate the data and visualize it.

...which brings us to the question:

## Can I do it myself?

Certainly! All you need to get is

* an antenna to receive the signals from the aircraft

* an SDR (which in it's cheapest form is a USB dongle) capable of processing the signals coming from the antenna

* a PC, preferably a small SBC (Single-Board Computer)

As for the antenna and the SDR, you probably get the most bang for you buck if you go with [this bundle](https://www.rtl-sdr.com/product/rtl-sdr-blog-v4-r828d-rtl2832u-1ppm-tcxo-sma-software-defined-radio-with-dipole-antenna/). As for the PC, the recommended choice a [Raspberry Pi](https://www.raspberrypi.com/) (get at least the Pi 4 with 4 GB of RAM).

The antenna is connected to the SDR and the SDR to the Pi.