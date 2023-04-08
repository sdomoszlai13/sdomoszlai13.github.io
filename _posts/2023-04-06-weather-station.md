# Simple Arduino-based Weather Station

*See also: why buy a professional weather station for 30$ if you can build your own for 40$, although with less functionality and more frustration?*

The screen of the lovely weather station with a backlit sun and clouds, purchased a few weeks ago from Aliexpress for 15$, reads (as ever so often): "Indoor temperature: 20.6 C, indoor humidity: 56%. Outdoor temperature: -.-- C, outdoor humidty: --%".
This journey begins on a Tuesday morning as I am looking at our certainly cute but not too reliable weather station. The base station lost connection to the outdoor unit - again. "Why not build your own then?", I said to myself, and so I did.

I wanted to build a weather station similar to those available commercially. The system consists of a base station that can be put on a shelf or table indoor and a remote unit that can be placed outdoor, though it should not come in direct contact with water. Both units measure temperature, pressure and humidity. The remote unit transmits the data to the base station that displays both indoor and outdoor measurements. The remote unit features a display too, which is off by default and displays current measurement values for a few seconds when the button on the unit is pressed. The display of the base station is always on.


## Hardware

Temperature, pressure, and humidity are measured by a widely available combined **AHT20 + BMP280 sensor**. The former chip can measure temperature and humidity, while the latter can measure temperature and pressure. The two chips are built on a single board and have a common I2C breakout (more about that later). Such devices are available for around 3$ online.

The measured values than can be displayed on a display, e.g. a 128 X 32 LCD, which I used. These are available for around 6$.

The wireless communication between the two units is realized with the help of two nRF24L01 radio modules. These modules allow for wireless communication of 5-50 m, depending on the circumstances (walls are bad for range, obviously). One module goes for about 3$ online.

Last but not least: the "brain" of the weather station: two Arduinos.

{% include info.html text="**What is an Arduino?**
<br>
The Arduino platform is an 8-bit, user-friendly environment for hobby, ultra-fast prototyping and experimenting. There is a wide variety of boards available, ranging from the Arduino Nano with a flash memory of 32 kB and a clock speed of 16 MHz all the way up to the 32-bit, STM32-based Arduino G1 Wi-Fi with 2 MB of flash memory and a clock speed of 480 MHz.
" %}

Although ESP32-based or STM32-based boards would be a great option too, I went with the Arduinos as neither Wi-Fi, nor a significant computing capacity was a requirement.
