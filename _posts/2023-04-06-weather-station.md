# Simple Arduino-based Weather Station

*See also: why buy a professional weather station for 30\$ if you can build your own for 40\$, although with less functionality and more frustration?*

The screen of the lovely weather station with a backlit sun and clouds, purchased a few weeks ago from Aliexpress for 15$, reads (as ever so often): "Indoor temperature: 20.6 C, indoor humidity: 56%. Outdoor temperature: -.-- C, outdoor humidty: --%".
This journey begins on a Tuesday morning as I am looking at our certainly cute but not too reliable weather station. The base station lost connection to the outdoor unit - again. "Why not build your own then?", I said to myself, and so I did.

I wanted to build a weather station similar to those available commercially. The system consists of a base station that can be put on a shelf or table indoor and a remote unit that can be placed outdoor, though it should not come in direct contact with water. Both units measure temperature, pressure and humidity. The remote unit transmits the data to the base station that displays both indoor and outdoor measurements. The remote unit features a display too, which is off by default and displays current measurement values for a few seconds when the button on the unit is pressed. The display of the base station is always on.


## Hardware

The hardware has to complete the following tasks:

Base station:
* measure temperature, pressure and humidity
* display both measured and received values on an LCD


Remote unit:
* measure temperature, pressure and humidity
* send measured values to the base station
* display measured values on an LCD if a button is pressed


Temperature, pressure, and humidity are measured by a widely available combined **AHT20 + BMP280 sensor**. The former chip can measure temperature and humidity, while the latter can measure temperature and pressure. The two chips are built on a single board and have a common I2C breakout (more about that later). Such devices are available for around 3$ online.

The measured values than can be displayed on a display, e.g. a 128 X 32 LCD, which I used. These are available for around 6$.

The wireless communication between the two units is realized with the help of two nRF24L01 radio modules. These modules allow for wireless communication of 5-50 m, depending on the circumstances (walls are bad for range, obviously). One module goes for about 3$ online.

Last but not least: the "brain" of the weather station, which consists of two Arduinos.

{% include info.html text="<b>What is an Arduino?<b>
<br>
The Arduino platform is an 8-bit, user-friendly environment for hobby, ultra-fast prototyping and experimenting. There is a wide variety of boards available, ranging from the Arduino Nano with a flash memory of 32 kB and a clock speed of 16 MHz all the way up to the 32-bit, STM32-based Arduino G1 Wi-Fi with 2 MB of flash memory and a clock speed of 480 MHz.
" %}

Although ESP32-based or STM32-based boards would be a great option too, I went with the Arduinos as neither Wi-Fi, nor a significant computing capacity is required for this project.
  
The modules are connected to the Arduino using a prototyping PCB. The devices are packaged in a 3D printed housing (designed and printed by the author too).
  

## Software
  
The software is required to do the following tasks:
  
* make a measurement with the sensors and store the returned values
* display measured values
* transmit/receive measured values
* monitor the state of a push button
  
To make the measurements, the *Sparkfun AHT20*, and the *Adafruit BMP280* libraries were used for the two sensors. To drive the display, the libraries *LiquidCrystalI2C* and *Adafruit SSD1306* were used. To control the transceiver, the *NRFLite* library was used.
  
### Choosing a communication protocol
  
When using peripheral devices with microcomputers like, you can choose from a number of communication protocols to use: SPI, I2C, UART etc. Each of these have their advantages and disadvantages, which won't be discussed here in detail. As I2C is a bit slower than than the others, but only requires 2 wires (besides VCC and GND), this protocol is used with the AHT20 and the BMP280 sensors, as well as with the LCD. However, as the nRF24L01 doesn't have I2C available, communications happen with the help of SPI. The devices are wired up accordingly, as shown in Fig. x.
  
When writing 
  
  
```c++
// BASE STATION


// Instantiate an AHT20 sensor
AHT20 aht20;

// Instantiate a BMP280 sensor
Adafruit_BMP280 bmp280; // Use I2C interface
Adafruit_Sensor *bmp_temp = bmp280.getTemperatureSensor();
Adafruit_Sensor *bmp_pressure = bmp280.getPressureSensor();

// Instantiate a display
LiquidCrystal_I2C lcd(0x27,20,4);   // Set address to 0x27 for a 16 X 2 display

// Instantiate a transceiver
NRFLite _radio;
const static uint8_t RADIO_ID = 0;              // This transceiver
const static uint8_t DESTINATION_RADIO_ID = 1;  // Other transceiver
const static uint8_t PIN_RADIO_CE = 9;
const static uint8_t PIN_RADIO_CSN = 10;

struct RadioPacket  // Packet to be received
{
    float temp;
    float pres;
    float hum;
};

enum TypeOfVal {temp, pres, hum}; // Determines which display function to use
```

