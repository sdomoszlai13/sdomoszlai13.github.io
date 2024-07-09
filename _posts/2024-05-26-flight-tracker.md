# Show Me Your ADS-B signature and I'll Tell You Who You Are - DIY Flight Tracking

Ever wondered where a plane is heading when you looked up in the sky and saw a tiny, distant airplane zipping in the sky? Think about it. On every commercial flight up there, there are sitting 100+ people from probably a different country then yours, all with different families, jobs, backgrounds, goals... Most of them never met. But they came together for that single flights, unified in their goal for a few hours. They all paid to get to their distant destination, some visiting family members, others hoping to close a deal, apply for a job or just relax on the beach. So, where are they heading?

As with nearly everyhting in the 21st century, turns out there's also an app as well as a website for that (multiple, to be precise). Here, take a look:

* [FlightRadar24](https://flightradar24.com)
* [planefinder](https://planefinder.net/)
* [FlightAware](https://www.flightaware.com/live/)

You can even click on any aircraft to see some details: What type of aircraft is flying there? Where is it heading to? How long does its flight take? If you pay a subscription fee, you can even see live meteorological data, the aircraft's age etc. If you now look up in the sky, determine the rough heading of an aircraft you see and find it based on the heading in a flight tracker, you can tell where the flight started and where it will end. Try it!

![](/images/flight-tracker/miami-a380.png "Miami airport with an approaching Airbus A380 from London")

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

### Hardware

The hardware setup is really simple: connect the dipole antenna (length: 13-14 cm in total) to the SDR dongle and plug the dongle into the Pi. Make sure the antenna is aligned vertically (as the signal emitted by the aircraft is vertically polarized).

### Software

To make life easier, the whole flight tracker is set up in a [Docker Compose](https://docs.docker.com/compose/) file. This helps isolating all the software components from programs running on the host machine as well as avoiding potential dependency conflicts. The Docker Compose file contains the following components:

* Decoder & feeder
* Flight visualizer
* Database
* Data visualizer

These components are explained in detail below. You can access the complete Docker Compose file in the [Flight Tracker](https://github.com/sdomoszlai13/flight-tracker) repo.

#### The decoder & feeder

For this, the `docker-adsb-ultrafeeder` image from SDR Enthusiasts is used. Note that the `telegraf` tag is required to enable data feeding to the database.

```docker
  ultrafeeder:
    image: ghcr.io/sdr-enthusiasts/docker-adsb-ultrafeeder:telegraf
    container_name: ultrafeeder
    hostname: ultrafeeder
    tty: true
    restart: unless-stopped
    device_cgroup_rules:
      - 'c 189:* rwm'
    ports:
      - 8080:80
    environment:
      - LOGLEVEL=error
      - TZ=${FEEDER_TZ}

      # SDR-related parameters:
      - READSB_DEVICE_TYPE=rtlsdr
      - READSB_RTLSDR_DEVICE=${ADSB_SDR_SERIAL}
      - READSB_RTLSDR_PPM=${ADSB_SDR_PPM}

      # readsb/decoder parameters:
      - READSB_LAT=${FEEDER_LAT}
      - READSB_LON=${FEEDER_LONG}
      - READSB_ALT=${FEEDER_ALT_M}m
      - READSB_GAIN=${ADSB_SDR_GAIN}
      - READSB_RX_LOCATION_ACCURACY=2
      - READSB_STATS_RANGE=true

      # Sources and aggregator connections
      - ULTRAFEEDER_CONFIG=
          adsb,dump978,30978,uat_in;
          mlathub,piaware,30105,beast_in;
      - UUID=${ULTRAFEEDER_UUID}
      - MLAT_USER=${FEEDER_NAME}
      - READSB_FORWARD_MLAT_SBS=true

      # TAR1090 (map web page) parameters:
      - UPDATE_TAR1090=true
      - TAR1090_DEFAULTCENTERLAT=${FEEDER_LAT}
      - TAR1090_DEFAULTCENTERLON=${FEEDER_LONG}
      - TAR1090_MESSAGERATEINTITLE=true
      - TAR1090_PAGETITLE=${FEEDER_NAME}
      - TAR1090_PLANECOUNTINTITLE=true
      - TAR1090_ENABLE_AC_DB=true
      - TAR1090_FLIGHTAWARELINKS=true
      - HEYWHATSTHAT_PANORAMA_ID=${FEEDER_HEYWHATSTHAT_ID}
      - HEYWHATSTHAT_ALTS=${FEEDER_HEYWHATSTHAT_ALTS}
      - TAR1090_SITESHOW=true
      - TAR1090_RANGE_OUTLINE_COLORED_BY_ALTITUDE=true
      - TAR1090_RANGE_OUTLINE_WIDTH=2.0
      - TAR1090_RANGERINGSDISTANCES=50,100,150,200
      - TAR1090_RANGERINGSCOLORS='#1A237E','#0D47A1','#42A5F5','#64B5F6'
      - TAR1090_USEROUTEAPI=true

      # GRAPHS1090 (decoder and system status web page) parameters:
      - GRAPHS1090_DARKMODE=true

      # InfluxDB config
      - INFLUXDBV2_URL=http://influxdb:8086
      - INFLUXDBV2_BUCKET=ultrafeeder
      - INFLUXDBV2_ORG=ultrafeeder
      - INFLUXDBV2_TOKEN=${INFLUXDB_ADMIN_TOKEN}

```

The variables used in the above Docker Compose file are defined in a file called *.env*. Here's a sample .env file:

```python
FEEDER_ALT_FT=YourAltitudeInFt
FEEDER_ALT_M=YourAltitudeInM
FEEDER_LAT=YourLatitude
FEEDER_LONG=YourLongitude
FEEDER_TZ=Europe/Zurich
FEEDER_NAME=Flight-Tracker
ADSB_SDR_SERIAL=1090
ADSB_SDR_GAIN=autogain
ADSB_SDR_PPM=
ULTRAFEEDER_UUID=YourUltraFeederUUID
FEEDER_HEYWHATSTHAT_ID=YourHeyWhatsThatID
FEEDER_HEYWHATSTHAT_ALTS=3000,12000
FR24_SHARING_KEY=YourFR24SharingKey
INFLUXDB_USER=flight-tracker
INFLUXDB_PASSWORD=YourFancySecurePassword
INFLUXDB_ADMIN_TOKEN=YourInfluxDBToken
```

#### The flight visualizer

To visualize the real-time tracked flights on a map, one can use e.g. TAR1090. This tool gives a visualizaton of our data similar to the professional flight trackers like FlightRadar24. See for yourself:

![](/images/flight-tracker/own-tracker.png "DIY flight tracker map")

#### The database

As for the database, [InfluxDB](https://www.influxdata.com/) is well-suited for our type of data called *time series data*, referring to data that comes in regular or irregular time intervals and stored with a time stamp. IoT devices and sensors tipically generate data of this kind.

Compared to a relational database system like [PostgreSQL](https://www.postgresql.org/), a time series database has some additional features like downsampling or using retention policies. InfluxDB also has a free plan for which no registration is required.

![](/images/flight-tracker/influx-db.png "Visualization of the *adsb_icao* field of the *readsb_stats* measurement in InfluxDB's native data explorer")

#### The data visualizer

To visualize the collected raw data, [Grafana](https://grafana.com/) is used. This is a simple to use, versatile tool for data visualization used in the industry for monitoring and data analysis. Visualizations can be created easily with SQL queries.


## Observations

I've been running this setup on a Pi 4 for over a month now. The antenna is located several meters away from the nearest window and therefore, coverage isn't great. My observations so far:

* on average, about 6 aircraft are detected simultaneously

![](/images/flight-tracker/no-tracked-flights.png "Number of tracked flights over time")

* terrain (mountains) has a significant effect on the coverage

* even with a simple dipole antenna placed inside a concrete building, aircraft that are 60-70 km away can be detected

* writing the data received to a database (with Telegraf in this case) has a significant effect on processor and memory usage. As expected, during the day the number of airborne aircraft is significantly higher than at night. This also means higher data rates and therefore the time of the day correlates with writes.

![](/images/flight-tracker/writes.png "Writes over time")

Using Grafana, many more interesting insights can be gained. Try & explore - *(not even) the sky's the limit!*
