# Radio Frequency Underwater Localization @433MHz using SDR

Short installation guide:

```
sudo apt-get update
sudo apt-get install cmake build-essential python-pip libusb-1.0-0-dev python-numpy git pandoc
```
Download and install RTL-SDR library:
```
cd ~ (navigate to your src-directory)
git clone git://git.osmocom.org/rtl-sdr.git
cd rtl-sdr
mkdir build
cd build
cmake ../ -DINSTALL_UDEV_RULES=ON -DDETACH_KERNEL_DRIVER=ON
make
sudo make install
sudo ldconfig
```
Install Python RTL-SDR wrapper
```
sudo pip install pyrtlsdr
```
From here on you should be able to run `rf.py`.


Feel free to use our code for your own work.
If you use this code, please drop a message to daniel.duecker@tuhh.de to maintain a research in rf-localization using software defined radio and cite:

```
@article{Duecker2017,
  title = {Embedded Spherical Localization for Micro Underwater Vehicles Based on Attenuation of Electro-Magnetic Carrier Signals},  
  author = {D A Duecker and A R Geist and M Hengeler and E Kreuzer and M A Pick and V Rausch and E Solowjow},
  doi = {https://doi.org/10.3390/s17050959},
  url = {https://doi.org/10.3390/s17050959},
  pages = {1--22},
  year = {2017},
  month = 5,
  publisher = {MDPI},
  journal = {Sensors}
}
@inproceedings{Duecker2019,
  title = {An Integrated Approach to Navigation and Control in Micro Underwater Robotics using Radio-Frequency Localization},  
  author = {Duecker, Daniel A and Johannink, Tobias and Kreuzer, Edwin and Rausch, Viktor and Solowjow, Eugen},  
  pages = {6846--6852},
  year  = {2019},
  month = {may},
  publisher = {{IEEE}},
  booktitle = {2019 International Conference on Robotics and Automation (ICRA)}
}
```